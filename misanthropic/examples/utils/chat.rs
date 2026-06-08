//! A small, reusable chat *event loop* for the examples — in the spirit of
//! `winit`'s loop, but for a conversation.
//!
//! Most chat-shaped examples are the same skeleton: init the tools, then per
//! round seat one user-side beat, run the model to *quiescence* (answer every
//! tool call until the assistant stops calling tools), repeat. [`Chat`] owns
//! all of that — the [`ToolBox`](misanthropic::tool::ToolBox) lifecycle, the
//! tool-dispatch sub-loop, append-only (cache-friendly) prompt mutation,
//! interleaving tool-pushed notifications with the user's input, and
//! teardown-even-on-error. The example supplies only the part that *varies*:
//! how to read the next line of user input.
//!
//! ```ignore
//! utils::Chat::new(client, Prompt::default(), toolbox)
//!     .on_assistant(move |_state, msg| printer.line(format!("claude ▸ {}", msg.content)))
//!     .run((), async move |_state| {
//!         Ok(lines.recv().await.map(|line| vec![(Role::User, line).into()]))
//!     })
//!     .await?;
//! ```

use misanthropic::{
    Client, Prompt,
    prompt::message::{AssistantMessage, Block, Content, Message, Role},
    tool::{Notification, Notifications, Tool, ToolBox, Use},
};

/// Boxed, thread-safe error — matches the `Tool` lifecycle-hook error type and
/// the crate's `Client` errors, so both flow through `?` unchanged.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Default ceiling on consecutive tool-dispatch rounds within a single user
/// beat. A runaway model that keeps calling tools is stopped here; real agents
/// (Claude Code) run uncapped, so override with
/// [`Chat::max_consecutive_tool_calls`].
const DEFAULT_MAX_TOOL_CALLS: usize = 8;

/// An append-only chat driver, generic over a caller-owned `State` threaded
/// through the per-turn closure and the [`Chat::on_assistant`] display hook.
///
/// Build it, optionally tune it, then [`run`](Chat::run) it with your `State`
/// and a closure that produces the next user-side beat.
pub struct Chat<State> {
    client: Client,
    prompt: Prompt,
    toolbox: ToolBox,
    max_tool_calls: usize,
    #[allow(clippy::type_complexity)]
    on_assistant: Option<Box<dyn FnMut(&mut State, &AssistantMessage) + Send>>,
}

impl<State> Chat<State> {
    /// A driver for `client`, seeded with `prompt` and driving `toolbox`. The
    /// `prompt` should *not* carry tools — [`run`](Chat::run) installs the
    /// box's method definitions itself.
    pub fn new(client: Client, prompt: Prompt, toolbox: ToolBox) -> Self {
        Self {
            client,
            prompt,
            toolbox,
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            on_assistant: None,
        }
    }

    /// Cap consecutive tool-dispatch rounds within one user beat (default
    /// [`DEFAULT_MAX_TOOL_CALLS`]). Hitting the cap ends [`run`](Chat::run)
    /// with an error.
    pub fn max_consecutive_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = max;
        self
    }

    /// Display hook, called with each assistant [`Message`](AssistantMessage)
    /// as the loop runs the model — the example's output side (the input side
    /// is the [`run`](Chat::run) closure). Shares `&mut State` with that
    /// closure, so a counter set in one is visible to the other.
    pub fn on_assistant(
        mut self,
        hook: impl FnMut(&mut State, &AssistantMessage) + Send + 'static,
    ) -> Self {
        self.on_assistant = Some(Box::new(hook));
        self
    }

    /// Drive the conversation until `next_beat` returns `None`, then return the
    /// final [`Prompt`] and `State`.
    ///
    /// `next_beat` produces the next user-side turn(s) — a human line, a
    /// scripted prompt — as `Some(messages)`, or `None` to stop. It owns its
    /// own input source (typically captured by `move`), so the driver stays
    /// I/O-agnostic. Returning several messages seats them in order.
    ///
    /// Tool-pushed notifications are handled by the driver itself: it races
    /// them against `next_beat`, so the losing future is cancelled. Keep
    /// `next_beat` cancel-safe (await a channel `recv`, don't hold
    /// non-restartable state across the await) — the canonical stdin reader is.
    pub async fn run<H>(
        mut self,
        mut state: State,
        next_beat: H,
    ) -> Result<(Prompt, State), BoxError>
    where
        H: AsyncFnMut(&mut State) -> Result<Option<Vec<Message>>, BoxError>,
    {
        // Install the box's method definitions and run each tool's `on_init`.
        self.toolbox.prepare(&mut self.prompt).await?;

        // The driver owns notification interleaving: subscribe to the box once
        // and race pushes against the caller's input inside `drive`.
        let notifications = self.toolbox.subscribe();

        // Drive to completion, then tear down *even on the error path* — async
        // teardown can't ride `Drop`, so we sequence it by hand and don't let
        // it mask the original outcome.
        let outcome = self.drive(&mut state, next_beat, notifications).await;
        let _ = self.toolbox.teardown_tools(&mut self.prompt).await;

        outcome.map(|()| (self.prompt, state))
    }

    /// The loop body: per round, let the tools see the turn, take the next beat
    /// (racing caller input against tool-pushed notifications), then run the
    /// model to quiescence.
    async fn drive<H>(
        &mut self,
        state: &mut State,
        mut next_beat: H,
        mut notifications: Option<Notifications>,
    ) -> Result<(), BoxError>
    where
        H: AsyncFnMut(&mut State) -> Result<Option<Vec<Message>>, BoxError>,
    {
        loop {
            // Tools see the turn first — a push-only tool may drop a
            // notification into its mailbox here, which the `select!` below can
            // then pick up in the same round.
            self.toolbox.update_turn_context(&mut self.prompt).await?;

            // Race the caller's next beat against any tool-pushed notification.
            // The losing future is cancelled; both arms await a cancel-safe
            // channel `recv`, so a beat or note that loses simply stays
            // buffered for the next round.
            tokio::select! {
                result = next_beat(state) => match result? {
                    None => return Ok(()), // graceful stop (e.g. Ctrl-D)
                    Some(beat) => {
                        for message in beat {
                            self.seat(message)?;
                        }
                    }
                },
                note = recv_note(&mut notifications) => match note {
                    // The channel closed (all tools torn down): stop selecting
                    // it and carry on with caller input alone.
                    None => {
                        notifications = None;
                        continue;
                    }
                    Some(note) => {
                        let message = self.note_to_message(note);
                        self.seat(message)?;
                    }
                },
            }

            self.quiesce(state).await?;
        }
    }

    /// Turn a pushed [`Notification`] into a seat-able [`Message`], resolving
    /// its preferred role against the model with a placement guard: a `System`
    /// turn is only legal immediately after a `user` turn (the API also allows
    /// it last, or after a server-tool assistant turn), so if the tail isn't a
    /// `User` we fall back to `User` (always legal) rather than push an
    /// unplaceable turn.
    ///
    /// This fallback honors a `[System, User]` preference list (the tool itself
    /// declared `User` an acceptable fallback). The agreed direction for a
    /// *dedicated* `[System]`-only note with no placeable slot is to **defer**
    /// it and flush it after the next `User` turn — a buffer we'll add when a
    /// System-only pusher exists; downgrade is the placeholder until then.
    fn note_to_message(&self, note: Notification) -> Message {
        let mut role = self.prompt.resolve_role(&note.preferred_roles);
        let tail_is_user = matches!(
            self.prompt.messages.last().map(|message| message.role),
            Some(Role::User)
        );
        if role == Role::System && !tail_is_user {
            role = Role::User;
        }
        (role, note.content).into()
    }

    /// Call the model, answer every tool call, and loop until the assistant
    /// stops calling tools — so the caller's beat is the *last* thing seated
    /// before control returns.
    async fn quiesce(&mut self, state: &mut State) -> Result<(), BoxError> {
        let mut dispatched = 0usize;
        loop {
            let response = self.client.message(&self.prompt).await?;

            if let Some(hook) = self.on_assistant.as_mut() {
                hook(state, &response.inner);
            }

            // Collect *every* client-side tool call (the model may emit several
            // in one turn; server tools carry their own results and just flow
            // through) before the message is moved into the prompt.
            let calls: Vec<Use> = response
                .inner
                .content
                .iter()
                .filter_map(|block| block.tool_use().cloned())
                .collect();

            self.seat(response)?;

            if calls.is_empty() {
                return Ok(()); // assistant is done; back to the caller
            }

            if dispatched >= self.max_tool_calls {
                return Err(format!(
                    "exceeded {} consecutive tool-call rounds",
                    self.max_tool_calls
                )
                .into());
            }
            dispatched += 1;

            // Dispatch each call and seat all results as one user turn.
            let mut results = Vec::with_capacity(calls.len());
            for call in calls {
                results.push(Block::from(self.toolbox.call(call).await));
            }
            self.seat((Role::User, Content(results)))?;
        }
    }

    /// Append `message`, keeping the conversation legal *and* portable to
    /// third-party servers: consecutive same-role turns are concatenated
    /// client-side (exactly what Anthropic does server-side) rather than pushed
    /// as a second turn. So a tool result followed by an injected user note, or
    /// two assistant turns, never trips turn-order validation.
    fn seat(&mut self, message: impl Into<Message>) -> Result<(), BoxError> {
        let message = message.into();
        if let Some(last) = self.prompt.messages.last_mut()
            && last.role == message.role
        {
            last.content.extend(message.content);
            return Ok(());
        }
        self.prompt.push_message(message)?;
        Ok(())
    }
}

/// Await the next notification, or never resolve when there's no notification
/// stream — so it can sit in a `select!` arm whether or not the box pushes.
async fn recv_note(
    notifications: &mut Option<Notifications>,
) -> Option<Notification> {
    match notifications {
        Some(notifications) => notifications.recv().await,
        None => std::future::pending().await,
    }
}
