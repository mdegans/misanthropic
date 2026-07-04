//! A small, reusable chat *event loop* for the examples — in the spirit of
//! `winit`'s loop, but for a conversation.
//!
//! Most chat-shaped examples are the same skeleton: init the tools, then per
//! round seat one user-side beat, run the model to *quiescence* (answer every
//! tool call until the assistant stops calling tools), repeat. [`Chat`] owns
//! all of that — the [`ToolBox`](misanthropic::tool::ToolBox) lifecycle, the
//! tool-dispatch sub-loop, append-only (cache-friendly) prompt mutation,
//! interleaving tool-pushed notifications with the user's input, paused
//! server-tool turns, and teardown-even-on-error. The example supplies only
//! the part that *varies*: how to read the next line of user input, and (via
//! [`Chat::on_assistant`]) what each assistant turn becomes.
//!
//! ```ignore
//! utils::Chat::new(client, Prompt::default(), toolbox)
//!     .on_assistant(move |_state, msg| {
//!         printer.line(format!("claude ▸ {}", msg.content));
//!         [msg.into()] // seat the turn unchanged
//!     })
//!     .run((), async move |_state| {
//!         Ok(lines.recv().await.map(|line| vec![(Role::User, line).into()]))
//!     })
//!     .await?;
//! ```
//!
//! # System messages
//!
//! Operator ([`System`](Role::System)) content — from a tool-pushed
//! [`Notification`] or an [`on_assistant`](Chat::on_assistant) return — is
//! seated through [`Prompt::seat`], the crate's wire-legality kernel. It places
//! the note the moment the tail permits a system turn (after a user turn, or an
//! assistant turn
//! [ending in a server-tool result](Message::ends_in_server_tool_result) — the
//! wire rule, stricter than the docs) and otherwise holds it in the driver's
//! `pending_system` buffer until a later seat opens a slot. It is **never**
//! downgraded to the user role: operator content riding the user channel
//! misattributes authorship and erodes the channel-authority distinction the
//! system role exists to provide. The buffer is the *only* state the driver
//! keeps for this — the seat/merge/buffer legality all lives in the crate.

use misanthropic::{
    Client, Prompt,
    prompt::message::{
        AssistantMessage, Block, Content, Message, Role, SystemMessage,
    },
    response::StopReason,
    tool::{self, Notification, Notifications, Tool, ToolBox, Use},
};

/// Boxed, thread-safe error — matches the `Tool` lifecycle-hook error type and
/// the crate's `Client` errors, so both flow through `?` unchanged.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Default ceiling on consecutive model rounds (tool dispatches and paused
/// server-tool continuations both count) within a single user beat. A runaway
/// model is stopped here; real agents (Claude Code) run uncapped, so override
/// with [`Chat::max_consecutive_tool_calls`]. What happens at the cap is the
/// [`BudgetPolicy`].
const DEFAULT_MAX_TOOL_CALLS: usize = 8;

/// What [`Chat::run`] does when one user beat exhausts
/// [`max_consecutive_tool_calls`](Chat::max_consecutive_tool_calls). Either
/// way, every dangling tool call is answered with a synthetic `is_error`
/// result explaining the situation, so the prompt stays legal and the model
/// learns *why* nothing ran.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BudgetPolicy {
    /// Seat the synthetic results and hand control back to the caller
    /// silently; the model sees the explanation on the next beat.
    #[default]
    HandBack,
    /// Make exactly one more call so the assistant can wrap up verbally. If
    /// that turn calls tools *again*, those are synthetic-errored too and
    /// control is handed back unconditionally.
    FinalWord,
}

/// An append-only chat driver, generic over a caller-owned `State` threaded
/// through the per-turn closure and the [`Chat::on_assistant`] hook.
///
/// Build it, optionally tune it, then [`run`](Chat::run) it with your `State`
/// and a closure that produces the next user-side beat.
pub struct Chat<State> {
    client: Client,
    prompt: Prompt,
    toolbox: ToolBox,
    max_tool_calls: usize,
    budget_policy: BudgetPolicy,
    /// Pending-system buffer threaded into [`Prompt::seat`] — see the
    /// module-level notes on system messages.
    pending_system: Option<SystemMessage>,
    #[allow(clippy::type_complexity)]
    on_assistant: Option<
        Box<dyn FnMut(&mut State, AssistantMessage) -> Vec<Message> + Send>,
    >,
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
            budget_policy: BudgetPolicy::default(),
            pending_system: None,
            on_assistant: None,
        }
    }

    /// Cap consecutive model rounds within one user beat (default
    /// [`DEFAULT_MAX_TOOL_CALLS`]). Hitting the cap triggers the
    /// [`BudgetPolicy`].
    pub fn max_consecutive_tool_calls(mut self, max: usize) -> Self {
        self.max_tool_calls = max;
        self
    }

    /// What to do at the [`max_consecutive_tool_calls`] cap (default
    /// [`BudgetPolicy::HandBack`]).
    ///
    /// [`max_consecutive_tool_calls`]: Chat::max_consecutive_tool_calls
    pub fn on_budget_exhausted(mut self, policy: BudgetPolicy) -> Self {
        self.budget_policy = policy;
        self
    }

    /// The assistant-turn hook: receives each assistant
    /// [`Message`](AssistantMessage) the model produces and returns the
    /// message(s) actually seated — the loop's output side (the input side is
    /// the [`run`](Chat::run) closure). Shares `&mut State` with that closure.
    ///
    /// Return `[msg.into()]` to seat the turn unchanged (display-only hooks),
    /// something else to replace it (redaction, a classifier verdict), extra
    /// messages to append context, or an assistant message carrying
    /// `tool_use` blocks to *force* tool calls — the driver dispatches
    /// whatever client tool calls are in the **seated** assistant turns,
    /// regardless of provenance. A returned [`System`](Role::System) message
    /// goes through [`Prompt::seat`] like any other — seated when the tail
    /// permits, otherwise buffered — never re-attributed to the user role.
    ///
    /// Without a hook the response is seated unchanged.
    pub fn on_assistant<I>(
        mut self,
        mut hook: impl FnMut(&mut State, AssistantMessage) -> I + Send + 'static,
    ) -> Self
    where
        I: IntoIterator<Item = Message>,
    {
        self.on_assistant = Some(Box::new(move |state, msg| {
            hook(state, msg).into_iter().collect()
        }));
        self
    }

    /// Drive the conversation until `next_beat` returns `None`, then return the
    /// final [`Prompt`] and `State`.
    ///
    /// `next_beat` produces the next user-side turn(s) — a human line, a
    /// scripted prompt — as `Some(messages)`, or `None` to stop. It owns its
    /// own input source (typically captured by `move`), so the driver stays
    /// I/O-agnostic. Returning several messages seats them in order; a beat
    /// that seats nothing new (empty, or all-[`System`](Role::System) and thus
    /// buffered) is a no-op round — the model is not called.
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
        if let Err(error) = self.toolbox.teardown_tools(&mut self.prompt).await
        {
            log::warn!("tool teardown failed: {error}");
        }

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
            let seated_before = self.prompt.messages.len();
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
                        log::debug!("interleaving a tool-pushed notification");
                        self.seat_note(note)?;
                    }
                },
            }

            // A beat that seated nothing (all-System → buffered, or empty)
            // gives the model nothing new: don't call it. The buffer flushes
            // with the next beat that does.
            if self.prompt.messages.len() == seated_before {
                continue;
            }

            self.quiesce(state).await?;
        }
    }

    /// Seat a pushed [`Notification`], resolving its preferred role against
    /// the model.
    ///
    /// A note resolving to [`System`](Role::System) goes through
    /// [`Prompt::seat`], which places it as soon as the tail permits and
    /// otherwise buffers it (never on the user channel) to ride the next
    /// request. Either way it does not force a model round on its own — it is
    /// operator context folded into the next call — whereas a
    /// [`User`](Role::User)-resolved note appends a user turn and drives an
    /// immediate round (right for a job completion, versus an operator fact).
    ///
    /// # Panics
    /// A `[System]`-only preference on a model with no system role is a
    /// programming error in the tool itself — there is nothing legal to seat,
    /// ever — so this panics naming the offender rather than silently
    /// re-attributing operator content.
    fn seat_note(&mut self, note: Notification) -> Result<(), BoxError> {
        let role = self.prompt.resolve_role(&note.preferred_roles);
        let downgraded = role != Role::System
            && note.preferred_roles.contains(&Role::System);
        let has_fallback = note
            .preferred_roles
            .iter()
            .any(|r| matches!(r, Role::User | Role::Assistant));
        assert!(
            !downgraded || has_fallback,
            "tool `{}` pushed a [System]-only notification, but model `{}` \
             has no system role — give the tool a fallback role or gate it \
             on Model::supports_system_role",
            note.source,
            self.prompt.model,
        );

        self.seat((role, note.content))
    }

    /// Call the model, answer every tool call, and loop until the assistant
    /// stops calling tools *and* the turn isn't paused on a server tool — so
    /// the caller's beat is the *last* thing seated before control returns.
    async fn quiesce(&mut self, state: &mut State) -> Result<(), BoxError> {
        let mut rounds = 0usize;
        loop {
            log::trace!("quiesce round {rounds}: calling the model");
            // A pending system note was already seated by `seat` the moment a
            // legal tail appeared, so the prompt is request-ready here.
            let response = self.client.message(&self.prompt).await?;
            // `pause_turn` means a server tool is still running: the turn
            // must be continued, even though there's nothing to dispatch.
            let paused =
                matches!(response.stop_reason, Some(StopReason::PauseTurn));

            let calls = self.seat_assistant(state, response.inner)?;

            if calls.is_empty() && !paused {
                return Ok(()); // assistant is done; back to the caller
            }

            if rounds >= self.max_tool_calls {
                if paused {
                    // The wire forbids abandoning an in-flight server tool:
                    // a `server_tool_use` without its result 400s the moment
                    // any turn follows it (verified live — see the
                    // count_tokens placement probes). The only legal exit is
                    // to drop the paused turn entirely; with it go any
                    // continuations merged into it. The policy doesn't get a
                    // FinalWord here — a fresh call could just pause again.
                    log::warn!(
                        "budget exhausted mid-pause: dropping the in-flight \
                         server-tool turn (the wire forbids abandoning it \
                         in place)"
                    );
                    self.prompt.messages.pop();
                    return Ok(());
                }
                return self.exhaust_budget(state, calls).await;
            }
            rounds += 1;

            if !calls.is_empty() {
                log::debug!(
                    "dispatching {} client-side tool call(s)",
                    calls.len()
                );
                self.dispatch(calls).await?;
            }
            // Paused with no client calls: loop — the next request resumes
            // the in-flight server tool.
        }
    }

    /// Run the assistant turn through the [`on_assistant`](Chat::on_assistant)
    /// hook (or seat it unchanged), then collect the client tool calls **from
    /// what was seated** — single source of truth, so a hook that replaces or
    /// redacts the turn naturally governs which tools run.
    fn seat_assistant(
        &mut self,
        state: &mut State,
        message: AssistantMessage,
    ) -> Result<Vec<Use>, BoxError> {
        let seated: Vec<Message> = match self.on_assistant.as_mut() {
            Some(hook) => hook(state, message),
            None => vec![message.into()],
        };
        let calls = seated
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .flat_map(|m| m.content.iter())
            .filter_map(|block| block.tool_use().cloned())
            .collect();
        for message in seated {
            self.seat(message)?;
        }
        Ok(calls)
    }

    /// Dispatch each call through the [`ToolBox`] and seat all results as one
    /// user turn.
    async fn dispatch(&mut self, calls: Vec<Use>) -> Result<(), BoxError> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(Block::from(self.toolbox.call(call).await));
        }
        self.seat((Role::User, Content(results)))
    }

    /// The beat hit [`max_consecutive_tool_calls`]: answer every dangling
    /// call with a synthetic error result (keeping the prompt legal and
    /// telling the model why), then apply the [`BudgetPolicy`].
    ///
    /// [`max_consecutive_tool_calls`]: Chat::max_consecutive_tool_calls
    async fn exhaust_budget(
        &mut self,
        state: &mut State,
        calls: Vec<Use>,
    ) -> Result<(), BoxError> {
        log::warn!(
            "beat exhausted {} consecutive model rounds ({:?})",
            self.max_tool_calls,
            self.budget_policy,
        );
        self.synthesize_results(&calls)?;

        if self.budget_policy == BudgetPolicy::FinalWord && !calls.is_empty() {
            let response = self.client.message(&self.prompt).await?;
            let again = self.seat_assistant(state, response.inner)?;
            // No second chance: error these too and hand back regardless.
            self.synthesize_results(&again)?;
        }
        Ok(())
    }

    /// Seat one user turn of `is_error` results answering `calls` — the
    /// "tool budget exhausted, wait for the user" explanation. No-op when
    /// there are no dangling calls (a paused turn that ran out of budget).
    fn synthesize_results(&mut self, calls: &[Use]) -> Result<(), BoxError> {
        if calls.is_empty() {
            return Ok(());
        }
        let results: Vec<Block> = calls
            .iter()
            .map(|call| {
                Block::from(
                    tool::Result::new(
                        call.id.clone(),
                        format!(
                            "Not run: this turn already used {} consecutive \
                             tool-call rounds (the loop's budget). Stop \
                             calling tools and wait for the user.",
                            self.max_tool_calls
                        ),
                    )
                    .error(),
                )
            })
            .collect();
        self.seat((Role::User, Content(results)))
    }

    /// Append `message` through [`Prompt::seat`] — the crate's wire-legality
    /// kernel. Same-role tails concatenate (portable to strict-alternation
    /// backends); [`System`](Role::System) content seats the moment the tail
    /// permits and otherwise buffers in `pending_system` (never downgraded to
    /// the user channel — see the module-level notes). A merge that would
    /// trail a `tool_result` behind other content, or any other illegal
    /// placement, errors loudly ([`TurnOrderError`]) — a programming error in
    /// the caller's beat or hook.
    ///
    /// [`TurnOrderError`]: misanthropic::prompt::TurnOrderError
    fn seat(&mut self, message: impl Into<Message>) -> Result<(), BoxError> {
        self.prompt.seat(message, &mut self.pending_system)?;
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
