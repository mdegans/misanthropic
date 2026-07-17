//! A small, reusable chat *event loop* — in the spirit of `winit`'s loop,
//! but for a conversation.
//!
//! Most chat-shaped programs are the same skeleton: init the tools, then per
//! round seat one user-side beat, run the model to *quiescence* (answer every
//! tool call until the assistant stops calling tools), repeat. [`Chat`] owns
//! all of that — the [`ToolBox`] lifecycle, the tool-dispatch sub-loop,
//! append-only (cache-friendly) prompt mutation, interleaving tool-pushed
//! notifications with the user's input, paused server-tool turns, and
//! teardown-even-on-error. The caller supplies only the part that *varies*:
//! how to read the next line of user input, and (via [`Chat::on_assistant`])
//! what each assistant turn becomes.
//!
//! The driver is generic over its [`Transport`] — an API [`Client`] and a
//! local inference engine drive the same loop.
//!
//! ```ignore
//! Chat::new(client, Prompt::default(), toolbox)
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
//!
//! # Caching
//!
//! Opt in with [`Chat::cache`] and the driver places `cache_control`
//! breakpoints where the transport's [`Quirks`](crate::Quirks) say they pay: canonical
//! Anthropic endpoints get server-side [auto caching]
//! ([`Prompt::auto_cache`] semantics); a transport whose prefix reuse keys
//! on the end-of-assistant render
//! ([`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant))
//! gets a budget-aware rolling window ([`Prompt::cache_windowed_with`])
//! re-marked after each assistant turn; a transport that
//! [ignores markers](crate::Quirks::cache_markers_ignored) gets none. Without the
//! knob the driver stays out of caching entirely — pre-configured markers
//! on the prompt are untouched either way.
//!
//! [auto caching]: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
//! [`Client`]: crate::Client

use std::sync::{Arc, Mutex};

use futures::FutureExt;

use crate::{
    Prompt, Transport,
    prompt::message::{
        AssistantMessage, Block, CacheControl, Content, Message, Role,
        SystemMessage,
    },
    response::{StopReason, TokenCounts},
    tool::{self, Notification, Notifications, Tool, ToolBox, Use},
};

/// Boxed, thread-safe error — matches the [`Tool`] lifecycle-hook error type
/// and any [`Transport::Error`], so both flow through `?` unchanged.
pub type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Default ceiling on consecutive model rounds (tool dispatches and paused
/// server-tool continuations both count) within a single user beat. A runaway
/// model is stopped here; real agents (Claude Code) run uncapped, so override
/// with [`Chat::max_consecutive_tool_calls`]. What happens at the cap is the
/// [`BudgetPolicy`].
pub const DEFAULT_MAX_TOOL_CALLS: usize = 8;

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

/// What the round's `select!` produced — computed first, acted on after, so
/// the racing futures' borrows end before the driver mutates anything.
enum Turn {
    /// The caller's beat: `None` is a graceful stop.
    Beat(Option<Vec<Message>>),
    /// A tool-pushed note: `None` means the channel closed.
    Note(Option<Notification>),
}

/// An append-only chat driver, generic over its [`Transport`] and over a
/// caller-owned `State` threaded through the per-turn closure and the
/// [`Chat::on_assistant`] hook.
///
/// Build it, optionally tune it, then [`run`](Chat::run) it with your `State`
/// and a closure that produces the next user-side beat.
pub struct Chat<State, T: Transport> {
    transport: T,
    prompt: Prompt,
    toolbox: ToolBox,
    max_tool_calls: usize,
    budget_policy: BudgetPolicy,
    /// `Some` while the driver owns cache placement — see [`Chat::cache`].
    /// [`run`](Chat::run) resolves the strategy against the transport's
    /// [`Quirks`](crate::Quirks) once, up front.
    cache: Option<CacheControl>,
    /// Pending-system buffer threaded into [`Prompt::seat`] — see the
    /// module-level notes on system messages.
    pending_system: Option<SystemMessage>,
    #[allow(clippy::type_complexity)]
    on_assistant: Option<
        Box<dyn FnMut(&mut State, AssistantMessage) -> Vec<Message> + Send>,
    >,
    /// Cumulative token-usage sink — see [`track_usage`](Chat::track_usage).
    usage: Option<Arc<Mutex<TokenCounts>>>,
}

impl<State, T: Transport> Chat<State, T> {
    /// A driver for `transport`, seeded with `prompt` and driving `toolbox`.
    /// The `prompt` should *not* carry tools — [`run`](Chat::run) installs
    /// the box's method definitions itself.
    pub fn new(transport: T, prompt: Prompt, toolbox: ToolBox) -> Self {
        Self {
            transport,
            prompt,
            toolbox,
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            budget_policy: BudgetPolicy::default(),
            cache: None,
            pending_system: None,
            on_assistant: None,
            usage: None,
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

    /// Let the driver own `cache_control` placement, quirk-aware — see the
    /// module-level notes on caching. Without this the driver stays out of
    /// caching (the prior behavior: callers pre-configure the prompt).
    pub fn cache(mut self, cache_control: CacheControl) -> Self {
        self.cache = Some(cache_control);
        self
    }

    /// Accumulate every model round's token usage into `sink`. The driver
    /// adds each response's counts as it arrives — including tool-dispatch
    /// rounds the caller never sees — so the sink is the true per-seat cost.
    /// Keep a clone of the `Arc` and read it whenever; [`TokenCounts`] is
    /// `Copy` + `AddAssign` precisely for cheap accumulation.
    pub fn track_usage(mut self, sink: Arc<Mutex<TokenCounts>>) -> Self {
        self.usage = Some(sink);
        self
    }

    /// Add `response`'s counts to the [`track_usage`](Chat::track_usage)
    /// sink, if one is installed. Called at every [`Transport::send`] site.
    fn record_usage(&self, response: &crate::response::Message) {
        if let Some(sink) = &self.usage {
            *sink.lock().expect("usage sink poisoned") += response.usage.counts;
        }
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
        // Resolve the caching strategy against the transport's quirks once.
        // `self.cache` stays `Some` only for the per-assistant-turn windowed
        // marking path; the other strategies act here (or never).
        if let Some(cache_control) = self.cache.take() {
            let quirks = self.transport.quirks();
            if quirks.cache_markers_ignored {
                log::debug!("transport ignores cache markers; placing none");
            } else if quirks.breakpoint_after_assistant {
                self.cache = Some(cache_control);
            } else {
                // Canonical Anthropic: the server places the breakpoint on
                // the last cacheable block at request time.
                self.prompt.cache_control = Some(cache_control);
            }
        }

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
            // buffered for the next round. The result is computed in an inner
            // scope so the racing futures' borrows (`state`, `notifications`)
            // end before the driver acts on it.
            let seated_before = self.prompt.messages.len();
            let turn = {
                let beat = next_beat(state).fuse();
                let note = recv_note(&mut notifications).fuse();
                futures::pin_mut!(beat, note);
                futures::select! {
                    result = beat => Turn::Beat(result?),
                    note = note => Turn::Note(note),
                }
            };
            match turn {
                Turn::Beat(None) => return Ok(()), // graceful stop (Ctrl-D)
                Turn::Beat(Some(beat)) => {
                    for message in beat {
                        self.seat(message)?;
                    }
                }
                // The channel closed (all tools torn down): stop selecting
                // it and carry on with caller input alone.
                Turn::Note(None) => {
                    notifications = None;
                    continue;
                }
                Turn::Note(Some(note)) => {
                    log::debug!("interleaving a tool-pushed notification");
                    self.seat_note(note)?;
                }
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
            let response = self.transport.send(&self.prompt).await?;
            self.record_usage(&response);
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
    ///
    /// When the driver owns caching for a
    /// [`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant)
    /// transport, the seated assistant tail is (re-)marked here with a
    /// 2-deep rolling window — the end-of-assistant render is what such
    /// backends hash, and the second trailing breakpoint is what keeps a
    /// later tail merge re-paying only the last segment.
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

        if let Some(cache_control) = &self.cache {
            self.prompt.cache_windowed_with(2, cache_control.clone());
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
            let response = self.transport.send(&self.prompt).await?;
            self.record_usage(&response);
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
    /// [`TurnOrderError`]: crate::prompt::TurnOrderError
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        response,
        tool::{CustomMethodDef, MethodDef},
        transport::tests::Script,
    };

    /// A tool that answers every call with "echoed" and remembers the calls.
    struct Echo {
        calls: Vec<Use>,
    }

    #[async_trait::async_trait]
    impl Tool for Echo {
        fn name(&self) -> &str {
            "Echo"
        }

        fn definitions(&self) -> Vec<MethodDef> {
            vec![MethodDef::Custom(CustomMethodDef {
                name: "Echo__echo".into(),
                description: "Echo".into(),
                schema: serde_json::json!({"type": "object"}),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })]
        }

        async fn call(&mut self, call: Use) -> tool::Result {
            let id = call.id.clone();
            self.calls.push(call);
            tool::Result::new(id, "echoed")
        }
    }

    fn text_response(text: &str) -> response::Message {
        let inner: AssistantMessage =
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            }))
            .unwrap();
        response::Message::builder("test-model", inner)
            .stop_reason(StopReason::EndTurn)
            .build()
    }

    fn tool_response(call_id: &str) -> response::Message {
        let mut inner: AssistantMessage =
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": "calling"}],
            }))
            .unwrap();
        inner.content.push(
            Use::new("toolbox__Echo__echo", serde_json::json!({}))
                .with_id(call_id.to_string()),
        );
        response::Message::builder("test-model", inner)
            .stop_reason(StopReason::ToolUse)
            .build()
    }

    fn paused_response() -> response::Message {
        let inner: AssistantMessage =
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": "searching…"}],
            }))
            .unwrap();
        response::Message::builder("test-model", inner)
            .stop_reason(StopReason::PauseTurn)
            .build()
    }

    /// A `next_beat` that feeds the given beats in order, then stops.
    fn beats(
        beats: Vec<Vec<Message>>,
    ) -> impl AsyncFnMut(&mut ()) -> Result<Option<Vec<Message>>, BoxError>
    {
        let mut queue: std::collections::VecDeque<_> = beats.into();
        async move |_: &mut ()| Ok(queue.pop_front())
    }

    fn user(text: &str) -> Vec<Message> {
        vec![(Role::User, text).into()]
    }

    #[test]
    fn one_beat_seats_one_assistant_turn() {
        let script = Script::new([text_response("hello")]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new());

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("hi")])))
                .unwrap();

        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(prompt.messages[0].role, Role::User);
        assert_eq!(prompt.messages[1].role, Role::Assistant);
    }

    #[test]
    fn tool_calls_round_trip_through_the_toolbox() {
        let script =
            Script::new([tool_response("call_1"), text_response("done")]);
        let toolbox = ToolBox::new().add(Echo { calls: Vec::new() });
        let chat = Chat::new(script, Prompt::default(), toolbox);

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        // user, assistant(tool_use), user(tool_result), assistant(done)
        assert_eq!(prompt.messages.len(), 4);
        assert_eq!(prompt.messages[2].role, Role::User);
        let result = prompt.messages[2].content.iter().next().unwrap();
        if let Block::ToolResult { result } = result {
            assert!(!result.is_error, "unexpected error result: {result:?}");
        } else {
            panic!("expected a tool result block, got: {result:?}");
        }
    }

    /// At the budget cap every dangling call is answered with a synthetic
    /// `is_error` result and (HandBack) control returns without another
    /// model call — the prompt stays wire-legal.
    #[test]
    fn budget_hand_back_seats_synthetic_errors() {
        let script =
            Script::new([tool_response("call_1"), tool_response("call_2")]);
        let toolbox = ToolBox::new().add(Echo { calls: Vec::new() });
        let chat = Chat::new(script, Prompt::default(), toolbox)
            .max_consecutive_tool_calls(1);

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(matches!(
            last.content.iter().last().unwrap(),
            Block::ToolResult { result } if result.is_error
        ));
    }

    /// FinalWord grants exactly one wrap-up call after the cap.
    #[test]
    fn budget_final_word_makes_one_more_call() {
        let script = Script::new([
            tool_response("call_1"),
            tool_response("call_2"),
            text_response("to summarize: echoed"),
        ]);
        let toolbox = ToolBox::new().add(Echo { calls: Vec::new() });
        let chat = Chat::new(script, Prompt::default(), toolbox)
            .max_consecutive_tool_calls(1)
            .on_budget_exhausted(BudgetPolicy::FinalWord);

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
    }

    /// The wire forbids abandoning an in-flight server tool: exhausting the
    /// budget mid-pause drops the paused turn entirely.
    #[test]
    fn budget_exhausted_mid_pause_drops_the_paused_turn() {
        let script = Script::new([paused_response()]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .max_consecutive_tool_calls(0);

        let (prompt, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("search")])),
        )
        .unwrap();

        // The paused assistant turn is gone; the user's beat is the tail.
        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.messages[0].role, Role::User);
    }

    /// Default quirks + `.cache(…)`: server-side auto placement — the
    /// request-level `cache_control` is set and no message carries a marker.
    #[test]
    fn cache_canonical_uses_auto_cache() {
        let script = Script::new([text_response("hello")]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .cache(CacheControl::ephemeral());

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("hi")])))
                .unwrap();

        assert!(prompt.cache_control.is_some());
        assert!(!prompt.messages.iter().any(|m| m.content.has_cache()));
    }

    /// `breakpoint_after_assistant`: the marker lands on the assistant tail
    /// (the end-of-assistant render is what such backends hash), not on the
    /// request level.
    #[test]
    fn cache_breakpoint_after_assistant_marks_the_tail() {
        let mut script = Script::new([text_response("hello")]);
        script.quirks.breakpoint_after_assistant = true;
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .cache(CacheControl::ephemeral());

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("hi")])))
                .unwrap();

        assert!(prompt.cache_control.is_none());
        let tail = prompt.messages.last().unwrap();
        assert_eq!(tail.role, Role::Assistant);
        assert!(tail.content.has_cache());
    }

    /// `cache_markers_ignored`: the driver places nothing at all.
    #[test]
    fn cache_markers_ignored_places_nothing() {
        let mut script = Script::new([text_response("hello")]);
        script.quirks.cache_markers_ignored = true;
        // Ignored even if the endpoint would also prefer assistant markers.
        script.quirks.breakpoint_after_assistant = true;
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .cache(CacheControl::ephemeral());

        let (prompt, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("hi")])))
                .unwrap();

        assert!(prompt.cache_control.is_none());
        assert!(!prompt.messages.iter().any(|m| m.content.has_cache()));
    }
}
