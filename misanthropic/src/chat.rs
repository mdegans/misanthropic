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
//! Each response is classified by its [`Disposition`], and client tool calls
//! run only from a complete [`ToolUse`](Disposition::ToolUse) turn (or a
//! [paused](Disposition::Paused) one). A turn the driver can't use is never
//! seated and its calls never run — it stops with an [`Error`] that hands
//! everything back (the [`Prompt`], the tools, the configuration, the
//! `State`), so the caller can adjust and [resume](Error::resume):
//!
//! - [`Clipped`](Stop::Clipped) (`max_tokens`): its tool calls can be valid
//!   JSON missing arguments the model never emitted.
//! - [`Unusable`](Stop::Unusable): a finished turn — a `refusal` above all —
//!   that still calls client tools or leaves a server tool in flight. A
//!   refusal can cut either short.
//!
//! Handing back never strands an in-flight paused turn.
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
//! keeps for this — the seat/merge/buffer legality all lives in the crate —
//! and a hand-back returns it ([`Parts::pending`]) rather than drop a note.
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
    prompt::{
        Seated, TurnOrderError,
        message::{
            AssistantMessage, Block, CacheControl, Content, Message, Role,
            SystemMessage,
        },
    },
    response::{self, Disposition, TokenCounts},
    tool::{self, Notification, Notifications, Tool, ToolBox, Use},
    utils::cold_path,
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
#[non_exhaustive]
pub enum BudgetPolicy {
    /// Seat the synthetic results and hand control back to the caller
    /// silently; the model sees the explanation on the next beat.
    #[default]
    HandBack,
    /// Make exactly one more call, with `tool_choice: none`, so the
    /// assistant wraps up in words. If that turn calls tools anyway (a
    /// backend that ignores `tool_choice`), those are synthetic-errored too
    /// and control is handed back unconditionally.
    FinalWord,
}

/// Why [`Chat::run`] stopped early — the [`kind`](Error::kind) of an
/// [`Error`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Stop {
    /// A [`Clipped`](Disposition::Clipped) (`max_tokens`) turn. Its tool calls
    /// may be missing arguments the model never emitted, so nothing was
    /// seated or run: the prompt is exactly as sent. Raise
    /// [`max_tokens`](Prompt::max_tokens) (or ask for brevity) and resume.
    #[error(
        "the model's turn was clipped at max_tokens; nothing was seated \
         (raise max_tokens and resume)"
    )]
    Clipped(Box<response::Message>),
    /// A finished ([`Done`](Disposition::Done)) turn — a `refusal` above all
    /// — that still calls client tools, or leaves a server tool in flight
    /// (its own, or the paused turn's it continued): a refusal can cut
    /// either short. Nothing runs and the whole turn is dropped (with any
    /// paused turn it continued), never stripped — stripping could strand a
    /// `server_tool_use`. Inspect its
    /// [`stop_reason`](response::Message::stop_reason) and resume.
    #[error(
        "the model finished its turn ({:?}) with a tool call unanswerable; \
         nothing ran",
        .0.stop_reason
    )]
    Unusable(Box<response::Message>),
    /// [`Transport::send`] failed; the prompt is as sent.
    #[error(transparent)]
    Transport(BoxError),
    /// The [`run`](Chat::run) closure returned an error.
    #[error(transparent)]
    Beat(BoxError),
    /// A [`Tool`] lifecycle hook (init or turn context) failed.
    #[error(transparent)]
    Tool(BoxError),
    /// A beat, hook return or notification broke turn order — a programming
    /// error in the caller. A beat or a hook's return is seated whole or not
    /// at all, so the prompt is as it was before it.
    #[error(transparent)]
    TurnOrder(#[from] TurnOrderError),
}

/// A [`Chat`] without its transport — what [`run`](Chat::run) hands back
/// (inside an [`Error`] when it stops early), and what [`Chat::from_parts`]
/// resumes from, configured (hook, budget, caching, usage sink) as the
/// `Chat` that made them.
pub struct Parts<State = ()> {
    /// The conversation — see [`Stop`] for what an early stop dropped.
    pub prompt: Prompt,
    /// System content the tail doesn't admit yet (see the module-level
    /// notes). A `Chat` from these parts seats it after the next beat.
    pub pending: Option<SystemMessage>,
    /// The tools, torn down. A `Chat` from these parts prepares them again —
    /// a reload, so [`on_init`](Tool::on_init) runs once per run — and
    /// delivers the notifications they pushed in between.
    pub toolbox: ToolBox,
    config: Config<State>,
}

impl<State> std::fmt::Debug for Parts<State> {
    /// [`Prompt`]'s own `Debug` hides the conversation; the rest is opaque.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parts")
            .field("prompt", &self.prompt)
            .finish_non_exhaustive()
    }
}

/// [`Chat::run`] stopped early — see [`Stop`]. Hands back everything a
/// resume needs: the [`Prompt`], legal to resend; the pending system notes;
/// the `State`; and the rest of the [`Parts`]. [`resume`](Error::resume)
/// answers the prompt before asking for the next beat.
///
/// ```
/// use std::num::NonZeroU32;
///
/// use misanthropic::{
///     Prompt, Transport,
///     chat::{BoxError, Chat, Stop},
///     prompt::message::{Message, Role},
///     tool::ToolBox,
/// };
///
/// /// Chat through `lines`, doubling `max_tokens` whenever a turn clips.
/// async fn converse<T: Transport + Clone>(
///     transport: T,
///     lines: Vec<&str>,
/// ) -> Result<Prompt, BoxError> {
///     let mut lines = lines.into_iter();
///     let mut next_beat = async |_: &mut ()| {
///         let line = lines.next();
///         let beat = line.map(|l| vec![Message::from((Role::User, l))]);
///         Ok::<_, BoxError>(beat)
///     };
///
///     let prompt = Prompt::default();
///     let mut chat = Chat::new(transport.clone(), prompt, ToolBox::new());
///     loop {
///         let mut error = match chat.run((), &mut next_beat).await {
///             Ok((parts, ())) => return Ok(parts.prompt),
///             Err(error) => error,
///         };
///         // Anything else still converts with `?` (or `.into()`).
///         let Stop::Clipped(_) = error.kind else {
///             return Err(error.into());
///         };
///         // Nothing was seated or run: raise the limit and resume.
///         let two = NonZeroU32::new(2).unwrap();
///         let max_tokens = error.prompt.max_tokens.saturating_mul(two);
///         error.prompt.max_tokens = max_tokens;
///         (chat, _) = error.resume(transport.clone());
///     }
/// }
/// ```
pub struct Error<State = ()> {
    /// Why the driver stopped.
    pub kind: Stop,
    /// The prompt, legal to resend as is — see [`Stop`] for what (if
    /// anything) was dropped.
    pub prompt: Prompt,
    /// See [`Parts::pending`].
    pub pending: Option<SystemMessage>,
    /// The caller's state, as the driver last left it.
    pub state: State,
    /// The rest of the [`Parts`]. A `Mutex` — never locked — only so an
    /// `Error` stays `Sync`, as a [`BoxError`] must be: neither a
    /// [`ToolBox`] nor the hook is.
    rest: Mutex<(ToolBox, Config<State>)>,
}

impl<State> Error<State> {
    fn new(kind: Stop, parts: Parts<State>, state: State) -> Self {
        let Parts {
            prompt,
            pending,
            toolbox,
            config,
        } = parts;
        Self {
            kind,
            prompt,
            pending,
            state,
            rest: Mutex::new((toolbox, config)),
        }
    }

    /// See [`Parts::toolbox`].
    pub fn toolbox_mut(&mut self) -> &mut ToolBox {
        let rest = self.rest.get_mut();
        &mut rest.unwrap_or_else(std::sync::PoisonError::into_inner).0
    }

    /// The [`Parts`] (with any edits made to `prompt` or `pending`) and the
    /// state.
    pub fn into_parts(self) -> (Parts<State>, State) {
        let rest = self.rest.into_inner();
        let (toolbox, config) =
            rest.unwrap_or_else(std::sync::PoisonError::into_inner);
        let parts = Parts {
            prompt: self.prompt,
            pending: self.pending,
            toolbox,
            config,
        };
        (parts, self.state)
    }

    /// A [`Chat`] over `transport` that picks up where this one stopped —
    /// [`Chat::from_parts`] on [`into_parts`](Self::into_parts) — with the
    /// state to [`run`](Chat::run) it with.
    pub fn resume<T: Transport>(self, transport: T) -> (Chat<State, T>, State) {
        let (parts, state) = self.into_parts();
        (Chat::from_parts(transport, parts), state)
    }
}

impl<State> std::fmt::Debug for Error<State> {
    /// `state` is the caller's (and needn't be `Debug`); [`Prompt`]'s own
    /// `Debug` hides the conversation.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Error")
            .field("kind", &self.kind)
            .field("prompt", &self.prompt)
            .finish_non_exhaustive()
    }
}

impl<State> std::fmt::Display for Error<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}

impl<State> std::error::Error for Error<State> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Display is the kind's, so the chain continues below it.
        std::error::Error::source(&self.kind)
    }
}
// `?` into a `BoxError` needs both.
static_assertions::assert_impl_all!(Error<()>: Send, Sync);

/// The prompt's length and tail, and the pending buffer, before a seating
/// that may need undoing.
struct Checkpoint {
    len: usize,
    tail: Option<Message>,
    pending: Option<SystemMessage>,
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
    /// Pending-system buffer threaded into [`Prompt::seat`] — see the
    /// module-level notes on system messages.
    pending_system: Option<SystemMessage>,
    toolbox: ToolBox,
    config: Config<State>,
}

/// The `on_assistant` hook, boxed.
type Hook<State> =
    Box<dyn FnMut(&mut State, AssistantMessage) -> Vec<Message> + Send>;

/// A [`Chat`]'s tuning, carried through its [`Parts`] so a resumed run is
/// configured as the one that stopped.
struct Config<State> {
    max_tool_calls: usize,
    budget_policy: BudgetPolicy,
    /// `Some` while the driver owns cache placement — see [`Chat::cache`].
    /// [`run`](Chat::run) resolves the strategy against the transport's
    /// [`Quirks`](crate::Quirks) once, up front.
    cache: Option<CacheControl>,
    on_assistant: Option<Hook<State>>,
    /// Cumulative token-usage sink — see [`track_usage`](Chat::track_usage).
    usage: Option<Arc<Mutex<TokenCounts>>>,
}

impl<State> Default for Config<State> {
    fn default() -> Self {
        Self {
            max_tool_calls: DEFAULT_MAX_TOOL_CALLS,
            budget_policy: BudgetPolicy::default(),
            cache: None,
            on_assistant: None,
            usage: None,
        }
    }
}

impl<State, T: Transport> Chat<State, T> {
    /// A driver for `transport`, seeded with `prompt` and driving `toolbox`.
    /// The `prompt` should *not* carry tools — [`run`](Chat::run) installs
    /// the box's method definitions itself.
    pub fn new(transport: T, prompt: Prompt, toolbox: ToolBox) -> Self {
        Self {
            transport,
            prompt,
            pending_system: None,
            toolbox,
            config: Config::default(),
        }
    }

    /// A driver for `transport` that resumes `parts` — what a
    /// [`run`](Chat::run) handed back — configured as the `Chat` that made
    /// them. A prompt that awaits the model is answered first.
    pub fn from_parts(transport: T, parts: Parts<State>) -> Self {
        let Parts {
            prompt,
            pending,
            toolbox,
            config,
        } = parts;
        Self {
            transport,
            prompt,
            pending_system: pending,
            toolbox,
            config,
        }
    }

    /// Cap consecutive model rounds within one user beat (default
    /// [`DEFAULT_MAX_TOOL_CALLS`]). Hitting the cap triggers the
    /// [`BudgetPolicy`].
    pub fn max_consecutive_tool_calls(mut self, max: usize) -> Self {
        self.config.max_tool_calls = max;
        self
    }

    /// What to do at the [`max_consecutive_tool_calls`] cap (default
    /// [`BudgetPolicy::HandBack`]).
    ///
    /// [`max_consecutive_tool_calls`]: Chat::max_consecutive_tool_calls
    pub fn on_budget_exhausted(mut self, policy: BudgetPolicy) -> Self {
        self.config.budget_policy = policy;
        self
    }

    /// Let the driver own `cache_control` placement, quirk-aware — see the
    /// module-level notes on caching. Without this the driver stays out of
    /// caching (the prior behavior: callers pre-configure the prompt).
    pub fn cache(mut self, cache_control: CacheControl) -> Self {
        self.config.cache = Some(cache_control);
        self
    }

    /// Accumulate every model round's token usage into `sink`. The driver
    /// adds each response's counts as it arrives — including tool-dispatch
    /// rounds the caller never sees — so the sink is the true per-seat cost.
    /// Keep a clone of the `Arc` and read it whenever; [`TokenCounts`] is
    /// `Copy` + `AddAssign` precisely for cheap accumulation.
    pub fn track_usage(mut self, sink: Arc<Mutex<TokenCounts>>) -> Self {
        self.config.usage = Some(sink);
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
    /// regardless of provenance. (A turn the driver can't use — see [`Stop`]
    /// — never reaches the hook.) A returned [`System`](Role::System) message
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
        self.config.on_assistant = Some(Box::new(move |state, msg| {
            hook(state, msg).into_iter().collect()
        }));
        self
    }

    /// Drive the conversation until `next_beat` returns `None`, then hand
    /// back the [`Parts`] — the final [`Prompt`] among them — and `State`.
    ///
    /// A seeded `prompt` that awaits the model — ending in a user or system
    /// turn, or a paused one — is answered first; that is how a run resumes
    /// after an [`Error`].
    ///
    /// `next_beat` produces the next user-side turn(s) — a human line, a
    /// scripted prompt — as `Some(messages)`, or `None` to stop. It owns its
    /// own input source (typically captured by `move`; pass `&mut` to keep it
    /// across a resume), so the driver stays I/O-agnostic. Returning several
    /// messages seats them in order; a beat that seats nothing new (empty, or
    /// all-[`System`](Role::System) and thus buffered) is a no-op round — the
    /// model is not called.
    ///
    /// Tool-pushed notifications are handled by the driver itself: it races
    /// them against `next_beat`, so the losing future is cancelled, and a note
    /// that reaches the prompt drives a round as a beat does (see
    /// [`Notification`] for the buffered system case). Keep
    /// `next_beat` cancel-safe (await a channel `recv`, don't hold
    /// non-restartable state across the await) — the canonical stdin reader is.
    ///
    /// The [`ToolBox`] is prepared at the start of every run and torn down at
    /// the end, whatever the outcome: async teardown can't ride `Drop`, so a
    /// caller that gives up on an [`Error`] leaks nothing.
    ///
    /// # Errors
    /// An [`Error`] handing everything back for a resume — see [`Stop`].
    // The `Err` arm is large because it hands the `Parts` back — as the `Ok`
    // arm does.
    #[allow(clippy::result_large_err)]
    pub async fn run<H>(
        mut self,
        mut state: State,
        next_beat: H,
    ) -> Result<(Parts<State>, State), Error<State>>
    where
        H: AsyncFnMut(&mut State) -> Result<Option<Vec<Message>>, BoxError>,
    {
        self.resolve_cache();

        // The driver owns notification interleaving: subscribe to the box once
        // (a resumed box hands back the stream a previous run parked) and race
        // pushes against the caller's input inside `drive`.
        let mut notifications = self.toolbox.subscribe();

        // Install the box's method definitions and run each tool's `on_init`,
        // then drive to completion.
        let outcome = match self.toolbox.prepare(&mut self.prompt).await {
            Ok(()) => {
                self.drive(&mut state, next_beat, &mut notifications).await
            }
            Err(error) => {
                cold_path();
                Err(Stop::Tool(error))
            }
        };

        // Tear down *even on the error path*, without letting it mask the
        // outcome. The stream goes back to the box, pushes still queued in
        // it included, for a `Chat` from the parts to deliver.
        if let Err(error) = self.toolbox.teardown_tools(&mut self.prompt).await
        {
            log::warn!("tool teardown failed: {error}");
        }
        if let Some(notifications) = notifications {
            self.toolbox.park(notifications);
        }

        let parts = Parts {
            prompt: self.prompt,
            pending: self.pending_system,
            toolbox: self.toolbox,
            config: self.config,
        };
        match outcome {
            Ok(()) => Ok((parts, state)),
            Err(kind) => {
                cold_path();
                Err(Error::new(kind, parts, state))
            }
        }
    }

    /// Resolve the caching strategy against the transport's quirks, once per
    /// run: `config.cache` stays `Some` only for the per-assistant-turn
    /// windowed marking; the other strategies act here (or never).
    fn resolve_cache(&mut self) {
        let Some(cache_control) = self.config.cache.take() else {
            return;
        };
        let quirks = self.transport.quirks();
        if quirks.cache_markers_ignored {
            log::debug!("transport ignores cache markers; placing none");
        } else if quirks.breakpoint_after_assistant {
            self.config.cache = Some(cache_control);
        } else {
            // Canonical Anthropic: the server places the breakpoint on the
            // last cacheable block at request time.
            self.prompt.cache_control = Some(cache_control);
        }
    }

    /// The loop body: per round, let the tools see the turn, take the next beat
    /// (racing caller input against tool-pushed notifications), then run the
    /// model to quiescence.
    async fn drive<H>(
        &mut self,
        state: &mut State,
        mut next_beat: H,
        notifications: &mut Option<Notifications>,
    ) -> Result<(), Stop>
    where
        H: AsyncFnMut(&mut State) -> Result<Option<Vec<Message>>, BoxError>,
    {
        // A seeded prompt awaiting the model (a resume) is answered before
        // the first beat — once: a later user tail (a budget hand-back's
        // synthetic results) waits for the caller.
        let mut resume = self.prompt.messages.last().is_some_and(awaits_model);
        loop {
            // Tools see the turn first — a push-only tool may drop a
            // notification into its mailbox here, which the `select!` below can
            // then pick up in the same round.
            self.toolbox
                .update_turn_context(&mut self.prompt)
                .await
                .map_err(Stop::Tool)?;

            if std::mem::take(&mut resume) {
                self.quiesce(state).await?;
                continue;
            }

            // Race the caller's next beat against any tool-pushed notification.
            // The losing future is cancelled; both arms await a cancel-safe
            // channel `recv`, so a beat or note that loses simply stays
            // buffered for the next round. The result is computed in an inner
            // scope so the racing futures' borrows (`state`, `notifications`)
            // end before the driver acts on it.
            let turn = {
                let beat = next_beat(state).fuse();
                let note = recv_note(notifications).fuse();
                futures::pin_mut!(beat, note);
                futures::select! {
                    result = beat => Turn::Beat(result.map_err(Stop::Beat)?),
                    note = note => Turn::Note(note),
                }
            };
            // Whether anything reached the prompt — a merge into the tail
            // counts (lengths don't show it), a buffered system note doesn't.
            let advanced = match turn {
                Turn::Beat(None) => return Ok(()), // graceful stop (Ctrl-D)
                Turn::Beat(Some(beat)) => self.seat_all(beat)?,
                // The channel closed (all tools torn down): stop selecting
                // it and carry on with caller input alone.
                Turn::Note(None) => {
                    *notifications = None;
                    continue;
                }
                Turn::Note(Some(note)) => {
                    log::debug!("interleaving a tool-pushed notification");
                    self.seat_note(note)?.advanced()
                }
            };

            // A beat that seated nothing (all-System → buffered, or empty)
            // gives the model nothing new: don't call it. The buffer flushes
            // with the next beat that does.
            if !advanced {
                continue;
            }

            self.quiesce(state).await?;
        }
    }

    /// Seat a pushed [`Notification`], resolving its preferred role against
    /// the model.
    ///
    /// A note that reaches the prompt drives a model round, as a beat does —
    /// a [`User`](Role::User)-resolved note always (a job completion, a
    /// letter: `swarm`'s workers run on nothing else). A
    /// [`System`](Role::System) note goes through [`Prompt::seat`]: seated as
    /// soon as the tail permits (and then it drives a round too), otherwise
    /// buffered — never on the user channel — to ride the next beat's request
    /// without a round of its own. After an assistant turn, that's the
    /// common case.
    ///
    /// # Panics
    /// A `[System]`-only preference on a model with no system role is a
    /// programming error in the tool itself — there is nothing legal to seat,
    /// ever — so this panics naming the offender rather than silently
    /// re-attributing operator content.
    fn seat_note(&mut self, note: Notification) -> Result<Seated, Stop> {
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
    ///
    /// Client calls run only from a [`ToolUse`](Disposition::ToolUse) or
    /// [`Paused`](Disposition::Paused) turn; any other turn carrying them is
    /// a [`Stop`], never seated (see `unusable`).
    ///
    /// Whatever it returns, a round never leaves a legal prompt illegal: on
    /// `Ok` the caller's next beat can follow it, and on a [`Stop`] a resume
    /// can. Debug builds assert that at every exit.
    async fn quiesce(&mut self, state: &mut State) -> Result<(), Stop> {
        let was_legal =
            cfg!(debug_assertions) && self.prompt.check_turn_order().is_ok();
        let outcome = self.rounds(state).await;
        debug_assert!(
            !was_legal
                || match outcome {
                    Ok(()) => takes_a_beat(&self.prompt),
                    Err(_) => resumable(&self.prompt),
                },
            "Chat left a prompt the caller can't carry on from ({:?})",
            self.prompt.check_turn_order().err()
        );
        outcome
    }

    /// The body of [`quiesce`](Self::quiesce): model rounds until the turn
    /// settles or stops.
    async fn rounds(&mut self, state: &mut State) -> Result<(), Stop> {
        let mut rounds = 0usize;
        // Where the in-flight paused turn sits, while the last seated turn
        // paused — a hand-back must drop it whole. A resumed prompt may start
        // on one.
        let mut paused_at = self.paused_tail_start();
        loop {
            log::trace!("quiesce round {rounds}: calling the model");
            // A pending system note was already seated by `seat` the moment a
            // legal tail appeared, so the prompt is request-ready here.
            let response = self.send().await?;

            // `pause_turn` means a server tool is still running: the turn
            // must be continued, even though there's nothing to dispatch.
            let paused = match response.disposition() {
                Disposition::Clipped => {
                    cold_path();
                    log::warn!(
                        "turn clipped at max_tokens = {}: handing back \
                         without seating it",
                        self.prompt.max_tokens
                    );
                    return Err(Stop::Clipped(Box::new(response)));
                }
                Disposition::Done if self.unusable_done(&response) => {
                    return Err(self.unusable(response, paused_at));
                }
                Disposition::Paused => true,
                Disposition::ToolUse | Disposition::Done => false,
            };

            // This round's turn starts here: a continuation merges into an
            // assistant tail; anything else appends.
            let seated_from = self.prompt.messages.len()
                - usize::from(
                    self.prompt
                        .messages
                        .last()
                        .is_some_and(|m| m.role == Role::Assistant),
                );
            let calls = self.seat_assistant(state, response.inner)?;
            if calls.is_empty() && !paused {
                // The assistant is done; back to the caller.
                self.settle(paused_at);
                return Ok(());
            }

            // The paused turn starts where it was first seated — a later
            // continuation (a separate turn after a flushed note) doesn't
            // move it — and only this round's seating counts, so a hook that
            // redacts the turn can't point it at an earlier beat.
            paused_at = paused
                .then(|| {
                    paused_at.or_else(|| {
                        self.prompt
                            .messages
                            .iter()
                            .enumerate()
                            .skip(seated_from)
                            .find(|(_, m)| m.role == Role::Assistant)
                            .map(|(at, _)| at)
                    })
                })
                .flatten();

            if rounds >= self.config.max_tool_calls {
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
                    self.restore_tail(paused_at);
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

    /// Where the paused turn a seeded prompt ends on starts — its tail an
    /// assistant turn awaiting a server tool — so a hand-back can drop it
    /// whole: the first assistant turn after the last user turn (a note
    /// flushed mid-pause may split it).
    fn paused_tail_start(&self) -> Option<usize> {
        let messages = &self.prompt.messages;
        let tail = messages.last()?;
        if tail.role != Role::Assistant
            || tail.unfinished_server_tool_uses().next().is_none()
        {
            return None;
        }
        let after_user = messages
            .iter()
            .rposition(|m| m.role == Role::User)
            .map_or(0, |at| at + 1);
        messages[after_user..]
            .iter()
            .position(|m| m.role == Role::Assistant)
            .map(|at| after_user + at)
    }

    /// One model call, its usage recorded. (`&mut self`: a `&self` held
    /// across the await would need `Chat: Sync` for the future to be `Send`.)
    async fn send(&mut self) -> Result<response::Message, Stop> {
        let response =
            self.transport.send(&self.prompt).await.map_err(|error| {
                cold_path();
                Stop::Transport(Box::new(error))
            })?;
        self.record_usage(&response);
        Ok(response)
    }

    /// Add `response`'s counts to the [`track_usage`](Chat::track_usage)
    /// sink, if one is installed.
    fn record_usage(&self, response: &response::Message) {
        if let Some(sink) = &self.config.usage {
            *sink.lock().expect("usage sink poisoned") += response.usage.counts;
        }
    }

    /// Whether a finished `response` can't be seated: it calls client
    /// tools, or leaves a server tool in flight — one of its own, or the
    /// paused tail's it continues (merging into it). Either way the wire
    /// would have no answer for the call.
    fn unusable_done(&self, response: &response::Message) -> bool {
        let tail = self
            .prompt
            .messages
            .last()
            .filter(|tail| tail.role == Role::Assistant);
        let blocks = tail
            .into_iter()
            .flat_map(|tail| tail.content.iter())
            .chain(response.inner.content.iter())
            .cloned()
            .collect();
        let turn = Message::from((Role::Assistant, Content(blocks)));
        calls_tools(response) || turn.unfinished_server_tool_uses().count() > 0
    }

    /// A finished turn [the driver can't seat](Self::unusable_done): nothing
    /// runs, and the whole turn goes — with the paused turn it continued
    /// (from `paused_at`), so the prompt ends where the caller left it.
    /// System notes seated inside that turn are kept.
    fn unusable(
        &mut self,
        response: response::Message,
        paused_at: Option<usize>,
    ) -> Stop {
        cold_path();
        log::warn!(
            "turn finished ({:?}) with a tool call unanswerable: running \
             nothing, handing back without seating it",
            response.stop_reason
        );
        let notes = self.drop_paused_turn(paused_at);
        self.rebuffer(notes);
        // After the drop the tail is the caller's (a user or system turn), so
        // the rescued notes seat at once.
        if let Some(note) = self.pending_system.take()
            && let Err(error) = self.seat(note)
        {
            return error;
        }
        Stop::Unusable(Box::new(response))
    }

    /// Leave a tail the caller's next beat can legally follow when handing
    /// back without seating the last response.
    ///
    /// - An in-flight paused turn (starting at `paused_at`) is dropped whole:
    ///   the wire forbids abandoning a server tool in place. System notes
    ///   seated inside it are kept.
    /// - A trailing [`System`](Role::System) turn (seated right before the
    ///   model call) is taken off too: only an assistant turn may follow
    ///   one, so the next user beat would be a [`BadTransition`].
    ///
    /// Both go back to the front of `pending_system`, and [`Prompt::seat`]
    /// re-places them after the next beat — the same buffering any note gets
    /// while the tail forbids it.
    ///
    /// [`BadTransition`]: crate::prompt::TurnOrderError::BadTransition
    fn restore_tail(&mut self, paused_at: Option<usize>) {
        let dropped = self.drop_paused_turn(paused_at);
        let tail = self.prompt.messages.pop_if(|m| m.role == Role::System);
        // The trailing note was seated before the paused turn began.
        self.rebuffer(tail.into_iter().flat_map(|m| m.content).chain(dropped));
    }

    /// Leave a tail the caller's next beat can follow when the beat ends. A
    /// round that seats nothing new — a bare refusal, an empty `end_turn`, a
    /// hook returning nothing or only a note — leaves the tail as the round
    /// found it: a system note seated right before the call, or (a hook
    /// dropping the continuation) the paused turn starting at `paused_at`.
    /// Both go back to `pending_system` — see `restore_tail`.
    fn settle(&mut self, paused_at: Option<usize>) {
        let in_flight = self.prompt.messages.last().is_some_and(|tail| {
            tail.role == Role::Assistant
                && tail.unfinished_server_tool_uses().next().is_some()
        });
        self.restore_tail(paused_at.filter(|_| in_flight));
    }

    /// Truncate the paused turn starting at `at` (continuations, dispatched
    /// results and all), returning the system notes seated inside it.
    fn drop_paused_turn(&mut self, at: Option<usize>) -> Vec<Block> {
        at.map(|at| self.prompt.messages.split_off(at))
            .unwrap_or_default()
            .into_iter()
            .filter(|m| m.role == Role::System)
            .flat_map(|m| m.content)
            .collect()
    }

    /// Put `notes` back at the front of `pending_system` — they were seated
    /// before anything still buffered.
    fn rebuffer(&mut self, notes: impl IntoIterator<Item = Block>) {
        let notes: Vec<Block> = notes
            .into_iter()
            .chain(
                self.pending_system
                    .take()
                    .into_iter()
                    .flat_map(|p| p.content),
            )
            .collect();
        self.pending_system =
            (!notes.is_empty()).then(|| SystemMessage::from(Content(notes)));
    }

    /// Run the assistant turn through the [`on_assistant`](Chat::on_assistant)
    /// hook (or seat it unchanged), then collect the client tool calls **from
    /// what was seated** — single source of truth, so a hook that replaces or
    /// redacts the turn naturally governs which tools run. A turn with no
    /// content — a bare refusal, an empty `end_turn` — is not seated: the
    /// API rejects an empty turn, so the next request would fail.
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
    ) -> Result<Vec<Use>, Stop> {
        let seated: Vec<Message> = match self.config.on_assistant.as_mut() {
            Some(hook) => hook(state, message),
            None => vec![message.into()],
        }
        .into_iter()
        .filter(|m| !m.content.is_empty())
        .collect();
        let calls = seated
            .iter()
            .filter(|m| m.role == Role::Assistant)
            .flat_map(|m| m.tool_uses().cloned())
            .collect();
        self.seat_all(seated)?;

        if let Some(cache_control) = &self.config.cache {
            self.prompt.cache_windowed_with(2, cache_control.clone());
        }

        Ok(calls)
    }

    /// Seat `messages` in order, all or none: a beat's or a hook's turns
    /// that break turn order part-way mustn't leave the earlier ones seated
    /// (a `tool_use` turn nothing answers, a system turn no beat may follow).
    /// Returns whether any reached the prompt — see [`Seated::advanced`].
    fn seat_all(&mut self, messages: Vec<Message>) -> Result<bool, Stop> {
        // A single message seats atomically already.
        let checkpoint = (messages.len() > 1).then(|| self.checkpoint());
        let outcome =
            messages.into_iter().try_fold(false, |advanced, message| {
                Ok::<_, Stop>(self.seat(message)?.advanced() || advanced)
            });
        if let (Err(error), Some(checkpoint)) = (&outcome, checkpoint) {
            cold_path();
            log::warn!("rolling back turns seated out of order: {error}");
            self.rollback(checkpoint);
        }
        outcome
    }

    /// Enough to undo one round's seating — see [`Checkpoint`].
    fn checkpoint(&self) -> Checkpoint {
        Checkpoint {
            len: self.prompt.messages.len(),
            tail: self.prompt.messages.last().cloned(),
            pending: self.pending_system.clone(),
        }
    }

    /// Put the prompt and the pending buffer back as `checkpoint` found them.
    fn rollback(&mut self, checkpoint: Checkpoint) {
        self.prompt.messages.truncate(checkpoint.len);
        if let (Some(last), Some(tail)) =
            (self.prompt.messages.last_mut(), checkpoint.tail)
        {
            *last = tail; // a merge edits the tail in place
        }
        self.pending_system = checkpoint.pending;
    }

    /// Dispatch each call through the [`ToolBox`] and seat all results as one
    /// user turn.
    async fn dispatch(&mut self, calls: Vec<Use>) -> Result<(), Stop> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(Block::from(self.toolbox.call(call).await));
        }
        self.seat((Role::User, Content(results))).map(drop)
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
    ) -> Result<(), Stop> {
        log::warn!(
            "beat exhausted {} consecutive model rounds ({:?})",
            self.config.max_tool_calls,
            self.config.budget_policy,
        );
        self.synthesize_results(&calls)?;

        if self.config.budget_policy == BudgetPolicy::FinalWord
            && !calls.is_empty()
        {
            self.final_word(state).await?;
        }
        // Seating the results may have flushed a buffered system note.
        self.restore_tail(None);
        Ok(())
    }

    /// [`BudgetPolicy::FinalWord`]'s one wrap-up call, sent with
    /// [`tool_choice: none`](tool::Choice::None) so the model answers in
    /// words (the prompt's own `tool_choice` is restored after). A transport
    /// that [doesn't honor it](crate::Quirks::tool_choice_not_respected) gets
    /// the prompt unchanged. Calls the wrap-up makes anyway are answered with
    /// synthetic errors — no second chance.
    async fn final_word(&mut self, state: &mut State) -> Result<(), Stop> {
        let honored = !self.transport.quirks().tool_choice_not_respected;
        let choice = honored
            .then(|| self.prompt.tool_choice.replace(tool::Choice::none()));
        let response = self.send().await;
        if let Some(choice) = choice {
            self.prompt.tool_choice = choice;
        }
        let response = response?;

        match response.disposition() {
            Disposition::Clipped => {
                cold_path();
                log::warn!("final word clipped at max_tokens: handing back");
                Err(Stop::Clipped(Box::new(response)))
            }
            Disposition::Done if self.unusable_done(&response) => {
                Err(self.unusable(response, None))
            }
            // A paused wrap-up would leave its server tool in flight with no
            // budget to resume it: not seated.
            Disposition::Paused => {
                log::warn!("final word paused on a server tool: not seated");
                Ok(())
            }
            Disposition::ToolUse | Disposition::Done => {
                let again = self.seat_assistant(state, response.inner)?;
                self.synthesize_results(&again)
            }
        }
    }

    /// Seat one user turn of `is_error` results answering `calls` — the
    /// "tool budget exhausted, wait for the user" explanation. No-op when
    /// there are no dangling calls (a paused turn that ran out of budget).
    fn synthesize_results(&mut self, calls: &[Use]) -> Result<(), Stop> {
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
                            self.config.max_tool_calls
                        ),
                    )
                    .error(),
                )
            })
            .collect();
        self.seat((Role::User, Content(results))).map(drop)
    }

    /// Append `message` through [`Prompt::seat`] — the crate's wire-legality
    /// kernel. Same-role tails concatenate (portable to strict-alternation
    /// backends); [`System`](Role::System) content seats the moment the tail
    /// permits and otherwise buffers in `pending_system` (never downgraded to
    /// the user channel — see the module-level notes). A merge that would
    /// trail a `tool_result` behind other content, or any other illegal
    /// placement, is a [`Stop::TurnOrder`] — a programming error in the
    /// caller's beat or hook.
    fn seat(&mut self, message: impl Into<Message>) -> Result<Seated, Stop> {
        Ok(self.prompt.seat(message, &mut self.pending_system)?)
    }
}

/// Whether `response` carries client calls — raw, whatever its stop reason:
/// [`response::Message::tool_uses`] is empty off a `tool_use` stop.
fn calls_tools(response: &response::Message) -> bool {
    response.inner.content.tool_uses().next().is_some()
}

/// Whether a seeded prompt's tail awaits the model: anything but an
/// assistant turn — or one paused on a server tool.
fn awaits_model(tail: &Message) -> bool {
    tail.role != Role::Assistant
        || tail.unfinished_server_tool_uses().next().is_some()
}

/// Whether the caller's next beat — a user turn — may follow `prompt`, as
/// an `Ok` hand-back owes it. A user tail takes it by merging.
fn takes_a_beat(prompt: &Prompt) -> bool {
    let beat = Message::from((Role::User, "…"));
    prompt.check_turn_order().is_ok()
        && prompt.messages.last().is_none_or(|tail| {
            tail.role == Role::User || tail.may_precede(&beat).is_ok()
        })
}

/// Whether the caller can carry on from `prompt`: legal turn order, and a
/// tail that awaits the model (a resume answers it) or takes a user beat —
/// never one with client calls left unanswered.
fn resumable(prompt: &Prompt) -> bool {
    prompt.check_turn_order().is_ok()
        && prompt.messages.last().is_none_or(|tail| {
            awaits_model(tail) || tail.tool_uses().next().is_none()
        })
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

// The checks serve the mock-driven scenarios.
#[cfg(all(test, feature = "mock"))]
mod checks;
#[cfg(all(test, feature = "mock"))]
mod scenarios;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        response::StopReason,
        tool::{CustomMethodDef, MethodDef},
        transport::tests::Script,
    };

    /// A tool that answers every call with "echoed" and remembers the calls
    /// (shared, so a test can inspect them after the box is moved).
    #[derive(Default)]
    struct Echo {
        calls: Arc<Mutex<Vec<Use>>>,
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
            self.calls.lock().unwrap().push(call);
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

    /// A turn paused mid-search: its `server_tool_use` is still in flight.
    fn paused_response() -> response::Message {
        let mut inner: AssistantMessage =
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [{"type": "text", "text": "searching…"}],
            }))
            .unwrap();
        inner.content.push(
            serde_json::from_str::<Block>(include_str!(
                "../test/data/server_tools/server_tool_use.json"
            ))
            .unwrap(),
        );
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

        let (Parts { prompt, .. }, ()) =
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
        let toolbox = ToolBox::new().add(Echo::default());
        let chat = Chat::new(script, Prompt::default(), toolbox);

        let (Parts { prompt, .. }, ()) =
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
        let toolbox = ToolBox::new().add(Echo::default());
        let chat = Chat::new(script, Prompt::default(), toolbox)
            .max_consecutive_tool_calls(1);

        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(matches!(
            last.content.iter().last().unwrap(),
            Block::ToolResult { result } if result.is_error
        ));
    }

    /// A beat that merges into the tail (the synthetic results a budget
    /// hand-back leaves) is new content, so it still drives a round.
    #[test]
    fn beat_merged_into_the_tail_drives_a_round() {
        let script = Script::new([
            tool_response("call_1"),
            tool_response("call_2"),
            text_response("ok"),
        ]);
        let toolbox = ToolBox::new().add(Echo::default());
        let chat = Chat::new(script, Prompt::default(), toolbox)
            .max_consecutive_tool_calls(1);

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("go"), user("carry on")])),
        )
        .unwrap();

        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(last.content.to_string(), "ok");
    }

    /// FinalWord grants exactly one wrap-up call after the cap.
    #[test]
    fn budget_final_word_makes_one_more_call() {
        let script = Script::new([
            tool_response("call_1"),
            tool_response("call_2"),
            text_response("to summarize: echoed"),
        ]);
        let toolbox = ToolBox::new().add(Echo::default());
        let chat = Chat::new(script, Prompt::default(), toolbox)
            .max_consecutive_tool_calls(1)
            .on_budget_exhausted(BudgetPolicy::FinalWord);

        let (Parts { prompt, .. }, ()) =
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

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("search")])),
        )
        .unwrap();

        // The paused assistant turn is gone; the user's beat is the tail.
        assert_eq!(prompt.messages.len(), 1);
        assert_eq!(prompt.messages[0].role, Role::User);
    }

    #[cfg(feature = "mock")]
    /// The continuation of [`paused_response`]: the search's result, then
    /// `done`.
    fn continued_response() -> response::Message {
        let mut inner = AssistantMessage::from(Content(vec![
            serde_json::from_str::<Block>(include_str!(
                "../test/data/server_tools/web_search_result.json"
            ))
            .unwrap(),
        ]));
        inner.content.push("done");
        response::Message::builder("test-model", inner)
            .stop_reason(StopReason::EndTurn)
            .build()
    }

    /// A paused turn that ends in a server-tool result, its use answered —
    /// a system note may follow it.
    fn paused_on_a_result() -> response::Message {
        let block = |json: &str| serde_json::from_str::<Block>(json).unwrap();
        let mut inner = AssistantMessage::text("searching…");
        inner.content.push(block(include_str!(
            "../test/data/server_tools/server_tool_use.json"
        )));
        inner.content.push(block(include_str!(
            "../test/data/server_tools/web_search_result.json"
        )));
        response::Message::builder("test-model", inner)
            .stop_reason(StopReason::PauseTurn)
            .build()
    }

    /// An `on_assistant` hook that seats a system note after the first turn.
    fn note_after_first()
    -> impl FnMut(&mut (), AssistantMessage) -> Vec<Message> + Send + 'static
    {
        let mut first = true;
        move |_: &mut (), msg| {
            let mut seated = vec![Message::from(msg)];
            if std::mem::take(&mut first) {
                seated.push((Role::System, "note").into());
            }
            seated
        }
    }

    /// A hook that redacts a paused turn away doesn't make a budget
    /// hand-back drop earlier beats: only this round's seating can be the
    /// paused turn.
    #[test]
    fn redacted_pause_keeps_earlier_beats() {
        let script = Script::new([text_response("hello"), paused_response()]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .max_consecutive_tool_calls(0)
            .on_assistant(|_: &mut (), msg: AssistantMessage| {
                msg.server_tool_use().is_none().then(|| msg.into())
            });

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("hi"), user("search")])),
        )
        .unwrap();

        let roles: Vec<_> = prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::Assistant, Role::User]);
        assert_eq!(prompt.messages[1].content.to_string(), "hello");
    }

    /// A note seated inside a paused turn survives the budget dropping that
    /// turn: it re-seats after the next beat.
    #[test]
    fn budget_mid_pause_keeps_notes_seated_inside_the_turn() {
        let script = Script::new([paused_on_a_result(), text_response("done")]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .max_consecutive_tool_calls(0)
            .on_assistant(note_after_first());

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("search"), user("next")])),
        )
        .unwrap();

        let roles: Vec<_> = prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::System, Role::Assistant]);
        assert_eq!(prompt.messages[1].content.to_string(), "note");
        prompt.check_turn_order().unwrap();
    }

    /// A continuation seated after a flushed note is a separate assistant
    /// turn, but the paused turn still starts at the first one — so the
    /// budget drops it all.
    #[test]
    fn budget_mid_pause_drops_from_the_turn_start() {
        let script = Script::new([
            paused_on_a_result(),
            paused_response(),
            text_response("done"),
        ]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .max_consecutive_tool_calls(1)
            .on_assistant(note_after_first());

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(
            chat.run((), beats(vec![user("search"), user("next")])),
        )
        .unwrap();

        let roles: Vec<_> = prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::System, Role::Assistant]);
        assert_eq!(prompt.messages[2].content.to_string(), "done");
        prompt.check_turn_order().unwrap();
    }

    /// `FinalWord` asks for words: the wrap-up goes out with `tool_choice:
    /// none`, and the prompt's own choice is back afterwards.
    #[cfg(feature = "mock")]
    #[test]
    fn final_word_sends_tool_choice_none() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(mock::message(tool_response("call_1")))
                .then(mock::message(tool_response("call_2")))
                .then(mock::text("to summarize")),
        );
        let prompt = Prompt::default().tool_choice(tool::Choice::any());
        let chat = Chat::new(
            transport.clone(),
            prompt,
            ToolBox::new().add(Echo::default()),
        )
        .max_consecutive_tool_calls(1)
        .on_budget_exhausted(BudgetPolicy::FinalWord);

        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        let choices: Vec<_> = transport
            .requests()
            .iter()
            .map(|r| r["tool_choice"]["type"].clone())
            .collect();
        assert_eq!(choices, ["any", "any", "none"]);
        assert!(matches!(prompt.tool_choice, Some(tool::Choice::Any { .. })));
        assert_eq!(prompt.messages.last().unwrap().role, Role::Assistant);
    }

    /// A transport that ignores `tool_choice` gets the prompt unchanged, and
    /// the calls its wrap-up makes anyway are answered with synthetic errors.
    #[cfg(feature = "mock")]
    #[test]
    fn final_word_without_tool_choice_synthesizes() {
        use crate::{
            Quirks,
            mock::{self, MockTransport},
        };

        let quirks = Quirks {
            tool_choice_not_respected: true,
            ..Quirks::default()
        };
        let transport = Arc::new(
            MockTransport::new()
                .with_quirks(quirks)
                .then(mock::message(tool_response("call_1")))
                .then(mock::message(tool_response("call_2")))
                .then(mock::message(tool_response("call_3"))),
        );
        let echo = Echo::default();
        let calls = echo.calls.clone();
        let chat = Chat::new(
            transport.clone(),
            Prompt::default(),
            ToolBox::new().add(echo),
        )
        .max_consecutive_tool_calls(1)
        .on_budget_exhausted(BudgetPolicy::FinalWord);

        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        assert!(
            transport
                .requests()
                .iter()
                .all(|r| r.get("tool_choice").is_none())
        );
        // Only the first call ran; the wrap-up's call was answered with an
        // error, keeping the prompt legal.
        assert_eq!(calls.lock().unwrap().len(), 1);
        let last = prompt.messages.last().unwrap();
        assert!(matches!(
            last.content.iter().next().unwrap(),
            Block::ToolResult { result } if result.is_error
                && result.tool_use_id == "call_3"
        ));
        prompt.check_turn_order().unwrap();
    }

    /// A `FinalWord` wrap-up that pauses isn't seated: nothing could resume
    /// its server tool.
    #[test]
    fn paused_final_word_is_not_seated() {
        let script = Script::new([
            tool_response("call_1"),
            tool_response("call_2"),
            paused_response(),
        ]);
        let chat = Chat::new(
            script,
            Prompt::default(),
            ToolBox::new().add(Echo::default()),
        )
        .max_consecutive_tool_calls(1)
        .on_budget_exhausted(BudgetPolicy::FinalWord);

        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap();

        let last = prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(last.content.iter().all(|b| b.is_tool_result()));
        prompt.check_turn_order().unwrap();
    }

    #[cfg(feature = "mock")]
    /// A tool that hands its mailbox to the test, to push on cue.
    struct Pusher(Arc<Mutex<Option<tool::Mailbox>>>);

    #[cfg(feature = "mock")]
    #[async_trait::async_trait]
    impl Tool for Pusher {
        fn name(&self) -> &str {
            "Pusher"
        }

        fn definitions(&self) -> Vec<MethodDef> {
            Vec::new()
        }

        async fn call(&mut self, call: Use) -> tool::Result {
            tool::Result::new(call.id, "unused")
        }

        fn connect(&mut self, mailbox: tool::Mailbox) {
            *self.0.lock().unwrap() = Some(mailbox);
        }
    }

    #[cfg(feature = "mock")]
    /// Pending once (after asking to be woken), then ready.
    #[derive(Default)]
    struct YieldOnce(bool);

    #[cfg(feature = "mock")]
    impl std::future::Future for YieldOnce {
        type Output = ();

        fn poll(
            mut self: std::pin::Pin<&mut Self>,
            cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<()> {
            if std::mem::replace(&mut self.0, true) {
                return std::task::Poll::Ready(());
            }
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    }

    /// Beats "hi", then "more": the first reply's hook pushes one note
    /// preferring `role`, which the second `select!` picks up (the beat
    /// yields first, so the queued note wins). Returns the prompt and every
    /// request sent.
    #[cfg(feature = "mock")]
    fn chat_with_a_note(
        model: crate::Id,
        role: Role,
    ) -> (Prompt, Vec<serde_json::Value>) {
        use crate::mock::{self, MockTransport};

        let transport =
            Arc::new(MockTransport::with(|_: &Prompt| mock::text("ok")));
        let slot = Arc::new(Mutex::new(None));
        let pusher = Arc::clone(&slot);
        let mut queue: std::collections::VecDeque<_> =
            [user("hi"), user("more")].into();
        let chat = Chat::new(
            transport.clone(),
            Prompt::default().model(model),
            ToolBox::new().add(Pusher(slot)),
        )
        .on_assistant(move |_: &mut (), msg: AssistantMessage| {
            if let Some(mailbox) = pusher.lock().unwrap().take() {
                mailbox.send("job done", [role]).unwrap();
            }
            [msg.into()]
        });

        let (Parts { prompt, .. }, ()) = futures::executor::block_on(chat.run(
            (),
            async move |_: &mut ()| {
                if queue.len() < 2 {
                    YieldOnce::default().await;
                }
                Ok(queue.pop_front())
            },
        ))
        .unwrap();
        (prompt, transport.requests())
    }

    /// A user-role notification drives a round of its own — `swarm`'s
    /// workers run on nothing else.
    #[cfg(feature = "mock")]
    #[test]
    fn user_notification_drives_a_round() {
        let (prompt, requests) =
            chat_with_a_note(crate::Id::Sonnet46, Role::User);

        assert_eq!(requests.len(), 3);
        let roles: Vec<_> = prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            [
                Role::User,
                Role::Assistant,
                Role::User,
                Role::Assistant,
                Role::User,
                Role::Assistant
            ]
        );
        assert_eq!(prompt.messages[2].content.to_string(), "job done");
    }

    /// A system note can't follow an assistant turn: it buffers without a
    /// round and rides the next beat's request.
    #[cfg(feature = "mock")]
    #[test]
    fn buffered_system_notification_waits_for_the_next_beat() {
        let (_, requests) = chat_with_a_note(crate::Id::Opus48, Role::System);

        assert_eq!(requests.len(), 2);
        let roles: Vec<_> = requests[1]["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| m["role"].as_str().unwrap().to_owned())
            .collect();
        assert_eq!(roles, ["user", "assistant", "user", "system"]);
    }

    /// Default quirks + `.cache(…)`: server-side auto placement — the
    /// request-level `cache_control` is set and no message carries a marker.
    #[test]
    fn cache_canonical_uses_auto_cache() {
        let script = Script::new([text_response("hello")]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new())
            .cache(CacheControl::ephemeral());

        let (Parts { prompt, .. }, ()) =
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

        let (Parts { prompt, .. }, ()) =
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

        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![user("hi")])))
                .unwrap();

        assert!(prompt.cache_control.is_none());
        assert!(!prompt.messages.iter().any(|m| m.content.has_cache()));
    }

    /// #124: a `max_tokens`-clipped turn's tool calls may be missing
    /// arguments, so it is neither seated nor dispatched — the driver hands
    /// the un-advanced prompt back, and a fresh `Chat` on it answers it
    /// before asking for a beat.
    #[cfg(feature = "mock")]
    #[test]
    fn clipped_turn_hands_back_and_resumes() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(
                    mock::text("calling")
                        .tool_use("toolbox__Echo__echo", serde_json::json!({}))
                        .stop_reason(StopReason::MaxTokens),
                )
                .then(mock::text("done")),
        );
        let echo = Echo::default();
        let calls = echo.calls.clone();
        let chat = Chat::new(
            transport.clone(),
            Prompt::default(),
            ToolBox::new().add(echo),
        );

        let error =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap_err();

        assert!(matches!(error.kind, Stop::Clipped(_)), "{error}");
        assert!(calls.lock().unwrap().is_empty(), "clipped calls ran");
        // The clipped turn never reached the prompt.
        assert_eq!(error.prompt.messages.len(), 1);
        assert_eq!(transport.len(), 1);

        // Resume with more room: answered at once, no beat needed.
        let mut error = error;
        error.prompt.max_tokens = std::num::NonZeroU32::new(8192).unwrap();
        let (chat, ()) = error.resume(transport.clone());
        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![]))).unwrap();

        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(prompt.messages[1].content.to_string(), "done");
        let requests = transport.requests();
        assert_eq!(requests[0]["messages"], requests[1]["messages"]);
        assert_eq!(requests[1]["max_tokens"], 8192);
    }

    /// A clipped resume of a paused turn hands back the prompt as sent — the
    /// paused turn is its tail, legal to resend — and resuming continues it.
    #[cfg(feature = "mock")]
    #[test]
    fn clipped_continuation_hands_back_the_paused_turn() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(mock::message(paused_response()))
                .then(mock::max_tokens("and the"))
                .then(mock::message(continued_response())),
        );
        let chat =
            Chat::new(transport.clone(), Prompt::default(), ToolBox::new());

        let error = futures::executor::block_on(
            chat.run((), beats(vec![user("search")])),
        )
        .unwrap_err();

        assert!(matches!(error.kind, Stop::Clipped(_)));
        let roles: Vec<_> =
            error.prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::Assistant]);

        let (chat, ()) = error.resume(transport.clone());
        let (Parts { prompt, .. }, ()) =
            futures::executor::block_on(chat.run((), beats(vec![]))).unwrap();

        // The continuation merged into the paused turn.
        assert_eq!(transport.len(), 3);
        assert_eq!(prompt.messages.len(), 2);
        let turn = &prompt.messages[1].content;
        assert_eq!(turn.len(), 4);
        assert_eq!(turn.last().unwrap().to_string(), "done");
    }

    /// A clip right after a system note hands back with the note as the
    /// (legal to resend) tail.
    #[cfg(feature = "mock")]
    #[test]
    fn clipped_hand_back_keeps_a_system_tail() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(mock::max_tokens("and the"))
                .then(mock::text("done")),
        );
        let chat =
            Chat::new(transport.clone(), Prompt::default(), ToolBox::new());
        let first =
            vec![(Role::User, "go").into(), (Role::System, "be brief").into()];

        let error = futures::executor::block_on(
            chat.run((), beats(vec![first, user("again")])),
        )
        .unwrap_err();

        let roles: Vec<_> =
            error.prompt.messages.iter().map(|m| m.role).collect();
        assert_eq!(roles, [Role::User, Role::System]);
        error.prompt.check_turn_order().unwrap();
    }

    /// `FinalWord`'s wrap-up call hands back when it clips, leaving the
    /// synthetic results as the tail.
    #[cfg(feature = "mock")]
    #[test]
    fn clipped_final_word_hands_back() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(mock::message(tool_response("call_1")))
                .then(mock::message(tool_response("call_2")))
                .then(mock::max_tokens("to summ")),
        );
        let chat = Chat::new(
            transport.clone(),
            Prompt::default(),
            ToolBox::new().add(Echo::default()),
        )
        .max_consecutive_tool_calls(1)
        .on_budget_exhausted(BudgetPolicy::FinalWord);

        let error =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap_err();

        assert!(matches!(error.kind, Stop::Clipped(_)));
        assert_eq!(transport.len(), 3);
        let last = error.prompt.messages.last().unwrap();
        assert_eq!(last.role, Role::User);
        assert!(last.content.iter().all(|b| b.is_tool_result()));
    }

    /// A refusal can cut a `tool_use` short: a finished turn carrying client
    /// calls runs none and is never seated.
    #[cfg(feature = "mock")]
    #[test]
    fn refused_tool_turn_runs_nothing() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new().then(
                mock::tool_use("toolbox__Echo__echo", serde_json::json!({}))
                    .refusal("cyber", "no"),
            ),
        );
        let echo = Echo::default();
        let calls = echo.calls.clone();
        let chat = Chat::new(
            transport.clone(),
            Prompt::default(),
            ToolBox::new().add(echo),
        );

        let error =
            futures::executor::block_on(chat.run((), beats(vec![user("go")])))
                .unwrap_err();

        let Stop::Unusable(response) = &error.kind else {
            panic!("expected Unusable, got {error:?}");
        };
        assert_eq!(response.stop_reason, Some(StopReason::Refusal));
        assert!(calls.lock().unwrap().is_empty(), "refused calls ran");
        assert_eq!(error.prompt.messages.len(), 1);
        assert_eq!(transport.len(), 1);
    }

    /// A refused continuation drops the whole turn — the paused turn it
    /// continued included — so the caller's next beat is legal.
    #[cfg(feature = "mock")]
    #[test]
    fn refused_continuation_drops_the_paused_turn() {
        use crate::mock::{self, MockTransport};

        let transport = Arc::new(
            MockTransport::new()
                .then(mock::message(paused_response()))
                .then(
                    mock::tool_use(
                        "toolbox__Echo__echo",
                        serde_json::json!({}),
                    )
                    .refusal("cyber", "no"),
                ),
        );
        let chat = Chat::new(
            transport,
            Prompt::default(),
            ToolBox::new().add(Echo::default()),
        );

        let error = futures::executor::block_on(
            chat.run((), beats(vec![user("search")])),
        )
        .unwrap_err();

        assert!(matches!(error.kind, Stop::Unusable(_)));
        assert_eq!(error.prompt.messages.len(), 1);
        assert_eq!(error.prompt.messages[0].role, Role::User);
    }

    /// An error hands the caller's state back along with the prompt, and
    /// still converts to a `BoxError` with the beat's message.
    #[test]
    fn beat_error_hands_back_prompt_and_state() {
        let script = Script::new([text_response("hello")]);
        let chat = Chat::new(script, Prompt::default(), ToolBox::new());
        let mut beat = 0u8;

        let error = futures::executor::block_on(chat.run(
            41u8,
            async move |state: &mut u8| {
                beat += 1;
                *state += 1;
                match beat {
                    1 => Ok(Some(user("hi"))),
                    _ => Err("stdin closed".into()),
                }
            },
        ))
        .unwrap_err();

        assert!(matches!(error.kind, Stop::Beat(_)));
        assert_eq!(error.state, 43);
        assert_eq!(error.prompt.messages.len(), 2);
        // `Debug` needn't see the state; `source` continues below the kind.
        assert!(format!("{error:?}").starts_with("Error { kind: Beat("));
        assert!(std::error::Error::source(&error).is_none());
        let boxed: BoxError = error.into();
        assert_eq!(boxed.to_string(), "stdin closed");
    }
}
