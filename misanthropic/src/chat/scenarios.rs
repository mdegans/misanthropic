//! [`Chat`]'s scenario table. Each [`Row`] scripts a [`MockTransport`],
//! drives a `Chat` through it (resuming after a hand-back if the row says
//! so), and asserts the requests sent, the final turn shape, the calls
//! dispatched, the usage tracked, and the [`checks`] invariants on every
//! request and every hand-back. Rows marked `live` also run against a
//! local Anthropic-compatible server — see `live`.

#[cfg(feature = "blallama")]
mod live;

use std::{
    collections::VecDeque,
    num::NonZeroU32,
    panic::AssertUnwindSafe,
    sync::{Arc, Mutex},
};

use super::{
    BoxError, BudgetPolicy, Chat, Stop,
    checks::{self, Checked, Log},
};
use crate::{
    Id, Prompt, Quirks, Transport,
    mock::{self, MockTransport, Reply},
    model::Model,
    prompt::message::{
        AssistantMessage, Block, Content, Message, Role, SystemMessage,
    },
    response::{self, StopReason, TokenCounts},
    stream::tests::{assembled, assembled_sse},
    tool::{self, Choice, CustomMethodDef, MethodDef, Tool, ToolBox, Use},
};

/// The wire name of [`Echo`]'s one method, as the [`ToolBox`] routes it.
const ECHO: &str = "toolbox__Echo__echo";

/// A tool that echoes, failing when asked to, and remembers every call.
#[derive(Clone, Default)]
struct Echo {
    calls: Arc<Mutex<Vec<Use>>>,
}

/// [`Echo`]'s arguments.
#[derive(serde::Serialize)]
struct EchoArgs {
    text: &'static str,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    fail: bool,
}

#[async_trait::async_trait]
impl Tool for Echo {
    fn name(&self) -> &str {
        "Echo"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Custom(CustomMethodDef::with_string_param(
            "Echo__echo",
            "Echo `text` back.",
            "text",
            "What to echo.",
            true,
        ))]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        let fail = call.input.get("fail").is_some_and(|f| f == true);
        let result = tool::Result::new(call.id.clone(), "echoed");
        self.calls.lock().unwrap().push(call);
        if fail { result.error() } else { result }
    }
}

/// A call to [`Echo`] with id `id`.
fn echo(id: &'static str) -> Use {
    let args = EchoArgs {
        text: "hi",
        fail: false,
    };
    Use::new(ECHO, serde_json::to_value(args).unwrap()).with_id(id)
}

/// A call to [`Echo`] that fails.
fn echo_failing(id: &'static str) -> Use {
    let args = EchoArgs {
        text: "hi",
        fail: true,
    };
    Use::new(ECHO, serde_json::to_value(args).unwrap()).with_id(id)
}

/// Where [`Pusher`] leaves its mailbox for a [`Hook::Push`] to send through.
type Slot = Arc<Mutex<Option<tool::Mailbox>>>;

/// A tool with no methods that hands its mailbox to the test.
struct Pusher(Slot);

#[async_trait::async_trait]
impl Tool for Pusher {
    fn name(&self) -> &str {
        "Pusher"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        Vec::new()
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        tool::Result::new(call.id, "unused").error()
    }

    fn connect(&mut self, mailbox: tool::Mailbox) {
        *self.0.lock().unwrap() = Some(mailbox);
    }
}

/// A tool whose `on_init` fails.
struct Broken;

#[async_trait::async_trait]
impl Tool for Broken {
    fn name(&self) -> &str {
        "Broken"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        Vec::new()
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        tool::Result::new(call.id, "unused").error()
    }

    async fn on_init(&mut self, _: &mut Prompt) -> Result<(), BoxError> {
        Err("no sandbox".into())
    }
}

/// A live `pause_turn`: ten `web_fetch` rounds, the eleventh in flight.
fn paused() -> Reply {
    mock::message(assembled(include_str!(
        "../../test/data/server_tools/pause_turn.sse.stream.jsonl"
    )))
}

/// The live continuation of [`paused`]: the in-flight result, one more
/// fetch, and `end_turn`.
fn resumed() -> Reply {
    mock::message(assembled(include_str!(
        "../../test/data/server_tools/pause_turn_resume.sse.stream.jsonl"
    )))
}

/// A live programmatic-tool-calling turn: a `code_execution` container
/// calls `query_sales` — a client call it waits on — and the turn stops
/// `tool_use`. No tool here serves `query_sales`, so the box answers each
/// call with an error result.
fn ptc() -> Reply {
    mock::message(assembled(include_str!(
        "../../test/data/server_tools/ptc.sse.stream.jsonl"
    )))
}

/// [`ptc`]'s turn, stopped `pause_turn` instead: a paused turn carrying a
/// client call.
fn ptc_paused() -> Reply {
    ptc().stop_reason(StopReason::PauseTurn)
}

/// The live continuation of [`ptc`]: the container's next call.
fn ptc_resumed() -> Reply {
    mock::message(assembled(include_str!(
        "../../test/data/server_tools/ptc_resume.sse.stream.jsonl"
    )))
}

/// The live end of [`ptc`]'s exchange: the container's result, then text.
fn ptc_done() -> Reply {
    mock::message(assembled(include_str!(
        "../../test/data/server_tools/code_execution_result.sse.stream.jsonl"
    )))
}

/// The ids of the client calls in the final prompt's `n`th turn, and of the
/// results in the turn after it.
fn answered(run: &Run, n: usize) -> (Vec<String>, Vec<String>) {
    let calls = run.turn(n).tool_uses().map(|c| c.id.to_string()).collect();
    let results = run
        .turn(n + 1)
        .content
        .iter()
        .filter_map(|block| match block {
            Block::ToolResult { result } => {
                Some(result.tool_use_id.to_string())
            }
            _ => None,
        })
        .collect();
    (calls, results)
}

/// A paused turn whose server tool has answered — a system turn may follow
/// it. Built from captured blocks.
fn paused_on_result() -> Reply {
    let block = |json: &str| serde_json::from_str::<Block>(json).unwrap();
    let mut turn = AssistantMessage::text("searching…");
    turn.content.push(block(include_str!(
        "../../test/data/server_tools/server_tool_use.json"
    )));
    turn.content.push(block(include_str!(
        "../../test/data/server_tools/web_search_result.json"
    )));
    mock::message(
        response::Message::builder(Model::default(), turn)
            .stop_reason(StopReason::PauseTurn)
            .build(),
    )
}

/// A turn that stops (`reason`) with a search in flight — cutting it short
/// unless `reason` is `pause_turn`. Built from captured blocks.
fn cut_short(reason: StopReason) -> Reply {
    let block = |json: &str| serde_json::from_str::<Block>(json).unwrap();
    let mut turn = AssistantMessage::text("searching…");
    turn.content.push(block(include_str!(
        "../../test/data/server_tools/server_tool_use.json"
    )));
    mock::message(
        response::Message::builder(Model::default(), turn)
            .stop_reason(reason)
            .build(),
    )
}

/// A live (Haiku 4.5) turn: a response fixture under `test/data/stop/`.
fn captured(json: &str) -> Reply {
    mock::message(serde_json::from_str(json).unwrap())
}

/// A live forced `write_file` call the stop sequence `print(` cut short:
/// valid, closed JSON, truncated at the match, stopped `stop_sequence`.
fn stopped_mid_call() -> Reply {
    captured(include_str!(
        "../../test/data/stop/stop_sequence_tool.response.json"
    ))
}

/// [`stopped_mid_call`]'s shape after text, assembled from a live stream.
fn stopped_mid_call_streamed() -> Reply {
    mock::message(assembled_sse(include_str!(
        "../../test/data/stop/stop_sequence_text_tool.sse.stream.txt"
    )))
}

/// A live forced call clipped at `max_tokens`: valid JSON missing a
/// required argument.
fn clipped_live() -> Reply {
    captured(include_str!("../../test/data/stop/clip_tool.response.json"))
}

/// [`clipped_live`]'s stream: the call's block never closes, so it
/// doesn't assemble — the turn arrives empty, stopped `max_tokens`.
fn clipped_live_streamed() -> Reply {
    mock::message(assembled_sse(include_str!(
        "../../test/data/stop/clip_tool.sse.stream.txt"
    )))
}

/// A live forced call clipped 140 tokens into its `contents` string: the
/// input keeps only the completed `path`, dropping `contents` whole.
fn clipped_long_live() -> Reply {
    captured(include_str!(
        "../../test/data/stop/clip_long_tool.response.json"
    ))
}

/// A turn calling [`Echo`] once per id.
fn calls(ids: &[&'static str]) -> Reply {
    ids.iter()
        .fold(mock::text("calling"), |reply, id| reply.call(echo(id)))
}

/// A turn cut off at `max_tokens` mid-call.
fn clipped_call() -> Reply {
    calls(&["cut"]).stop_reason(StopReason::MaxTokens)
}

/// A finished turn with no content at all.
fn empty_turn() -> Reply {
    let turn = AssistantMessage::from(Content(Vec::new()));
    mock::message(
        response::Message::builder(Model::default(), turn)
            .stop_reason(StopReason::EndTurn)
            .build(),
    )
}

/// `text`, stopped by `sequence`.
fn stopped_at(text: &'static str, sequence: &'static str) -> Reply {
    let mut message = mock::text(text)
        .stop_reason(StopReason::StopSequence)
        .build();
    message.stop_sequence = Some(sequence.into());
    mock::message(message)
}

/// A scripted reply: a message, or an HTTP error status.
#[allow(clippy::large_enum_variant)] // a short, test-only script
enum Scripted {
    Reply(Reply),
    Status(u16),
}

impl From<Reply> for Scripted {
    fn from(reply: Reply) -> Self {
        Self::Reply(reply)
    }
}

/// One user-side beat.
#[derive(Clone, Copy)]
enum Beat {
    User(&'static str),
    System(&'static str),
    /// A user line with an operator note after it.
    Both(&'static str, &'static str),
    /// A user line, then a result that can't lead — the second message
    /// breaks turn order.
    Torn,
    /// The beat source fails.
    Fail,
}

impl Beat {
    fn messages(self) -> Result<Vec<Message>, BoxError> {
        match self {
            Beat::User(text) => Ok(vec![(Role::User, text).into()]),
            Beat::System(text) => Ok(vec![(Role::System, text).into()]),
            Beat::Both(user, system) => Ok(vec![
                (Role::User, user).into(),
                (Role::System, system).into(),
            ]),
            Beat::Torn => Ok(vec![
                (Role::User, "hi").into(),
                (Role::User, tool::Result::new("x", "out")).into(),
            ]),
            Beat::Fail => Err("the beat source failed".into()),
        }
    }
}

impl From<&'static str> for Beat {
    fn from(text: &'static str) -> Self {
        Beat::User(text)
    }
}

/// The caller's state: what the driver threads through every run.
#[derive(Default)]
struct Tally {
    /// Beats taken from the source.
    beats: usize,
    /// Assistant turns the hook saw.
    turns: usize,
}

/// What the [`on_assistant`](Chat::on_assistant) hook does.
#[derive(Clone, Copy)]
enum Hook {
    /// No hook installed.
    None,
    /// Seat every turn unchanged.
    PassThrough,
    /// Replace every text block with `[redacted]`.
    Redact,
    /// Seat nothing.
    Drop,
    /// Replace the first turn with a forced [`Echo`] call, id `forced`.
    Force,
    /// Seat a system note after the first turn.
    NoteAfterFirst,
    /// On the first turn, push a notification preferring these roles.
    Push(&'static [Role]),
    /// Seat a user turn after the first — a caller bug when that turn calls
    /// tools.
    Stray,
    /// Replace the first turn with a system verdict alone.
    Verdict,
}

impl Hook {
    fn install<T: Transport>(
        self,
        chat: Chat<Tally, T>,
        slot: Slot,
    ) -> Chat<Tally, T> {
        if let Hook::None = self {
            return chat;
        }
        // "First" across resumes too: the tally is the caller's state.
        chat.on_assistant(move |tally: &mut Tally, turn| {
            tally.turns += 1;
            self.apply(tally.turns == 1, turn, &slot)
        })
    }

    fn apply(
        self,
        first: bool,
        mut turn: AssistantMessage,
        slot: &Slot,
    ) -> Vec<Message> {
        match self {
            Hook::Redact => {
                for block in turn.content.iter_mut() {
                    if matches!(block, Block::Text { .. }) {
                        *block = "[redacted]".into();
                    }
                }
                vec![turn.into()]
            }
            Hook::Drop => Vec::new(),
            Hook::Force if first => {
                let mut forced = AssistantMessage::text("forcing");
                forced.content.push(echo("forced"));
                vec![forced.into()]
            }
            Hook::NoteAfterFirst if first => {
                vec![turn.into(), (Role::System, "note").into()]
            }
            Hook::Push(roles) if first => {
                let mailbox = slot.lock().unwrap();
                let mailbox = mailbox.as_ref().expect("the box connected");
                mailbox.send("job done", roles.to_vec()).unwrap();
                vec![turn.into()]
            }
            Hook::Verdict if first => vec![(Role::System, "verdict").into()],
            Hook::Stray if first => {
                vec![turn.into(), (Role::User, "stray").into()]
            }
            _ => vec![turn.into()],
        }
    }
}

/// A [`Stop`] without its payload, to compare.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Clipped,
    Unusable,
    Transport,
    Beat,
    Tool,
    TurnOrder,
}

impl Kind {
    fn of(stop: &Stop) -> Self {
        match stop {
            Stop::Clipped(_) => Kind::Clipped,
            Stop::Unusable(_) => Kind::Unusable,
            Stop::Transport(_) => Kind::Transport,
            Stop::Beat(_) => Kind::Beat,
            Stop::Tool(_) => Kind::Tool,
            Stop::TurnOrder(_) => Kind::TurnOrder,
        }
    }
}

/// One scenario: its setup, then what it must produce.
struct Row {
    name: &'static str,
    /// Also run against a local server (see `live`).
    live: bool,
    /// The mock's model; a live run uses the server's.
    model: Id,
    prompt: fn(Prompt) -> Prompt,
    quirks: Quirks,
    budget: Option<(usize, BudgetPolicy)>,
    hook: Hook,
    /// Add a [`Broken`] tool.
    broken: bool,
    replies: Vec<Scripted>,
    beats: Vec<Beat>,
    /// Resume after a hand-back (raising `max_tokens` after a clip).
    resume: bool,
    /// The kinds of the hand-backs, in order.
    stops: Vec<Kind>,
    requests: usize,
    /// The final prompt's turn roles (see [`checks::roles`]).
    roles: &'static str,
    /// The ids [`Echo`] ran (live: only how many).
    dispatched: Vec<&'static str>,
    /// The final turn's text (mock only).
    last: Option<&'static str>,
    /// Anything else to assert.
    extra: Option<fn(&Run)>,
}

fn row(name: &'static str) -> Row {
    Row {
        name,
        live: false,
        model: Id::Opus48,
        prompt: |prompt| prompt,
        quirks: Quirks::default(),
        budget: None,
        hook: Hook::None,
        broken: false,
        replies: Vec::new(),
        beats: vec![Beat::User("go")],
        resume: false,
        stops: Vec::new(),
        requests: 0,
        roles: "",
        dispatched: Vec::new(),
        last: None,
        extra: None,
    }
}

impl Row {
    fn live(self) -> Self {
        Self { live: true, ..self }
    }

    fn model(self, model: Id) -> Self {
        Self { model, ..self }
    }

    fn prompt(self, prompt: fn(Prompt) -> Prompt) -> Self {
        Self { prompt, ..self }
    }

    fn quirks(self, quirks: Quirks) -> Self {
        Self { quirks, ..self }
    }

    fn budget(self, max: usize, policy: BudgetPolicy) -> Self {
        let budget = Some((max, policy));
        Self { budget, ..self }
    }

    fn hook(self, hook: Hook) -> Self {
        Self { hook, ..self }
    }

    fn broken(self) -> Self {
        Self {
            broken: true,
            ..self
        }
    }

    fn reply(mut self, reply: impl Into<Scripted>) -> Self {
        self.replies.push(reply.into());
        self
    }

    fn status(self, status: u16) -> Self {
        self.reply(Scripted::Status(status))
    }

    fn beats<B: Into<Beat>>(self, beats: impl IntoIterator<Item = B>) -> Self {
        let beats = beats.into_iter().map(Into::into).collect();
        Self { beats, ..self }
    }

    fn resume(self) -> Self {
        Self {
            resume: true,
            ..self
        }
    }

    fn stops(self, stops: impl Into<Vec<Kind>>) -> Self {
        let stops = stops.into();
        Self { stops, ..self }
    }

    fn requests(self, requests: usize) -> Self {
        Self { requests, ..self }
    }

    fn roles(self, roles: &'static str) -> Self {
        Self { roles, ..self }
    }

    fn dispatched(self, ids: impl Into<Vec<&'static str>>) -> Self {
        let dispatched = ids.into();
        Self { dispatched, ..self }
    }

    fn last(self, text: &'static str) -> Self {
        let last = Some(text);
        Self { last, ..self }
    }

    fn extra(self, extra: fn(&Run)) -> Self {
        let extra = Some(extra);
        Self { extra, ..self }
    }
}

/// What a row's run produced.
struct Run {
    prompt: Prompt,
    /// The system notes handed back still buffered.
    pending: Option<SystemMessage>,
    log: Log,
    calls: Vec<Use>,
    tally: Tally,
    stops: Vec<Kind>,
    /// The stop reasons of the turns `stops` handed back, in order.
    stop_reasons: Vec<Option<StopReason>>,
}

impl Run {
    /// The roles of the `n`th request's turns.
    fn sent_roles(&self, n: usize) -> String {
        checks::roles(&self.log.sent[n])
    }

    /// The final prompt's `n`th turn.
    fn turn(&self, n: usize) -> &Message {
        &self.prompt.messages[n]
    }
}

/// The `n`th block of `turn` as a tool result.
fn result(turn: &Message, n: usize) -> &tool::Result {
    match &turn.content[n] {
        Block::ToolResult { result } => result,
        other => panic!("expected a tool result, got {other:?}"),
    }
}

/// Pending once, then ready: lets a queued notification win the driver's
/// `select!` against the next beat.
async fn yield_once() {
    let mut yielded = false;
    futures::future::poll_fn(move |cx| {
        if std::mem::replace(&mut yielded, true) {
            return std::task::Poll::Ready(());
        }
        cx.waker().wake_by_ref();
        std::task::Poll::Pending
    })
    .await
}

/// Drive `row` through `transport` from `base` (tuned by the row), resuming
/// as the row says, checking every hand-back and the usage sink.
async fn drive<T>(row: &Row, transport: Checked<T>, base: Prompt) -> Run
where
    T: Transport + Clone,
{
    let echo = Echo::default();
    let sink = Arc::new(Mutex::new(TokenCounts::default()));
    let mut queue: VecDeque<Beat> = row.beats.iter().copied().collect();
    let mut next_beat = async |tally: &mut Tally| {
        yield_once().await;
        let beat = queue.pop_front();
        tally.beats += usize::from(beat.is_some());
        beat.map(Beat::messages).transpose()
    };

    // One toolbox and one configuration for the whole row: a resume carries
    // both through the hand-back.
    let slot = Slot::default();
    let mut toolbox =
        ToolBox::new().add(echo.clone()).add(Pusher(slot.clone()));
    if row.broken {
        toolbox = toolbox.add(Broken);
    }
    let mut chat = Chat::new(transport.clone(), (row.prompt)(base), toolbox)
        .track_usage(Arc::clone(&sink));
    if let Some((max, policy)) = row.budget {
        chat = chat
            .max_consecutive_tool_calls(max)
            .on_budget_exhausted(policy);
    }
    let mut chat = row.hook.install(chat, slot);

    let mut tally = Tally::default();
    let mut stops = Vec::new();
    let mut stop_reasons = Vec::new();
    let (prompt, pending, tally) = loop {
        let mut error = match chat.run(tally, &mut next_beat).await {
            Ok((parts, state)) => {
                checks::assert_beat_may_follow(&parts.prompt);
                let received = &transport.log().received;
                checks::assert_in_flight_paused(&parts.prompt, received);
                break (parts.prompt, parts.pending, state);
            }
            Err(error) => error,
        };
        checks::assert_handback_legal(&error.prompt);
        let received = &transport.log().received;
        checks::assert_in_flight_paused(&error.prompt, received);
        stops.push(Kind::of(&error.kind));
        if let Stop::Clipped(turn) | Stop::Unusable(turn) = &error.kind {
            stop_reasons.push(turn.stop_reason);
        }
        if !row.resume || stops.len() > 2 {
            break (error.prompt, error.pending, error.state);
        }
        if let Stop::Clipped(_) = error.kind {
            let two = NonZeroU32::new(2).unwrap();
            error.prompt.max_tokens =
                error.prompt.max_tokens.saturating_mul(two);
        }
        (chat, tally) = error.resume(transport.clone());
    };

    let consumed = row.beats.len() - queue.len();
    assert_eq!(tally.beats, consumed, "the state threads through each run");
    let log = std::mem::take(&mut *transport.log());
    assert_eq!(*sink.lock().unwrap(), log.usage(), "track_usage");
    let calls = echo.calls.lock().unwrap().clone();
    Run {
        prompt,
        pending,
        log,
        calls,
        tally,
        stops,
        stop_reasons,
    }
}

/// Assert `run` is what `row` expects. A `live` run checks only what a
/// real model can't vary: dispatch counts, not ids or text.
fn expect(row: &Row, run: &Run, live: bool) {
    assert_eq!(run.stops, row.stops, "hand-backs");
    assert_eq!(run.log.sent.len(), row.requests, "requests sent");
    assert_eq!(checks::roles(&run.prompt), row.roles, "final turn roles");
    let ids: Vec<&str> = run.calls.iter().map(|c| c.id.as_ref()).collect();
    match live {
        true => assert_eq!(ids.len(), row.dispatched.len(), "calls run"),
        false => assert_eq!(ids, row.dispatched, "calls run"),
    }
    if let (false, Some(last)) = (live, row.last) {
        let tail = run.prompt.messages.last().expect("a final turn");
        assert_eq!(text(tail), last, "final turn text");
    }
    if let Some(extra) = row.extra {
        extra(run);
    }
}

/// Run `row` against a [`MockTransport`] scripted with its replies.
fn run_mock(row: Row) {
    // A live row can only script what a real server does on its own.
    let scripts_errors =
        row.replies.iter().any(|r| matches!(r, Scripted::Status(_)));
    assert!(
        !row.live || !(row.resume || scripts_errors),
        "not live-able"
    );
    let mock = row.replies.iter().fold(
        MockTransport::new().with_quirks(row.quirks),
        |mock, scripted| match scripted {
            Scripted::Reply(reply) => mock.then(reply.clone().usage(3, 2)),
            Scripted::Status(status) => {
                mock.then(mock::http_error(*status, None))
            }
        },
    );
    let mock = Arc::new(mock);
    let transport = Checked::new(Arc::clone(&mock));
    let base = Prompt::default().model(row.model);
    let run = futures::executor::block_on(drive(&row, transport, base));
    assert_eq!(mock.remaining(), 0, "every scripted reply is used");
    expect(&row, &run, false);
}

/// Every row passes, reported together.
#[test]
fn scenarios() {
    let rows = rows();
    let mut names: Vec<_> = rows.iter().map(|r| r.name).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), rows.len(), "row names are unique");

    let total = rows.len();
    let failed: Vec<&str> = rows
        .into_iter()
        .filter_map(|row| {
            let name = row.name;
            let outcome =
                std::panic::catch_unwind(AssertUnwindSafe(|| run_mock(row)));
            outcome.is_err().then(|| {
                eprintln!("^ scenario row `{name}`");
                name
            })
        })
        .collect();
    assert!(
        failed.is_empty(),
        "{} of {total} scenario rows failed (panics above): {failed:?}",
        failed.len()
    );
}

/// The table.
fn rows() -> Vec<Row> {
    use BudgetPolicy::{FinalWord, HandBack};

    vec![
        // Plain turns.
        row("plain_end_turn")
            .live()
            .beats(["Say hello in one word."])
            .reply(mock::text("Hello."))
            .requests(1)
            .roles("UA")
            .last("Hello."),
        row("two_beats")
            .live()
            .beats(["Say hi in one word.", "Now say bye in one word."])
            .reply(mock::text("Hi."))
            .reply(mock::text("Bye."))
            .requests(2)
            .roles("UAUA")
            .last("Bye."),
        row("no_beats").beats::<Beat>([]).requests(0).roles(""),
        row("beat_error")
            .beats([Beat::User("hi"), Beat::Fail])
            .reply(mock::text("hi"))
            .stops([Kind::Beat])
            .requests(1)
            .roles("UA"),
        row("stop_sequence")
            .live()
            .prompt(|p| p.stop_sequences(["STOP"]))
            .beats(["Reply with exactly these words: alpha beta STOP gamma"])
            .reply(stopped_at("alpha beta ", "STOP"))
            .requests(1)
            .roles("UA")
            .extra(|run| {
                let reply = &run.log.received[0];
                assert_eq!(reply.stop_reason, Some(StopReason::StopSequence));
                assert_eq!(reply.stop_sequence.as_deref(), Some("STOP"));
            }),
        row("broken_tool_init")
            .broken()
            .stops([Kind::Tool])
            .requests(0)
            .roles(""),
        // Client tools.
        row("tool_use_single")
            .reply(calls(&["a"]))
            .reply(mock::text("done"))
            .requests(2)
            .roles("UAUA")
            .dispatched(["a"])
            .last("done")
            .extra(|run| assert!(!result(run.turn(2), 0).is_error)),
        row("tool_use_parallel")
            .reply(calls(&["a", "b"]))
            .reply(mock::text("done"))
            .requests(2)
            .roles("UAUA")
            .dispatched(["a", "b"])
            .extra(|run| {
                let turn = run.turn(2);
                assert_eq!(turn.content.len(), 2, "one turn of results");
                assert_eq!(result(turn, 0).tool_use_id, "a");
                assert_eq!(result(turn, 1).tool_use_id, "b");
            }),
        row("tool_use_chain")
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(mock::text("done"))
            .requests(3)
            .roles("UAUAUA")
            .dispatched(["a", "b"]),
        row("tool_error")
            .reply(mock::text("calling").call(echo_failing("a")))
            .reply(mock::text("it failed"))
            .requests(2)
            .roles("UAUA")
            .dispatched(["a"])
            .extra(|run| assert!(result(run.turn(2), 0).is_error)),
        row("unknown_method")
            .reply(
                mock::text("calling").call(
                    Use::new("toolbox__Nope__nope", serde_json::Value::Null)
                        .with_id("x"),
                ),
            )
            .reply(mock::text("oops"))
            .requests(2)
            .roles("UAUA")
            .extra(|run| assert!(result(run.turn(2), 0).is_error)),
        // Refusals: a finished turn never runs client calls.
        row("refusal_with_call")
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(1)
            .roles("U"),
        row("refusal_with_text")
            .reply(mock::text("I can't help with that.").refusal("cyber", "no"))
            .requests(1)
            .roles("UA")
            .last("I can't help with that."),
        row("refusal_after_tool_round")
            .reply(calls(&["a"]))
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(2)
            .roles("UAU")
            .dispatched(["a"]),
        row("refused_continuation")
            .reply(paused())
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(2)
            .roles("U"),
        row("refused_continuation_rescues_its_note")
            .hook(Hook::NoteAfterFirst)
            .reply(paused_on_result())
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(2)
            .roles("US")
            .extra(|run| assert_eq!(run.sent_roles(1), "UAS")),
        // A finished turn with a server tool in flight: nothing answers it.
        row("refusal_cuts_a_server_tool_short")
            .reply(cut_short(StopReason::Refusal))
            .stops([Kind::Unusable])
            .requests(1)
            .roles("U"),
        row("end_turn_cuts_a_server_tool_short")
            .reply(cut_short(StopReason::EndTurn))
            .requests(1)
            .stops([Kind::Unusable])
            .roles("U"),
        // Live: a stop sequence matched inside a call's input closes it
        // truncated — a finished turn cutting a call short.
        row("stop_sequence_cuts_a_call_short")
            .prompt(|p| p.stop_sequences(["print("]))
            .reply(stopped_mid_call())
            .stops([Kind::Unusable])
            .requests(1)
            .roles("U")
            .extra(|run| {
                let stopped = [Some(StopReason::StopSequence)];
                assert_eq!(run.stop_reasons, stopped);
            }),
        row("stop_sequence_cuts_a_streamed_call_short")
            .prompt(|p| p.stop_sequences(["print("]))
            .reply(stopped_mid_call_streamed())
            .stops([Kind::Unusable])
            .requests(1)
            .roles("U")
            .extra(|run| {
                let stopped = [Some(StopReason::StopSequence)];
                assert_eq!(run.stop_reasons, stopped);
            }),
        row("refused_continuation_strands_the_pause")
            .reply(paused())
            .reply(mock::text("I can't continue.").refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(2)
            .roles("U")
            .extra(|run| assert_eq!(run.sent_roles(1), "UA")),
        row("refusal_without_content")
            .beats(["hi", "again"])
            .reply(mock::refusal("cyber", "no"))
            .reply(mock::text("ok"))
            .requests(2)
            .roles("UA"),
        row("empty_turn_is_not_seated")
            .beats(["hi", "again"])
            .reply(empty_turn())
            .reply(mock::text("ok"))
            .requests(2)
            .roles("UA"),
        // A round that seats nothing leaves a note seated before the call
        // trailing; it goes back to the buffer, to follow the next beat.
        row("note_before_a_bare_refusal")
            .beats([Beat::Both("hi", "be brief"), Beat::User("again")])
            .reply(mock::refusal("cyber", "no"))
            .reply(mock::text("ok"))
            .requests(2)
            .roles("USA")
            .extra(|run| assert_eq!(run.sent_roles(1), "US")),
        row("note_before_an_empty_turn")
            .beats([Beat::Both("hi", "be brief"), Beat::User("again")])
            .reply(empty_turn())
            .reply(mock::text("ok"))
            .requests(2)
            .roles("USA")
            .extra(|run| assert_eq!(run.sent_roles(1), "US")),
        row("hook_drops_a_turn_after_a_note")
            .hook(Hook::Drop)
            .beats([Beat::Both("hi", "be brief"), Beat::User("again")])
            .reply(mock::text("x"))
            .reply(mock::text("y"))
            .requests(2)
            .roles("U")
            .extra(|run| {
                assert_eq!(run.sent_roles(1), "US");
                assert_eq!(pending(run), "be brief");
            }),
        row("hook_returns_only_a_verdict")
            .hook(Hook::Verdict)
            .beats(["hi", "again"])
            .reply(mock::text("x"))
            .reply(mock::text("ok"))
            .requests(2)
            .roles("USA")
            .last("ok")
            .extra(|run| {
                assert_eq!(run.sent_roles(1), "US");
                assert_eq!(text(run.turn(1)), "verdict");
            }),
        row("empty_turn_after_tool_round")
            .beats(["go", "again"])
            .reply(calls(&["a"]))
            .reply(empty_turn())
            .reply(mock::text("ok"))
            .requests(3)
            .roles("UAUA")
            .dispatched(["a"]),
        // Clips: never seated, never dispatched.
        row("clip_first_round")
            .live()
            .prompt(|p| {
                p.max_tokens(NonZeroU32::new(16).unwrap())
                    .tool_choice(Choice::any())
            })
            .beats([
                "Use the echo tool to echo the full first paragraph of the \
                 US Declaration of Independence.",
            ])
            .reply(clipped_call())
            .stops([Kind::Clipped])
            .requests(1)
            .roles("U")
            .extra(|run| {
                let stop = run.log.received[0].stop_reason;
                assert_eq!(stop, Some(StopReason::MaxTokens));
            }),
        row("clip_captured")
            .reply(clipped_live())
            .stops([Kind::Clipped])
            .requests(1)
            .roles("U"),
        row("clip_captured_long")
            .reply(clipped_long_live())
            .stops([Kind::Clipped])
            .requests(1)
            .roles("U")
            .extra(|run| {
                assert!(run.calls.is_empty());
                assert_eq!(run.stop_reasons, [Some(StopReason::MaxTokens)]);
            }),
        row("clip_captured_streamed")
            .reply(clipped_live_streamed())
            .stops([Kind::Clipped])
            .requests(1)
            .roles("U"),
        row("clip_then_resume")
            .resume()
            .reply(clipped_call())
            .reply(mock::text("done"))
            .stops([Kind::Clipped])
            .requests(2)
            .roles("UA")
            .last("done")
            .extra(|run| {
                let [first, second] = &run.log.sent[..] else {
                    panic!("two requests");
                };
                assert_eq!(second.max_tokens.get(), 2 * first.max_tokens.get());
                let wire = |p: &Prompt| serde_json::to_value(&p.messages);
                assert_eq!(wire(first).unwrap(), wire(second).unwrap());
            }),
        row("clip_after_tool_round")
            .reply(calls(&["a"]))
            .reply(clipped_call())
            .stops([Kind::Clipped])
            .requests(2)
            .roles("UAU")
            .dispatched(["a"]),
        row("clip_after_pause")
            .resume()
            .reply(paused())
            .reply(mock::max_tokens("and the"))
            .reply(resumed())
            .stops([Kind::Clipped])
            .requests(3)
            .roles("UA")
            .extra(|run| {
                // The clip handed back the paused turn; the resume merged
                // its continuation in.
                assert_eq!(run.sent_roles(2), "UA");
                assert_settled(run.turn(1));
            }),
        row("clip_at_budget_cap")
            .budget(1, HandBack)
            .reply(calls(&["a"]))
            .reply(clipped_call())
            .stops([Kind::Clipped])
            .requests(2)
            .roles("UAU")
            .dispatched(["a"]),
        row("clip_on_final_word")
            .budget(1, FinalWord)
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(mock::max_tokens("to summ"))
            .stops([Kind::Clipped])
            .requests(3)
            .roles("UAUAU")
            .dispatched(["a"]),
        // Paused server tools: resumed, or dropped whole.
        row("pause_resume")
            .reply(paused())
            .reply(resumed())
            .requests(2)
            .roles("UA")
            .extra(|run| {
                assert_eq!(run.sent_roles(1), "UA", "the resume");
                assert_settled(run.turn(1));
            }),
        row("pause_at_budget_cap")
            .budget(0, HandBack)
            .reply(paused())
            .requests(1)
            .roles("U"),
        row("pause_at_cap_after_tool_round")
            .budget(1, HandBack)
            .reply(calls(&["a"]))
            .reply(paused())
            .requests(2)
            .roles("UAU")
            .dispatched(["a"]),
        row("pause_note_rides_the_next_beat")
            .hook(Hook::NoteAfterFirst)
            .beats(["search", "next"])
            .reply(paused())
            .reply(resumed())
            .reply(mock::text("ok"))
            .requests(3)
            .roles("UAUSA")
            .extra(|run| assert_eq!(run.sent_roles(1), "UA")),
        row("pause_at_cap_keeps_its_note")
            .budget(0, HandBack)
            .hook(Hook::NoteAfterFirst)
            .beats(["search", "next"])
            .reply(paused_on_result())
            .reply(mock::text("done"))
            .requests(2)
            .roles("USA"),
        // Programmatic tool calling (captured): a container's client calls
        // settle its `server_tool_use` until its result arrives.
        row("ptc_round_trip")
            .reply(ptc())
            .reply(ptc_resumed())
            .reply(ptc_done())
            .requests(3)
            .roles("UAUAUA")
            .extra(|run| {
                for n in [1, 3] {
                    let (calls, results) = answered(run, n);
                    assert_eq!(calls.len(), 1);
                    assert_eq!(calls, results, "turn {n}'s call answered");
                }
                assert_settled(run.turn(5));
            }),
        row("ptc_paused_turn_carrying_calls")
            .reply(ptc_paused())
            .reply(ptc_resumed())
            .reply(ptc_done())
            .requests(3)
            .roles("UAUAUA")
            .extra(|run| {
                assert_eq!(run.sent_roles(1), "UAU", "answered, then resumed");
                let (calls, results) = answered(run, 1);
                assert_eq!(calls, results);
            }),
        row("ptc_paused_then_refused")
            .reply(ptc_paused())
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(2)
            .roles("U"),
        // The budget.
        row("budget_hand_back")
            .budget(1, HandBack)
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .requests(2)
            .roles("UAUAU")
            .dispatched(["a"])
            .extra(|run| {
                let synthetic = result(run.turn(4), 0);
                assert!(synthetic.is_error && synthetic.tool_use_id == "b");
            }),
        row("budget_hand_back_then_beat")
            .budget(1, HandBack)
            .beats(["go", "carry on"])
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(mock::text("ok"))
            .requests(3)
            .roles("UAUAUA")
            .dispatched(["a"])
            .last("ok"),
        row("budget_zero")
            .budget(0, HandBack)
            .reply(calls(&["a"]))
            .requests(1)
            .roles("UAU"),
        row("forced_tool_final_word")
            .live()
            .prompt(|p| p.tool_choice(Choice::any()))
            .budget(1, FinalWord)
            .beats(["Use the echo tool to echo 'ping', then say 'pong'."])
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(mock::text("pong"))
            .requests(3)
            .roles("UAUAUA")
            .dispatched(["a"])
            .extra(|run| {
                let choice = |n: usize| run.log.sent[n].tool_choice.clone();
                assert!(matches!(choice(0), Some(Choice::Any { .. })));
                assert!(matches!(choice(1), Some(Choice::Any { .. })));
                assert!(
                    matches!(choice(2), Some(Choice::None)),
                    "the wrap-up asks for words"
                );
                assert!(matches!(
                    run.prompt.tool_choice,
                    Some(Choice::Any { .. })
                ));
                assert_eq!(run.turn(5).tool_uses().count(), 0);
            }),
        row("final_word_ignored_tool_choice")
            .quirks(Quirks {
                tool_choice_not_respected: true,
                ..Quirks::default()
            })
            .prompt(|p| p.tool_choice(Choice::any()))
            .budget(1, FinalWord)
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(calls(&["c"]))
            .requests(3)
            .roles("UAUAUAU")
            .dispatched(["a"])
            .extra(|run| {
                assert!(run.log.sent.iter().all(|p| matches!(
                    p.tool_choice,
                    Some(Choice::Any { .. })
                )));
                let synthetic = result(run.turn(6), 0);
                assert!(synthetic.is_error && synthetic.tool_use_id == "c");
            }),
        row("final_word_refused")
            .budget(1, FinalWord)
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(calls(&["c"]).refusal("cyber", "no"))
            .stops([Kind::Unusable])
            .requests(3)
            .roles("UAUAU")
            .dispatched(["a"]),
        row("final_word_paused")
            .budget(1, FinalWord)
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(paused())
            .requests(3)
            .roles("UAUAU")
            .dispatched(["a"]),
        // System notes: seated when the tail allows, buffered until then.
        row("note_after_tool_turn")
            .hook(Hook::NoteAfterFirst)
            .reply(calls(&["a"]))
            .reply(mock::text("done"))
            .requests(2)
            .roles("UAUSA")
            .dispatched(["a"])
            .extra(|run| assert_eq!(run.sent_roles(1), "UAUS")),
        row("note_rides_the_next_beat")
            .live()
            .hook(Hook::NoteAfterFirst)
            .beats(["Say hi in one word.", "Now say bye in one word."])
            .reply(mock::text("Hi."))
            .reply(mock::text("Bye."))
            .requests(2)
            .roles("UAUSA")
            .extra(|run| assert_eq!(run.sent_roles(1), "UAUS")),
        row("system_only_beat_waits")
            .beats([Beat::System("be brief"), Beat::User("hi")])
            .reply(mock::text("ok"))
            .requests(1)
            .roles("USA"),
        row("user_and_system_beat")
            .beats([Beat::Both("hi", "be brief")])
            .reply(mock::text("ok"))
            .requests(1)
            .roles("USA"),
        // A note still buffered at a hand-back comes back in the parts, and
        // a resume seats it after the next beat.
        row("trailing_note_is_handed_back")
            .hook(Hook::NoteAfterFirst)
            .reply(mock::text("hi"))
            .requests(1)
            .roles("UA")
            .extra(|run| assert_eq!(pending(run), "note")),
        row("buffered_note_survives_a_resume")
            .resume()
            .hook(Hook::NoteAfterFirst)
            .beats(["search", "next"])
            .reply(paused())
            .status(529)
            .reply(resumed())
            .reply(mock::text("ok"))
            .stops([Kind::Transport])
            .requests(4)
            .roles("UAUSA")
            .extra(|run| assert_eq!(run.sent_roles(3), "UAUS")),
        row("notification_survives_a_resume")
            .resume()
            .hook(Hook::Push(&[Role::User]))
            .reply(calls(&["a"]))
            .status(529)
            .reply(mock::text("done"))
            .reply(mock::text("noted"))
            .stops([Kind::Transport])
            .requests(4)
            .roles("UAUAUA")
            .dispatched(["a"])
            .extra(|run| assert_eq!(text(run.turn(4)), "job done")),
        // The on_assistant hook.
        row("hook_pass_through")
            .hook(Hook::PassThrough)
            .reply(mock::text("hi"))
            .requests(1)
            .roles("UA")
            .last("hi")
            .extra(|run| assert_eq!(run.tally.turns, 1)),
        row("hook_redacts")
            .hook(Hook::Redact)
            .reply(calls(&["a"]))
            .reply(mock::text("secret"))
            .requests(2)
            .roles("UAUA")
            .dispatched(["a"])
            .last("[redacted]"),
        row("hook_drops_every_turn")
            .hook(Hook::Drop)
            .beats(["hi", "again"])
            .reply(mock::text("x"))
            .reply(mock::text("y"))
            .requests(2)
            .roles("U"),
        row("hook_forces_a_call")
            .hook(Hook::Force)
            .reply(mock::text("no tools"))
            .reply(mock::text("done"))
            .requests(2)
            .roles("UAUA")
            .dispatched(["forced"])
            .last("done"),
        row("beat_breaking_turn_order_seats_nothing")
            .beats([Beat::User("hi"), Beat::Torn])
            .reply(mock::text("hello"))
            .stops([Kind::TurnOrder])
            .requests(1)
            .roles("UA"),
        row("hook_strays_onto_the_user_channel")
            .hook(Hook::Stray)
            .reply(calls(&["a"]))
            .stops([Kind::TurnOrder])
            .requests(1)
            .roles("U"),
        // Notifications: a seated note drives a round.
        row("user_note_drives_a_round")
            .live()
            .model(Id::Sonnet46)
            .hook(Hook::Push(&[Role::User]))
            .beats(["Say hi in one word.", "Now say bye in one word."])
            .reply(mock::text("Hi."))
            .reply(mock::text("Noted."))
            .reply(mock::text("Bye."))
            .requests(3)
            .roles("UAUAUA")
            .extra(|run| {
                assert_eq!(text(run.turn(2)), "job done");
            }),
        row("system_note_waits_for_a_beat")
            .hook(Hook::Push(&[Role::System]))
            .beats(["hi", "more"])
            .reply(mock::text("hi"))
            .reply(mock::text("ok"))
            .requests(2)
            .roles("UAUSA"),
        row("system_note_after_hand_back_drives_a_round")
            .budget(1, HandBack)
            .hook(Hook::Push(&[Role::System]))
            .reply(calls(&["a"]))
            .reply(calls(&["b"]))
            .reply(mock::text("ok"))
            .requests(3)
            .roles("UAUAUSA")
            .dispatched(["a"]),
        row("system_note_falls_back_to_user")
            .model(Id::Sonnet46)
            .hook(Hook::Push(&[Role::System, Role::User]))
            .beats(["hi", "more"])
            .reply(mock::text("hi"))
            .reply(mock::text("noted"))
            .reply(mock::text("ok"))
            .requests(3)
            .roles("UAUAUA"),
        // Transport errors hand back the prompt as sent; resuming answers it.
        row("transport_error_resumes")
            .resume()
            .status(529)
            .reply(mock::text("ok"))
            .stops([Kind::Transport])
            .requests(2)
            .roles("UA")
            .extra(|run| assert_eq!(run.tally.beats, 1)),
        row("transport_error_after_tool_round")
            .resume()
            .reply(calls(&["a"]))
            .status(500)
            .reply(mock::text("ok"))
            .stops([Kind::Transport])
            .requests(3)
            .roles("UAUA")
            .dispatched(["a"]),
        row("transport_error_mid_pause")
            .resume()
            .reply(paused())
            .status(529)
            .reply(resumed())
            .stops([Kind::Transport])
            .requests(3)
            .roles("UA")
            .extra(|run| assert_settled(run.turn(1))),
        // A resumed paused turn is still dropped whole.
        row("refusal_after_a_resumed_pause")
            .resume()
            .reply(paused())
            .status(529)
            .reply(calls(&["r"]).refusal("cyber", "no"))
            .reply(mock::text("ok"))
            .stops([Kind::Transport, Kind::Unusable])
            .requests(4)
            .roles("UA")
            .last("ok")
            .extra(|run| {
                assert_eq!(run.sent_roles(2), "UA", "the resumed pause");
                assert_eq!(run.sent_roles(3), "U", "the paused turn dropped");
            }),
    ]
}

/// `turn`'s text blocks, joined — raw, where `Display` may render markdown.
fn text(turn: &Message) -> String {
    turn.content
        .iter()
        .filter_map(|block| match block {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        })
        .collect()
}

/// The handed-back pending notes' text.
fn pending(run: &Run) -> String {
    let pending = run.pending.as_ref().expect("a pending note");
    pending.content.to_string()
}

/// `turn` has no server tool in flight.
fn assert_settled(turn: &Message) {
    let unfinished: Vec<_> = turn.unfinished_server_tool_uses().collect();
    assert!(unfinished.is_empty(), "in flight: {unfinished:?}");
}
