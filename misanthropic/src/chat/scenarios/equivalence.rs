//! A warm-vs-cold equivalence check for a server's KV cache: under greedy
//! decoding (`top_k: 1`) a reply is a function of its prompt alone, so
//! however much of the prompt the server restored from its prefix cache,
//! the reply must match the same prompt prefilled from scratch. A mismatch
//! is the one visible sign of silent KV corruption — a restored prefix
//! whose cells don't hold what a fresh prefill would put there.
//!
//! [`converse`] runs [`steps`] warm, cached the way a long conversation is
//! on blallama (a marker on the system, a window on assistant turns, and
//! the server's own tip), recording each request exactly as sent. Its
//! beats aim at reuse's riskier paths: a long assistant turn, a long
//! grammar-constrained tool input (the segmentation-drift class of
//! drama_llama#91), a turn cut by a stop sequence, and a clipped turn
//! retried in full — the last two leave a tip the next prompt must not
//! trust past what was seated. A step clipped at `max_tokens` or still
//! calling after [`MAX_ROUNDS`] is dropped, and the table says so; the run
//! stops early ([`assert_forced_call_seated`]) unless the forced call and
//! its result round were both seated, since otherwise the #91 path went
//! unexercised. [`replay_cold`] then sends each recorded prompt again with
//! nothing to reuse, and [`assert_equivalent`] demands the same reply, byte
//! for byte, and the same prompt size.
//!
//! **Forcing a cold prefill.** blallama has no request-level bypass: its
//! cache-off mode is a session setting, and a prompt without markers still
//! reuses the internal tip. So [`cold`] evicts instead: it sends
//! [`FLUSHES`] tiny prompts unrelated to anything cached, each of which
//! misses and takes a slot from the least recently used, and the replay
//! must then report `cache_read_input_tokens` of zero — checked, and
//! retried with more flushes, never assumed. The flushes evict *every*
//! slot, so don't run this beside other cache-sensitive work on the same
//! server.
//!
//! **Reading a mismatch.** llama.cpp's kernels aren't batch-invariant: a
//! logit depends on the schedule that computed the KV behind it — where
//! the ubatch boundaries fell, whether a token was decoded alone or
//! prefilled in a batch, which other sequences shared the unified KV cache
//! — and greedy decoding flips on a near-tie. Restoring a breakpoint costs
//! nothing here (its cells were prefilled on the same ubatch grid a cold
//! prefill uses), but a request that reads back the previous turn's
//! generated tokens (the **tip**: `read_k > T_{k-1}`, counted as the
//! `cache` scenario's `read_past` counts it) restores cells warm decoded
//! one at a time, where cold prefills them in a batch. drama_llama's
//! `logits_determinism` TIP experiment shows the decoder restores a
//! stepped tip losslessly, but not that blallama's tip matching is sound
//! (the history re-rendered against the generated tokens, the final
//! sampled token, a stop sequence's trim, drama_llama#91). So each
//! mismatching request gets a control ([`recheck`]) and a class
//! ([`Mismatch`]):
//!
//! - A **tip** request `k` gets the **matched-schedule** control
//!   ([`matched`]): request `k-1` replayed cold, which must match warm
//!   `k-1` byte for byte (regenerating the tip one token at a time, as
//!   warm did), then request `k` sent with no flush, on that fresh tip,
//!   which must match warm `k`. Both match: **tip schedule (explained)**,
//!   passed with a note. `k-1` matches but `k` doesn't: **cache failure**,
//!   failed. `k-1` doesn't: **upstream nondeterminism**, which leaves `k`
//!   unchecked — explained, and warned about, when warm `k-1` already
//!   differed from its own cold replay (its row says why); otherwise cold
//!   disagrees with itself, which fails on a server declared to run one
//!   slot (`BLALLAMA_CACHE_SLOTS=1`) and warns on any other.
//! - Any other request is replayed cold a second time. The two cold
//!   replies agree: **cache suspect**, failed. They don't:
//!   **nondeterminism**, no evidence of corruption, warned about loudly
//!   with a reminder to run one slot.
//!
//! The matched control can't tell the schedule from a fault the tip match
//! commits the same way every time, and it prefills the turns before
//! `k-1` in a batch where warm decoded them. So every tip is also bounded:
//! it can't exceed the previous request's output tokens plus
//! [`TIP_SLACK`] (a stop sequence's overrun), or the server restored cells
//! nothing generated there — a **tip overread**, failed.
//!
//! A mismatch with no control fails, as does any difference in prompt
//! size, which no schedule explains. The report prints each divergence's
//! position and context, the warm request's usage (what it restored, and
//! how much of it was tip), and the control's verdict.
//!
//! **Start the server with `--no-penalty` and `--cache-slots 1`.** A
//! repetition penalty is sampler state: warm reuse resumes it from the
//! cached stream, while a cold prefill rebuilds it from the prompt's prose
//! (drama_llama's `seed_prose_fold`), so the two can penalize different
//! tokens and reply differently with no KV corruption at all. And other
//! sequences in the unified KV cache change a request's logits, so a
//! second slot is a neighbor one run has and the other doesn't; one slot
//! removes it. blallama reports neither setting over its API, so the test
//! can't check them; it prints a reminder instead, and so does a failure.
//! `BLALLAMA_CACHE_SLOTS` declares the slot count (`just
//! test-equivalence` sets it, `1` by default); only a declared single slot
//! fails cold disagreeing with itself upstream of a tip.
//!
//! - `blallama::replays_cold`: live, skipped unless `BLALLAMA_URL` is set
//!   (see `live`) **and** `BLALLAMA_EQUIVALENCE=1`, since its flushes evict
//!   every slot: an exported `BLALLAMA_URL` alone must never let the
//!   pre-commit gate (`cargo test --all-features`) run it. Run with `just
//!   test-equivalence`, which sets both.
//! - `simulated_*`: offline, through a [`MockTransport`] standing in for a
//!   healthy cache, a stale one, a server whose tip reuse shifts replies by
//!   its schedule, one whose tip reuse drifts with the cache's history,
//!   one that overreads its tip, and ones that aren't deterministic cold.

use std::{
    collections::HashMap,
    num::{NonZeroU16, NonZeroU32},
    panic::AssertUnwindSafe,
    sync::Mutex,
};

use crate::{
    Prompt, Transport,
    mock::{self, MockTransport, Reply},
    prompt::message::{Block, Role},
    response::{self, StopReason, TokenCounts},
    tool::{self, Choice, CustomMethodDef, MethodDef},
};

/// The one tool: files a titled text.
const ARCHIVE: &str = "archive";
/// `top_k: 1` — greedy, whatever the server's sampling chain.
const GREEDY: NonZeroU16 = NonZeroU16::MIN;
/// The clipped step's output budget.
const CLIP_TOKENS: NonZeroU32 = NonZeroU32::new(40).unwrap();
/// Tool rounds a step may take before its turns are dropped.
const MAX_ROUNDS: usize = 3;
/// Flushes before a cold replay, doubled on each retry. Twice blallama's
/// usual slot count, and cheap: a flush prefills a few dozen tokens.
const FLUSHES: usize = 8;
/// Tries at a cold replay before giving up.
const COLD_ATTEMPTS: u32 = 3;
/// Characters of context printed either side of a divergence.
const CONTEXT: usize = 60;
/// Why a warm and a cold reply may differ with a healthy cache.
const PENALTY: &str = "Unless the server runs with `--no-penalty`, a \
    repetition penalty resumed warm but rebuilt cold also explains a \
    mismatch.";
/// Why a run should have one slot.
const SLOTS: &str = "Start blallama with `--cache-slots 1`: other sequences \
    in its unified KV cache change a request's logits, so a second slot is \
    a neighbor one replay has and another doesn't.";
/// What a [`Mismatch::Schedule`] means.
const TIP_SCHEDULE: &str = "tip schedule (explained): replayed on warm's \
    schedule (the request before cold, then this one on the tip that \
    replay generated), both match warm byte for byte, so the gap to cold \
    is the tip decoded one token at a time warm but prefilled in a batch \
    cold. Note: a fault the tip match commits the same way every time \
    would match too; the tip bound and drama_llama's logits_determinism \
    TIP experiment narrow that down";
/// Tokens a tip may run past the previous request's output: a stop
/// sequence's overrun, decoded, then trimmed from the reply.
const TIP_SLACK: u64 = 8;

/// What the forced call must copy into its input: quotes, backslashes,
/// escapes spelled out, nested JSON, tabs, newlines, and multi-byte text,
/// so the call's JSON re-renders unlike the tokens that generated it.
const PASSAGE: &str = "Keeper's log, 3 March. Wind \"backing\" SW→NW, \
    force 7–8; glass at 29.6\" and falling. Café tab: €4.50 (paid).\n\
    Path: C:\\logs\\2026\\03.txt\n\
    Seen on the relay: {\"lamp\": {\"on\": true, \"rpm\": 12}, \
    \"tags\": [\"fog\", \"é\"]}\n\
    A tab:\there, an escape written out: \\n, a lone quote: \".\n\
    Naïve résumé of the night: 🌫️ fog at 02:10, cleared by 04:45 — two \
    ships sighted, both answered.";

/// The clipped step's beat, asked again in full by the next step.
const STORY: &str = "Write a 400-word story about a night shift in the \
    archive, in plain prose.";

/// Rules in the system prompt: about 1,500 tokens in all.
const RULES: usize = 30;

const SUBJECTS: [&str; 8] = [
    "intake desk",
    "map room",
    "cold store",
    "reading room",
    "bindery",
    "loading bay",
    "catalogue",
    "strongroom",
];

const DUTIES: [&str; 6] = [
    "Sign the day sheet before touching anything.",
    "Wear cotton gloves for paper older than fifty years.",
    "Log each move in the ledger with the time in UTC.",
    "Keep food and drink outside the door, always.",
    "Report damp or pests to the conservator the same day.",
    "Return every box to its shelf before the shift ends.",
];

/// The system prompt: a brief, then numbered house rules. Deterministic,
/// so every run caches the same prefix.
fn system() -> String {
    let brief = "You are the duty archivist's assistant at the Harbour \
        Archive, a fictional records office. Answer briefly and cite rules \
        by number. File anything the user asks to archive with the \
        `archive` tool.\n\n# House rules";
    let rule = |i: usize| {
        let subject = SUBJECTS[i % SUBJECTS.len()];
        let duty = DUTIES[(i * 7 + 3) % DUTIES.len()];
        let shelf = 3 + (i * 37) % 90;
        format!(
            "{i}. The {subject}, part {i}. Its boxes live on shelf {shelf}; \
             ask the senior archivist before moving more than {} of them. \
             {duty}",
            2 + i % 5
        )
    };
    std::iter::once(brief.to_string())
        .chain((1..=RULES).map(rule))
        .collect::<Vec<_>>()
        .join("\n")
}

/// [`ARCHIVE`]'s definition.
fn archive() -> MethodDef {
    CustomMethodDef::builder(ARCHIVE)
        .description("File a text in the archive under a title.")
        .string_param("title", "A short title for the item.", true)
        .string_param("body", "The text to file, exactly as given.", true)
        .build()
        .expect("a valid tool")
        .into()
}

/// [`ARCHIVE`]'s answer to `call`: deterministic, so both runs see the
/// same history.
fn archived(call: &tool::Use) -> Block {
    let field = |key: &str| {
        call.input
            .get(key)
            .and_then(|v| v.as_str())
            .unwrap_or_default()
    };
    let text = format!(
        "Filed \"{}\" ({} characters).",
        field("title"),
        field("body").chars().count()
    );
    tool::Result::new(call.id.clone(), text).into()
}

/// One user beat, and the knobs its first request carries (tool rounds
/// after it go out as the conversation stands).
struct Step {
    /// The step's name in the report.
    name: &'static str,
    say: String,
    knobs: fn(Prompt) -> Prompt,
}

/// The beats, in order.
fn steps() -> Vec<Step> {
    let step = |name, say: &str, knobs| Step {
        name,
        say: say.to_string(),
        knobs,
    };
    let forced = format!(
        "Archive this passage with the tool, titled \"Log 3 March\", \
         copying it into `body` exactly, character for character:\n\n\
         {PASSAGE}"
    );
    vec![
        step(
            "long",
            "In about 250 words, explain how intake, cataloguing and \
             retrieval fit together here, citing rules by number.",
            |p| p,
        ),
        step("forced_call", &forced, |p| {
            p.tool_choice(Choice::method(ARCHIVE))
        }),
        step(
            "auto_call",
            "Archive one more item titled \"Keys\": a short list of where \
             the three spare keys hang (make it up).",
            |p| p,
        ),
        step(
            "stop",
            "Reply with exactly these words: alpha beta STOP gamma",
            |p| p.tool_choice(Choice::none()).stop_sequences(["STOP"]),
        ),
        step("clip", STORY, |p| {
            p.tool_choice(Choice::none()).max_tokens(CLIP_TOKENS)
        }),
        step("retry", STORY, |p| p.tool_choice(Choice::none())),
        step(
            "summary",
            "Summarize our conversation in three sentences, naming both \
             archived items.",
            |p| p,
        ),
    ]
}

/// One request of the warm run.
struct Exchange {
    /// The [`Step`] it belongs to.
    step: &'static str,
    /// The request, serialized exactly as sent.
    json: String,
    reply: response::Message,
    /// Why its step's turns were dropped, if they were.
    dropped: Option<Dropped>,
}

/// Why [`converse`] dropped a step's turns rather than seat them.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dropped {
    /// A reply stopped at `max_tokens`.
    MaxTokens,
    /// The step was still calling tools after [`MAX_ROUNDS`] rounds.
    MaxRounds,
}

impl std::fmt::Display for Dropped {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MaxTokens => "max_tokens",
            Self::MaxRounds => "MAX_ROUNDS",
        })
    }
}

/// Send `prompt`, or panic naming `what`.
async fn send<T: Transport>(
    transport: &T,
    prompt: &Prompt,
    what: &str,
) -> response::Message {
    transport
        .send(prompt)
        .await
        .unwrap_or_else(|error| panic!("{what}: {error}"))
}

/// Run [`steps`] from `base` (its model and `max_tokens`), greedy and
/// cached, and hand back every request with its reply. A clipped reply is
/// never seated (as [`Chat`](crate::chat::Chat) never seats one), nor is a
/// step still calling after [`MAX_ROUNDS`]: its turns are dropped, marked
/// [`Dropped`], and the conversation moves on without them.
async fn converse<T: Transport>(transport: &T, base: Prompt) -> Vec<Exchange> {
    // Marked while it has no messages, so the mark lands on the system.
    let mut prompt = base
        .system(system())
        .add_tool(archive())
        .top_k(GREEDY)
        .cache();
    let mut log = Vec::new();
    for step in steps() {
        let (start, first) = (prompt.messages.len(), log.len());
        prompt
            .push_message((Role::User, step.say.as_str()))
            .expect("a beat follows an assistant turn");
        let mut request = (step.knobs)(prompt.clone());
        for round in 0.. {
            let what = format!("step `{}`, round {round}", step.name);
            let reply = send(transport, &request, &what).await;
            log.push(Exchange {
                step: step.name,
                json: serde_json::to_string(&request).unwrap(),
                reply: reply.clone(),
                dropped: None,
            });
            let calls: Vec<tool::Use> =
                reply.inner.content.tool_uses().cloned().collect();
            let clipped = reply.stop_reason == Some(StopReason::MaxTokens);
            let dropped = match () {
                _ if clipped => Some(Dropped::MaxTokens),
                _ if !calls.is_empty() && round + 1 == MAX_ROUNDS => {
                    Some(Dropped::MaxRounds)
                }
                _ => None,
            };
            if let Some(why) = dropped {
                log[first..]
                    .iter_mut()
                    .for_each(|exchange| exchange.dropped = Some(why));
                prompt.messages.truncate(start);
                break;
            }
            prompt
                .push_message(reply)
                .expect("a reply follows its prompt");
            prompt.cache_windowed(2);
            if calls.is_empty() {
                break;
            }
            let results: Vec<Block> = calls.iter().map(archived).collect();
            prompt
                .push_message((Role::User, results))
                .expect("results follow their calls");
            request = prompt.clone();
        }
    }
    log
}

/// The step whose call carries [`PASSAGE`].
const FORCED: &str = "forced_call";

/// `warm`'s forced call ran its course: a `tool_use`, then a result round,
/// both seated. Otherwise the long grammar-constrained input (drama_llama
/// #91's path) was never read back from the cache, and the run proves
/// nothing about it.
fn assert_forced_call_seated(warm: &[Exchange]) {
    let forced: Vec<&Exchange> =
        warm.iter().filter(|e| e.step == FORCED).collect();
    let called = forced
        .first()
        .is_some_and(|e| e.reply.inner.content.tool_uses().next().is_some());
    let seated =
        forced.len() >= 2 && forced.iter().all(|e| e.dropped.is_none());
    let rounds: Vec<String> = forced
        .iter()
        .map(|e| {
            format!(
                "{:?}, {} call(s), {}",
                e.reply.stop_reason,
                e.reply.inner.content.tool_uses().count(),
                e.dropped
                    .map_or("seated".into(), |why| format!("dropped: {why}"))
            )
        })
        .collect();
    assert!(
        called && seated,
        "the `{FORCED}` step didn't seat a call and its result round, so \
         the #91 path went unexercised: {rounds:?}"
    );
}

/// Tokens `reply` restored from the server's cache.
fn read(reply: &response::Message) -> u64 {
    reply
        .usage
        .counts
        .cache_read_input_tokens
        .unwrap_or_default()
}

/// The whole prompt behind `counts`: `input + creation + read`.
fn prompt_size(counts: &TokenCounts) -> u64 {
    counts.input_tokens
        + counts.cache_creation_input_tokens.unwrap_or_default()
        + counts.cache_read_input_tokens.unwrap_or_default()
}

/// Tokens of `before`'s generation that `now` read back, its tip:
/// `read_k - T_{k-1}`, or zero with nothing before.
fn tip(before: Option<&Exchange>, now: &Exchange) -> u64 {
    before.map_or(0, |before| {
        let prompt = prompt_size(&before.reply.usage.counts);
        super::cache::read_past(prompt, read(&now.reply))
    })
}

/// A prompt that shares nothing with any other: `id` leads its system,
/// it carries no marker, and its tip ends past anything another prompt
/// could match. Sending one takes a slot, evicting the least recently
/// used.
fn flush(model: &crate::model::Model, id: &str) -> Prompt {
    Prompt::default()
        .model(model.clone())
        .max_tokens(NonZeroU32::MIN)
        .top_k(GREEDY)
        .system(format!("{id}: cache flush."))
        .add_message((Role::User, "ok"))
        .expect("a user turn")
}

/// Flushes sent so far by this process, so no two are alike.
static FLUSHED: Mutex<u64> = Mutex::new(0);

/// A name for the next flush, unique to this run.
fn next_flush_id() -> String {
    let mut flushed = FLUSHED.lock().unwrap();
    *flushed += 1;
    let run = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{run:x}-{flushed}")
}

/// Request `n`, `exchange`, as recorded, checked to round-trip exactly.
fn replayable(n: usize, exchange: &Exchange) -> Prompt {
    let prompt: Prompt =
        serde_json::from_str(&exchange.json).expect("a recorded request");
    assert_eq!(
        serde_json::to_string(&prompt).unwrap(),
        exchange.json,
        "request {n} doesn't round-trip, so it can't be replayed exactly"
    );
    prompt
}

/// `exchange`'s request replayed with nothing to reuse: after enough
/// [`flush`]es that the replay reads nothing from the cache. Returns the
/// reply and the flushes it took.
async fn cold<T: Transport>(
    transport: &T,
    n: usize,
    exchange: &Exchange,
) -> (response::Message, usize) {
    let prompt = replayable(n, exchange);
    let mut flushes = FLUSHES;
    for _ in 0..COLD_ATTEMPTS {
        for _ in 0..flushes {
            let flush = flush(&prompt.model, &next_flush_id());
            // A failed flush is harmless: the replay's read says whether
            // the flushes as a whole worked.
            if let Err(error) = transport.send(&flush).await {
                eprintln!("request {n}: a flush failed: {error}");
            }
        }
        let what = format!("request {n}, replayed cold");
        let reply = send(transport, &prompt, &what).await;
        if read(&reply) == 0 {
            return (reply, flushes);
        }
        eprintln!(
            "request {n}: the replay still read {} cached tokens after {} \
             flushes; flushing more",
            read(&reply),
            flushes
        );
        flushes *= 2;
    }
    panic!(
        "request {n}: couldn't force a cold prefill in {COLD_ATTEMPTS} \
         tries: the replay kept reading from the cache"
    )
}

/// A cold replay's reply, and the flushes it took.
type Cold = (response::Message, usize);

/// Every request of `warm`, replayed [`cold`], in order.
async fn replay_cold<T: Transport>(
    transport: &T,
    warm: &[Exchange],
) -> Vec<Cold> {
    let mut cold_replies = Vec::with_capacity(warm.len());
    for (n, exchange) in warm.iter().enumerate() {
        cold_replies.push(cold(transport, n + 1, exchange).await);
    }
    cold_replies
}

/// The control replayed for a request whose cold replay disagreed with it
/// (see [`recheck`]).
#[derive(Clone)]
enum Control {
    /// No tip: the request replayed [`cold`] once more.
    Cold(response::Message),
    /// A tip: warm's schedule, matched ([`matched`]).
    Matched {
        /// The request before, replayed cold.
        before: response::Message,
        /// This request, sent on the tip `before` generated; `None` when
        /// `before` differed from warm, which leaves no warm tip to send it
        /// on. Boxed, as the variant is otherwise twice [`Self::Cold`].
        now: Option<Box<response::Message>>,
    },
}

/// The controls: for each request of `warm` whose `cold` replay disagreed
/// with it, the [`matched`] schedule where it read a tip, and otherwise
/// one more [`cold`] replay; `None` where they agreed.
async fn recheck<T: Transport>(
    transport: &T,
    warm: &[Exchange],
    cold_replies: &[Cold],
) -> Vec<Option<Control>> {
    let mut controls = Vec::with_capacity(warm.len());
    for (n, (exchange, (first, _))) in warm.iter().zip(cold_replies).enumerate()
    {
        let before = n.checked_sub(1).map(|before| &warm[before]);
        let tipped = tip(before, exchange) > 0;
        let control = match (agree(&exchange.reply, first), tipped) {
            (true, _) => None,
            (false, true) => Some(matched(transport, n + 1, warm).await),
            (false, false) => {
                let (again, _) = cold(transport, n + 1, exchange).await;
                Some(Control::Cold(again))
            }
        };
        controls.push(control);
    }
    controls
}

/// The matched-schedule control for request `k` (from one), which read
/// request `k - 1`'s tip: that request replayed [`cold`], which generates
/// its tip one token at a time as warm did, then, if it matched warm,
/// request `k` sent with no flush, so it reads that fresh tip as warm read
/// its own.
async fn matched<T: Transport>(
    transport: &T,
    k: usize,
    warm: &[Exchange],
) -> Control {
    let (previous, exchange) = (&warm[k - 2], &warm[k - 1]);
    let (before, _) = cold(transport, k - 1, previous).await;
    let now = match agree(&previous.reply, &before) {
        true => {
            let prompt = replayable(k, exchange);
            let what = format!("request {k}, on a matched tip");
            Some(Box::new(send(transport, &prompt, &what).await))
        }
        false => None,
    };
    Control::Matched { before, now }
}

/// Whether `a` and `b` are the same reply to prompts of the same size.
fn agree(a: &response::Message, b: &response::Message) -> bool {
    Divergence::find(a, b).is_none()
        && prompt_size(&a.usage.counts) == prompt_size(&b.usage.counts)
}

/// A block as the model wrote it: the prose of text and thought, a call's
/// name and input, the wire JSON of anything else.
fn written(block: &Block) -> String {
    match block {
        Block::Text { text, .. } => text.to_string(),
        Block::Thought { thought, .. } => thought.to_string(),
        Block::ToolUse { call } => format!("{}({})", call.name, call.input),
        other => serde_json::to_string(other).unwrap(),
    }
}

/// `block`'s wire `type`, e.g. `tool_use`.
fn kind(block: &Block) -> String {
    let value = serde_json::to_value(block).unwrap();
    value["type"].as_str().unwrap_or("block").to_string()
}

/// The first char at which `a` and `b` differ, if they do.
fn divergence(a: &str, b: &str) -> Option<usize> {
    let (mut a, mut b) = (a.chars(), b.chars());
    (0..).find_map(|at| match (a.next(), b.next()) {
        (None, None) => Some(None),
        (x, y) if x != y => Some(Some(at)),
        _ => None,
    })?
}

/// `text` around char `at`, quoted.
fn context(text: &str, at: usize) -> String {
    let start = at.saturating_sub(CONTEXT);
    let window: String = text.chars().skip(start).take(2 * CONTEXT).collect();
    format!("{window:?} (from char {start})")
}

/// Where a warm reply and its cold replay first differ.
enum Divergence {
    /// Block `block` differs from char `at` of the two sides, as
    /// [`written`] (or as wire JSON, when only a field outside that
    /// differs). A block one side lacks reads as empty.
    Block {
        block: usize,
        kind: String,
        at: usize,
        warm: String,
        cold: String,
    },
    /// The blocks agree; the stop reason or sequence doesn't.
    Stop,
}

impl Divergence {
    /// Where `warm` and `cold` first differ, or `None` if they don't:
    /// everything but the id and usage is compared.
    fn find(
        warm: &response::Message,
        cold: &response::Message,
    ) -> Option<Self> {
        let (w, c) = (&warm.inner.content, &cold.inner.content);
        let value = |block: Option<&Block>| {
            block.map(|block| serde_json::to_value(block).unwrap())
        };
        let block = (0..w.len().max(c.len())).find_map(|n| {
            let (a, b) = (w.get(n), c.get(n));
            if value(a) == value(b) {
                return None;
            }
            let text =
                |block: Option<&Block>| block.map(written).unwrap_or_default();
            let (mut warm, mut cold) = (text(a), text(b));
            if warm == cold {
                // A field outside the written text differs: show the JSON.
                let json = |block| value(block).unwrap_or_default().to_string();
                (warm, cold) = (json(a), json(b));
            }
            Some(Self::Block {
                block: n,
                kind: a.or(b).map(kind).unwrap_or_default(),
                at: divergence(&warm, &cold).unwrap_or_default(),
                warm,
                cold,
            })
        });
        let stopped = (&warm.stop_reason, &warm.stop_sequence)
            != (&cold.stop_reason, &cold.stop_sequence);
        block.or(stopped.then_some(Self::Stop))
    }

    /// A few words for the table.
    fn brief(&self) -> String {
        match self {
            Self::Block {
                block, kind, at, ..
            } => format!("block {block} ({kind}) at char {at}"),
            Self::Stop => "stop differs".into(),
        }
    }
}

/// What a mismatch means, read from the warm request's reuse and its
/// control (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Mismatch {
    /// A tip, and on warm's schedule ([`matched`]) the request before and
    /// this one both match warm. Passes, with a note.
    Schedule,
    /// A tip, and the request before, replayed cold, doesn't match warm,
    /// which leaves this request unchecked. `explained` when that request's
    /// own cold replay already differed from warm; otherwise cold disagrees
    /// with itself, which fails on a declared single slot and warns on any
    /// other.
    Upstream { explained: bool },
    /// A tip, and on warm's schedule the request before matches warm but
    /// this one doesn't. Fails.
    TipCache,
    /// No tip, and the two cold replies agree. Fails.
    Cache,
    /// No tip, and the two cold replies differ. Warned about.
    Nondeterminism,
    /// No control to read it by. Fails.
    Unchecked,
}

/// What a request's reading does to the check.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Outcome {
    /// Passes, with a note.
    Pass,
    /// Passes, with a warning.
    Warn,
    /// Fails.
    Fail,
}

impl std::fmt::Display for Outcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Pass => "pass",
            Self::Warn => "warn",
            Self::Fail => "FAIL",
        })
    }
}

impl Mismatch {
    /// What this does to the check, on a server declared to run one slot
    /// or not.
    fn outcome(self, single_slot: bool) -> Outcome {
        match self {
            Self::Schedule => Outcome::Pass,
            Self::Upstream { explained: false } if single_slot => Outcome::Fail,
            Self::Upstream { .. } | Self::Nondeterminism => Outcome::Warn,
            Self::TipCache | Self::Cache | Self::Unchecked => Outcome::Fail,
        }
    }

    /// Whether this is cold disagreeing with itself.
    fn noisy(self) -> bool {
        matches!(
            self,
            Self::Nondeterminism | Self::Upstream { explained: false }
        )
    }

    /// The class's name.
    fn name(self) -> &'static str {
        match self {
            Self::Schedule => "tip schedule (explained)",
            Self::Upstream { explained: true } => {
                "upstream nondeterminism (explained)"
            }
            Self::Upstream { explained: false } => "upstream nondeterminism",
            Self::TipCache => "cache failure",
            Self::Cache => "cache suspect",
            Self::Nondeterminism => "nondeterminism",
            Self::Unchecked => "unchecked",
        }
    }

    /// The reading, in full, for the report.
    fn explain(self, single_slot: bool) -> String {
        match self {
            Self::Schedule => TIP_SCHEDULE.to_string(),
            Self::Upstream { explained: true } => "upstream nondeterminism \
                (explained): the warm request before already differed from \
                its cold replay (its row says why), so cold can't generate \
                warm's tip, and this request is unchecked"
                .to_string(),
            Self::Upstream { explained: false } => format!(
                "upstream nondeterminism: the request before matched warm on \
                 its first cold replay but not on the control's, so cold \
                 disagrees with itself, and this request is unchecked; {}",
                match single_slot {
                    true => "on the single slot declared, that fails",
                    false =>
                        "only a single slot declared \
                        (BLALLAMA_CACHE_SLOTS=1) fails that",
                }
            ),
            Self::TipCache => "cache failure: on warm's schedule, the request \
                before matches warm but this one doesn't, so no schedule \
                explains it: the cache's history does"
                .to_string(),
            Self::Cache => "cache suspect: the warm request reused no \
                generated tokens and cold is self-consistent"
                .to_string(),
            Self::Nondeterminism => "nondeterminism: cold disagrees with \
                itself, so this is no evidence of corruption"
                .to_string(),
            Self::Unchecked => "unchecked: no control replay to tell the \
                cache from a schedule or nondeterminism"
                .to_string(),
        }
    }
}

/// How `other` compares with `reply`, called `name`, in a few words.
fn compared(
    reply: &response::Message,
    name: &str,
    other: &response::Message,
) -> String {
    match (agree(reply, other), Divergence::find(reply, other)) {
        (true, _) => format!("matches {name}"),
        (false, Some(divergence)) => {
            format!("differs from {name} ({})", divergence.brief())
        }
        (false, None) => format!("differs from {name} in prompt size"),
    }
}

/// How a warm reply and its cold replay compare.
struct Verdict {
    step: &'static str,
    warm: response::Message,
    cold: response::Message,
    flushes: usize,
    /// Tokens of the previous request's generation the warm request read
    /// back (the tip): `read_k - T_{k-1}`, or zero.
    reused: u64,
    /// The previous request's warm reply, if there was one.
    before: Option<response::Message>,
    /// Whether the previous request's first cold replay matched it.
    before_agreed: bool,
    divergence: Option<Divergence>,
    /// Made where the cold replay disagreed (see [`recheck`]).
    control: Option<Control>,
    /// Why the warm request's step was dropped, if it was.
    dropped: Option<Dropped>,
    /// Whether the server was declared to run one slot.
    single_slot: bool,
}

impl Verdict {
    /// `exchange`, sent after `before` (the previous request and its first
    /// cold replay, if any), against its `cold` replay and `control`.
    fn new(
        exchange: &Exchange,
        before: Option<(&Exchange, &response::Message)>,
        (cold, flushes): &Cold,
        control: Option<&Control>,
        single_slot: bool,
    ) -> Self {
        Self {
            step: exchange.step,
            warm: exchange.reply.clone(),
            cold: cold.clone(),
            flushes: *flushes,
            reused: tip(before.map(|(before, _)| before), exchange),
            before: before.map(|(before, _)| before.reply.clone()),
            before_agreed: before
                .is_some_and(|(before, cold)| agree(&before.reply, cold)),
            divergence: Divergence::find(&exchange.reply, cold),
            control: control.cloned(),
            dropped: exchange.dropped,
            single_slot,
        }
    }

    /// The control's reading, in a few words for the table.
    fn control_brief(&self) -> String {
        match &self.control {
            None => String::new(),
            Some(Control::Cold(control)) => match agree(&self.cold, control) {
                true => "; cold2 agrees".into(),
                false => "; cold2 differs".into(),
            },
            Some(Control::Matched { now: None, .. }) => {
                "; matched: k-1 differs".into()
            }
            Some(Control::Matched { now: Some(now), .. }) => {
                match agree(&self.warm, now) {
                    true => "; matched: k-1 same, k same".into(),
                    false => "; matched: k-1 same, k differs".into(),
                }
            }
        }
    }

    /// Whether both replies came from prompts of the same size.
    fn sizes_agree(&self) -> bool {
        prompt_size(&self.warm.usage.counts)
            == prompt_size(&self.cold.usage.counts)
    }

    fn passed(&self) -> bool {
        self.divergence.is_none() && self.sizes_agree()
    }

    /// The most tip [`reused`](Self::reused) may be: the previous
    /// request's output, plus [`TIP_SLACK`].
    fn bound(&self) -> Option<u64> {
        let before = self.before.as_ref()?;
        Some(before.usage.counts.output_tokens + TIP_SLACK)
    }

    /// Whether the tip runs past its [`bound`](Self::bound): cells restored
    /// where nothing generated them.
    fn overreads(&self) -> bool {
        self.bound().is_some_and(|bound| self.reused > bound)
    }

    /// What the mismatch means, if the replies differ.
    fn mismatch(&self) -> Option<Mismatch> {
        if self.passed() {
            return None;
        }
        Some(match &self.control {
            None => Mismatch::Unchecked,
            Some(Control::Cold(control)) => match agree(&self.cold, control) {
                true => Mismatch::Cache,
                false => Mismatch::Nondeterminism,
            },
            Some(Control::Matched { now: None, .. }) => Mismatch::Upstream {
                explained: !self.before_agreed,
            },
            Some(Control::Matched { now: Some(now), .. }) => {
                match agree(&self.warm, now) {
                    true => Mismatch::Schedule,
                    false => Mismatch::TipCache,
                }
            }
        })
    }

    /// What the mismatch does to the check, if the replies differ.
    fn outcome(&self) -> Option<Outcome> {
        Some(self.mismatch()?.outcome(self.single_slot))
    }

    /// Whether this request fails the check: a mismatch whose
    /// [outcome](Mismatch::outcome) fails, a tip that
    /// [overreads](Self::overreads), or prompts of different sizes, which
    /// no schedule explains.
    fn fails(&self) -> bool {
        !self.sizes_agree()
            || self.overreads()
            || self.outcome() == Some(Outcome::Fail)
    }

    /// The control's reading, in full, for the report on request `n`.
    fn control_report(&self, n: usize) -> String {
        match (&self.control, &self.before) {
            (None, _) => "\n  control: not replayed".to_string(),
            (Some(Control::Cold(control)), _) => {
                match agree(&self.cold, control) {
                    true => "\n  control: a second cold replay matches the \
                        first, so cold is self-consistent"
                        .to_string(),
                    false => format!(
                        "\n  control: a second cold replay {}, so the server \
                         isn't deterministic even cold",
                        compared(&self.cold, "the first", control)
                    ),
                }
            }
            (Some(Control::Matched { before, now }), warm_before) => {
                let replayed = warm_before
                    .as_ref()
                    .map_or("has no warm twin".into(), |warm| {
                        compared(warm, "warm", before)
                    });
                let sent = match now {
                    Some(now) => format!(
                        "request {n}, sent on its tip, {}, reading {} (warm \
                         read {})",
                        compared(&self.warm, "warm", now),
                        read(now),
                        read(&self.warm)
                    ),
                    None => format!(
                        "request {n} wasn't sent: no warm tip to send it on"
                    ),
                };
                format!(
                    "\n  control (matched schedule): request {}, replayed \
                     cold, {replayed}; {sent}",
                    n - 1
                )
            }
        }
    }

    /// The report for request `n`, if its replies differ or its tip
    /// overreads.
    fn report(&self, n: usize) -> Option<String> {
        let mismatch = self.mismatch();
        if mismatch.is_none() && !self.overreads() {
            return None;
        }
        let (warm, cold) = (self.warm.usage.counts, self.cold.usage.counts);
        let usage = format!(
            "request {n} (`{}`): warm prompt {} tokens, cold {}; the warm \
             one restored {} from cache ({} of them the previous request's \
             generated tokens), wrote {}, paid {} fresh; {} / {} output \
             tokens (warm / cold), stops {:?} / {:?}",
            self.step,
            prompt_size(&warm),
            prompt_size(&cold),
            read(&self.warm),
            self.reused,
            warm.cache_creation_input_tokens.unwrap_or_default(),
            warm.input_tokens,
            warm.output_tokens,
            cold.output_tokens,
            self.warm.stop_reason,
            self.cold.stop_reason,
        );
        let detail = match &self.divergence {
            Some(Divergence::Block {
                block,
                kind,
                at,
                warm,
                cold,
            }) => format!(
                "\n  block {block} ({kind}) of {} / {} diverges at char {at} \
                 of {} / {}\n  warm: {}\n  cold: {}",
                self.warm.inner.content.len(),
                self.cold.inner.content.len(),
                warm.chars().count(),
                cold.chars().count(),
                context(warm, *at),
                context(cold, *at),
            ),
            Some(Divergence::Stop) => format!(
                "\n  same blocks, but stop_sequence {:?} / {:?}",
                self.warm.stop_sequence, self.cold.stop_sequence
            ),
            None => String::new(),
        };
        let class = mismatch.map_or(String::new(), |mismatch| {
            format!(
                "{}\n  class: {}",
                self.control_report(n),
                mismatch.explain(self.single_slot)
            )
        });
        let sizes = match self.sizes_agree() {
            true => String::new(),
            false => "\n  FAIL: the prompt sizes differ, which no schedule \
                explains"
                .to_string(),
        };
        let overread = match (self.overreads(), self.bound()) {
            (true, Some(bound)) => format!(
                "\n  FAIL: tip overread: it read {} tokens past the previous \
                 request's prompt, which generated only {} (+{TIP_SLACK} \
                 slack, {bound} in all), so the server restored cells \
                 nothing generated there",
                self.reused,
                bound - TIP_SLACK
            ),
            _ => String::new(),
        };
        Some(usage + &detail + &class + &sizes + &overread)
    }
}

/// The per-request table.
fn table(verdicts: &[Verdict]) -> String {
    let header = format!(
        "{:>3} {:<11} {:>7} {:>7} {:>5} {:>6} {:>7} {:>8} {:>8} {:>5} \
         {:<10} verdict",
        "req",
        "step",
        "prompt",
        "read",
        "tip",
        "input",
        "written",
        "warm_out",
        "cold_out",
        "flush",
        "dropped",
    );
    let rows = verdicts.iter().enumerate().map(|(n, v)| {
        let counts = v.warm.usage.counts;
        let class = v.mismatch().map_or(String::new(), |mismatch| {
            format!("; {}: {}", mismatch.name(), v.outcome().unwrap())
        });
        let overread = match v.overreads() {
            true => "; tip overread: FAIL",
            false => "",
        };
        let verdict = match (&v.divergence, v.sizes_agree()) {
            (None, true) => "same".to_string(),
            (None, false) => "prompt sizes differ".to_string(),
            (Some(divergence), _) => divergence.brief(),
        } + &v.control_brief()
            + &class
            + overread;
        format!(
            "{:>3} {:<11} {:>7} {:>7} {:>5} {:>6} {:>7} {:>8} {:>8} {:>5} \
             {:<10} {verdict}",
            n + 1,
            v.step,
            prompt_size(&counts),
            read(&v.warm),
            v.reused,
            counts.input_tokens,
            counts.cache_creation_input_tokens.unwrap_or_default(),
            counts.output_tokens,
            v.cold.usage.counts.output_tokens,
            v.flushes,
            v.dropped.map_or("-".into(), |why| why.to_string()),
        )
    });
    let mut dropped: Vec<String> = verdicts
        .iter()
        .filter_map(|v| v.dropped.map(|why| format!("`{}` ({why})", v.step)))
        .collect();
    dropped.dedup();
    let dropped = match dropped.is_empty() {
        true => "dropped steps: none".to_string(),
        false => format!("dropped steps: {}", dropped.join(", ")),
    };
    let legend = "prompt / read / input / written: the warm request's \
        tokens; tip: how many of those read were the previous request's \
        generated tokens; warm_out / cold_out: output tokens of the warm \
        reply and of its cold replay; flush: flushes before that replay; \
        dropped: why the step's turns weren't seated; cold2: a second cold \
        replay; matched: request k-1 replayed cold, then k on its tip"
        .to_string();
    std::iter::once(header)
        .chain(rows)
        .chain([dropped, legend])
        .collect::<Vec<_>>()
        .join("\n")
}

/// Each request of `warm` against its `cold` replay and `controls`, on a
/// server declared to run one slot or not.
fn verdicts(
    warm: &[Exchange],
    cold: &[Cold],
    controls: &[Option<Control>],
    single_slot: bool,
) -> Vec<Verdict> {
    assert_eq!(warm.len(), cold.len(), "a cold replay per warm request");
    assert_eq!(warm.len(), controls.len(), "a control per warm request");
    let replies = warm.iter().zip(cold.iter().map(|(reply, _)| reply));
    let before = std::iter::once(None).chain(replies.map(Some));
    warm.iter()
        .zip(before)
        .zip(cold.iter().zip(controls))
        .map(|((w, b), (c, control))| {
            Verdict::new(w, b, c, control.as_ref(), single_slot)
        })
        .collect()
}

/// The mismatches that don't fail the check: a `NOTE` block for each that
/// passes, a `WARN` block for each warned about, and [`SLOTS`] after them
/// where cold disagreed with itself.
fn warnings(verdicts: &[Verdict]) -> Vec<String> {
    let warned: Vec<(Mismatch, String)> = verdicts
        .iter()
        .enumerate()
        .filter(|(_, verdict)| !verdict.fails())
        .filter_map(|(n, verdict)| {
            let report = verdict.report(n + 1)?;
            let mismatch = verdict.mismatch()?;
            let label = match verdict.outcome()? {
                Outcome::Pass => "NOTE",
                _ => "WARN",
            };
            let warning = format!("{label} {}: {report}", mismatch.name());
            Some((mismatch, warning))
        })
        .collect();
    let noisy = warned.iter().any(|(mismatch, _)| mismatch.noisy());
    let advice = noisy.then(|| {
        format!(
            "WARN nondeterminism: cold replays of the same request \
             disagreed, so this run can't vouch for the cache on those \
             requests. {SLOTS}"
        )
    });
    warned
        .into_iter()
        .map(|(_, warning)| warning)
        .chain(advice)
        .collect()
}

/// The reports of the requests that fail the check.
fn failures(verdicts: &[Verdict]) -> Vec<String> {
    verdicts
        .iter()
        .enumerate()
        .filter(|(_, verdict)| verdict.fails())
        .filter_map(|(n, verdict)| verdict.report(n + 1))
        .collect()
}

/// Every warm reply must equal its cold replay, byte for byte, from a
/// prompt of the same size, unless its control explains the difference
/// without the cache ([`Mismatch`]); no tip may run past what the previous
/// request generated; and the warm run must have reused the cache at all,
/// or the check proves nothing. `single_slot`: the server was declared to
/// run one slot.
fn assert_equivalent(
    warm: &[Exchange],
    cold: &[Cold],
    controls: &[Option<Control>],
    single_slot: bool,
) {
    let verdicts = verdicts(warm, cold, controls, single_slot);
    eprintln!("{}", table(&verdicts));

    assert!(
        warm.iter()
            .skip(1)
            .any(|exchange| read(&exchange.reply) > 0),
        "no warm request read from the cache, so the check proves nothing"
    );
    let warned = warnings(&verdicts);
    if !warned.is_empty() {
        eprintln!(
            "\n=== WARNING: requests replied differently cold without \
             failing the check. A divergence at the first token, or a \
             garbled warm reply, still points at the cache: read them. ==="
        );
    }
    warned.iter().for_each(|warning| eprintln!("\n{warning}"));
    let failures = failures(&verdicts);
    assert!(
        failures.is_empty(),
        "{} of {} requests failed: a reply that differs cold which no \
         control explains, a tip past what was generated, or a prompt of \
         another size:\n{}\n\nA divergence at the first token, or a garbled \
         warm reply, points at the cache the more surely. {PENALTY} {SLOTS}",
        failures.len(),
        verdicts.len(),
        failures.join("\n"),
    );
}

/// A [`Client`](crate::Client) riding out a 529 from a busy server.
#[derive(Clone)]
struct Retrying(crate::Client);

#[async_trait::async_trait]
impl Transport for Retrying {
    type Error = crate::client::Error;

    async fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<response::Message, Self::Error> {
        let send = || self.0.message(prompt);
        crate::utils::retry_transient("equivalence", send).await
    }

    async fn models(&self) -> Result<crate::model::Models, Self::Error> {
        self.0.models().await
    }
}

/// The check against a local blallama; see the module docs.
mod blallama {
    use super::*;

    /// Room for a local model's thinking.
    const MAX_TOKENS: NonZeroU32 = NonZeroU32::new(4096).unwrap();

    /// Whether the run is opted into, with `BLALLAMA_EQUIVALENCE=1`: it
    /// evicts every slot of the server's cache.
    fn opted() -> bool {
        std::env::var("BLALLAMA_EQUIVALENCE").is_ok_and(|v| v == "1")
    }

    #[tokio::test]
    async fn replays_cold() {
        let Some((url, model)) = super::super::live::target() else {
            return eprintln!("skipping `replays_cold`: BLALLAMA_URL is unset");
        };
        if !opted() {
            return eprintln!(
                "skipping `replays_cold`: it evicts every cache slot, and \
                 BLALLAMA_EQUIVALENCE=1 is unset (`just test-equivalence` \
                 sets it)"
            );
        }
        let slots = std::env::var("BLALLAMA_CACHE_SLOTS").ok();
        let single_slot = slots.as_deref().is_some_and(|v| v.trim() == "1");
        eprintln!(
            "NOTE: this check assumes blallama runs with `--no-penalty` and \
             `--cache-slots 1`, which it can't see. {PENALTY} {SLOTS} \
             BLALLAMA_CACHE_SLOTS is {slots:?}, so cold disagreeing with \
             itself upstream of a tip {}.",
            match single_slot {
                true => "fails",
                false => "only warns",
            }
        );
        let client = Retrying(super::super::live::client(&url));
        let base = Prompt::default().model(model).max_tokens(MAX_TOKENS);
        let warm = converse(&client, base).await;
        assert_forced_call_seated(&warm);
        let cold = replay_cold(&client, &warm).await;
        let controls = recheck(&client, &warm, &cold).await;
        assert_equivalent(&warm, &cold, &controls, single_slot);
    }
}

/// The system prompt's first line, which marks a [`flush`].
const FLUSH_MARK: &str = ": cache flush.";

/// How [`simulated`] departs from a healthy cache.
#[derive(Clone, Copy, PartialEq)]
enum Fault {
    /// None: every reply is a function of its prompt alone.
    Healthy,
    /// A reply built on a read comes back a char longer, as from a corrupt
    /// KV cache. Reads no tip, so each mismatch meets the plain control.
    Stale,
    /// A reply built on a tip comes back a char longer, as from a tip
    /// decoded token by token rather than in a batch: a function of the
    /// schedule, which the matched control reproduces.
    Tip,
    /// A reply built on a tip comes back a char longer once the slot has
    /// served reads two requests running, as from cells an earlier turn
    /// left stale: a function of the cache's history, which the matched
    /// control doesn't reproduce.
    Drift,
    /// [`Tip`](Self::Tip), and a reply that read nothing gains a `~` from
    /// the third sight of its prompt on, so a request's first cold replay
    /// agrees with warm and the matched control's doesn't.
    Flaky,
    /// A tip runs [`TIP_SLACK`] and one past the previous reply's output,
    /// cells nothing generated; the replies are untouched.
    Overread,
    /// A reply that read nothing gains a `~` for each earlier sight of its
    /// prompt, as from a server that isn't deterministic even cold. Reads
    /// no tip.
    Noisy,
}

impl Fault {
    /// Whether a read reaches into the previous reply where the prompt
    /// seats it, as blallama's tip does.
    fn tips(self) -> bool {
        !matches!(self, Self::Stale | Self::Noisy)
    }
}

/// Output tokens of every [`simulated`] reply.
const SIM_OUTPUT: u64 = 12;

/// A stand-in for blallama with a one-slot cache and a deterministic
/// model, but for its `fault`. A request reads the previous one's size back
/// when it extends it (and, where the fault [`tips`](Fault::tips), the
/// previous reply's output when it seats that reply).
fn simulated(fault: Fault) -> MockTransport {
    // The last prompt's turns and size, and the reply to it.
    let last: Mutex<Option<(Vec<String>, u64, String)>> = Mutex::default();
    let seen: Mutex<HashMap<String, usize>> = Mutex::default();
    // Requests in a row that read the cache.
    let streak: Mutex<usize> = Mutex::default();
    MockTransport::with(move |prompt: &Prompt| {
        let system = prompt.system.as_ref().map(ToString::to_string);
        let turns: Vec<String> = system
            .iter()
            .cloned()
            .chain(prompt.messages.iter().map(|m| m.content.to_string()))
            .collect();
        let json = serde_json::to_string(prompt).unwrap();
        let size = json.len() as u64 / 4;
        let sightings = {
            let mut seen = seen.lock().unwrap();
            let sightings = seen.entry(json).or_default();
            *sightings += 1;
            *sightings - 1
        };
        let mut last = last.lock().unwrap();
        let (read, tip) = match last.as_ref() {
            Some((before, size, reply)) if turns.starts_with(before) => {
                let seated = turns.get(before.len()) == Some(reply);
                let tip = fault.tips() && seated;
                let past = match (tip, fault) {
                    (false, _) => 0,
                    (true, Fault::Overread) => SIM_OUTPUT + TIP_SLACK + 1,
                    (true, _) => SIM_OUTPUT,
                };
                (size + past, tip)
            }
            _ => (0, false),
        };
        let read = read.min(size - 3);
        let streak = {
            let mut streak = streak.lock().unwrap();
            *streak = if read > 0 { *streak + 1 } else { 0 };
            *streak
        };

        let mut counts = TokenCounts::new(3, SIM_OUTPUT);
        counts.cache_read_input_tokens = Some(read);
        counts.cache_creation_input_tokens = Some(size - read - 3);
        let mut reply = simulated_reply(prompt, system.as_deref())
            .counts(counts)
            .build();
        // A late slip: past the first text.
        let slip = match fault {
            Fault::Stale if read > 0 => "!".to_string(),
            Fault::Tip | Fault::Flaky if tip => "!".to_string(),
            Fault::Drift if tip && streak >= 2 => "!".to_string(),
            Fault::Flaky if read == 0 && sightings >= 2 => "~".to_string(),
            Fault::Noisy if read == 0 => "~".repeat(sightings),
            _ => String::new(),
        };
        if let Some(Block::Text { text, .. }) =
            reply.inner.content.0.first_mut()
        {
            *text = format!("{text}{slip}").into();
        }
        *last = Some((turns, size, reply.inner.content.to_string()));
        reply
    })
}

/// What [`simulated`]'s model says to `prompt`, a function of it alone.
fn simulated_reply(prompt: &Prompt, system: Option<&str>) -> Reply {
    let tail = prompt.messages.last().expect("a request has a tail");
    let answered = tail
        .content
        .iter()
        .any(|block| matches!(block, Block::ToolResult { .. }));
    let stops = prompt.stop_sequences.iter().flatten();
    let stopped = stops.map(AsRef::as_ref).any(|stop: &str| stop == "STOP");
    let forced = matches!(prompt.tool_choice, Some(Choice::Method { .. }));
    let turns = prompt.messages.len();
    match () {
        _ if system.is_some_and(|s| s.contains(FLUSH_MARK)) => {
            mock::max_tokens("ok")
        }
        _ if answered => mock::text("Filed."),
        _ if forced => {
            let input = serde_json::to_value(ArchiveArgs {
                title: "Log 3 March",
                body: PASSAGE,
            })
            .unwrap();
            let call = tool::Use::new(ARCHIVE, input)
                .with_id(format!("call_{turns}_{ARCHIVE}"));
            mock::text("Filing.").call(call)
        }
        _ if stopped => {
            let mut reply = mock::text("alpha beta ")
                .stop_reason(StopReason::StopSequence)
                .build();
            reply.stop_sequence = Some("STOP".into());
            mock::message(reply)
        }
        _ if prompt.max_tokens == CLIP_TOKENS => mock::max_tokens("Once"),
        _ => mock::text(format!("Answer to {turns} turns.")),
    }
}

/// [`ARCHIVE`]'s arguments, as [`simulated`] calls it.
#[derive(serde::Serialize)]
struct ArchiveArgs {
    title: &'static str,
    body: &'static str,
}

/// The warm run through [`simulated`], its cold replays, and their
/// controls.
type Simulated = (Vec<Exchange>, Vec<Cold>, Vec<Option<Control>>);

/// [`simulated`]`(fault)`'s warm run, cold replays and controls.
fn simulate(fault: Fault) -> Simulated {
    let server = simulated(fault);
    futures::executor::block_on(async {
        let warm = converse(&server, Prompt::default()).await;
        let cold = replay_cold(&server, &warm).await;
        let controls = recheck(&server, &warm, &cold).await;
        (warm, cold, controls)
    })
}

/// Each verdict's class.
fn classes(verdicts: &[Verdict]) -> Vec<Option<Mismatch>> {
    verdicts.iter().map(Verdict::mismatch).collect()
}

/// The message `assert_equivalent` fails with.
fn failed(
    (warm, cold, controls): &Simulated,
    single_slot: bool,
    why: &str,
) -> String {
    let checked = std::panic::catch_unwind(AssertUnwindSafe(|| {
        assert_equivalent(warm, cold, controls, single_slot)
    }));
    let panic = checked.expect_err(why);
    panic.downcast_ref::<String>().expect("a message").clone()
}

/// Every step goes out as designed, and a healthy cache passes.
#[test]
fn simulated_healthy_cache_passes() {
    let (warm, cold, controls) = simulate(Fault::Healthy);

    let steps: Vec<_> = warm.iter().map(|exchange| exchange.step).collect();
    assert_eq!(
        steps,
        [
            "long",
            "forced_call",
            "forced_call",
            "auto_call",
            "stop",
            "clip",
            "retry",
            "summary"
        ],
        "a tool round after the forced call"
    );
    let first: Prompt = serde_json::from_str(&warm[0].json).unwrap();
    assert_eq!(first.top_k, Some(GREEDY));
    assert!(first.system.as_ref().is_some_and(|s| s.has_cache()));
    // The clip was dropped: the retry follows the stop step's reply.
    let retry: Prompt = serde_json::from_str(&warm[6].json).unwrap();
    let roles: String = retry
        .messages
        .iter()
        .map(|m| if m.role.is_user() { 'U' } else { 'A' })
        .collect();
    assert_eq!(roles, "UAUAUAUAUAU");
    let retried = retry.messages.last().map(|m| m.content.to_string());
    assert_eq!(retried.as_deref(), Some(STORY));
    // Each replay is cold, and flushing took one round.
    assert!(
        cold.iter().all(|(reply, flushes)| {
            read(reply) == 0 && *flushes == FLUSHES
        })
    );

    assert!(controls.iter().all(Option::is_none), "nothing to recheck");
    assert_forced_call_seated(&warm);
    let dropped: Vec<_> = warm.iter().map(|e| e.dropped).collect();
    let clip = Some(Dropped::MaxTokens);
    assert_eq!(dropped[5], clip, "the clip step");
    assert_eq!(dropped.iter().filter(|d| d.is_some()).count(), 1);
    let verdicts = verdicts(&warm, &cold, &controls, true);
    // The tips were read, within their bound.
    assert_eq!(verdicts[1].reused, SIM_OUTPUT, "the long reply, read back");
    assert!(verdicts.iter().all(|v| !v.overreads()));
    let printed = table(&verdicts);
    assert!(printed.contains("dropped steps: `clip` (max_tokens)"));
    assert!(printed.lines().nth(6).unwrap().contains(" max_tokens "));

    assert_equivalent(&warm, &cold, &controls, true);
}

/// A reply that changes when built on the cache fails, with the
/// divergence and the warm request's reuse in the report.
#[test]
fn simulated_stale_cache_fails() {
    let simulated = simulate(Fault::Stale);
    let (warm, cold, controls) = &simulated;
    assert!(controls[0].is_none(), "request 1 read nothing, so matched");
    assert!(
        matches!(controls[1], Some(Control::Cold(_))),
        "request 2 read no tip, so it's replayed cold again"
    );
    let message = failed(&simulated, true, "a stale reply must fail");
    assert!(
        message.contains("(text) of 2 / 2 diverges at char"),
        "{message}"
    );
    assert!(message.contains("restored"), "{message}");
    assert!(message.contains("(0 of them the previous"), "{message}");
    assert!(message.contains("turns.!"), "{message}");
    // The first request had nothing to read, so it matched.
    assert!(!message.contains("request 1 "), "{message}");
    // The stale reply is deterministic cold, so the control blames the cache.
    assert!(message.contains("cold is self-consistent"), "{message}");
    assert!(message.contains("class: cache suspect"), "{message}");
    assert!(message.contains("--cache-slots 1"), "{message}");
    let verdicts = verdicts(warm, cold, controls, true);
    assert!(warnings(&verdicts).is_empty(), "every mismatch failed");
}

/// A reply that changes only where the warm request read back the
/// previous request's generated tokens, and the same way on warm's
/// schedule, is a tip schedule: passed with a note. A tip request after
/// one is upstream of it: explained, and warned about.
#[test]
fn simulated_tip_schedule_passes_with_a_note() {
    let (warm, cold, controls) = simulate(Fault::Tip);
    let verdicts = verdicts(&warm, &cold, &controls, true);
    let schedule = Some(Mismatch::Schedule);
    let upstream = Some(Mismatch::Upstream { explained: true });
    assert_eq!(
        classes(&verdicts),
        [
            None, schedule, upstream, upstream, upstream, upstream, None,
            schedule
        ],
        "{}",
        table(&verdicts)
    );
    // The retry re-sends the clipped prompt, whose reply was never seated:
    // it read the cache, but none of the clip's output.
    assert!(read(&warm[6].reply) > 0, "the retry read the cache");
    assert_eq!(verdicts[6].reused, 0, "but no tip");
    // Request 2's control: request 1 cold, then request 2 on its tip.
    let Some(Control::Matched {
        before,
        now: Some(now),
    }) = &controls[1]
    else {
        panic!("request 2 read a tip, so its schedule is matched");
    };
    assert!(agree(&warm[0].reply, before) && agree(&warm[1].reply, now));
    assert_eq!(read(now), read(&warm[1].reply), "it read the fresh tip");

    assert!(failures(&verdicts).is_empty());
    let warned = warnings(&verdicts);
    assert_eq!(warned.len(), 6, "{warned:#?}");
    let notes: Vec<_> = warned
        .iter()
        .filter(|w| w.starts_with("NOTE tip schedule (explained): "))
        .collect();
    assert_eq!(notes.len(), 2, "{warned:#?}");
    assert!(notes.iter().all(|w| w.contains(TIP_SCHEDULE)), "{notes:#?}");
    let generated = format!("({SIM_OUTPUT} of them the previous request's");
    assert!(notes[0].contains(&generated), "{notes:#?}");
    assert!(
        notes[0].contains(
            "control (matched schedule): request 1, replayed cold, \
             matches warm; request 2, sent on its tip, matches warm"
        ),
        "{notes:#?}"
    );
    let explained = "WARN upstream nondeterminism (explained): ";
    assert_eq!(
        warned.iter().filter(|w| w.starts_with(explained)).count(),
        4,
        "{warned:#?}"
    );
    let printed = table(&verdicts);
    assert!(
        printed.contains(
            "matched: k-1 same, k same; tip schedule (explained): pass"
        ),
        "{printed}"
    );
    assert!(
        printed.contains(
            "matched: k-1 differs; upstream nondeterminism (explained): warn"
        ),
        "{printed}"
    );

    assert_equivalent(&warm, &cold, &controls, true);
}

/// A tip reply that changes with the cache's history, not its schedule,
/// is a cache failure: on warm's schedule the request before matches warm
/// and this one doesn't.
#[test]
fn simulated_tip_cache_failure_fails() {
    let simulated = simulate(Fault::Drift);
    let (warm, cold, controls) = &simulated;
    let verdicts = verdicts(warm, cold, controls, false);
    let failure = Some(Mismatch::TipCache);
    let upstream = Some(Mismatch::Upstream { explained: true });
    assert_eq!(
        classes(&verdicts),
        [
            None, None, failure, upstream, upstream, upstream, None, failure
        ],
        "{}",
        table(&verdicts)
    );
    let message = failed(&simulated, false, "a drifting tip must fail");
    assert!(message.starts_with("2 of 8 requests failed"), "{message}");
    assert!(message.contains("class: cache failure"), "{message}");
    assert!(
        message.contains(
            "request 2, replayed cold, matches warm; request 3, sent on its \
             tip, differs from warm (block 0 (text) at char"
        ),
        "{message}"
    );
    let printed = table(&verdicts);
    assert!(
        printed.contains("matched: k-1 same, k differs; cache failure: FAIL"),
        "{printed}"
    );
}

/// Cold disagreeing with itself on the request before a tip leaves the tip
/// unchecked: warned about on a server not declared to run one slot,
/// failed on one that is.
#[test]
fn simulated_upstream_nondeterminism_fails_on_one_slot() {
    let simulated = simulate(Fault::Flaky);
    let (warm, cold, controls) = &simulated;
    let unexplained = Some(Mismatch::Upstream { explained: false });
    let explained = Some(Mismatch::Upstream { explained: true });
    let many = verdicts(warm, cold, controls, false);
    assert_eq!(
        classes(&many),
        [
            None,
            unexplained,
            explained,
            explained,
            explained,
            explained,
            None,
            unexplained
        ],
        "{}",
        table(&many)
    );
    assert!(failures(&many).is_empty());
    let warned = warnings(&many);
    assert!(
        warned[0].starts_with("WARN upstream nondeterminism: "),
        "{warned:#?}"
    );
    assert!(warned[0].contains("only a single slot declared"));
    assert!(warned[0].contains("request 1, replayed cold, differs"));
    assert!(warned[0].contains("request 2 wasn't sent"));
    let advice = warned.last().unwrap();
    assert!(advice.contains("--cache-slots 1"), "{advice}");
    assert_equivalent(warm, cold, controls, false);

    let one = verdicts(warm, cold, controls, true);
    assert_eq!(failures(&one).len(), 2);
    let message = failed(&simulated, true, "one slot must be deterministic");
    assert!(message.contains("on the single slot declared"), "{message}");
    let printed = table(&one);
    assert!(
        printed.contains("matched: k-1 differs; upstream nondeterminism: FAIL"),
        "{printed}"
    );
}

/// A tip past what the previous request generated fails, though every
/// reply matches its cold replay.
#[test]
fn simulated_tip_overread_fails() {
    let simulated = simulate(Fault::Overread);
    let (warm, cold, controls) = &simulated;
    assert!(controls.iter().all(Option::is_none), "every reply matches");
    let verdicts = verdicts(warm, cold, controls, true);
    assert_eq!(verdicts[1].reused, SIM_OUTPUT + TIP_SLACK + 1);
    assert!(verdicts[1].passed() && verdicts[1].overreads());
    let tipped = verdicts.iter().filter(|v| v.reused > 0).count();
    assert_eq!(failures(&verdicts).len(), tipped);
    assert!(warnings(&verdicts).is_empty());
    let message = failed(&simulated, true, "an overread must fail");
    let generated = format!(
        "which generated only {SIM_OUTPUT} (+{TIP_SLACK} slack, {} in all)",
        SIM_OUTPUT + TIP_SLACK
    );
    assert!(message.contains(&generated), "{message}");
    assert!(message.contains("FAIL: tip overread"), "{message}");
    assert!(!message.contains("class:"), "no reply differs: {message}");
    assert!(table(&verdicts).contains("same; tip overread: FAIL"));
}

/// Cold replays that disagree with each other are nondeterminism: warned
/// about loudly, with the one-slot advice, not failed.
#[test]
fn simulated_nondeterminism_warns() {
    let (warm, cold, controls) = simulate(Fault::Noisy);
    let verdicts = verdicts(&warm, &cold, &controls, true);
    assert!(
        verdicts
            .iter()
            .all(|v| v.mismatch() == Some(Mismatch::Nondeterminism)),
        "{}",
        table(&verdicts)
    );

    assert!(failures(&verdicts).is_empty());
    let warned = warnings(&verdicts);
    assert_eq!(warned.len(), verdicts.len() + 1, "{warned:#?}");
    assert!(warned[0].starts_with("WARN nondeterminism"), "{warned:#?}");
    assert!(warned[0].contains("isn't deterministic even cold"));
    let advice = warned.last().unwrap();
    assert!(advice.contains("--cache-slots 1"), "{advice}");
    let printed = table(&verdicts);
    assert!(printed.contains("cold2 differs; nondeterminism: warn"));

    assert_equivalent(&warm, &cold, &controls, true);
}

/// A mismatch on a tip is read by its matched schedule: explained where
/// both requests match warm, a cache failure where only the one before
/// does, upstream nondeterminism where that one doesn't (failed on one
/// slot unless its own cold replay already differed). Elsewhere it's the
/// cache when the cold replies agree, nondeterminism when they don't, and
/// unchecked without a control. A tip past its bound, and any difference in
/// prompt size, fail whatever the replies.
#[test]
fn mismatches_are_classed() {
    let exchange = |reply: Reply| Exchange {
        step: "long",
        json: String::new(),
        reply: reply.build(),
        dropped: None,
    };
    // A 100-token prompt generating 20, then one reading 97 of it, or 10
    // past it.
    let before = exchange(mock::text("before").usage(100, 20));
    let (breakpoint, past) = (
        exchange(mock::text("warm").usage(3, 1).cache_read(97)),
        exchange(mock::text("warm").usage(3, 1).cache_read(110)),
    );
    let reply = |text: &'static str, size| mock::text(text).usage(size, 1);
    let verdict = |warm: &Exchange,
                   control: Option<Control>,
                   agreed: bool,
                   single_slot: bool| {
        let size = prompt_size(&warm.reply.usage.counts);
        let first = (reply("cold", size).build(), FLUSHES);
        let before_cold = match agreed {
            true => before.reply.clone(),
            false => reply("other", 120).build(),
        };
        let before = Some((&before, &before_cold));
        Verdict::new(warm, before, &first, control.as_ref(), single_slot)
    };
    let again = |text| Some(Control::Cold(reply(text, 100).build()));
    let matched = |now: Option<&'static str>| {
        Some(Control::Matched {
            before: before.reply.clone(),
            now: now.map(|text| Box::new(reply(text, 113).build())),
        })
    };

    let suspect = verdict(&breakpoint, again("cold"), true, true);
    assert_eq!(suspect.reused, 0);
    assert_eq!(suspect.mismatch(), Some(Mismatch::Cache));
    assert!(suspect.fails());
    let report = suspect.report(1).expect("a mismatch");
    assert!(report.contains("class: cache suspect"), "{report}");
    assert!(table(&[suspect]).contains("cold2 agrees; cache suspect: FAIL"));

    let noisy = verdict(&breakpoint, again("cool"), true, true);
    assert_eq!(noisy.mismatch(), Some(Mismatch::Nondeterminism));
    assert!(!noisy.fails(), "even on one slot");
    let report = noisy.report(1).expect("a mismatch");
    assert!(report.contains("no evidence of corruption"), "{report}");
    assert!(report.contains("block 0 (text) at char 2"), "{report}");
    assert!(table(&[noisy]).contains("cold2 differs; nondeterminism: warn"));

    let unchecked = verdict(&breakpoint, None, true, true);
    assert_eq!(unchecked.mismatch(), Some(Mismatch::Unchecked));
    assert!(unchecked.fails());
    let report = unchecked.report(1).expect("a mismatch");
    assert!(report.contains("control: not replayed"), "{report}");
    let tip_unchecked = verdict(&past, None, true, false);
    assert_eq!(tip_unchecked.mismatch(), Some(Mismatch::Unchecked));
    assert!(tip_unchecked.fails(), "a tip no longer excuses itself");

    let schedule = verdict(&past, matched(Some("warm")), true, false);
    assert_eq!(schedule.reused, 10, "read 110 of a 100-token prompt");
    assert_eq!(schedule.mismatch(), Some(Mismatch::Schedule));
    assert_eq!(schedule.outcome(), Some(Outcome::Pass));
    assert!(!schedule.fails());
    let report = schedule.report(2).expect("a mismatch");
    assert!(report.contains(TIP_SCHEDULE), "{report}");
    assert!(
        report.contains("request 1, replayed cold, matches warm"),
        "{report}"
    );
    let printed = table(&[schedule]);
    assert!(printed.contains("k same; tip schedule (explained): pass"));

    let failure = verdict(&past, matched(Some("cold")), true, false);
    assert_eq!(failure.mismatch(), Some(Mismatch::TipCache));
    assert!(failure.fails());
    let report = failure.report(2).expect("a mismatch");
    assert!(report.contains("class: cache failure"), "{report}");
    assert!(
        report.contains(
            "on its tip, differs from warm (block 0 (text) at char 0)"
        ),
        "{report}"
    );
    let printed = table(&[failure]);
    assert!(printed.contains("k-1 same, k differs; cache failure: FAIL"));

    let cases = [
        (true, true, Outcome::Fail),
        (true, false, Outcome::Warn),
        (false, true, Outcome::Warn),
        (false, false, Outcome::Warn),
    ];
    for (agreed, single_slot, outcome) in cases {
        let upstream = verdict(&past, matched(None), agreed, single_slot);
        let explained = !agreed;
        let case = format!("{agreed}, {single_slot}");
        assert_eq!(
            upstream.mismatch(),
            Some(Mismatch::Upstream { explained }),
            "{case}"
        );
        assert_eq!(upstream.outcome(), Some(outcome), "{case}");
        assert_eq!(upstream.fails(), outcome == Outcome::Fail, "{case}");
        let report = upstream.report(2).expect("a mismatch");
        assert!(report.contains("request 2 wasn't sent"), "{report}");
    }

    // The first request has nothing before it: no tip, and no bound.
    let first = (reply("cold", 113).build(), FLUSHES);
    let alone = Verdict::new(&past, None, &first, None, true);
    assert_eq!((alone.reused, alone.bound()), (0, None));

    // A tip may run TIP_SLACK past the 20 tokens generated, no further,
    // however well the replies match.
    let matching = |past_by: u64| {
        let warm = exchange(
            mock::text("warm")
                .usage(3, 1)
                .cache_read(100 + 20 + past_by),
        );
        let size = prompt_size(&warm.reply.usage.counts);
        let cold = (reply("warm", size).build(), FLUSHES);
        let before = Some((&before, &before.reply));
        Verdict::new(&warm, before, &cold, None, true)
    };
    let within = matching(TIP_SLACK);
    assert!(within.passed() && !within.fails() && within.report(1).is_none());
    let beyond = matching(TIP_SLACK + 1);
    assert!(beyond.passed() && beyond.overreads() && beyond.fails());
    let report = beyond.report(2).expect("an overread");
    assert!(report.contains("FAIL: tip overread"), "{report}");
    let printed = table(&[beyond]);
    assert!(printed.contains("same; tip overread: FAIL"), "{printed}");

    // No schedule explains a prompt of another size, tip or not.
    let resized = (reply("warm", 111).build(), FLUSHES);
    let before_pair = Some((&before, &before.reply));
    let resized = Verdict::new(&past, before_pair, &resized, None, true);
    assert!(resized.fails());
    let report = resized.report(1).expect("a mismatch");
    assert!(report.contains("no schedule explains"), "{report}");
    assert!(failures(&[resized]).len() == 1);
}

/// A server whose cache never clears is reported, not trusted.
#[test]
fn a_cache_that_wont_clear_is_reported() {
    let server = MockTransport::with(|_: &Prompt| {
        mock::text("hi").usage(3, 1).cache_read(100)
    });
    let exchange = Exchange {
        step: "long",
        json: serde_json::to_string(
            &Prompt::default().add_message((Role::User, "hi")).unwrap(),
        )
        .unwrap(),
        reply: mock::text("hi").build(),
        dropped: None,
    };
    let replay = std::panic::catch_unwind(AssertUnwindSafe(|| {
        futures::executor::block_on(cold(&server, 1, &exchange))
    }));
    let panic = replay.expect_err("a warm replay must not pass as cold");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(
        message.contains("couldn't force a cold prefill"),
        "{message}"
    );
    let flushes: usize = (0..COLD_ATTEMPTS).map(|n| FLUSHES << n).sum();
    let sent = flushes + COLD_ATTEMPTS as usize;
    assert_eq!(server.len(), sent, "flushes doubled each try");
}

#[test]
fn divergence_finds_the_first_differing_char() {
    assert_eq!(divergence("same", "same"), None);
    assert_eq!(divergence("", ""), None);
    assert_eq!(divergence("abc", "abd"), Some(2));
    assert_eq!(
        divergence("ab", "abc"),
        Some(2),
        "a prefix diverges at its end"
    );
    assert_eq!(divergence("é🌫️x", "é🌫️y"), Some(3), "counted in chars");
}

#[test]
fn context_is_quoted_and_clamped() {
    let text = "a".repeat(10) + "\"b\n";
    assert_eq!(context(&text, 10), format!("{text:?} (from char 0)"));
    let long = "x".repeat(200);
    assert!(context(&long, 150).ends_with("(from char 90)"));
}

#[test]
fn divergence_names_the_block_or_the_stop() {
    let call = |body: &'static str| {
        let input =
            serde_json::to_value(ArchiveArgs { title: "t", body }).unwrap();
        mock::text("Filing.")
            .call(tool::Use::new(ARCHIVE, input).with_id("call_0_archive"))
            .build()
    };
    assert!(Divergence::find(&call("a\"b"), &call("a\"b")).is_none());
    let found = Divergence::find(&call("a\"b"), &call("a\"c"));
    let found = found.expect("a divergence");
    assert_eq!(found.brief(), "block 1 (tool_use) at char 32");

    let stopped = |sequence: &'static str| {
        let mut reply = mock::text("alpha beta ")
            .stop_reason(StopReason::StopSequence)
            .build();
        reply.stop_sequence = Some(sequence.into());
        reply
    };
    let found = Divergence::find(&stopped("STOP"), &stopped("END"));
    assert_eq!(found.map(|d| d.brief()).as_deref(), Some("stop differs"));
}

/// A forced call that was clipped, went uncalled, or never got its result
/// round seated leaves the #91 path unexercised, and fails.
#[test]
fn an_unseated_forced_call_fails() {
    let unexercised = |warm: Vec<Exchange>, why: &str| {
        let checked = std::panic::catch_unwind(AssertUnwindSafe(|| {
            assert_forced_call_seated(&warm)
        }));
        let panic = checked.expect_err(why);
        let message = panic.downcast_ref::<String>().expect("a message");
        assert!(message.contains("went unexercised"), "{message}");
    };
    let warm = || simulate(Fault::Healthy).0;

    let mut clipped = warm();
    clipped[2].dropped = Some(Dropped::MaxTokens);
    clipped[1].dropped = Some(Dropped::MaxTokens);
    unexercised(clipped, "a clipped result round");

    let mut uncalled = warm();
    uncalled[1].reply = mock::text("No.").build();
    unexercised(uncalled, "no call");

    let mut unanswered = warm();
    unanswered.remove(2);
    unexercised(unanswered, "no result round");
}
