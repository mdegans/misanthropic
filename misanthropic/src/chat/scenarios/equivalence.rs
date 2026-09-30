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
//! trust past what was seated. [`replay_cold`] then sends each recorded
//! prompt again with nothing to reuse, and [`assert_equivalent`] demands
//! the same reply, byte for byte, and the same prompt size.
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
//! token decoded alone and the same token prefilled in a batch can land
//! logits a rounding error apart, and greedy decoding flips on a near-tie.
//! A late divergence between two fluent replies may be that; one at the
//! first token, or a garbled warm reply, points at the cache. The report
//! prints each divergence's position and context, and the warm request's
//! usage (what it restored).
//!
//! **Start the server with `--no-penalty`.** A repetition penalty is
//! sampler state: warm reuse resumes it from the cached stream, while a
//! cold prefill rebuilds it from the prompt's prose (drama_llama's
//! `seed_prose_fold`), so the two can penalize different tokens and reply
//! differently with no KV corruption at all. blallama reports no sampling
//! configuration over its API, so the test can't check this; it prints a
//! reminder instead, and so does a failure.
//!
//! - `blallama::replays_cold`: live, skipped unless `BLALLAMA_URL` is set
//!   (see `live`) **and** `BLALLAMA_EQUIVALENCE=1`, since its flushes evict
//!   every slot: an exported `BLALLAMA_URL` alone must never let the
//!   pre-commit gate (`cargo test --all-features`) run it. Run with `just
//!   test-equivalence`, which sets both.
//! - `simulated_*`: offline, through a [`MockTransport`] standing in for a
//!   healthy cache and a stale one.

use std::{
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
/// never seated (as [`Chat`](crate::chat::Chat) never seats one): its
/// step's turns are dropped and the conversation moves on without them.
async fn converse<T: Transport>(transport: &T, base: Prompt) -> Vec<Exchange> {
    // Marked while it has no messages, so the mark lands on the system.
    let mut prompt = base
        .system(system())
        .add_tool(archive())
        .top_k(GREEDY)
        .cache();
    let mut log = Vec::new();
    for step in steps() {
        let start = prompt.messages.len();
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
            });
            let calls: Vec<tool::Use> =
                reply.inner.content.tool_uses().cloned().collect();
            let clipped = reply.stop_reason == Some(StopReason::MaxTokens);
            if clipped || (!calls.is_empty() && round + 1 == MAX_ROUNDS) {
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

/// `exchange`'s request replayed with nothing to reuse: after enough
/// [`flush`]es that the replay reads nothing from the cache. Returns the
/// reply and the flushes it took.
async fn cold<T: Transport>(
    transport: &T,
    n: usize,
    exchange: &Exchange,
) -> (response::Message, usize) {
    let prompt: Prompt =
        serde_json::from_str(&exchange.json).expect("a recorded request");
    assert_eq!(
        serde_json::to_string(&prompt).unwrap(),
        exchange.json,
        "request {n} doesn't round-trip, so it can't be replayed exactly"
    );
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

/// Every request of `warm`, replayed [`cold`], in order.
async fn replay_cold<T: Transport>(
    transport: &T,
    warm: &[Exchange],
) -> Vec<(response::Message, usize)> {
    let mut cold_replies = Vec::with_capacity(warm.len());
    for (n, exchange) in warm.iter().enumerate() {
        cold_replies.push(cold(transport, n + 1, exchange).await);
    }
    cold_replies
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

/// How a warm reply and its cold replay compare.
struct Verdict {
    step: &'static str,
    warm: response::Message,
    cold: response::Message,
    flushes: usize,
    divergence: Option<Divergence>,
}

impl Verdict {
    fn new(
        exchange: &Exchange,
        (cold, flushes): &(response::Message, usize),
    ) -> Self {
        Self {
            step: exchange.step,
            warm: exchange.reply.clone(),
            cold: cold.clone(),
            flushes: *flushes,
            divergence: Divergence::find(&exchange.reply, cold),
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

    /// The failure report for request `n`, if it failed.
    fn failure(&self, n: usize) -> Option<String> {
        if self.passed() {
            return None;
        }
        let (warm, cold) = (self.warm.usage.counts, self.cold.usage.counts);
        let usage = format!(
            "request {n} (`{}`): warm prompt {} tokens, cold {}; the warm \
             one restored {} from cache, wrote {}, paid {} fresh; {} / {} \
             output tokens (warm / cold), stops {:?} / {:?}",
            self.step,
            prompt_size(&warm),
            prompt_size(&cold),
            read(&self.warm),
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
        Some(usage + &detail)
    }
}

/// The per-request table.
fn table(verdicts: &[Verdict]) -> String {
    let header = format!(
        "{:>3} {:<11} {:>7} {:>7} {:>6} {:>7} {:>5} {:>5} {:>5} verdict",
        "req",
        "step",
        "prompt",
        "read",
        "input",
        "written",
        "out",
        "cold",
        "flush",
    );
    let rows = verdicts.iter().enumerate().map(|(n, v)| {
        let counts = v.warm.usage.counts;
        let verdict = match (&v.divergence, v.sizes_agree()) {
            (None, true) => "same".to_string(),
            (None, false) => "prompt sizes differ".to_string(),
            (Some(divergence), _) => divergence.brief(),
        };
        format!(
            "{:>3} {:<11} {:>7} {:>7} {:>6} {:>7} {:>5} {:>5} {:>5} {verdict}",
            n + 1,
            v.step,
            prompt_size(&counts),
            read(&v.warm),
            counts.input_tokens,
            counts.cache_creation_input_tokens.unwrap_or_default(),
            counts.output_tokens,
            v.cold.usage.counts.output_tokens,
            v.flushes,
        )
    });
    std::iter::once(header)
        .chain(rows)
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every warm reply must equal its cold replay, byte for byte, from a
/// prompt of the same size — and the warm run must have reused the cache
/// at all, or the check proves nothing.
fn assert_equivalent(warm: &[Exchange], cold: &[(response::Message, usize)]) {
    assert_eq!(warm.len(), cold.len(), "a cold replay per warm request");
    let verdicts: Vec<Verdict> = warm
        .iter()
        .zip(cold)
        .map(|(w, c)| Verdict::new(w, c))
        .collect();
    eprintln!("{}", table(&verdicts));

    assert!(
        warm.iter()
            .skip(1)
            .any(|exchange| read(&exchange.reply) > 0),
        "no warm request read from the cache, so the check proves nothing"
    );
    let failures: Vec<String> = verdicts
        .iter()
        .enumerate()
        .filter_map(|(n, verdict)| verdict.failure(n + 1))
        .collect();
    assert!(
        failures.is_empty(),
        "{} of {} requests replied differently cold:\n{}\n\nA late \
         divergence between two fluent replies may be llama.cpp's \
         batch-variant kernels breaking a near-tie; one at the first \
         token, or a garbled warm reply, points at the cache. {PENALTY}",
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
        eprintln!(
            "NOTE: this check assumes blallama runs with `--no-penalty`, \
             which it can't see. {PENALTY}"
        );
        let client = Retrying(super::super::live::client(&url));
        let base = Prompt::default().model(model).max_tokens(MAX_TOKENS);
        let warm = converse(&client, base).await;
        let cold = replay_cold(&client, &warm).await;
        assert_equivalent(&warm, &cold);
    }
}

/// The system prompt's first line, which marks a [`flush`].
const FLUSH_MARK: &str = ": cache flush.";

/// A stand-in for blallama with a one-slot cache and a deterministic
/// model. A request reads the previous one's size back when it extends
/// it; when `stale`, a reply built on a read comes back a char longer, as
/// from a corrupt KV cache.
fn simulated(stale: bool) -> MockTransport {
    let last: Mutex<Option<(Vec<String>, u64)>> = Mutex::default();
    MockTransport::with(move |prompt: &Prompt| {
        let system = prompt.system.as_ref().map(ToString::to_string);
        let turns: Vec<String> = system
            .iter()
            .cloned()
            .chain(prompt.messages.iter().map(|m| m.content.to_string()))
            .collect();
        let size = serde_json::to_string(prompt).unwrap().len() as u64 / 4;
        let mut last = last.lock().unwrap();
        let read = match last.as_ref() {
            Some((before, size)) if turns.starts_with(before) => *size,
            _ => 0,
        }
        .min(size - 3);
        *last = Some((turns, size));

        let mut counts = TokenCounts::new(3, 12);
        counts.cache_read_input_tokens = Some(read);
        counts.cache_creation_input_tokens = Some(size - read - 3);
        let mut reply = simulated_reply(prompt, system.as_deref())
            .counts(counts)
            .build();
        // A late slip, as from a stale cell: one char past the first text.
        let first = reply.inner.content.0.first_mut();
        if let Some(Block::Text { text, .. }) =
            first.filter(|_| stale && read > 0)
        {
            *text = format!("{text}!").into();
        }
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

/// The warm run through [`simulated`]`(stale)`, and its cold replays.
fn simulate(stale: bool) -> (Vec<Exchange>, Vec<(response::Message, usize)>) {
    let server = simulated(stale);
    futures::executor::block_on(async {
        let warm = converse(&server, Prompt::default()).await;
        let cold = replay_cold(&server, &warm).await;
        (warm, cold)
    })
}

/// Every step goes out as designed, and a healthy cache passes.
#[test]
fn simulated_healthy_cache_passes() {
    let (warm, cold) = simulate(false);

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

    assert_equivalent(&warm, &cold);
}

/// A reply that changes when built on the cache fails, with the
/// divergence and the warm request's reuse in the report.
#[test]
fn simulated_stale_cache_fails() {
    let (warm, cold) = simulate(true);
    let checked = std::panic::catch_unwind(|| assert_equivalent(&warm, &cold));
    let panic = checked.expect_err("a stale reply must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(
        message.contains("(text) of 2 / 2 diverges at char"),
        "{message}"
    );
    assert!(message.contains("restored"), "{message}");
    assert!(message.contains("turns.!"), "{message}");
    // The first request had nothing to read, so it matched.
    assert!(!message.contains("request 1 "), "{message}");
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
