//! A multi-turn prompt-caching check: one [`Chat`] run of scripted beats
//! over a long, stable system prompt, cached the way a growing conversation
//! should be on Anthropic — [`Chat::cache`] (automatic, so the breakpoint
//! follows the tail) plus a marker on the system (so tools and system
//! survive anything that rewrites the tail). [`assert_caches`] then holds
//! every request to a healthy loop's signature, whichever backend served it:
//!
//! - the prompt, `P = input + creation + read`, grows every request;
//! - from the second on, `read` covers the previous request's prompt, less
//!   a small moved tail, so only the new turns are paid for fresh;
//! - `input` stays small however long the prompt grows;
//! - latency per output token doesn't climb with the prompt.
//!
//! A broken cache fails the second and third: the whole prompt is
//! re-prefilled every turn, which is what makes a long conversation crawl.
//! blallama may read past the new breakpoint (a tip hit: the previous turn's
//! generated KV), which only moves tokens from `creation` or `input` to
//! `read`.
//!
//! Every entry point runs the same beats:
//!
//! - `blallama::canonical` and `blallama::after_assistant`: a local
//!   drama_llama server, skipped unless `BLALLAMA_URL` is set (see `live`).
//!   The first is configured exactly as for Anthropic; the second reports
//!   [`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant)
//!   so the driver marks assistant turns instead. Each request's `P` must
//!   also equal the server's `count_tokens`. Run with `just test-cache`.
//! - `anthropic`: `#[ignore]`d and **paid** — `claude-haiku-4-5`, about
//!   three cents — with `misanthropic/api.key`.
//! - `simulated_*`: offline, through a [`MockTransport`] standing in for a
//!   healthy cache and a broken one.

use std::{
    num::NonZeroU32,
    sync::{Arc, Mutex},
};

use serde_json::Value;

use crate::{
    Client, Id, Prompt, Transport,
    chat::{
        BoxError, Chat,
        checks::{Checked, Log},
    },
    mock::{self, MockTransport, Reply},
    model,
    prompt::message::{Block, CacheControl, Role},
    response::{self, TokenCounts},
    tool::{self, CustomMethodDef, MethodDef, Tool, ToolBox, Use},
};

/// The largest minimum cacheable prefix among current models (Haiku 4.5,
/// Opus 4.5 and 4.6). A shorter prefix silently caches nothing.
const MIN_CACHEABLE: u64 = 4096;
/// What the first prompt must reach: well clear of [`MIN_CACHEABLE`].
const PREFIX_FLOOR: u64 = 5000;
/// Tokens a request may re-pay from the end of the previous prompt (the
/// framing after its last cacheable block).
const MOVED_TAIL: u64 = 64;
/// The most uncached `input` a request after the first may pay: a beat or
/// a tool result, never the conversation.
const INPUT_CAP: u64 = 1024;
/// How many times the early median the late median of milliseconds per
/// output token may be, plus [`LATENCY_SLACK_MS`]. Generous: the token
/// counts are the load-bearing check, latency a coarse one.
const LATENCY_RATIO: f64 = 3.0;
/// See [`LATENCY_RATIO`].
const LATENCY_SLACK_MS: f64 = 20.0;

/// The user's beats. Four ask for a tool (`ledger`, `stores`), which a
/// model takes as a round before it answers.
const BEATS: [&str; 10] = [
    "Who checks the fog signal, and how often?",
    "Read the ledger entry for day 3 and tell me who was on watch.",
    "What is the lamp's service code?",
    "Count the lamp mantles in the stores. Should I reorder?",
    "Which shelf holds the radio's spares?",
    "Read the ledger entry for day 7 and sum up the weather in a few words.",
    "If the cistern reads low, whom do I tell?",
    "How much paraffin is in the stores?",
    "Sum up our conversation so far in one sentence.",
    "Last one: what does section 12 cover?",
];

/// Handbook sections in the system prompt: about 6,500 tokens in all.
const SECTIONS: u64 = 72;

const SUBJECTS: [&str; 12] = [
    "lamp",
    "lens",
    "fog signal",
    "radio",
    "cistern",
    "generator",
    "boat davit",
    "weather station",
    "landing stage",
    "battery bank",
    "tide gauge",
    "gallery rail",
];

const KEEPERS: [&str; 6] = ["Ada", "Birgit", "Cormac", "Dilys", "Euan", "Fen"];

const DUTIES: [&str; 10] = [
    "Wipe any glass with the chamois from the brass locker, never paper.",
    "Write each reading in the day book in ink, with the time in UTC.",
    "Wear the harness on the gallery whenever the wind is above force 6.",
    "Keep the spares tagged with the date they came off the supply boat.",
    "Report a fault by radio at the next scheduled call, not before.",
    "Never leave the tower door open while the lamp is lit.",
    "Test the backup before touching the primary, and log both results.",
    "Return every tool to its hook on the shadow board by the stairs.",
    "Note the barometer beside each entry; falling pressure means rounds.",
    "Hand over at the change of watch in person, never by note alone.",
];

/// A deterministic choice of `n` for section `i` under `salt`.
fn pick(i: u64, salt: u64, n: usize) -> usize {
    let mixed = i
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(salt.wrapping_mul(0xBF58_476D_1CE4_E5B9));
    ((mixed >> 29) % n as u64) as usize
}

/// Section `i` of the station handbook.
fn section(i: u64) -> String {
    let subject = SUBJECTS[pick(i, 1, SUBJECTS.len())];
    let keeper = KEEPERS[pick(i, 2, KEEPERS.len())];
    let other = KEEPERS[pick(i, 3, KEEPERS.len())];
    let days = 1 + pick(i, 4, 7);
    let shelf = 1 + pick(i, 5, 30);
    let percent = 20 + 5 * pick(i, 6, 10);
    let code = 1000 + pick(i, 7, 9000);
    let duty = DUTIES[pick(i, 8, DUTIES.len())];
    let also = DUTIES[pick(i, 9, DUTIES.len())];
    format!(
        "Section {i}. The {subject}, part {i}. {keeper} checks it every \
         {days} days and signs its card. Its service code is SK-{code}, and \
         its spares are on shelf {shelf}. If it reads below {percent} \
         percent, tell {other} before the next watch and log the time. \
         {duty} {also}"
    )
}

/// The system prompt: a short brief, then the handbook. Deterministic, so
/// every run (and every backend) caches the same prefix.
fn system() -> String {
    let brief = "You are the relief keeper's assistant at Skerry Light, a \
        fictional lighthouse station. Answer in one short sentence, citing \
        handbook sections by number. When asked to read the ledger or count \
        the stores, use the tool; never guess what it would return.\n\n\
        # Station handbook";
    std::iter::once(brief.to_string())
        .chain((1..=SECTIONS).map(section))
        .collect::<Vec<_>>()
        .join("\n\n")
}

/// The wire name of [`Ledger`]'s method, as the [`ToolBox`] routes it.
const LEDGER: &str = "toolbox__Ledger__entry";
/// The wire name of [`Stores`]' method.
const STORES: &str = "toolbox__Stores__count";

/// The station ledger: one day's entry, about 80 words.
struct Ledger;

/// [`Ledger`]'s arguments.
#[derive(serde::Serialize)]
struct DayArgs {
    day: &'static str,
}

/// The station stores: how many of an item are left.
struct Stores;

/// [`Stores`]' arguments.
#[derive(serde::Serialize)]
struct ItemArgs {
    item: &'static str,
}

/// `call`'s string argument `key`, however the model typed it.
fn argument(call: &Use, key: &str) -> String {
    match call.input.get(key) {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// A stable number from `text`, for the tools' made-up records.
fn seed(text: &str) -> u64 {
    text.bytes()
        .fold(7, |seed, byte| seed.wrapping_mul(31) + u64::from(byte))
}

#[async_trait::async_trait]
impl Tool for Ledger {
    fn name(&self) -> &str {
        "Ledger"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Custom(CustomMethodDef::with_string_param(
            "Ledger__entry",
            "Read the station ledger's entry for a day.",
            "day",
            "The day number, e.g. `3`.",
            true,
        ))]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        let day = argument(&call, "day");
        let n = seed(&day);
        let wind = ["north", "east", "south", "west"][pick(n, 1, 4)];
        let force = 1 + pick(n, 2, 9);
        let keeper = KEEPERS[pick(n, 3, KEEPERS.len())];
        let blasts = pick(n, 4, 40);
        let entry = format!(
            "Day {day}. Wind {wind} force {force}, sea moderate, pressure \
             falling slowly. Lamp lit at 19:{:02} and out at dawn. Keeper on \
             watch: {keeper}. Fog signal sounded {blasts} times after \
             midnight. One drum of paraffin opened; the old drum returned \
             to the store. Remarks: gallery rail wet, harness worn, nothing \
             else to report.",
            pick(n, 5, 60),
        );
        tool::Result::new(call.id, entry)
    }
}

#[async_trait::async_trait]
impl Tool for Stores {
    fn name(&self) -> &str {
        "Stores"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Custom(CustomMethodDef::with_string_param(
            "Stores__count",
            "Count an item in the station stores.",
            "item",
            "What to count, e.g. `paraffin`.",
            true,
        ))]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        let item = argument(&call, "item");
        let n = seed(&item.to_lowercase());
        let (have, reorder) = (pick(n, 1, 40), 5 + pick(n, 2, 10));
        let count = format!("{item}: {have} in store; reorder at {reorder}.");
        tool::Result::new(call.id, count)
    }
}

/// Drive [`BEATS`] through `transport` from `base` (its model and
/// `max_tokens`) with the caching knobs a long conversation wants, and
/// hand back what was sent and received.
async fn scenario<T>(transport: Checked<T>, base: Prompt) -> Log
where
    T: Transport + Clone,
{
    let sink = Arc::new(Mutex::new(TokenCounts::default()));
    let toolbox = ToolBox::new().add(Ledger).add(Stores);
    // Marked while it has no messages, so the mark lands on the system,
    // covering the tools (they render first) too.
    let prompt = base.system(system()).cache();
    let mut beats = BEATS.iter();
    let mut next_beat = async |_: &mut ()| {
        let beat = beats.next().map(|&beat| vec![(Role::User, beat).into()]);
        Ok::<_, BoxError>(beat)
    };

    let outcome = Chat::new(transport.clone(), prompt, toolbox)
        .cache(CacheControl::ephemeral())
        .track_usage(Arc::clone(&sink))
        .run((), &mut next_beat)
        .await;

    let log = std::mem::take(&mut *transport.log());
    if let Err(error) = outcome {
        eprintln!("{}", table(&requests(&log, None)));
        panic!("the scenario stopped: {error}");
    }
    assert_eq!(*sink.lock().unwrap(), log.usage(), "track_usage");
    log
}

/// One request's accounting.
struct Request {
    /// `tool` for a tool round, `beat` for a user beat.
    kind: &'static str,
    input: u64,
    written: u64,
    read: u64,
    /// The server's `count_tokens` for the request, when asked.
    counted: Option<u64>,
    output: u64,
    millis: f64,
}

impl Request {
    /// The whole prompt: `input + creation + read`, as Anthropic bills it.
    fn prompt(&self) -> u64 {
        self.input + self.written + self.read
    }

    /// Milliseconds per output token.
    fn per_token(&self) -> f64 {
        self.millis / self.output.max(1) as f64
    }
}

/// `log`'s requests, with `counted` alongside if given.
fn requests(log: &Log, counted: Option<&[u64]>) -> Vec<Request> {
    let sent = log.sent.iter().zip(&log.received).zip(&log.elapsed);
    sent.enumerate()
        .map(|(n, ((prompt, reply), elapsed))| {
            let counts = reply.usage.counts;
            let tail = prompt.messages.last().expect("a request has a tail");
            let tool_round = tail
                .content
                .iter()
                .any(|block| matches!(block, Block::ToolResult { .. }));
            Request {
                kind: if tool_round { "tool" } else { "beat" },
                input: counts.input_tokens,
                written: counts.cache_creation_input_tokens.unwrap_or_else(
                    || panic!("request {}: no cache_creation", n + 1),
                ),
                read: counts.cache_read_input_tokens.unwrap_or_else(|| {
                    panic!("request {}: no cache_read", n + 1)
                }),
                counted: counted.map(|counted| counted[n]),
                output: counts.output_tokens,
                millis: elapsed.as_secs_f64() * 1e3,
            }
        })
        .collect()
}

/// The per-request table, and the run's totals.
fn table(requests: &[Request]) -> String {
    let header = format!(
        "{:>3} {:<4} {:>7} {:>7} {:>6} {:>7} {:>7} {:>5} {:>7} {:>7} {:>5}",
        "req",
        "kind",
        "prompt",
        "counted",
        "input",
        "written",
        "read",
        "out",
        "ms",
        "ms/out",
        "hit%",
    );
    let rows = requests.iter().enumerate().map(|(n, r)| {
        let counted = r.counted.map_or("-".into(), |c| c.to_string());
        format!(
            "{:>3} {:<4} {:>7} {:>7} {:>6} {:>7} {:>7} {:>5} {:>7.0} \
             {:>7.1} {:>5.1}",
            n + 1,
            r.kind,
            r.prompt(),
            counted,
            r.input,
            r.written,
            r.read,
            r.output,
            r.millis,
            r.per_token(),
            100.0 * r.read as f64 / r.prompt().max(1) as f64,
        )
    });
    let sum = |f: fn(&Request) -> u64| requests.iter().map(f).sum::<u64>();
    let (read, prompt) = (sum(|r| r.read), sum(Request::prompt));
    let totals = format!(
        "total: prompt {prompt}, input {}, written {}, read {read} \
         ({:.1}% of all prompt tokens read from cache)",
        sum(|r| r.input),
        sum(|r| r.written),
        100.0 * read as f64 / prompt.max(1) as f64,
    );
    std::iter::once(header)
        .chain(rows)
        .chain([totals])
        .collect::<Vec<_>>()
        .join("\n")
}

/// The median of `values`.
fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.collect();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Hold `log` to a healthy cached loop's signature (see the module docs),
/// after printing its table. `counted` is each request's `count_tokens`,
/// which `input + creation + read` must equal exactly.
fn assert_caches(log: &Log, counted: Option<&[u64]>) {
    let requests = requests(log, counted);
    eprintln!("{}", table(&requests));

    assert!(requests.len() >= BEATS.len(), "a request per beat at least");
    assert!(
        requests.iter().any(|r| r.kind == "tool"),
        "no tool round: the scenario covers tool results too"
    );
    for (n, r) in requests.iter().enumerate() {
        if let Some(counted) = r.counted {
            assert_eq!(
                r.prompt(),
                counted,
                "request {}: input + creation + read vs count_tokens",
                n + 1
            );
        }
    }

    let first = &requests[0];
    assert!(
        first.prompt() >= PREFIX_FLOOR,
        "the first prompt ({}) is too short to cache on every model",
        first.prompt()
    );
    assert!(
        first.written + first.read >= MIN_CACHEABLE,
        "request 1 neither wrote nor read the cached prefix"
    );
    for (n, pair) in requests.windows(2).enumerate() {
        let (before, now, k) = (&pair[0], &pair[1], n + 2);
        assert!(
            now.prompt() > before.prompt(),
            "request {k}: the prompt didn't grow ({} after {})",
            now.prompt(),
            before.prompt()
        );
        assert!(
            now.read + MOVED_TAIL >= before.prompt(),
            "request {k} read {} tokens of the previous prompt's {}: the \
             prefix was re-prefilled",
            now.read,
            before.prompt()
        );
        assert!(
            now.input <= INPUT_CAP,
            "request {k} paid {} uncached input tokens (cap {INPUT_CAP})",
            now.input
        );
    }

    // Request 1 pays the cold prefill: compare the next three to the last.
    if requests.len() < 7 {
        return eprintln!("too few requests to compare latency");
    }
    let early = median(requests[1..4].iter().map(Request::per_token));
    let late = requests[requests.len() - 3..].iter();
    let late = median(late.map(Request::per_token));
    assert!(
        late <= LATENCY_RATIO * early + LATENCY_SLACK_MS,
        "late turns take {late:.1} ms per output token, early ones \
         {early:.1}: latency climbs with the prompt"
    );
}

/// Each of `log`'s requests counted by `client`'s `count_tokens`.
async fn counted(client: &Client, log: &Log) -> Vec<u64> {
    let mut counted = Vec::with_capacity(log.sent.len());
    for prompt in &log.sent {
        let count = client.count_tokens(prompt).await.expect("count_tokens");
        counted.push(u64::from(count));
    }
    counted
}

/// A [`Client`] reporting
/// [`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant),
/// as agentkit's blallama transport does.
#[cfg(feature = "blallama")]
#[derive(Clone)]
struct AfterAssistant(Client);

#[cfg(feature = "blallama")]
#[async_trait::async_trait]
impl Transport for AfterAssistant {
    type Error = crate::client::Error;

    async fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<response::Message, Self::Error> {
        self.0.message(prompt).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        self.0.models().await
    }

    fn quirks(&self) -> crate::Quirks {
        crate::Quirks {
            breakpoint_after_assistant: true,
            ..crate::Quirks::default()
        }
    }
}

/// The scenario against a local blallama; see the module docs.
#[cfg(feature = "blallama")]
mod blallama {
    use super::*;

    /// Run the scenario through `wrap`ped [`Client`], unless no server is
    /// configured.
    async fn run<T>(name: &str, wrap: impl FnOnce(Client) -> T)
    where
        T: Transport + Clone,
    {
        let Some((url, model)) = super::super::live::target() else {
            return eprintln!("skipping `{name}`: BLALLAMA_URL is unset");
        };
        let client = super::super::live::client(&url);
        // Room for a local model's thinking, and for the prompt beside it.
        let base = Prompt::default()
            .model(model)
            .max_tokens(NonZeroU32::new(8192).unwrap());
        let log = scenario(Checked::new(wrap(client.clone())), base).await;
        let counted = counted(&client, &log).await;
        assert_caches(&log, Some(&counted));
    }

    /// Configured exactly as for Anthropic.
    #[tokio::test]
    async fn canonical() {
        run("canonical", |client| client).await;
    }

    /// Markers on assistant turns, as agentkit places them for blallama.
    #[tokio::test]
    async fn after_assistant() {
        run("after_assistant", AfterAssistant).await;
    }
}

/// A [`Client`] riding out a transient 429 or 529, as the other live tests
/// do.
#[derive(Clone)]
struct Retrying(Client);

#[async_trait::async_trait]
impl Transport for Retrying {
    type Error = crate::client::Error;

    async fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<response::Message, Self::Error> {
        let send = || self.0.message(prompt);
        crate::utils::retry_transient("cache scenario", send).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        self.0.models().await
    }
}

/// The reference: Anthropic itself, on the model with the largest minimum.
#[tokio::test]
#[ignore = "PAID: live Anthropic API (claude-haiku-4-5, about 3 cents); \
            needs misanthropic/api.key"]
async fn anthropic() {
    let client = Client::new(crate::utils::load_api_key().await).unwrap();
    let base = Prompt::default()
        .model(Id::Haiku45)
        .max_tokens(NonZeroU32::new(512).unwrap());
    let log = scenario(Checked::new(Retrying(client)), base).await;
    assert_caches(&log, None);
}

/// A stand-in for Anthropic's cache and model. It sizes a request by its
/// serialized length and writes it all to the cache; when `healthy` it
/// first reads back the previous request's prompt if this one extends it,
/// broken it never does (a prefix that changes every turn). A beat naming
/// the ledger or the stores gets one call, anything else an answer.
fn simulated(healthy: bool) -> MockTransport {
    let last: Mutex<Option<(Vec<String>, u64)>> = Mutex::default();
    MockTransport::with(move |prompt: &Prompt| {
        let turns: Vec<String> = prompt
            .messages
            .iter()
            .map(|turn| serde_json::to_string(turn).unwrap())
            .collect();
        let size = serde_json::to_string(prompt).unwrap().len() as u64 / 4;
        let mut last = last.lock().unwrap();
        let read = match last.as_ref() {
            Some((before, size)) if healthy && turns.starts_with(before) => {
                *size
            }
            _ => 0,
        };
        *last = Some((turns, size));

        // Everything to the breakpoint on the last block is written; what
        // follows it is framing.
        let mut counts = TokenCounts::new(3, 12);
        counts.cache_read_input_tokens = Some(read);
        counts.cache_creation_input_tokens = Some(size - read - 3);
        simulated_reply(prompt).counts(counts)
    })
}

/// What [`simulated`]'s model says to `prompt`.
fn simulated_reply(prompt: &Prompt) -> Reply {
    let tail = prompt.messages.last().expect("a request has a tail");
    let beat = tail.content.to_string().to_lowercase();
    let answered = tail
        .content
        .iter()
        .any(|block| matches!(block, Block::ToolResult { .. }));
    let (ledger, stores) = (DayArgs { day: "3" }, ItemArgs { item: "oil" });
    match () {
        _ if answered => mock::text("Noted."),
        _ if beat.contains("ledger") => {
            let args = serde_json::to_value(ledger).unwrap();
            mock::text("Reading.").tool_use(LEDGER, args)
        }
        _ if beat.contains("stores") => {
            let args = serde_json::to_value(stores).unwrap();
            mock::text("Counting.").tool_use(STORES, args)
        }
        _ => mock::text("Section 1."),
    }
}

/// The scenario against [`simulated`]`(healthy)`, and its log.
fn simulate(healthy: bool) -> Log {
    let transport = Checked::new(Arc::new(simulated(healthy)));
    let base = Prompt::default().model(Id::Haiku45);
    futures::executor::block_on(scenario(transport, base))
}

/// The knobs reach the wire, the tools run, and a healthy cache passes.
#[test]
fn simulated_healthy_cache_passes() {
    let log = simulate(true);

    let first = &log.sent[0];
    assert!(first.cache_control.is_some(), "automatic caching is on");
    let system = first.system.as_ref().expect("a system prompt");
    assert!(system.has_cache(), "the system carries a marker");
    let names = first.tools.iter().flatten().map(MethodDef::name);
    assert_eq!(names.collect::<Vec<_>>(), [LEDGER, STORES]);
    assert_eq!(log.sent.len(), BEATS.len() + 4, "four tool rounds");

    assert_caches(&log, None);
}

/// A cache rewritten every turn fails, however well the rest goes.
#[test]
fn simulated_broken_cache_fails() {
    let log = simulate(false);
    let checked = std::panic::catch_unwind(|| assert_caches(&log, None));
    let panic = checked.expect_err("a re-prefilled prompt must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("re-prefilled"), "{message}");
}
