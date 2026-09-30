//! A multi-turn prompt-caching check: a [`Chat`] run of scripted beats
//! over a long, stable system prompt, cached the way a growing conversation
//! should be on Anthropic — [`Chat::cache`] (automatic, so the breakpoint
//! follows the tail) plus a marker on the system (so tools and system
//! survive anything that rewrites the tail). [`assert_caches`] prints a
//! per-request table, then holds every request to a healthy loop's
//! signature, whichever backend served it. With `T` the prompt
//! (`input + creation + read`, as Anthropic bills it) and `k` the request:
//!
//! - `read_k` covers what request `k - 1` cached, as Anthropic would read
//!   it, less a backend's slack: all of `T_{k-1}` when the breakpoint
//!   follows the tail (Anthropic re-pays only the few tokens of framing
//!   after it), or `T_{k-1} - input_{k-1}` when markers ride assistant
//!   turns (the beat after the marker was never cached);
//! - `input` stays small however long the prompt grows;
//! - the time a request spends beyond decoding fits prefilling only its
//!   uncached tokens (`input + creation`), at the rate the run's coldest
//!   request measured — not the whole prompt. A server that reports reads
//!   but re-prefills anyway fails here. When no request prefilled enough
//!   to measure (the server was warm from an earlier run), a backend's
//!   fallback rate stands in (blallama's: `BLALLAMA_PREFILL_RATE` tokens a
//!   second, 420 by default), and the table says which.
//!
//! A broken cache fails the first and last: the whole prompt is re-prefilled
//! every turn, which is what makes a long conversation crawl.
//!
//! Printed, and held only on Anthropic: the prompt grows every request. A
//! local chat template may drop earlier turns' reasoning, so elsewhere a
//! shrink is a note. Printed only: the **tip** — how far `read_k` reaches
//! past the Anthropic-equivalent bound. blallama may reuse the previous
//! turn's generated KV there, which only moves tokens from `creation` to
//! `read`; a warning flags a request that re-prefilled most of the previous
//! turn's output (read less than half of it past that turn's whole prompt,
//! `read_k - T_{k-1}`).
//!
//! Two scripts: [`SHORT`], ten beats, and [`LONG`], twenty with larger tool
//! results, so the conversation itself grows to many thousands of tokens.
//! The entry points:
//!
//! - `blallama::canonical`, `blallama::after_assistant` (both [`SHORT`])
//!   and `blallama::long`: a local drama_llama server, skipped unless
//!   `BLALLAMA_URL` is set (see `live`). `canonical` and `long` are
//!   configured exactly as for Anthropic; `after_assistant` reports
//!   [`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant)
//!   so the driver marks assistant turns instead. Each request's `T` must
//!   also equal the server's `count_tokens`. Run with `just test-cache`.
//! - `anthropic::canonical` ([`SHORT`]) and `anthropic::long`: **paid** —
//!   `claude-haiku-4-5`, about three and eight cents, with
//!   `misanthropic/api.key`. `#[ignore]`d, and skipped even then unless
//!   `MISANTHROPIC_PAID_CACHE=1`, so the live CI gate (which runs every
//!   ignored test) doesn't pay for them. Run with
//!   `just test-cache-anthropic [canonical|long]`.
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
/// How many times the calibrated prefill cost of a request's uncached
/// tokens its time beyond decode may reach. Generous: a re-prefilled prompt
/// costs tens of times more.
const LATENCY_RATIO: f64 = 3.0;
/// The share of a request's estimated decode time also allowed, since
/// decoding slows as the context grows.
const DECODE_DRIFT: f64 = 0.5;
/// The fewest tokens a request must prefill to calibrate the prefill rate.
const CALIBRATION_FLOOR: u64 = 1024;
/// The fewest output tokens a request must decode to estimate the decode
/// rate.
const DECODE_FLOOR: u64 = 16;
/// The fewest generated tokens worth warning about when a request's tip
/// re-prefills most of them.
const TIP_FLOOR: u64 = 64;

/// What a backend's cache is held to.
#[derive(Clone, Copy)]
struct Backend {
    /// Anthropic, or a stand-in for it: its prompt must grow every request
    /// (elsewhere a shrink is a note), and its tips go unwarned (Anthropic
    /// never reads past a breakpoint, so it has no generated KV to reuse).
    reference: bool,
    /// Markers ride assistant turns (the driver's window), not the tail.
    after_assistant: bool,
    /// Tokens `read` may fall short of the bound.
    slack: u64,
    /// Wall time allowed past the latency budget, for the network and
    /// scheduling; `None` prints the latency without holding it (Anthropic's
    /// accounting is ground truth, its latency noise and retries).
    latency_slack_ms: Option<f64>,
    /// The prefill rate, in tokens per second, to hold latency to when no
    /// request of a run prefilled enough to measure one (a server still
    /// warm from an earlier run); `None` leaves such a run unchecked.
    prefill_rate: Option<f64>,
}

/// Anthropic: `read` is the previous prompt less the framing after its
/// last block — single digits.
const ANTHROPIC: Backend = Backend {
    reference: true,
    after_assistant: false,
    slack: 16,
    latency_slack_ms: None,
    prefill_rate: None,
};

/// A local blallama, configured as for Anthropic. Its template may render
/// the moved tail a little differently.
#[cfg(feature = "blallama")]
const BLALLAMA: Backend = Backend {
    reference: false,
    after_assistant: false,
    slack: 64,
    latency_slack_ms: Some(1500.0),
    prefill_rate: Some(BLALLAMA_PREFILL_RATE),
};

/// blallama's fallback [`prefill_rate`](Backend::prefill_rate), in tokens
/// per second, unless `BLALLAMA_PREFILL_RATE` sets one: on 2026-09-30,
/// Qwen3.6-35B-A3B (`UD-Q4_K_S`) prefilled about 7,500 tokens cold in
/// about 18 s, some 420 a second. Set it for other models and machines; a
/// re-prefill costs tens of times the budget, so it needn't be exact.
#[cfg(feature = "blallama")]
const BLALLAMA_PREFILL_RATE: f64 = 420.0;

/// [`simulated`]: Anthropic's accounting, with a mock's latency (a second
/// of slack rides out a loaded CI runner).
const SIMULATED: Backend = Backend {
    latency_slack_ms: Some(1000.0),
    ..ANTHROPIC
};

/// A conversation to drive, and what it's held to.
struct Script {
    /// The user's beats, in order.
    beats: &'static [&'static str],
    /// Offer the [`Logbook`] too.
    logbook: bool,
    /// The most uncached `input` a request after the first may pay: a beat
    /// or a tool result, never the conversation.
    input_cap: u64,
}

/// Ten beats, four with a tool round: the conversation stays around a
/// thousand tokens beside the system prompt.
const SHORT: Script = Script {
    beats: &BEATS,
    logbook: false,
    input_cap: 1024,
};

/// Twenty beats, twelve with a tool round, nine of them reading ten days
/// of the [`Logbook`] (about a thousand tokens each): the conversation
/// grows to many thousands of tokens, the regime where a prefix that isn't
/// reused makes every turn crawl. Needs a context of about 32k.
const LONG: Script = Script {
    beats: &LONG_BEATS,
    logbook: true,
    input_cap: 2048,
};

/// [`SHORT`]'s beats. Four ask for a tool (`ledger`, `stores`), which a
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

/// [`LONG`]'s beats: nine read ten days of the logbook, three more ask for
/// the ledger or the stores.
const LONG_BEATS: [&str; 20] = [
    "Read the logbook for days 1 to 10. Which keeper stood the most watches?",
    "Who checks the fog signal, and how often?",
    "Read the logbook for days 11 to 20. Was the wind ever above force 7?",
    "What is the lamp's service code?",
    "Read the logbook for days 21 to 30 and total the fog-signal blasts.",
    "Count the lamp mantles in the stores. Should I reorder?",
    "Read the logbook for days 31 to 40. When was the lamp lit latest?",
    "Which shelf holds the radio's spares?",
    "Read the logbook for days 41 to 50 and name the calmest day.",
    "If the cistern reads low, whom do I tell?",
    "Read the logbook for days 51 to 60. Did anyone stand two nights running?",
    "How much paraffin is in the stores?",
    "Read the logbook for days 61 to 70 and sum up the weather.",
    "Which keeper signs the tide gauge's card?",
    "Read the logbook for days 71 to 80. How often did the wind back west?",
    "Sum up our conversation so far in one sentence.",
    "Read the logbook for days 81 to 90 and name the windiest day.",
    "What does section 40 cover?",
    "Read the ledger entry for day 95 and tell me who was on watch.",
    "Last one: which three sections matter most to a new keeper?",
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
/// The wire name of [`Logbook`]'s method.
const LOGBOOK: &str = "toolbox__Logbook__range";
/// The most days a [`Logbook`] range returns.
const RANGE_CAP: u64 = 14;

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

/// The station logbook: the [`Ledger`]'s entries for a run of days, up to
/// [`RANGE_CAP`] of them.
struct Logbook;

/// [`Logbook`]'s arguments.
#[derive(serde::Serialize)]
struct RangeArgs {
    days: &'static str,
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
        let entry = entry(&argument(&call, "day"));
        tool::Result::new(call.id, entry)
    }
}

/// The ledger's entry for `day`: about 80 words.
fn entry(day: &str) -> String {
    let n = seed(day);
    let wind = ["north", "east", "south", "west"][pick(n, 1, 4)];
    let force = 1 + pick(n, 2, 9);
    let keeper = KEEPERS[pick(n, 3, KEEPERS.len())];
    let blasts = pick(n, 4, 40);
    format!(
        "Day {day}. Wind {wind} force {force}, sea moderate, pressure \
         falling slowly. Lamp lit at 19:{:02} and out at dawn. Keeper on \
         watch: {keeper}. Fog signal sounded {blasts} times after midnight. \
         One drum of paraffin opened; the old drum returned to the store. \
         Remarks: gallery rail wet, harness worn, nothing else to report.",
        pick(n, 5, 60),
    )
}

#[async_trait::async_trait]
impl Tool for Logbook {
    fn name(&self) -> &str {
        "Logbook"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Custom(CustomMethodDef::with_string_param(
            "Logbook__range",
            "Read the station logbook's entries for a run of days, at most \
             14.",
            "days",
            "The first and last day, e.g. `1-10`.",
            true,
        ))]
    }

    async fn call(&mut self, call: Use) -> tool::Result {
        let days = argument(&call, "days");
        let mut bounds = days
            .split(|c: char| !c.is_ascii_digit())
            .filter_map(|n| n.parse::<u64>().ok());
        let first = bounds.next().unwrap_or(1);
        let last = bounds.next().unwrap_or(first).max(first);
        let entries = (first..=last.min(first + RANGE_CAP - 1))
            .map(|day| entry(&day.to_string()))
            .collect::<Vec<_>>();
        tool::Result::new(call.id, entries.join("\n\n"))
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

/// Drive `script` through `transport` from `base` (its model and
/// `max_tokens`) with the caching knobs a long conversation wants, and
/// hand back what was sent and received. `backend` is only for the table
/// printed if the run stops.
async fn scenario<T>(
    transport: Checked<T>,
    base: Prompt,
    script: &Script,
    backend: Backend,
) -> Log
where
    T: Transport + Clone,
{
    let sink = Arc::new(Mutex::new(TokenCounts::default()));
    let toolbox = ToolBox::new().add(Ledger).add(Stores);
    let toolbox = match script.logbook {
        true => toolbox.add(Logbook),
        false => toolbox,
    };
    // Marked while it has no messages, so the mark lands on the system,
    // covering the tools (they render first) too.
    let prompt = base.system(system()).cache();
    let mut beats = script.beats.iter();
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
        let requests = requests(&log, None);
        let latency = Latency::fit(&requests, backend.prefill_rate);
        eprintln!("{}", table(&requests, backend, &latency));
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
    /// `cache_creation_input_tokens`; `None` when the server left it out.
    written: Option<u64>,
    /// `cache_read_input_tokens`; `None` when the server left it out.
    read: Option<u64>,
    /// The server's `count_tokens` for the request (or why it failed),
    /// when asked.
    counted: Option<Result<u64, String>>,
    output: u64,
    millis: f64,
}

impl Request {
    /// The whole prompt, `T`: `input + creation + read`, as Anthropic bills
    /// it.
    fn prompt(&self) -> u64 {
        self.input + self.written.unwrap_or(0) + self.read()
    }

    /// Tokens read from the cache.
    fn read(&self) -> u64 {
        self.read.unwrap_or(0)
    }

    /// Tokens actually prefilled: `input + creation`.
    fn prefilled(&self) -> u64 {
        self.input + self.written.unwrap_or(0)
    }

    /// What Anthropic would read of this request on the next: everything
    /// through its last breakpoint.
    fn cached(&self) -> u64 {
        self.prompt() - self.input
    }
}

/// `log`'s requests, with each prompt's `counted` alongside if given
/// (indexed like `log.sent`, which a failed send leaves longer).
fn requests(log: &Log, counted: Option<&[Counted]>) -> Vec<Request> {
    let received = log.received.iter().zip(&log.answers).zip(&log.elapsed);
    received
        .map(|((reply, &n), elapsed)| {
            let counts = reply.usage.counts;
            let tail =
                log.sent[n].messages.last().expect("a request has a tail");
            let tool_round = tail
                .content
                .iter()
                .any(|block| matches!(block, Block::ToolResult { .. }));
            Request {
                kind: if tool_round { "tool" } else { "beat" },
                input: counts.input_tokens,
                written: counts.cache_creation_input_tokens,
                read: counts.cache_read_input_tokens,
                counted: counted.map(|counted| counted[n].clone()),
                output: counts.output_tokens,
                millis: elapsed.as_secs_f64() * 1e3,
            }
        })
        .collect()
}

/// The lowest `read_k` [`assert_caches`] accepts, before `slack`: what
/// request `k - 1` (`before`) left cached, as Anthropic reads it.
fn bound(before: &Request, backend: Backend) -> u64 {
    if backend.after_assistant {
        before.cached()
    } else {
        before.prompt()
    }
}

/// How far `now` read past what Anthropic would have (see [`Request::cached`]).
fn tip(before: &Request, now: &Request) -> u64 {
    now.read().saturating_sub(before.cached())
}

/// How much of what `before` generated `now` read back: its read past
/// `before`'s whole prompt, `read_k - T_{k-1}`, which only a server reusing
/// generated KV (blallama's tip) reaches.
fn reused(before: &Request, now: &Request) -> u64 {
    now.read().saturating_sub(before.prompt())
}

/// What a request's wall time should be, fitted to the run itself.
struct Latency {
    /// Where [`prefill_ms`](Self::prefill_ms) came from.
    prefill: Prefill,
    /// Milliseconds per prefilled token.
    prefill_ms: f64,
    /// Milliseconds per output token: the fastest other request's, as close
    /// to pure decode as the run gets.
    decode_ms: f64,
}

/// Where [`Latency`]'s prefill rate came from.
#[derive(Debug, PartialEq)]
enum Prefill {
    /// Measured on request `n` (from zero), the one that prefilled the
    /// most: its wall time less its decode.
    Measured(usize),
    /// No request prefilled [`CALIBRATION_FLOOR`] tokens (the server was
    /// warm from an earlier run; the most was `most`), so the backend's
    /// [`prefill_rate`](Backend::prefill_rate), in tokens per second.
    Fallback { rate: f64, most: u64 },
}

impl std::fmt::Display for Prefill {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Measured(n) => write!(f, "measured on request {}", n + 1),
            Self::Fallback { rate, most } => write!(
                f,
                "the backend's fallback of {rate:.0} tokens/s: no request \
                 prefilled {CALIBRATION_FLOOR} tokens (the most was {most})"
            ),
        }
    }
}

impl Latency {
    /// The model, or `None` (with the reason) when the run can't calibrate
    /// one: no request prefilled [`CALIBRATION_FLOOR`] tokens (all warm from
    /// an earlier run) and there's no `fallback` rate (tokens per second),
    /// or none else decoded [`DECODE_FLOOR`].
    fn fit(
        requests: &[Request],
        fallback: Option<f64>,
    ) -> Result<Self, String> {
        let (calibration, cold) = requests
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.prefilled())
            .ok_or("no requests")?;
        let most = cold.prefilled();
        let prefill = match (most >= CALIBRATION_FLOOR, fallback) {
            (true, _) => Prefill::Measured(calibration),
            (false, Some(rate)) => Prefill::Fallback { rate, most },
            (false, None) => {
                return Err(format!(
                    "no request prefilled {CALIBRATION_FLOOR} tokens (the \
                     most was {most}), and there's no fallback rate: nothing \
                     to calibrate on"
                ));
            }
        };
        let decode_ms = requests
            .iter()
            .enumerate()
            .filter(|&(n, _)| prefill != Prefill::Measured(n))
            .filter(|(_, r)| r.output >= DECODE_FLOOR)
            .map(|(_, r)| r.millis / r.output as f64)
            .min_by(f64::total_cmp)
            .ok_or("no other request decoded enough to time decoding")?;
        let prefill_ms = match prefill {
            Prefill::Measured(_) => {
                let rest = cold.millis - decode_ms * cold.output as f64;
                rest.max(0.0) / most as f64
            }
            Prefill::Fallback { rate, .. } => 1e3 / rate,
        };
        Ok(Self {
            prefill,
            prefill_ms,
            decode_ms,
        })
    }

    /// The part of `r`'s wall time that wasn't decoding.
    fn rest(&self, r: &Request) -> f64 {
        (r.millis - self.decode_ms * r.output as f64).max(0.0)
    }

    /// The most [`rest`](Self::rest) may be for a request that prefilled
    /// only its uncached tokens, before a backend's slack.
    fn budget(&self, r: &Request) -> f64 {
        LATENCY_RATIO * self.prefill_ms * r.prefilled() as f64
            + DECODE_DRIFT * self.decode_ms * r.output as f64
    }

    /// What re-prefilling `r`'s whole prompt would take.
    fn full(&self, r: &Request) -> f64 {
        self.prefill_ms * r.prompt() as f64
    }
}

/// `value`, or `-` when absent.
fn or_dash(value: Option<impl ToString>) -> String {
    value.map_or("-".into(), |value| value.to_string())
}

/// The per-request table, the latency model, and the run's totals.
fn table(
    requests: &[Request],
    backend: Backend,
    latency: &Result<Latency, String>,
) -> String {
    let header = format!(
        "{:>3} {:<4} {:>7} {:>7} {:>6} {:>7} {:>7} {:>6} {:>5} {:>7} {:>7} \
         {:>7} {:>5}",
        "req",
        "kind",
        "prompt",
        "counted",
        "input",
        "written",
        "read",
        "tip",
        "out",
        "ms",
        "rest",
        "budget",
        "hit%",
    );
    let before = std::iter::once(None).chain(requests.iter().map(Some));
    let rows = requests.iter().zip(before).enumerate().map(|(n, (r, b))| {
        let fitted = latency.as_ref().ok();
        let slack = backend.latency_slack_ms.unwrap_or(0.0);
        format!(
            "{:>3} {:<4} {:>7} {:>7} {:>6} {:>7} {:>7} {:>6} {:>5} {:>7.0} \
             {:>7} {:>7} {:>5.1}",
            n + 1,
            r.kind,
            r.prompt(),
            match &r.counted {
                Some(Ok(counted)) => counted.to_string(),
                Some(Err(_)) => "err".into(),
                None => "-".into(),
            },
            r.input,
            or_dash(r.written),
            or_dash(r.read),
            or_dash(b.map(|b| tip(b, r))),
            r.output,
            r.millis,
            or_dash(fitted.map(|l| format!("{:.0}", l.rest(r)))),
            or_dash(fitted.map(|l| format!("{:.0}", l.budget(r) + slack))),
            100.0 * r.read() as f64 / r.prompt().max(1) as f64,
        )
    });
    let model = match latency {
        Ok(l) => format!(
            "latency: {:.2} ms per prefilled token ({}), {:.1} ms per \
             output token; `rest` is wall time less decode",
            l.prefill_ms, l.prefill, l.decode_ms,
        ),
        Err(why) => format!("latency: not checked: {why}"),
    };
    let sum = |f: fn(&Request) -> u64| requests.iter().map(f).sum::<u64>();
    let (read, prompt) = (sum(Request::read), sum(Request::prompt));
    let totals = format!(
        "total: prompt {prompt}, input {}, written {}, read {read} \
         ({:.1}% of all prompt tokens read from cache)",
        sum(|r| r.input),
        sum(|r| r.written.unwrap_or(0)),
        100.0 * read as f64 / prompt.max(1) as f64,
    );
    std::iter::once(header)
        .chain(rows)
        .chain([model, totals])
        .collect::<Vec<_>>()
        .join("\n")
}

/// Hold `log`, a run of `script`, to a healthy cached loop's signature on
/// `backend` (see the module docs), after printing its table. `counted` is
/// each prompt's `count_tokens`, which `input + creation + read` must
/// equal exactly.
fn assert_caches(
    log: &Log,
    counted: Option<&[Counted]>,
    script: &Script,
    backend: Backend,
) {
    let requests = requests(log, counted);
    let latency = Latency::fit(&requests, backend.prefill_rate);
    eprintln!("{}", table(&requests, backend, &latency));

    let beats = script.beats.len();
    assert!(requests.len() >= beats, "a request per beat at least");
    assert!(
        requests.iter().any(|r| r.kind == "tool"),
        "no tool round: the scenario covers tool results too"
    );
    check_counts(&requests);
    check_growth(&requests, backend);
    check_reads(&requests, backend, script.input_cap);
    check_tips(&requests, backend);
    if let (Ok(latency), Some(slack)) = (&latency, backend.latency_slack_ms) {
        check_latency(&requests, latency, slack);
    }
}

/// Every request reports its cache fields, and they add up to
/// `count_tokens` where it was asked.
fn check_counts(requests: &[Request]) {
    for (n, r) in requests.iter().enumerate() {
        assert!(
            r.written.is_some() && r.read.is_some(),
            "request {}: no cache_creation or cache_read",
            n + 1
        );
        match &r.counted {
            Some(Ok(counted)) => assert_eq!(
                r.prompt(),
                *counted,
                "request {}: input + creation + read vs count_tokens",
                n + 1
            ),
            Some(Err(error)) => {
                panic!("request {}: count_tokens failed: {error}", n + 1)
            }
            None => {}
        }
    }
}

/// The prompt grows every request: held on Anthropic, a note elsewhere,
/// where a chat template may drop earlier turns' reasoning.
fn check_growth(requests: &[Request], backend: Backend) {
    for (n, pair) in requests.windows(2).enumerate() {
        let (before, now) = (pair[0].prompt(), pair[1].prompt());
        if now > before {
            continue;
        }
        let shrank = format!(
            "request {}: the prompt didn't grow ({now} after {before})",
            n + 2
        );
        assert!(!backend.reference, "{shrank}");
        eprintln!("note: {shrank}; the template may drop old reasoning");
    }
}

/// The first request caches the prefix, and each after reads what the one
/// before it cached, paying no more than `input_cap` fresh.
fn check_reads(requests: &[Request], backend: Backend, input_cap: u64) {
    let first = &requests[0];
    assert!(
        first.prompt() >= PREFIX_FLOOR,
        "the first prompt ({}) is too short to cache on every model",
        first.prompt()
    );
    assert!(
        first.written.unwrap_or(0) + first.read() >= MIN_CACHEABLE,
        "request 1 neither wrote nor read the cached prefix"
    );
    for (n, pair) in requests.windows(2).enumerate() {
        let (before, now, k) = (&pair[0], &pair[1], n + 2);
        let bound = bound(before, backend);
        assert!(
            now.read() + backend.slack >= bound,
            "request {k} read {} tokens where request {} cached {bound} \
             (slack {}): the prefix was re-prefilled",
            now.read(),
            k - 1,
            backend.slack
        );
        assert!(
            now.input <= input_cap,
            "request {k} paid {} uncached input tokens (cap {input_cap})",
            now.input
        );
    }
}

/// Warn, off Anthropic, where a request re-prefilled most of what the one
/// before it generated rather than reading its KV back.
fn check_tips(requests: &[Request], backend: Backend) {
    tip_warnings(requests, backend)
        .iter()
        .for_each(|warning| eprintln!("{warning}"));
}

/// [`check_tips`]' warnings: a request that read back less than half of
/// what the one before it generated (at least [`TIP_FLOOR`] tokens).
fn tip_warnings(requests: &[Request], backend: Backend) -> Vec<String> {
    if backend.reference {
        return Vec::new();
    }
    let warn = |(n, pair): (usize, &[Request])| {
        let (before, now, k) = (&pair[0], &pair[1], n + 2);
        let reused = reused(before, now);
        (before.output >= TIP_FLOOR && reused < before.output / 2).then(|| {
            format!(
                "WARN tip: request {k} read {reused} tokens past request {}'s \
                 whole prompt, which then generated {}: it re-prefilled most \
                 of them",
                k - 1,
                before.output
            )
        })
    };
    requests.windows(2).enumerate().filter_map(warn).collect()
}

/// Each request's time beyond decoding fits prefilling only its uncached
/// tokens, within `slack_ms`.
fn check_latency(requests: &[Request], latency: &Latency, slack_ms: f64) {
    for (n, r) in requests.iter().enumerate() {
        let (rest, budget) = (latency.rest(r), latency.budget(r) + slack_ms);
        assert!(
            rest <= budget,
            "request {} spent {rest:.0} ms beyond decoding, over the \
             {budget:.0} ms that prefilling its {} uncached tokens allows \
             ({:.2} ms per token, {}); re-prefilling all {} \
             would take about {:.0} ms: the reported cache read didn't save \
             the time",
            n + 1,
            r.prefilled(),
            latency.prefill_ms,
            latency.prefill,
            r.prompt(),
            latency.full(r),
        );
    }
}

/// A request's `count_tokens`, or why it failed.
type Counted = Result<u64, String>;

/// Each of `log`'s requests counted by `client`'s `count_tokens`. A failure
/// is kept, not raised, so [`assert_caches`] prints the table before
/// failing on it.
async fn counted(client: &Client, log: &Log) -> Vec<Counted> {
    let mut counted = Vec::with_capacity(log.sent.len());
    for prompt in &log.sent {
        let count = client.count_tokens(prompt).await;
        counted.push(count.map(u64::from).map_err(|e| e.to_string()));
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

/// `key` parsed from the environment, or `default` when it's unset.
#[cfg(feature = "blallama")]
fn from_env<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).map_or(default, |value| {
        value
            .parse()
            .unwrap_or_else(|_| panic!("{key}={value:?} isn't a number"))
    })
}

/// The scenario against a local blallama; see the module docs.
#[cfg(feature = "blallama")]
mod blallama {
    use super::*;

    /// Run `script` through `wrap`ped [`Client`], held to `backend`,
    /// unless no server is configured.
    async fn run<T>(
        name: &str,
        script: &Script,
        backend: Backend,
        wrap: impl FnOnce(Client) -> T,
    ) where
        T: Transport + Clone,
    {
        let Some((url, model)) = super::super::live::target() else {
            return eprintln!("skipping `{name}`: BLALLAMA_URL is unset");
        };
        let client = super::super::live::client(&url);
        let backend = Backend {
            prefill_rate: Some(from_env(
                "BLALLAMA_PREFILL_RATE",
                BLALLAMA_PREFILL_RATE,
            )),
            ..backend
        };
        // Room for a local model's thinking, and for the prompt beside it.
        let base = Prompt::default()
            .model(model)
            .max_tokens(NonZeroU32::new(8192).unwrap());
        let transport = Checked::new(wrap(client.clone()));
        let log = scenario(transport, base, script, backend).await;
        let counted = counted(&client, &log).await;
        assert_caches(&log, Some(&counted), script, backend);
    }

    /// Configured exactly as for Anthropic.
    #[tokio::test]
    async fn canonical() {
        run("canonical", &SHORT, BLALLAMA, |client| client).await;
    }

    /// [`LONG`], configured as for Anthropic.
    #[tokio::test]
    async fn long() {
        run("long", &LONG, BLALLAMA, |client| client).await;
    }

    /// Markers on assistant turns, as agentkit places them for blallama.
    #[tokio::test]
    async fn after_assistant() {
        let backend = Backend {
            after_assistant: true,
            ..BLALLAMA
        };
        run("after_assistant", &SHORT, backend, AfterAssistant).await;
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

/// Whether the paid Anthropic runs are opted into, with
/// `MISANTHROPIC_PAID_CACHE=1`: CI's live gate runs every `#[ignore]`d test,
/// and shouldn't pay for reference numbers.
fn paid(name: &str) -> bool {
    let opted =
        std::env::var("MISANTHROPIC_PAID_CACHE").is_ok_and(|v| v == "1");
    if !opted {
        eprintln!(
            "skipping `{name}`: PAID, and MISANTHROPIC_PAID_CACHE=1 is unset \
             (`just test-cache-anthropic` sets it)"
        );
    }
    opted
}

/// The reference: Anthropic itself, on the model with the largest minimum.
mod anthropic {
    use super::*;

    /// Run `script` on Haiku 4.5, if paid runs are opted into.
    async fn run(name: &str, script: &Script) {
        if !paid(name) {
            return;
        }
        let client = Client::new(crate::utils::load_api_key().await).unwrap();
        let base = Prompt::default()
            .model(Id::Haiku45)
            .max_tokens(NonZeroU32::new(512).unwrap());
        let transport = Checked::new(Retrying(client));
        let log = scenario(transport, base, script, ANTHROPIC).await;
        assert_caches(&log, None, script, ANTHROPIC);
    }

    /// Configured as a long conversation should be; see the module docs.
    #[tokio::test]
    #[ignore = "PAID: live Anthropic API (claude-haiku-4-5, about 3 cents); \
                needs misanthropic/api.key and MISANTHROPIC_PAID_CACHE=1"]
    async fn canonical() {
        run("anthropic::canonical", &SHORT).await;
    }

    /// [`LONG`], for reference numbers in the growth regime.
    #[tokio::test]
    #[ignore = "PAID: live Anthropic API (claude-haiku-4-5, about 8 cents); \
                needs misanthropic/api.key and MISANTHROPIC_PAID_CACHE=1"]
    async fn long() {
        run("anthropic::long", &LONG).await;
    }
}

/// A stand-in for Anthropic's cache and model. It sizes a request by its
/// serialized length and writes it all to the cache but the framing; when
/// `healthy` it first reads back what the previous request cached if this
/// one extends it, broken it never does (a prefix that changes every
/// turn). A beat naming the logbook, the ledger or the stores gets one
/// call, anything else an answer.
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
        // Everything to the breakpoint on the last block is written; what
        // follows it is framing, re-paid next time as on Anthropic.
        const FRAMING: u64 = 3;
        let read = match last.as_ref() {
            Some((before, size)) if healthy && turns.starts_with(before) => {
                size - FRAMING
            }
            _ => 0,
        };
        *last = Some((turns, size));

        let mut counts = TokenCounts::new(FRAMING, 20);
        counts.cache_read_input_tokens = Some(read);
        counts.cache_creation_input_tokens = Some(size - read - FRAMING);
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
    let logbook = RangeArgs { days: "1-10" };
    match () {
        _ if answered => mock::text("Noted."),
        _ if beat.contains("logbook") => {
            let args = serde_json::to_value(logbook).unwrap();
            mock::text("Reading.").tool_use(LOGBOOK, args)
        }
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

/// `script` against [`simulated`]`(healthy)`, and its log.
fn simulate(script: &Script, healthy: bool) -> Log {
    let transport = Checked::new(Arc::new(simulated(healthy)));
    let base = Prompt::default().model(Id::Haiku45);
    futures::executor::block_on(scenario(transport, base, script, SIMULATED))
}

/// The knobs reach the wire, the tools run, and a healthy cache passes.
#[test]
fn simulated_healthy_cache_passes() {
    let log = simulate(&SHORT, true);

    let first = &log.sent[0];
    assert!(first.cache_control.is_some(), "automatic caching is on");
    let system = first.system.as_ref().expect("a system prompt");
    assert!(system.has_cache(), "the system carries a marker");
    let names = first.tools.iter().flatten().map(MethodDef::name);
    assert_eq!(names.collect::<Vec<_>>(), [LEDGER, STORES]);
    assert_eq!(log.sent.len(), BEATS.len() + 4, "four tool rounds");

    assert_caches(&log, None, &SHORT, SIMULATED);
}

/// [`LONG`] offers the logbook, grows by many thousands of tokens, and a
/// healthy cache passes it.
#[test]
fn simulated_long_conversation_passes() {
    let log = simulate(&LONG, true);

    let names = log.sent[0].tools.iter().flatten().map(MethodDef::name);
    assert_eq!(names.collect::<Vec<_>>(), [LEDGER, LOGBOOK, STORES]);
    assert_eq!(log.sent.len(), LONG_BEATS.len() + 12, "twelve tool rounds");
    let requests = requests(&log, None);
    let grown = requests.last().unwrap().prompt() - requests[0].prompt();
    assert!(grown >= 8000, "the conversation grew only {grown} tokens");

    assert_caches(&log, None, &LONG, SIMULATED);
}

/// A logbook range is a day's entry per day, capped.
#[test]
fn logbook_reads_a_capped_range() {
    let read = |days: &'static str| {
        let args = serde_json::to_value(RangeArgs { days }).unwrap();
        let call = Use::new(LOGBOOK, args).with_id("toolu_1");
        let result = futures::executor::block_on(Logbook.call(call));
        result.content.to_string()
    };
    assert_eq!(read("1 to 10").matches("Day ").count(), 10);
    assert_eq!(read("5-5").matches("Day ").count(), 1);
    assert_eq!(read("1-100").matches("Day ").count(), RANGE_CAP as usize);
    assert!(read("3-4").starts_with(&entry("3")));
}

/// A cache rewritten every turn fails, however well the rest goes.
#[test]
fn simulated_broken_cache_fails() {
    let log = simulate(&SHORT, false);
    let checked = std::panic::catch_unwind(|| {
        assert_caches(&log, None, &SHORT, SIMULATED);
    });
    let panic = checked.expect_err("a re-prefilled prompt must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("re-prefilled"), "{message}");
}

/// A request billed `input`, `written` and `read`, that decoded `output`.
fn accounted(input: u64, written: u64, read: u64, output: u64) -> Request {
    Request {
        kind: "beat",
        input,
        written: Some(written),
        read: Some(read),
        counted: None,
        output,
        millis: 0.0,
    }
}

/// A synthetic request: `prefilled` uncached tokens (all written but a
/// 3-token tail) over `read` cached, `output` decoded, in `millis`.
fn synthetic(read: u64, prefilled: u64, output: u64, millis: f64) -> Request {
    Request {
        millis,
        ..accounted(3, prefilled - 3, read, output)
    }
}

/// A blallama-like run: a cold 6,600-token prefill at about 2.5 ms a
/// token, then warm turns prefilling a beat each, one of them a long
/// generation. `lag` is added to every warm turn.
fn synthetic_run(lag: f64) -> Vec<Request> {
    let turns = [(40, 150), (60, 300), (7000, 120), (30, 90), (80, 400)];
    let first = synthetic(0, 6600, 200, 6600.0 * 2.5 + 200.0 * 17.0);
    let warm = turns.iter().scan(6600, |read, &(output, prefilled)| {
        let request = synthetic(
            *read,
            prefilled,
            output,
            lag + 2.5 * prefilled as f64 + 17.0 * output as f64,
        );
        *read += prefilled;
        Some(request)
    });
    std::iter::once(first).chain(warm).collect()
}

/// The latency check passes a healthy run, and fails one whose server
/// reports cache reads but spends a re-prefill's time on every turn.
#[test]
fn latency_catches_reads_that_save_no_time() {
    let healthy = synthetic_run(0.0);
    let latency = Latency::fit(&healthy, None).unwrap();
    assert_eq!(latency.prefill, Prefill::Measured(0));
    check_latency(&healthy, &latency, 1500.0);

    // Reads reported, but each turn re-prefills the ~7k-token prompt.
    let lying = synthetic_run(2.5 * 7000.0);
    let latency = Latency::fit(&lying, None).unwrap();
    let checked = std::panic::catch_unwind(|| {
        check_latency(&lying, &latency, 1500.0);
    });
    let panic = checked.expect_err("a re-prefill's time must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("didn't save"), "{message}");
}

/// A shrinking prompt fails on Anthropic, and is only a note elsewhere.
#[test]
fn a_shrinking_prompt_is_a_note_off_anthropic() {
    let mut run = synthetic_run(0.0);
    // Request 3 is 100 tokens shorter than request 2.
    run[2] = synthetic(6600, run[1].prompt() - 6700, 60, 1000.0);
    let local = Backend {
        reference: false,
        ..ANTHROPIC
    };

    let shrink = std::panic::catch_unwind(|| check_growth(&run, ANTHROPIC));
    assert!(shrink.is_err(), "Anthropic's prompt must grow");
    check_growth(&run, local);
}

/// Markers on the tail are read back whole; markers on assistant turns
/// leave the previous request's uncached `input` (its beat) out.
#[test]
fn the_bound_follows_where_the_markers_sit() {
    let after_assistant = Backend {
        after_assistant: true,
        ..ANTHROPIC
    };
    // A 6,000-token prompt whose last 100 (the beat) weren't cached.
    let before = accounted(100, 5900, 0, 200);
    assert_eq!(bound(&before, ANTHROPIC), 6000);
    assert_eq!(bound(&before, after_assistant), 5900);

    let run = [before, accounted(50, 100, 5900, 20)];
    check_reads(&run, after_assistant, 1024);
    let tail = std::panic::catch_unwind(|| check_reads(&run, ANTHROPIC, 1024));
    let panic = tail.expect_err("a tail marker reads the whole prompt back");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("request 2 read 5900"), "{message}");
}

/// The tip is the read past Anthropic's bound; the reuse, past the whole
/// previous prompt (into what it generated). Both stop at zero.
#[test]
fn tips_and_reuse_count_past_their_bounds() {
    let before = accounted(100, 5900, 0, 200);
    let past = accounted(50, 100, 6150, 20);
    assert_eq!(tip(&before, &past), 250);
    assert_eq!(reused(&before, &past), 150);
    let short = accounted(50, 100, 5000, 20);
    assert_eq!(tip(&before, &short), 0);
    assert_eq!(reused(&before, &short), 0);
}

/// Off Anthropic, a request that reads back less than half of what the
/// previous one generated is warned about, even when its tip (which also
/// counts the previous beat) looks healthy.
#[test]
fn a_tip_that_re_prefills_generated_tokens_warns() {
    let local = Backend {
        reference: false,
        ..ANTHROPIC
    };
    let run = |output, read| {
        [
            accounted(100, 5900, 0, output),
            accounted(50, 100, read, 20),
        ]
    };
    assert!(
        tip_warnings(&run(200, 6150), local).is_empty(),
        "150 of 200"
    );

    let rereads = run(200, 6050);
    assert!(
        tip(&rereads[0], &rereads[1]) >= 100,
        "the tip looks healthy"
    );
    let warned = tip_warnings(&rereads, local);
    assert_eq!(warned.len(), 1, "{warned:?}");
    assert!(warned[0].contains("request 2 read 50 tokens"), "{warned:?}");
    assert!(warned[0].contains("generated 200"), "{warned:?}");

    assert!(
        tip_warnings(&rereads, ANTHROPIC).is_empty(),
        "the reference"
    );
    let short = run(TIP_FLOOR - 1, 6000);
    assert!(tip_warnings(&short, local).is_empty(), "too few to matter");
}

/// A failed send leaves `sent` longer than `received`: each reply pairs
/// with the prompt it answers, for its kind and its count.
#[test]
fn requests_pair_each_reply_with_what_it_answers() {
    let beat = Prompt::default().add_message((Role::User, "hi")).unwrap();
    let result: Block = tool::Result::new("toolu_1", "done").into();
    let mut tool_round = beat.clone();
    tool_round.messages.push((Role::User, vec![result]).into());
    let reply = |input| mock::text("ok").usage(input, 2).build();
    let log = Log {
        sent: vec![beat.clone(), beat, tool_round],
        received: vec![reply(10), reply(30)],
        answers: vec![0, 2],
        elapsed: vec![std::time::Duration::from_millis(5); 2],
    };
    let counted = [Ok(10), Err("refused".to_string()), Ok(30)];

    let requests = requests(&log, Some(&counted));
    let kinds: Vec<_> = requests.iter().map(|r| r.kind).collect();
    assert_eq!(kinds, ["beat", "tool"]);
    let counts: Vec<_> = requests.iter().map(|r| r.counted.clone()).collect();
    assert_eq!(counts, [Some(Ok(10)), Some(Ok(30))]);
    assert_eq!(requests[1].input, 30);
}

/// A reply without its cache fields, or a failed `count_tokens`, still
/// gets the table (with the gap marked), then fails the run.
#[test]
fn a_missing_cache_field_prints_the_table_then_fails() {
    let mut log = simulate(&SHORT, true);
    log.received[3].usage.counts.cache_creation_input_tokens = None;
    let rows = requests(&log, None);
    let printed = table(&rows, SIMULATED, &Latency::fit(&rows, None));
    let row = printed.lines().nth(4).expect("request 4's row");
    assert_eq!(row.split_whitespace().nth(5), Some("-"), "{row}");

    let checked = std::panic::catch_unwind(|| {
        assert_caches(&log, None, &SHORT, SIMULATED);
    });
    let panic = checked.expect_err("a missing cache field must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(
        message.contains("request 4: no cache_creation"),
        "{message}"
    );

    let log = simulate(&SHORT, true);
    let mut counted: Vec<Counted> = requests(&log, None)
        .iter()
        .map(|r| Ok(r.prompt()))
        .collect();
    counted[2] = Err("connection refused".into());
    let rows = requests(&log, Some(&counted));
    let printed = table(&rows, SIMULATED, &Latency::fit(&rows, None));
    let row = printed.lines().nth(3).expect("request 3's row");
    assert_eq!(row.split_whitespace().nth(3), Some("err"), "{row}");
    let checked = std::panic::catch_unwind(|| {
        assert_caches(&log, Some(&counted), &SHORT, SIMULATED);
    });
    let panic = checked.expect_err("a failed count must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(
        message.contains("request 3: count_tokens failed: connection refused"),
        "{message}"
    );
}

/// A run that found the server warm prefilled too little to measure a
/// rate; with a fallback it's still checked, and still fails re-prefills.
#[test]
fn a_warm_run_falls_back_to_the_backends_rate() {
    let warm = |lag| synthetic_run(lag).split_off(1);
    let unmeasured = Latency::fit(&warm(0.0), None);
    let why = unmeasured.err().expect("nothing to measure on");
    assert!(why.contains("no fallback rate"), "{why}");

    // blallama-like: 2.5 ms a token.
    let latency = Latency::fit(&warm(0.0), Some(400.0)).unwrap();
    assert!(matches!(
        latency.prefill,
        Prefill::Fallback { most: 400, .. }
    ));
    assert!(latency.prefill.to_string().contains("400 tokens/s"));
    check_latency(&warm(0.0), &latency, 1500.0);

    let lying = warm(2.5 * 7000.0);
    let latency = Latency::fit(&lying, Some(400.0)).unwrap();
    let checked = std::panic::catch_unwind(|| {
        check_latency(&lying, &latency, 1500.0);
    });
    let panic = checked.expect_err("a re-prefill's time must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("fallback of 400 tokens/s"), "{message}");
}
