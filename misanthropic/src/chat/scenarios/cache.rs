//! A multi-turn prompt-caching check: one [`Chat`] run of scripted beats
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
//!   but re-prefills anyway fails here.
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
//! turn's output.
//!
//! Every entry point runs the same beats:
//!
//! - `blallama::canonical` and `blallama::after_assistant`: a local
//!   drama_llama server, skipped unless `BLALLAMA_URL` is set (see `live`).
//!   The first is configured exactly as for Anthropic; the second reports
//!   [`breakpoint_after_assistant`](crate::Quirks::breakpoint_after_assistant)
//!   so the driver marks assistant turns instead. Each request's `T` must
//!   also equal the server's `count_tokens`. Run with `just test-cache`.
//! - `anthropic::canonical`: **paid** — `claude-haiku-4-5`, about three
//!   cents, with `misanthropic/api.key`. `#[ignore]`d, and skipped even
//!   then unless `MISANTHROPIC_PAID_CACHE=1`, so the live CI gate (which
//!   runs every ignored test) doesn't pay for it. Run with
//!   `just test-cache-anthropic`.
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
/// The most uncached `input` a request after the first may pay: a beat or
/// a tool result, never the conversation.
const INPUT_CAP: u64 = 1024;
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
    /// Anthropic, or a stand-in for it: the reference, held to its exact
    /// behavior (the prompt grows, the tip is Anthropic's zero).
    reference: bool,
    /// Markers ride assistant turns (the driver's window), not the tail.
    after_assistant: bool,
    /// Tokens `read` may fall short of the bound.
    slack: u64,
    /// Wall time allowed past the latency budget, for the network and
    /// scheduling; `None` prints the latency without holding it (Anthropic's
    /// accounting is ground truth, its latency noise and retries).
    latency_slack_ms: Option<f64>,
}

/// Anthropic: `read` is the previous prompt less the framing after its
/// last block — single digits.
const ANTHROPIC: Backend = Backend {
    reference: true,
    after_assistant: false,
    slack: 16,
    latency_slack_ms: None,
};

/// A local blallama, configured as for Anthropic. Its template may render
/// the moved tail a little differently.
#[cfg(feature = "blallama")]
const BLALLAMA: Backend = Backend {
    reference: false,
    after_assistant: false,
    slack: 64,
    latency_slack_ms: Some(1500.0),
};

/// [`simulated`]: Anthropic's accounting, with a mock's latency (a second
/// of slack rides out a loaded CI runner).
const SIMULATED: Backend = Backend {
    latency_slack_ms: Some(1000.0),
    ..ANTHROPIC
};

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
/// hand back what was sent and received. `backend` is only for the table
/// printed if the run stops.
async fn scenario<T>(
    transport: Checked<T>,
    base: Prompt,
    backend: Backend,
) -> Log
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
        let requests = requests(&log, None);
        let latency = Latency::fit(&requests);
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
    /// The server's `count_tokens` for the request, when asked.
    counted: Option<u64>,
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

/// `log`'s requests, with each prompt's `counted` alongside if given.
fn requests(log: &Log, counted: Option<&[u64]>) -> Vec<Request> {
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
                counted: counted.map(|counted| counted[n]),
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

/// What a request's wall time should be, fitted to the run itself.
struct Latency {
    /// The request calibrated on: the one that prefilled the most.
    calibration: usize,
    /// Milliseconds per prefilled token, from `calibration`'s wall time
    /// less its decode.
    prefill_ms: f64,
    /// Milliseconds per output token: the fastest other request's, as close
    /// to pure decode as the run gets.
    decode_ms: f64,
}

impl Latency {
    /// The model, or `None` (with the reason) when the run can't calibrate
    /// one: no request prefilled [`CALIBRATION_FLOOR`] tokens (all warm from
    /// an earlier run), or none else decoded [`DECODE_FLOOR`].
    fn fit(requests: &[Request]) -> Result<Self, String> {
        let (calibration, cold) = requests
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| r.prefilled())
            .ok_or("no requests")?;
        if cold.prefilled() < CALIBRATION_FLOOR {
            return Err(format!(
                "no request prefilled {CALIBRATION_FLOOR} tokens (the most \
                 was {}): nothing to calibrate on",
                cold.prefilled()
            ));
        }
        let decode_ms = requests
            .iter()
            .enumerate()
            .filter(|&(n, r)| n != calibration && r.output >= DECODE_FLOOR)
            .map(|(_, r)| r.millis / r.output as f64)
            .min_by(f64::total_cmp)
            .ok_or("no other request decoded enough to time decoding")?;
        let rest = cold.millis - decode_ms * cold.output as f64;
        Ok(Self {
            calibration,
            prefill_ms: rest.max(0.0) / cold.prefilled() as f64,
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
            or_dash(r.counted),
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
            "latency: {:.2} ms per prefilled token (request {}), {:.1} ms per \
             output token; `rest` is wall time less decode",
            l.prefill_ms,
            l.calibration + 1,
            l.decode_ms,
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

/// Hold `log` to a healthy cached loop's signature on `backend` (see the
/// module docs), after printing its table. `counted` is each prompt's
/// `count_tokens`, which `input + creation + read` must equal exactly.
fn assert_caches(log: &Log, counted: Option<&[u64]>, backend: Backend) {
    let requests = requests(log, counted);
    let latency = Latency::fit(&requests);
    eprintln!("{}", table(&requests, backend, &latency));

    assert!(requests.len() >= BEATS.len(), "a request per beat at least");
    assert!(
        requests.iter().any(|r| r.kind == "tool"),
        "no tool round: the scenario covers tool results too"
    );
    check_counts(&requests);
    check_growth(&requests, backend);
    check_reads(&requests, backend);
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
        if let Some(counted) = r.counted {
            assert_eq!(
                r.prompt(),
                counted,
                "request {}: input + creation + read vs count_tokens",
                n + 1
            );
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
/// before it cached, paying little fresh.
fn check_reads(requests: &[Request], backend: Backend) {
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
            now.input <= INPUT_CAP,
            "request {k} paid {} uncached input tokens (cap {INPUT_CAP})",
            now.input
        );
    }
}

/// Warn, off Anthropic, where a request re-prefilled most of what the one
/// before it generated rather than reading its KV back.
fn check_tips(requests: &[Request], backend: Backend) {
    if backend.reference {
        return;
    }
    for (n, pair) in requests.windows(2).enumerate() {
        let (before, now, k) = (&pair[0], &pair[1], n + 2);
        let tip = tip(before, now);
        if before.output >= TIP_FLOOR && tip < before.output / 2 {
            eprintln!(
                "WARN tip: request {k} read {tip} tokens past the \
                 Anthropic-equivalent bound; request {} generated {}, so it \
                 re-prefilled most of them",
                k - 1,
                before.output
            );
        }
    }
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
             ({:.2} ms per token, from request {}); re-prefilling all {} \
             would take about {:.0} ms: the reported cache read didn't save \
             the time",
            n + 1,
            r.prefilled(),
            latency.prefill_ms,
            latency.calibration + 1,
            r.prompt(),
            latency.full(r),
        );
    }
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

    /// Run the scenario through `wrap`ped [`Client`], held to `backend`,
    /// unless no server is configured.
    async fn run<T>(
        name: &str,
        backend: Backend,
        wrap: impl FnOnce(Client) -> T,
    ) where
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
        let transport = Checked::new(wrap(client.clone()));
        let log = scenario(transport, base, backend).await;
        let counted = counted(&client, &log).await;
        assert_caches(&log, Some(&counted), backend);
    }

    /// Configured exactly as for Anthropic.
    #[tokio::test]
    async fn canonical() {
        run("canonical", BLALLAMA, |client| client).await;
    }

    /// Markers on assistant turns, as agentkit places them for blallama.
    #[tokio::test]
    async fn after_assistant() {
        let backend = Backend {
            after_assistant: true,
            ..BLALLAMA
        };
        run("after_assistant", backend, AfterAssistant).await;
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

    /// Configured as a long conversation should be; see the module docs.
    #[tokio::test]
    #[ignore = "PAID: live Anthropic API (claude-haiku-4-5, about 3 cents); \
                needs misanthropic/api.key and MISANTHROPIC_PAID_CACHE=1"]
    async fn canonical() {
        if !paid("anthropic::canonical") {
            return;
        }
        let client = Client::new(crate::utils::load_api_key().await).unwrap();
        let base = Prompt::default()
            .model(Id::Haiku45)
            .max_tokens(NonZeroU32::new(512).unwrap());
        let transport = Checked::new(Retrying(client));
        let log = scenario(transport, base, ANTHROPIC).await;
        assert_caches(&log, None, ANTHROPIC);
    }
}

/// A stand-in for Anthropic's cache and model. It sizes a request by its
/// serialized length and writes it all to the cache but the framing; when
/// `healthy` it first reads back what the previous request cached if this
/// one extends it, broken it never does (a prefix that changes every
/// turn). A beat naming the ledger or the stores gets one call, anything
/// else an answer.
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
    futures::executor::block_on(scenario(transport, base, SIMULATED))
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

    assert_caches(&log, None, SIMULATED);
}

/// A cache rewritten every turn fails, however well the rest goes.
#[test]
fn simulated_broken_cache_fails() {
    let log = simulate(false);
    let checked =
        std::panic::catch_unwind(|| assert_caches(&log, None, SIMULATED));
    let panic = checked.expect_err("a re-prefilled prompt must fail");
    let message = panic.downcast_ref::<String>().expect("a message");
    assert!(message.contains("re-prefilled"), "{message}");
}

/// A synthetic request: `prefilled` uncached tokens (all written but a
/// 3-token tail) over `read` cached, `output` decoded, in `millis`.
fn synthetic(read: u64, prefilled: u64, output: u64, millis: f64) -> Request {
    Request {
        kind: "beat",
        input: 3,
        written: Some(prefilled - 3),
        read: Some(read),
        counted: None,
        output,
        millis,
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
    let latency = Latency::fit(&healthy).unwrap();
    assert_eq!(latency.calibration, 0);
    check_latency(&healthy, &latency, 1500.0);

    // Reads reported, but each turn re-prefills the ~7k-token prompt.
    let lying = synthetic_run(2.5 * 7000.0);
    let latency = Latency::fit(&lying).unwrap();
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
