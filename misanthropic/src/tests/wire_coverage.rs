//! The **wire-coverage gate**.
//!
//! Forces every wire-sourced [`Block`]/[`Caller`] variant to have a captured
//! fixture in `test/data/server_tools/` that round-trips exactly — so coverage
//! can't silently lapse when a new server tool or block type lands. The forcing
//! chain:
//!
//! 1. `EnumDiscriminants` derives a fieldless `BlockKind` from [`Block`]; adding
//!    a `Block` variant adds a `BlockKind` variant.
//! 2. [`needs_fixture`] is an **exhaustive** match on `BlockKind` — the new
//!    variant won't compile until it's classified wire / not-wire.
//! 3. If wire, [`every_wire_block_variant_has_a_fixture`] fails until a fixture
//!    that round-trips to that variant exists.
//!
//! No link is skippable. See `test/data/README.md` for the capture discipline.

use std::collections::HashSet;
use std::fs;

use strum::IntoEnumIterator;

use crate::prompt::message::{Block, BlockKind};
use crate::tool::{Caller, KnownCallerKind};
use crate::utils::{roundtrip_checked, roundtrip_sse};

/// The captured server-tool fixtures live here.
const DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/test/data/server_tools");

/// Whether a [`Block`] variant is a wire-sourced server-tool / caller-bearing
/// block that must have a captured fixture under `test/data/server_tools/`.
///
/// Exhaustive on purpose: adding a `Block` variant fails to compile until it's
/// classified — the compile-time half of the forcing function. Blocks we *send*
/// (`Text`/`Image`/`Document`/`ToolResult`) and model thinking output
/// (`Thought`/`RedactedThought`, covered by the top-level stream fixtures) are
/// not server-tool wire shapes.
fn needs_fixture(kind: BlockKind) -> bool {
    match kind {
        BlockKind::ServerToolUse
        | BlockKind::ToolUse // carries `caller` (PTC / client-executed memory)
        | BlockKind::WebSearchToolResult
        | BlockKind::WebFetchToolResult
        | BlockKind::ToolSearchToolResult
        | BlockKind::ToolReference
        | BlockKind::CodeExecutionToolResult
        | BlockKind::BashCodeExecutionToolResult
        | BlockKind::TextEditorCodeExecutionToolResult => true,
        BlockKind::Text
        | BlockKind::Thought
        | BlockKind::RedactedThought
        | BlockKind::Image
        | BlockKind::Document
        | BlockKind::ToolResult => false,
    }
}

/// Whether a known [`Caller`] shape must have a fixture exercising it.
///
/// Exhaustive (same forcing function, for callers). `code_execution_20250825`
/// is exempt: it's the superseded predecessor of `_20260120`, which the current
/// API emits instead, so it can't be re-captured live.
fn caller_needs_fixture(kind: KnownCallerKind) -> bool {
    match kind {
        KnownCallerKind::Direct | KnownCallerKind::CodeExecution20260120 => {
            true
        }
        KnownCallerKind::CodeExecution20250825 => false,
    }
}

/// `(file_name, contents)` for every non-streaming block fixture (`*.json`,
/// excluding the streaming `*.sse.stream.jsonl`), sorted for stable reports.
fn block_fixtures() -> Vec<(String, String)> {
    read_fixtures(|name| {
        name.ends_with(".json") && !name.ends_with(".sse.stream.jsonl")
    })
}

/// `(file_name, contents)` for every streaming fixture (`*.sse.stream.jsonl`).
fn stream_fixtures() -> Vec<(String, String)> {
    read_fixtures(|name| name.ends_with(".sse.stream.jsonl"))
}

fn read_fixtures(want: impl Fn(&str) -> bool) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = fs::read_dir(DIR)
        .expect("server_tools fixture dir exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(&want))
        .map(|p| {
            let name = p.file_name().unwrap().to_str().unwrap().to_string();
            (name, fs::read_to_string(&p).expect("read fixture"))
        })
        .collect();
    out.sort();
    out
}

/// Every non-streaming fixture round-trips exactly **and** is actually a
/// wire-sourced block (a misfiled non-wire block fails here too).
#[test]
fn block_fixtures_round_trip_and_are_wire() {
    let mut problems = Vec::new();
    for (name, json) in block_fixtures() {
        match roundtrip_checked::<Block>(&json) {
            Err(e) => problems.push(format!("  {name}: {e}")),
            Ok(block) => {
                let kind = BlockKind::from(&block);
                if !needs_fixture(kind) {
                    problems.push(format!(
                        "  {name}: round-trips as Block::{kind:?}, not a \
                         wire-sourced server-tool block — misfiled?"
                    ));
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "{} block fixture problem(s):\n{}",
        problems.len(),
        problems.join("\n"),
    );
}

/// The reverse check — the forcing function. Every wire-sourced `BlockKind` is
/// covered by at least one fixture; a new server tool with no capture fails.
#[test]
fn every_wire_block_variant_has_a_fixture() {
    let seen: HashSet<BlockKind> = block_fixtures()
        .into_iter()
        .filter_map(|(_, json)| roundtrip_checked::<Block>(&json).ok())
        .map(|b| BlockKind::from(&b))
        .collect();

    let missing: Vec<String> = BlockKind::iter()
        .filter(|k| needs_fixture(*k))
        .filter(|k| !seen.contains(k))
        .map(|k| format!("  Block::{k:?}"))
        .collect();

    assert!(
        missing.is_empty(),
        "wire-sourced Block variant(s) with no fixture in \
         test/data/server_tools/ — capture one (see README):\n{}",
        missing.join("\n"),
    );
}

/// Every known [`Caller`] shape is exercised by at least one fixture. Callers
/// are pulled from the fixtures' JSON wherever a `caller` appears.
#[test]
fn every_known_caller_has_a_fixture() {
    let mut seen: HashSet<KnownCallerKind> = HashSet::new();
    for (_, json) in block_fixtures() {
        let value: serde_json::Value =
            serde_json::from_str(&json).expect("fixture is JSON");
        let mut callers = Vec::new();
        collect_callers(&value, &mut callers);
        for c in callers {
            if let Ok(Caller::Known(kc)) = serde_json::from_value::<Caller>(c) {
                seen.insert(KnownCallerKind::from(&kc));
            }
        }
    }

    let missing: Vec<String> = KnownCallerKind::iter()
        .filter(|k| caller_needs_fixture(*k))
        .filter(|k| !seen.contains(k))
        .map(|k| format!("  KnownCaller::{k:?}"))
        .collect();

    assert!(
        missing.is_empty(),
        "known caller shape(s) with no fixture exercising them:\n{}",
        missing.join("\n"),
    );
}

/// Every streaming fixture round-trips per line (one report for all bad lines).
/// Vacuous until streaming fixtures are captured (see [`streaming_twins`]).
#[test]
fn stream_fixtures_round_trip() {
    let fixtures = stream_fixtures();
    let (mut byte_stable, mut total) = (0usize, 0usize);
    for (_, jsonl) in &fixtures {
        let f = roundtrip_sse(jsonl);
        f.assert_round_trips();
        byte_stable += f.byte_stable_count();
        total += f.lines.len();
    }
    if total > 0 {
        // Informational: ordering-stability signal, never gated on.
        eprintln!(
            "wire_coverage: {byte_stable}/{total} SSE lines byte-stable \
             across {} stream fixture(s)",
            fixtures.len(),
        );
    }
}

/// **Informational, never fails.** Lists block fixtures still lacking a
/// streaming `.sse.stream.jsonl` twin. The streaming captures are deferred to
/// #78; when they land, this flips to a hard assertion (the "force the capture"
/// gate).
#[test]
fn streaming_twins() {
    let have_stream: HashSet<String> = stream_fixtures()
        .into_iter()
        .map(|(n, _)| n.trim_end_matches(".sse.stream.jsonl").to_string())
        .collect();
    let missing: Vec<String> = block_fixtures()
        .into_iter()
        .map(|(n, _)| n.trim_end_matches(".json").to_string())
        .filter(|stem| !have_stream.contains(stem))
        .collect();
    if !missing.is_empty() {
        eprintln!(
            "wire_coverage: {} block fixture(s) still lack a streaming twin \
             (deferred capture):",
            missing.len(),
        );
        for stem in &missing {
            eprintln!("  {stem}.sse.stream.jsonl");
        }
    }
}

/// [`roundtrip_sse`] on faithful events — the ones the crate models 1:1 with
/// the wire (`ping`/`content_block_stop`/`message_stop`), so an exact per-line
/// round-trip is meaningful and `into_events` assembles them.
///
/// Note: *delta* events (`content_block_delta`) are deliberately **not**
/// round-trip-faithful — a `text_delta` deserializes into [`Block::Text`] (via
/// its `text_delta` alias) and re-serializes as `text`, because the crate
/// models streaming deltas as their assembled block. So full-stream exact
/// verification is an assemble-and-compare job (#78), not a round-trip; this
/// fixes the helper itself on the events where round-trip *is* the right check.
#[test]
fn roundtrip_sse_on_faithful_events() {
    let jsonl = "{\"type\":\"ping\"}\n\
                 {\"type\":\"content_block_stop\",\"index\":0}\n\
                 {\"type\":\"message_stop\"}\n";
    let f = roundtrip_sse(jsonl);
    f.assert_round_trips();
    assert_eq!(f.into_events().len(), 3, "every event line parsed");
}

/// Recursively collect every non-null value found under a `"caller"` key.
fn collect_callers(v: &serde_json::Value, out: &mut Vec<serde_json::Value>) {
    match v {
        serde_json::Value::Object(map) => {
            for (k, val) in map {
                if k == "caller" && !val.is_null() {
                    out.push(val.clone());
                }
                collect_callers(val, out);
            }
        }
        serde_json::Value::Array(arr) => {
            arr.iter().for_each(|x| collect_callers(x, out))
        }
        _ => {}
    }
}
