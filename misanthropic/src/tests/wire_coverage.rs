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

/// The data root — older streaming captures (e.g.
/// `redacted_thought.sse.stream.jsonl`) live here, predating `server_tools/`.
const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test/data");

/// Incremental-JSON captures (a tool call whose input is a list of objects,
/// and a structured-output generation of the same shape) — the parsing
/// substrate for #58. Round-trip-gated here like every stream fixture.
const INCREMENTAL_DIR: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/test/data/incremental");

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
    read_fixtures(DIR, |name| {
        name.ends_with(".json") && !name.ends_with(".sse.stream.jsonl")
    })
}

/// `(file_name, contents)` for every streaming fixture (`*.sse.stream.jsonl`),
/// from `server_tools/`, `incremental/`, and the data root (legacy captures).
fn stream_fixtures() -> Vec<(String, String)> {
    let mut out = read_fixtures(DIR, |n| n.ends_with(".sse.stream.jsonl"));
    out.extend(read_fixtures(DATA_DIR, |n| {
        n.ends_with(".sse.stream.jsonl")
    }));
    out.extend(read_fixtures(INCREMENTAL_DIR, |n| {
        n.ends_with(".sse.stream.jsonl")
    }));
    out.sort();
    out
}

fn read_fixtures(
    dir: &str,
    want: impl Fn(&str) -> bool,
) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = fs::read_dir(dir)
        .expect("fixture dir exists")
        .map(|e| e.expect("readable dir entry").path())
        .filter(|p| p.is_file())
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
/// are pulled from the fixtures' JSON wherever a `caller` appears — block
/// fixtures whole, stream fixtures per line — so streaming captures count
/// toward caller coverage too.
#[test]
fn every_known_caller_has_a_fixture() {
    let values = block_fixtures()
        .into_iter()
        .map(|(_, json)| serde_json::from_str(&json).expect("fixture is JSON"));
    let stream_values = stream_fixtures().into_iter().flat_map(|(_, jsonl)| {
        jsonl
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).expect("fixture line is JSON"))
            .collect::<Vec<serde_json::Value>>()
    });

    let mut seen: HashSet<KnownCallerKind> = HashSet::new();
    for value in values.chain(stream_values) {
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
/// Vacuous until streaming fixtures are captured (see
/// [`streaming_block_coverage`]).
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

/// The streaming coverage gate, content-based: every wire-sourced
/// [`BlockKind`] must arrive in the `content_block_start` of at least one
/// captured stream fixture. One captured stream covers every block it
/// contains (a single web_search stream covers both `ServerToolUse` and
/// `WebSearchToolResult`), so this needs far fewer captures than a
/// file-per-block convention — and it checks the block actually *streams*,
/// not that a file merely exists. The streaming "force the capture" gate
/// (#78): a new server tool can't ship without a captured stream.
///
/// `ToolReference` is exempt here (not in the block-fixture gate): on the
/// wire it only appears *nested* inside a `tool_search_tool_result`'s
/// `tool_references` — never as its own `content_block_start` — so the
/// tool_search capture covers its streaming shape.
#[test]
fn streaming_block_coverage() {
    let seen: HashSet<BlockKind> = stream_fixtures()
        .iter()
        .flat_map(|(_, jsonl)| {
            let f = roundtrip_sse(jsonl);
            f.assert_round_trips();
            f.into_events()
        })
        .filter_map(|event| match event {
            crate::stream::Event::ContentBlockStart {
                content_block, ..
            } => Some(BlockKind::from(&content_block)),
            _ => None,
        })
        .collect();

    let missing: Vec<String> = BlockKind::iter()
        .filter(|k| needs_fixture(*k))
        .filter(|k| *k != BlockKind::ToolReference) // nested-only; see above
        .filter(|k| !seen.contains(k))
        .map(|k| format!("  Block::{k:?}"))
        .collect();
    assert!(
        missing.is_empty(),
        "wire-sourced Block variant(s) not covered by any captured stream \
         fixture — capture one with test/data/capture.sh (see README):\n{}",
        missing.join("\n"),
    );
}

/// [`roundtrip_sse`] on the wrapped `{"Ok": …}` / `{"Err": …}` jsonl format —
/// both arms are typed, so error frames (hard to capture on purpose:
/// rate-limit and overloaded are intermittent) get real parse coverage. Delta
/// events round-trip exactly too since `Delta::Text` serializes as the wire's
/// `text_delta`.
#[test]
fn roundtrip_sse_wrapped_arms() {
    let jsonl = r#"{"Ok":{"type":"ping"}}
{"Ok":{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}}
{"Ok":{"type":"content_block_stop","index":0}}
{"Err":{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}}
{"Ok":{"type":"message_stop"}}
"#;
    let f = roundtrip_sse(jsonl);
    f.assert_round_trips();
    let results = f.into_results();
    assert_eq!(results.len(), 5, "every line parsed");
    assert_eq!(results.iter().filter(|r| r.is_err()).count(), 1);
    assert!(matches!(
        &results[3],
        Err(e) if matches!(
            e.error,
            crate::client::AnthropicError::Overloaded { .. }
        )
    ));
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
