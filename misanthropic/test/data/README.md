# Wire fixtures

Captured request/response shapes from the Anthropic API, replayed offline so
the crate's (de)serialization is verified against the **wire**, not the docs.

This exists because the docs drift from the wire, repeatedly:

- `tool_search_requests` is absent from the wire entirely (#72).
- the web_fetch error tag is `web_fetch_tool_result_error`, not the documented
  `web_fetch_tool_error`.
- live result blocks carry an undocumented `caller` field.

Trusting the tool-reference docs has cost real debugging time. A captured
fixture turns "I'm certain something's wrong somewhere in here" into a
checklist that fails loudly in the offline gate (`just test`).

## The discipline

1. **Capture** the real shape from the live API (see below), once.
2. **Save** it here under a descriptive name. Non-streaming blocks → `*.json`.
   Streaming → `*.sse.stream.txt` (raw SSE, with `event:`/`data:` framing) or
   `*.sse.stream.jsonl` (one JSON value per line). See *fixture formats* below.
3. **Replay offline** and assert an **exact** round-trip:
   - non-streaming: `crate::utils::roundtrip::<T>(include_str!(…))` —
     deserialize, re-serialize, assert value-equal to the captured bytes. Use
     `roundtrip_checked` for the non-panicking variant (collect many failures).
   - streaming: `crate::utils::roundtrip_sse(…)` asserts an exact per-line
     round-trip of an SSE-JSONL fixture and returns the events; or
     `stream::tests::mock_stream` (raw SSE) / `mock_stream_jsonl` to assert the
     **assembled** `Block`/`Message`.

## Fixture formats

| extension | contents | replayed by | what it guards |
| --- | --- | --- | --- |
| `*.json` | one response `Block` | `roundtrip` / `roundtrip_checked` | block (de)serialization, exact |
| `*.sse.stream.txt` | raw SSE bytes (`event:`/`data:` framing) | `mock_stream` | the SSE **parser** (framing → events) |
| `*.sse.stream.jsonl` | one value per line | `roundtrip_sse` / `mock_stream_jsonl` | event (de)serialization / assembly |

**On streaming round-trip:** it is exact only for events the crate models 1:1
with the wire (`ping`, `content_block_stop`, `message_stop`, …). It is *not*
exact for **delta** events — a `content_block_delta` of `text_delta`
deserializes into `Block::Text` (via its `text_delta` alias) and re-serializes
as `text`, because the crate models streaming deltas as their *assembled* block.
And **error** events (`{"type":"error",…}`) are the `Err` arm of the stream's
`Result<Event, _>`, not an `Event` variant — which is why the older
`*.sse.stream.jsonl` capture (`redacted_thought…`) is `{"Ok": …}`-wrapped.
So full-stream wire verification is an *assemble-and-compare* job (see #78), not
a byte round-trip; `roundtrip_sse` is for the faithful events and for per-event
checks within #78.

## The coverage gate

`src/tests/wire_coverage.rs` is the forcing function that keeps this from
lapsing. It auto-discovers every fixture in `server_tools/` (no hand-maintained
list) and:

- round-trips each `*.json` as a `Block`, asserting exactness **and** that it is
  a wire-sourced block (a misfiled one fails);
- asserts every *wire-sourced* `Block` variant — and every known `Caller` shape
  — is covered by at least one fixture. Adding a `Block`/`KnownCaller` variant
  fails to compile (`BlockKind`/`KnownCallerKind` via `strum::EnumDiscriminants`)
  until it is classified in `needs_fixture`/`caller_needs_fixture`, and then
  fails the test until a fixture exists. **You cannot add a server tool without
  capturing it.**

Streaming twins (a `.sse.stream.jsonl` per block fixture) are reported as a
pending list today and hard-gated once the streaming captures land.

The round-trip assertion is load-bearing: **no response type uses
`#[serde(deny_unknown_fields)]`**, so an undocumented, renamed, or mis-tagged
field is silently dropped on deserialize. Re-serializing and comparing to the
captured bytes is the only thing that catches that offline — a dropped field
makes the comparison fail.

## Capturing (Haiku, by preference)

Live capture is the source of truth. Use Haiku (`claude-haiku-4-5`) to keep it
cheap; server-tool shapes are model-independent. Either:

- `curl` the Messages API directly with the matching `anthropic-beta` header
  and dump the JSON, or
- run an `#[ignore]`d capture test under `just test-ignored` (needs
  `misanthropic/api.key`) and copy the block out of the response.

When you replace a doc-derived fixture with a real capture, the round-trip test
will fail loudly if the wire disagrees with what we assumed — that failure *is*
the win; fix the types, then commit the real bytes.

## Provenance

| fixture | source | status |
| --- | --- | --- |
| `redacted_thought.sse.stream.jsonl` | live | captured |
| `thinking.sse.stream.txt` | live | captured |
| `sse.stream.txt` | live | captured |
| `server_tools/web_fetch_result.json` | live (`web_fetch`, Haiku 4.5) | captured (`caller` verified; no-citations doc shape; text data truncated) |
| `server_tools/web_fetch_pdf.json` | docs | **pending live capture** |
| `server_tools/web_fetch_error.json` | live (`web_fetch`, curl) | captured (tag verified vs #72) |
| `server_tools/web_search_result.json` | live (`web_search`, Haiku 4.5) | captured (`caller`, `page_age: null` verified; trimmed to 2 results) |
| `server_tools/web_search_error.json` | docs | **pending live capture** |
| `server_tools/tool_search_result.json` | docs | **pending live capture** |
| `server_tools/tool_search_error.json` | docs | **pending live capture** |
| `server_tools/tool_reference.json` | docs | **pending live capture** |
| `server_tools/server_tool_use.json` | live (`web_search`, Haiku 4.5) | captured |
| `server_tools/ptc_tool_use.json` | live (programmatic tool calling, Sonnet 4.6) | captured (`caller` of `code_execution_20260120` verified; PTC is unavailable on Haiku) |
| `server_tools/code_execution_result.json` | live (programmatic tool calling, Sonnet 4.6) | captured (undocumented `abort_reason: null` verified; PTC completion block) |
| `server_tools/bash_code_execution_result.json` | live (`code_execution`, Haiku 4.5) | captured (`bash_code_execution` stdout/exit; in-band failure via `return_code`) |
| `server_tools/bash_code_execution_file_output.json` | live (`code_execution`, Sonnet 4.6) | captured (`bash_code_execution_output` + `file_id`; only surfaces for files written to `$OUTPUT_DIR`, not `/tmp`/cwd — see #32) |
| `server_tools/text_editor_code_execution_create_result.json` | live (`code_execution`, Haiku 4.5) | captured (`create` → `is_file_update`) |
| `server_tools/text_editor_code_execution_view_result.json` | live (`code_execution`, Haiku 4.5) | captured (snake_case `num_lines`/`start_line`/`total_lines`, **not** the docs' camelCase) |
| `server_tools/text_editor_code_execution_str_replace_result.json` | live (`code_execution`, Haiku 4.5) | captured (snake_case `old_start`… diff hunk, **not** the docs' camelCase) |
| `server_tools/text_editor_code_execution_error.json` | live (`code_execution`, Haiku 4.5) | captured (undocumented `error_message`; bash error shape is the parallel `*_tool_result_error`) |
| `server_tools/memory_tool_use.json` | live (`memory`, Haiku 4.5) | captured (client-executed → plain `tool_use` not `server_tool_use`; `caller: direct` verified in both non-streaming and SSE) |

"pending live capture" fixtures are our best current guess from the docs and
round-trip cleanly against today's types; they should be replaced with real
Haiku captures (which may surface drift — that's expected and wanted).
