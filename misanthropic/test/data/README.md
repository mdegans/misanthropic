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
   `*.sse.stream.jsonl` (wrapped: one `{"Ok": <event>}` / `{"Err": <error
   event>}` per line, **raw wire bytes inside the wrapper** — never
   re-serialized through our types, which would hide dropped fields). See
   *fixture formats* below.
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
| `*.sse.stream.jsonl` | one `{"Ok": <event>}` / `{"Err": <error event>}` per line | `roundtrip_sse` / `mock_stream_jsonl` | event **and error** (de)serialization / assembly, exact |

**The wrapped jsonl format:** an SSE stream is `Result<Event, Error>` to
callers — **error** events (`{"type":"error",…}`) are the `Err` arm, not an
`Event` variant. The wrapper captures both arms in one file: `Ok` lines
round-trip as `Event`, `Err` lines as the typed wire error
(`stream::ErrorEvent`, requiring the `"type":"error"` tag). Errors are
captured errors-and-all on purpose — rate-limit/overloaded are intermittent
and hard to reproduce, so a stream that died mid-capture is *more* valuable,
not less. The round-trip is **exact for every wire event including deltas**
(`Delta::Text` serializes as the wire's `text_delta`; the bare `text` tag is
a legacy alias). Exactness here is the *drift detector*, not a functional
need — what we re-submit to the API is assembled blocks, which the `*.json`
round-trip already guards — so if a future wire shape can't round-trip without
contorting the types, prefer exempting it and asserting assembly instead.

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

Streaming coverage is **content-based**: every wire-sourced `BlockKind` must
arrive in the `content_block_start` of at least one captured stream fixture
(`streaming_block_coverage`). One captured stream covers every block it
contains — a single web_search stream covers both `ServerToolUse` and
`WebSearchToolResult` — so there is no file-per-block convention. Reported as
a pending list today; hard-gated once the #78 captures land.

The round-trip assertion is load-bearing: **no response type uses
`#[serde(deny_unknown_fields)]`**, so an undocumented, renamed, or mis-tagged
field is silently dropped on deserialize. Re-serializing and comparing to the
captured bytes is the only thing that catches that offline — a dropped field
makes the comparison fail.

## Capturing (Haiku, by preference)

Live capture is the source of truth. Use Haiku (`claude-haiku-4-5`) to keep it
cheap; server-tool shapes are model-independent. Either:

- **streaming:** `./capture.sh requests/foo.json [beta,beta…] > foo.sse.stream.jsonl`
  — curls the API with `stream: true` and emits the wrapped jsonl by pure text
  transform (raw bytes preserved, never parsed through our types). Keep the
  request body under `requests/` so the capture is reproducible.
- **non-streaming:** `curl` the Messages API directly with the matching
  `anthropic-beta` header and dump the JSON, or run an `#[ignore]`d capture
  test under `just test-ignored` (needs `misanthropic/api.key`) and copy the
  block out of the response.

When you replace a doc-derived fixture with a real capture, the round-trip test
will fail loudly if the wire disagrees with what we assumed — that failure *is*
the win; fix the types, then commit the real bytes.

## Provenance

| fixture | source | status |
| --- | --- | --- |
| `text.sse.stream.jsonl` | live (plain text turn, Haiku 4.5, `capture.sh`, 2026-06-09) | captured (baseline message envelope: `message.type`, `stop_details: null`, `usage.cache_creation`/`service_tier`/`inference_geo`, SSE whitespace padding; request in `requests/text.json`) |
| `redacted_thought.sse.stream.jsonl` | live | captured (text-delta tags restored to the wire's `text_delta` — the original capture had been normalized through the crate's own types to the legacy `text`; the lines are otherwise the captured bytes) |
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
| `server_tools/web_search.sse.stream.jsonl` | live (`web_search`, Haiku 4.5, `capture.sh`, 2026-06-10) | captured (`server_tool_use` + `web_search_tool_result` + `citations_delta`; request in `requests/web_search.json`) |
| `server_tools/web_fetch.sse.stream.jsonl` | live (`web_fetch`, Haiku 4.5, `capture.sh`, 2026-06-10) | captured (`web_fetch_tool_result` with citations enabled + `citations_delta`; request in `requests/web_fetch.json`) |
| `server_tools/tool_search.sse.stream.jsonl` | live (`tool_search_regex`, Haiku 4.5, `capture.sh`, 2026-06-10) | captured (`tool_search_tool_result` with nested `tool_reference` — references never stream as their own content block — then a direct `tool_use` of the deferred tool, `caller: direct`; request in `requests/tool_search.json`) |
| `server_tools/code_execution.sse.stream.jsonl` | live (`code_execution`, Haiku 4.5, `capture.sh`, 2026-06-10) | captured (`bash_code_execution_tool_result` + 4× `text_editor_code_execution_tool_result` create/view/str_replace; request in `requests/code_execution.json`) |
| `server_tools/ptc.sse.stream.jsonl` | live (programmatic tool calling, Sonnet 4.6, `capture.sh`, 2026-06-10) | captured (PTC turn 1: `server_tool_use` via `input_json_delta`s, **complete** `tool_use` with `code_execution` caller in one `content_block_start`, **`container` in `message_delta`** — two wire shapes the docs don't mention; request in `requests/ptc.json`) |
| `server_tools/ptc_resume.sse.stream.jsonl` | live (programmatic tool calling, Sonnet 4.6, `capture.sh`, 2026-06-10) | captured (resumed PTC turn: `message_start` arrives **pre-populated** — content, `container`, `stop_reason` already set — then `message_stop`, no deltas at all) |
| `server_tools/code_execution_result.sse.stream.jsonl` | live (programmatic tool calling, Sonnet 4.6, `capture.sh`, 2026-06-10) | captured (final PTC turn: `code_execution_tool_result` completion block + text; produced by replaying the tool_result loop — see the `ptc` request and issue #78) |
| `incremental/tool_items.sse.stream.jsonl` | live (Haiku 4.5, `capture.sh`, 2026-06-10) | captured (a tool call whose input is a **list of objects**, arriving as 21 `input_json_delta` frames split mid-token — the #58 incremental-parsing substrate; request in `requests/incremental_tool.json`) |
| `incremental/structured_items.sse.stream.jsonl` | live (Haiku 4.5, `capture.sh`, 2026-06-10) | captured (a structured-output (`output_config`) generation of the **same list-of-items schema**, arriving as JSON in plain `text_delta`s — #58's other flavor of incremental JSON; request in `requests/incremental_structured.json`) |

"pending live capture" fixtures are our best current guess from the docs and
round-trip cleanly against today's types; they should be replaced with real
Haiku captures (which may surface drift — that's expected and wanted).
