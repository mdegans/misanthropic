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
2. **Save** it here under a descriptive name. Streaming → `*.sse.stream.txt`
   (raw SSE) or `*.sse.stream.jsonl` (one JSON `Event` per line);
   non-streaming blocks → `*.json`.
3. **Replay offline** and assert an **exact** round-trip:
   - non-streaming: `crate::utils::roundtrip::<T>(include_str!(…))` —
     deserialize, re-serialize, assert value-equal to the captured bytes.
   - streaming: `stream::tests::mock_stream` (raw SSE) or `mock_stream_jsonl`
     (JSONL), then assert the assembled `Block`/`Message`.

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

"pending live capture" fixtures are our best current guess from the docs and
round-trip cleanly against today's types; they should be replaced with real
Haiku captures (which may surface drift — that's expected and wanted).
