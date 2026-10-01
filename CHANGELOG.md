# Changelog

All notable changes to this crate are documented here. The format is loosely
based on [Keep a Changelog]. The crate follows [Semantic Versioning], with the
caveat that **while pre-1.0 (`0.x` / `1.0.0-alpha.*`), breaking changes may land
in any release** — they are collected under **Breaking** below so a downstream
upgrading across pre-releases has one place to look.

Entries marked **Breaking** require a downstream change. Conventional-commit
`!` markers (`feat(x)!: …`) in the git history are the authoritative per-commit
record; this file aggregates them.

[Keep a Changelog]: https://keepachangelog.com/en/1.1.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Breaking

- **Cache placements that Anthropic would reject now return a
  `CacheError` instead of building the request.** The 1-hour and automatic
  placements are fallible: `Prompt::cache_1h` / `cache_with` /
  `auto_cache` / `auto_cache_1h` / `auto_cache_with` return
  `Result<Prompt, CacheError>`, `Prompt::cache_windowed_1h` /
  `cache_windowed_with` return `Result<(), CacheError>`, and on
  `CachedPrompt` so do `cached_1h`, `cache_1h`, `cache_windowed_1h` /
  `_with` and `set_auto_cache` / `set_auto_cache_1h` (the `&mut` ones leave
  the prompt unchanged on an error). Add a `?` — `Prompt::default()
  .system(s).auto_cache()?`. The 5-minute block placements (`cache`,
  `cache_windowed`, `CachedPrompt::cached` / `cache`) keep their
  signatures: they can only clash with a 1-hour marker at or after theirs,
  which already caches that prefix for longer, so they skip that marker.
  `Stop` gains `Stop::Cache` (it is `#[non_exhaustive]`, so a `_` arm
  already covers it).

- **Deserializing a `Prompt` requires `model`, `messages` and `max_tokens`**,
  matching Anthropic, which 400s with `<field>: Field required`. The
  container-level `#[serde(default)]` is gone (every other field still
  defaults), so a body missing one is a serde `missing field` error instead
  of silently becoming e.g. a 4096-token request. This lets drama_llama's
  blallama return the same 400 as Anthropic. Anything a `Prompt` serialized
  (the chat demo's exports, persisted prompts) already has all three and
  still loads; add them to hand-written bodies. `CachedPrompt` deserializes
  through `Prompt`, so it is stricter the same way.

- **`chat::BudgetPolicy` and `tool::bash::Network` are `#[non_exhaustive]`.**
  Downstream `match` on either now needs a `_` arm. Planned variants — a
  dispatch-once final word (#136) and an egress `Allowlist` (#87) — can then
  land without further breaks.

- **With `schema-order-check`, an authored tool schema declaring a required
  property after an optional one is now rejected**: a compile error under
  `#[derive(ToolArgs)]`; `ToolBuildError::InvalidInputSchema` from
  `MethodBuilder::build` (and so from `TryFrom<MethodBuilder>` /
  `try_add_tool` given a builder); and a panic from `ToolArgs::definition`.
  `#[tool]` can't see its args' fields, so it emits a `#[cfg(test)]` test per
  method (`__misanthropic_schema_order_{tool}_{method}`) that builds the
  definition — a misordered args struct fails your `cargo test`. Received
  schemas are checked structurally only, so a `Prompt` written elsewhere
  still deserializes: deserializing a `Prompt` / `CustomMethodDef`,
  `CustomMethodDef::try_from(Value)` and `from_serializable`; opt an import
  in with the new `CustomMethodDef::try_from_checked`. Move required fields
  first, or reach for `MethodBuilder::build_unchecked` /
  `default-features = false`. Top-level properties only; `#[serde(flatten)]`ed
  fields and `with` types are left to the runtime check. A `#[tool]` impl
  inside a fn body trips `unnameable_test_items` in test builds; allow it on
  the enclosing fn, or move the impl to module level so its tests run.

- **`ToolBuildError` gains a `Json(serde_json::Error)` variant**, so an
  exhaustive `match` on it needs an arm. Its `InvalidInputSchema` message now
  reads "because", not "becuase".

- **Turn order rejects abandoning an in-flight server tool** — new
  `TurnOrderError::UnfinishedServerToolUse`. A `server_tool_use` no result in
  its turn answers (a `pause_turn` turn) admits only an assistant
  continuation; a user or system turn after it was accepted client-side and
  400'd on the wire (live-probed). `check_turn_order`, `push_message` and
  `Prompt::seat` now refuse it (a system note buffers instead). A
  programmatic call's container awaiting client `tool_result`s is not in
  flight, and a system turn after a result still needs that result to answer
  every use in the turn.
- **`Chat::run` hands everything back: `Ok((chat::Parts, State))` or
  `Err(chat::Error<State>)`** (was `Ok((Prompt, State))` or a bare
  `BoxError`, which dropped the prompt, the state, the tools and any
  buffered system note). `Parts { prompt, pending, toolbox, .. }` also
  carries the `Chat`'s configuration (hook, budget, caching, usage sink), and
  `Chat::from_parts(transport, parts)` rebuilds the same `Chat` from it — a
  note still buffered is seated after the next beat instead of lost. The
  `Error` carries the same (`kind`, `prompt`, `pending`, `state`, plus
  `toolbox_mut()` / `into_parts()`), implements `std::error::Error`, and
  stays `Send + Sync`, so `?` into a `BoxError` still works; `kind` is the
  new `chat::Stop`: `Clipped` and `Unusable` (see *Fixed*), or `Transport` /
  `Beat` / `Tool` / `TurnOrder` wrapping what used to be the bare error. The
  prompt is legal to resend, and a `Chat` whose prompt awaits the model
  (ending in a user or system turn, or a paused one) now **answers it before
  asking for a beat** — so resuming is a loop:

  ```rust
  let mut error = match chat.run((), &mut next_beat).await {
      Ok((parts, ())) => return Ok(parts.prompt),
      Err(error) => error,
  };
  // Clipped: nothing was seated or run — raise max_tokens, resume.
  error.prompt.max_tokens = error.prompt.max_tokens.saturating_mul(two);
  (chat, _) = error.resume(transport.clone());
  ```

  The toolbox is still torn down at the end of every run (so giving up on
  an `Error` leaks nothing) and prepared again by the next — `on_init` runs
  once per run — and what its tools pushed in between is delivered then.

### Added

- **`Prompt::check_cache`** checks a request's `cache_control` markers
  against the rules Anthropic 400s on, as a `CacheError` naming the
  offending `Breakpoint`s by Anthropic's own paths (`messages.2.content.0`):
  a tool result marked twice (on itself and in its content); more than 4
  markers (the automatic slot counted); a 1-hour marker after a 5-minute
  one (`tools` → `system` → `messages`, the automatic slot last); and an
  automatic slot whose TTL differs from a marker on the block it lands on
  (where Anthropic lands it: the last block that isn't thinking, a
  server-tool result included). The placements never fail it; hand-placed
  markers can, so `Chat` runs it before every request (unless the transport
  ignores markers) and stops with `Stop::Cache`, taking back the beat that
  request would have carried; a pushed note goes back to the front of the
  tools' queue, for a resumed `Chat` to deliver first. Under `Chat::cache`'s
  rolling window (a `breakpoint_after_assistant` transport), each request is
  also checked against the window its reply will get, where that turn will
  sit, so a turn the window can't legally mark is refused before it is paid
  for; a 1-hour window under a 5-minute `tools` / `system` marker, which no
  turn could carry, before a beat is taken. A turn that still can't be marked
  once seated (a hook reshaped it, or it ends in a server-tool result) is
  taken back rather than seated, its paid-for reply lost.
  Also `Block::cache_control`, `MethodDef::cache_control`,
  `CacheControl::ttl` and `CachedPrompt::cache_with`.

- **`schema-order-check` (default-on) enforces required-before-optional
  property order in tool input schemas** (#141). Anthropic and local grammar
  engines like drama_llama generate in `properties` order (the live probe
  found 0/24 optionals hoisted), while an engine following the
  structured-outputs docs hoists required properties first — so
  required-before-optional is the one layout every engine generates
  identically. It matters because field order changes what the model
  generates (reasoning must precede the answer), more so on smaller models.
  The feature enables `schema-order` (without `preserve_order` the check
  would see alphabetical order).

- **`CustomMethodDef::try_from_checked`** imports third-party tool JSON held
  to `MethodBuilder::build`'s authoring checks, property order included;
  `try_from` / `from_serializable` receive it as written. It returns a typed
  `ToolBuildError`: `InvalidInputSchema` for a misordered schema, the new
  `Json` variant for a value that isn't a tool definition.

- **`blallama` feature and `just test-blallama <model>`: live `Chat`
  scenarios against a local Anthropic-compatible server** (drama_llama's
  `blallama`; the recipes default to `http://127.0.0.1:11436`, overridden by
  `BLALLAMA_URL`). Local runs only: the tests skip unless `BLALLAMA_URL` is
  set, with `BLALLAMA_MODEL` naming the model, which keeps them out of CI's
  `--all-features` builds. They run the scenario table's
  live-able rows — a plain turn, stop sequences, a clip mid-call, a forced
  tool call through a `FinalWord` wrap-up, system notes, a notification —
  and hold every response to Anthropic's shape, so a server deviation
  surfaces as a failure. The offline table (68 rows over `MockTransport`)
  asserts the requests, turn shapes, calls run, usage and wire legality of
  every request and hand-back.
- **`just test-equivalence <model>`: a live warm-vs-cold KV-cache check.** A
  greedy (`top_k: 1`) conversation runs with blallama's cache reuse — a long
  answer, a forced tool call copying a long passage full of escapes and
  multi-byte text, a stop-sequence cut, a clipped turn retried in full — and
  then every recorded request is replayed cold, after tiny prompts evict
  every prefix-cache slot (the replay must read nothing from cache). Each
  reply must match its warm twin byte for byte, from a prompt of the same
  size; a mismatch names the block and char where they part, with context
  and the warm request's reuse, and gets a control. A request that read
  back the previous turn's generated tokens (the tip, `read_k > T_{k-1}`,
  which warm decoded one token at a time and cold prefills in a batch) gets
  the **matched-schedule** control: request `k-1` replayed cold must match
  warm `k-1` byte for byte, regenerating the tip as warm did, then request
  `k`, sent with no flush on that fresh tip, must match warm `k`. Both
  match: **tip schedule (explained)**, passed with a note. Only `k-1`
  matches: **cache failure**, failed. `k-1` doesn't: **upstream
  nondeterminism**, explained and warned about when warm `k-1` already
  differed from its own cold replay, and otherwise failed on a server
  declared to run one slot (`BLALLAMA_CACHE_SLOTS=1`, the recipe's
  default; `just test-equivalence <model> <slots>`), warned about on any
  other. Any other mismatch is replayed cold again: two cold replies that
  agree are **cache suspect** and fail, two that differ are
  **nondeterminism**, warned about loudly. Since the matched control can't
  tell a schedule from a fault the tip match repeats every time, every tip
  is also bounded by the previous request's output tokens plus a small
  slack for a stop sequence's overrun: a **tip overread** fails, even when
  the replies match. Any difference in prompt size fails, and so does a
  mismatch with no control. The table shows each request's tip, control
  and class, marks steps dropped for `max_tokens` or too many tool rounds,
  and the run fails early unless the forced call and its result round were
  both seated. Offline, simulated servers cover each class: a tip
  schedule passes with a note, a tip that drifts with the cache's history
  is a cache failure, cold nondeterminism upstream of a tip warns or fails
  by slot count, an overread tip fails, cold nondeterminism warns, and a
  stale cache fails. Start blallama with `--no-penalty` (a repetition
  penalty resumes warm but is rebuilt cold) and `--cache-slots 1` (other
  sequences in the unified KV cache change logits); the test skips unless
  `BLALLAMA_EQUIVALENCE=1` too, which only the recipe sets, since it
  evicts every cache slot.
- **`just test-cache <model>`: a live multi-turn prompt-caching check.** One
  `Chat` run of ten beats (four with a tool round) over a ~6.5k-token
  system prompt, cached as a long conversation should be (`Chat::cache` plus
  a system marker). Every request prints its input / written / read tokens,
  its tip (read past what Anthropic would) and latency, and must read back
  what the previous request cached, by Anthropic's own math for where the
  markers sit; keep `input` small; and spend no more time beyond decoding
  than prefilling its uncached tokens takes at the run's measured cold rate,
  so a server that reports reads but re-prefills fails. The same beats run
  against Anthropic as the paid reference (`just test-cache-anthropic`,
  about 3 cents on Haiku 4.5; `#[ignore]`d and also opt-in through
  `MISANTHROPIC_PAID_CACHE=1`, so CI's live gate doesn't pay for it), and
  offline against a simulated healthy and broken cache. A long variant —
  twenty beats, nine reading ten days of a logbook — grows the conversation
  itself by many thousands of tokens: on blallama by default (`just
  test-cache <model> long` alone), on Anthropic only as `just
  test-cache-anthropic long` (about 8 cents). On blallama, a run that finds
  the server warm (nothing prefilled enough to measure a rate) is timed
  against `BLALLAMA_PREFILL_RATE` tokens a second (420 by default, measured
  on Qwen3.6), and every request's prompt plus `max_tokens` must fit
  `BLALLAMA_N_CTX` (32768 by default; the long run asks for 4096 tokens, as
  Qwen3.6 renders each turn's reasoning back into the prompt).

- **`Message::unfinished_server_tool_uses()`**,
  **`Block::server_tool_result_id()`** and **`Caller::tool_id()`** — the pieces of the rule above: which server
  tool calls a turn leaves in flight, which use a server-tool result answers,
  and which container made a programmatic call.

- **`Prompt::user(content)` — an infallible opening turn** (#125), plus
  `impl From<&str> for Prompt`. A lone turn of text, image or document
  content is always legal, so the line every program starts with needs no
  `Role` import and no `?`: `client.message(Prompt::user("What is 2+2?"))`.
  (`tool_result` content is the exception; `add_message` is the checked
  path.)
- **`response::Disposition` + `response::Message::disposition()` — what a
  driver must do next with a turn** (#125): `Paused` (`pause_turn`),
  `Clipped` (`max_tokens` — never dispatch its tool calls), `ToolUse`, or
  `Done` (`end_turn` / `stop_sequence` / `refusal`). Exhaustive on purpose,
  so a new disposition breaks every driver's `match` at compile time. A
  response with no `stop_reason` (a provider that doesn't report one) is
  classified from its content: `ToolUse` if it calls tools, else `Done` — so
  a transport that omits it on a truncated turn bypasses the `Clipped` guard.
- **`tool_uses()` on `response::Message` and `Content`** — every client
  `tool::Use` in the turn, in order (the existing `tool_use()` returns only a
  trailing one). The response's is gated like `tool_use()`: empty unless
  `stop_reason` is `tool_use`, so a refused or truncated turn's calls never
  reach dispatch. `Content::tool_uses()` is the raw view (what `disposition`
  infers a stop-reason-less `ToolUse` from). The README, skills and tool
  examples (`strawberry`, `bash`, `text_editor`, `python`,
  `interleaved_thinking`) now dispatch through it and answer every call in one
  user turn; `tool_use()` is complete only when parallel tool use is disabled.

### Changed

- **`BudgetPolicy::FinalWord` asks for words.** The wrap-up call now goes out
  with `tool_choice: none` (the prompt's own `tool_choice` is restored after)
  instead of letting the model call tools only to answer them with synthetic
  errors. A transport with `Quirks::tool_choice_not_respected` gets the prompt
  unchanged, and calls a wrap-up makes anyway are still errored.
- **A streamed `max_tokens` clip's `Event::Message` now holds the open
  call.** Raw `Content::tool_uses()` on a clipped streamed turn now sees it,
  as on the non-streaming path; `response::Message::tool_uses()` stays empty
  and the turn is still `Disposition::Clipped`, so nothing dispatches.
  `with_tool_use` still never yields it. Only a `max_tokens` stop seats it:
  a call left open with no stop reason is still dropped.

### Fixed

- **Mixed cache TTLs built requests Anthropic rejects.** `cache_1h()` after
  `cache()`, `auto_cache_1h()` over a 5-minute marker, or `Chat::cache` with
  a 1-hour TTL over a seeded 5-minute marker each sent a 400 ("a ttl='1h'
  cache_control block must not come after a ttl='5m' cache_control block").
  So did a 5-minute automatic slot on a last block marked 1-hour — an
  undocumented rule the free `count_tokens` probes (claude-haiku-4-5,
  2026-09-30) turned up: "When both are specified on the same block, they
  must have matching TTLs". These are now `CacheError`s, before anything is
  sent.
- **A `cache_control` inside a tool result's content went uncounted.**
  Anthropic reads it as the tool result's own marker (its errors name the
  `tool_result` block), so it counts toward the 4 and the TTL rules, and a
  tool result takes one: on itself, or on one block of its content
  ("cache_control may not be specified within `tool_result.content`",
  probed on `count_tokens`). `check_cache` now counts it and reports a
  second one as `CacheError::Nested`; `Block::cache_with` on a tool result
  replaces the one in its content, and `uncache` (and so the window's
  eviction) clears it.
- **Cache markers past Anthropic's limit of 4.** A fifth `cache_control` is
  a 400 ("A maximum of 4 blocks with cache_control may be provided"), not
  the silent keep-the-last-4 the `CachedPrompt` docs promised, and the
  top-level automatic slot counts as one (probed on the free
  `count_tokens`). `Prompt::cache` / `CachedPrompt::cache` every turn used to
  pile up one marker per call; `cache_windowed*` counted a message once
  however many of its blocks were marked, ignored the automatic slot, and
  never trimmed its own window. Now `cache`, `cache_windowed*` and
  `auto_cache` / `set_auto_cache` (and so `Chat::cache`) never take a
  request past 4: they slide a window over the messages, always keeping
  the newest marker (the tail's, which the next request hits), then
  evicting the oldest 5-minute message markers first and a 1-hour one only
  when no other 5-minute one is left — never the `tools` / `system` ones or
  the automatic slot — and place nothing when those already hold every
  slot. So `cache` after four 1-hour message markers evicts the oldest of
  them rather than leaving the tail unmarked. An evicted marker costs no
  cache hits while a kept one sits within the API's ~20-block lookback
  after it and its entry is alive (its TTL, refreshed by each hit); a
  1-hour anchor is kept because, after a pause of more than five minutes,
  its entry is the one left. The docs also note the TTL rule:
  a 1-hour marker after a 5-minute one is a 400.
- **`ToolBox` offered its tools in a per-instance order.** Tools lead the
  cached prefix, but `definitions()` iterated a `HashMap`, so two boxes
  holding the same tools (identical agents, or one agent across a restart)
  usually rendered them differently and shared no prompt cache. They now
  render in insertion order (a re-added tool keeps its place), so a box
  registered in a fixed order renders the same prefix every time, and
  appending a tool leaves the earlier tools' cached prefix intact.
- **A streamed turn clipped mid-call assembled without the call.** On a
  `max_tokens` clip the wire streams a call's input only through its last
  completed member and never sends `content_block_stop`; `with_message`
  dropped the open block, so the turn differed from its non-streaming twin.
  Assembly now closes it at turn end — its open containers closed, a
  trailing scalar kept; failing that, completed members only (a member cut
  mid-value is dropped whole) — which matches the twin's `tool_use` input
  exactly in both captures (`test/data/stop/clip_*`).
- **A streamed turn's assembled usage double-counted.** `with_message` added
  the `message_delta` usage to `message_start`'s, but the delta's is
  cumulative for the turn — so a captured turn billed 685 in / 34 out
  assembled as 1370 / 44, and a server-tool turn summed its start and final
  input. The delta's counters now replace the start's, keeping any it omits
  (older deltas report only `output_tokens`; none carry the cache TTL
  breakdown). Every captured stream now assembles to its final report, and to
  its non-streaming twin's usage where one exists. `Chat::track_usage` sums
  whole responses, so it was off only behind a transport that streams.
- **Chat demo ran tool calls before the turn's stop reason arrived.** The
  frontend dispatched on `stream::Event::ToolUse`, which fires as the block
  closes — before `message_delta` — so a refused or truncated call could
  run. It now dispatches from the assembled `Event::Message` via
  `tool_uses()`, and answers parallel calls in one user turn instead of one
  turn per result. The `with_tool_use` / `Event::ToolUse` docs and the
  streaming skill now say to display from that event, not dispatch.
- **Chat demo seated refused / clipped turns.** A `Refusal` or `MaxTokens`
  turn holding an unanswered `tool_use` stayed in the history on both sides,
  so the next message 400'd. Both now drop the turn and rewind past the
  message that prompted it (a user turn can't follow a user turn), and the
  UI says why. The decision lives in `model::turn` (`Disposition`, `reply`,
  `rewind`), unit-tested against parallel, refused and clipped turns.
- **`Chat` no longer dispatches tool calls from a `max_tokens`-clipped turn**
  (#124). A clipped turn's `tool_use` can be valid JSON missing arguments the
  model never emitted; `Chat` seated it and ran the calls anyway. The loop now
  matches on `Disposition`: a `Clipped` turn (the `BudgetPolicy::FinalWord`
  wrap-up included) is never seated or dispatched, and `run` hands the
  un-advanced prompt back as `Stop::Clipped` — raising `max_tokens`,
  nudging the model, or giving up is the caller's policy.
- **`Chat` runs client tool calls only from a `tool_use` turn** (or a paused
  one). It dispatched by content, whatever the stop reason — but a
  `refusal` can cut a `tool_use` short, and so can a `stop_sequence`
  matched inside the call's input: captured live, the API closes the block
  as valid JSON truncated at the match, stopped `stop_sequence`, in both
  the non-streaming and SSE paths (`test/data/stop/`). A finished (`Done`)
  turn that still calls client tools is now `Stop::Unusable`: none run, and
  the whole turn is dropped (with any paused turn it continued), never
  stripped — stripping could strand a `server_tool_use`. Calls an
  `on_assistant` hook seats still run.
- **`Chat` never seats a finished turn that cuts a server tool short.** A
  refusal (or `end_turn`) leaving a `server_tool_use` unanswered — in its
  own content, or in the paused turn it continued — was seated as a dead
  turn nothing would ever answer, and the next beat 400'd. It is now
  `Stop::Unusable` like a finished turn with client calls, dropped whole.
- **A resumed `Chat` drops a paused turn it resumed, when it must.** A run
  seeded with a paused tail (a resume after, say, a transport error) didn't
  know where that turn started, so an unusable continuation or a budget
  hand-back left the in-flight turn in place. It now tracks the tail's
  paused turn from the start.
- **`Chat` hands back with a legal tail.** Exhausting the round budget
  mid-pause drops the in-flight paused turn whole, and a system turn left
  trailing by a hand-back (seated right before the call, or flushed by
  synthetic results) goes back to the pending buffer and re-seats after the
  next beat, instead of making that beat a `BadTransition`. The paused turn is
  tracked from where it was first seated and only within the round's own
  seating — a continuation seated after a flushed note no longer leaves the
  turn's start behind, and a hook that redacts a paused turn no longer makes
  the drop take earlier beats with it. System notes seated inside the dropped
  turn are re-buffered instead of lost, and a `FinalWord` wrap-up that pauses
  is no longer seated as an unresumable tail.
- **`Chat` drives a round on a beat that merges into the tail.** It skipped
  the model call whenever a beat left `messages.len()` unchanged, so a user
  beat merging into a user tail (e.g. the synthetic results a budget hand-back
  leaves) was silently never answered. It now asks `Seated::advanced`.
- **`Chat` no longer seats an empty assistant turn.** A refusal with no
  content, or an empty `end_turn`, was seated as a turn with no blocks, and
  the API rejects an empty turn — so the next request 400'd. Such a turn now
  reaches the `on_assistant` hook (which may still seat something) but is
  otherwise dropped, and the caller's next beat follows the previous tail.
- **A round that seats nothing no longer strands a system note.** A note
  seated right before the model call (a user + system beat, say) followed by
  a turn that seats nothing — a bare refusal, an empty `end_turn`, a hook
  returning nothing or only a system verdict — left a `[…, user, system]`
  tail, and the next beat failed with `Stop::TurnOrder` (system → user). The
  note now goes back to the pending buffer and follows the next beat, and a
  paused turn a hook left uncontinued is dropped whole.
- **A beat, or an `on_assistant` return, that breaks turn order is seated
  whole or not at all.** A hook returning, say, a `tool_use` turn followed by
  a user turn got `Stop::TurnOrder` with the `tool_use` turn already seated
  and unanswered — a hand-back no beat could legally follow — and a
  multi-message beat failing part-way left its first messages seated. The
  seating now rolls back.

## [1.0.0-alpha.20] — 2026-09-28

### Added

- **`MockTransport` — a scripted, recording `Transport` for offline tests**
  (new non-default `mock` feature, meant for downstream
  `[dev-dependencies]`; no new dependencies, runtime-agnostic). Generic over
  the prompt type (`MockTransport<Prompt>`, `MockTransport<CachedPrompt>`),
  it records every request as the JSON that would have hit the wire
  (`requests()`, `last()`, `len()`), answers from a FIFO script of
  `mock::Outcome`s — replies or `client::Error`s — and, with
  `MockTransport::with(|prompt| …)`, from a closure once the script runs
  out. Request *n* is always paired with the *n*th scripted outcome, however
  sends interleave. An exhausted script with no closure **panics** rather
  than returning an error, so retry or fallback logic under test can't
  swallow a missing reply. `yields(n)`, `events()` and `peak_in_flight()`
  let a test assert on overlap and ordering; `with_quirks` and
  `with_concurrency` set what the transport reports.

  `mock::Reply` builds the canned `response::Message`s — `mock::text`,
  `tool_use` (or `.call(tool::Use)` for a chosen id), `refusal` (with
  `StopDetails`, which downstream can't otherwise construct), `max_tokens`,
  or `message` for a captured fixture — with synthetic usage via `.usage`,
  `.cache_read`, `.cache_write` and `.counts`. `mock::http_error(status,
  retry_after)` scripts the `AnthropicError` the `Client` reports for a
  status, `retry-after` included on `429`/`529`.

  ```rust
  let mock = MockTransport::new()
      .then(mock::text("Hello!").usage(10, 2).cache_read(100))
      .then(mock::http_error(529, Some(3)));
  ```

## [1.0.0-alpha.19] — 2026-09-25

### Fixed

- **An error with an unrecognized `type` now keeps its HTTP status.** A
  non-OK response whose body parses as an error of a `type` the crate
  doesn't know (e.g. `"unknown"` from an Anthropic-compatible server) became
  `AnthropicError::Unknown { code: None, .. }`, dropping the status, so
  `status()` returned `None` and a caller's "retry 5xx" rule couldn't fire.
  `Client::get`/`post` now fill `code` from the response status when the
  body didn't supply one. Known variants are unchanged (their `status()` is
  still the one implied by their `type`). An SSE `error` event mid-stream
  has no HTTP status to take, so an unknown `type` there is still
  `code: None`.

## [1.0.0-alpha.18] — 2026-08-30

### Fixed

- **Batch `errored` results now surface the real API error instead of
  `"unknown error: error: "`.** The batch results JSONL wraps an errored
  item's error in the same envelope as a top-level HTTP error —
  `{"type":"errored","error":{"type":"error","error":{"type":"...","message":"..."}}}`
  — but `BatchResult`'s `Deserialize` fed the whole envelope to
  `client::AnthropicError`, whose `Deserialize` expects the inner object
  directly. The mismatched `"type": "error"` fell through to
  `AnthropicError::Unknown` with an empty message, discarding the actual
  error. The `errored` arm now unwraps the inner `error` object when the
  envelope shape is present, falling back to the bare shape otherwise.

## [1.0.0-alpha.17] — 2026-08-21

### Breaking

- **`Client::batch_poll` returns `batch::Error<P>` instead of `client::Error`,
  and `Pending<P>`/`Prompts<P>` no longer implement `Clone`.** `batch_poll`
  takes its [`Pending`] by value, so any transient failure — a gateway 503, a
  reset connection — destroyed a batch that was already submitted, already
  being billed, and not reconstructible from its id. `batch::Error` carries the
  `Pending` back out in the error:

  ```rust
  match client.batch_poll(pending).await {
      Ok(batch) => { /* … */ }
      Err(batch::Error { client_error, pending }) => {
          // The batch survived. Retry it, or persist it before giving up.
      }
  }
  ```

  There is deliberately **no** `From<batch::Error<P>> for client::Error`, so
  `?` cannot silently discard a live batch; use `Error::into_pending`,
  `Error::decompose`, or the `From<Error<P>> for Pending<P>` impl once the
  cause has been classified.

  `Clone` (added in #51 for exactly this retry problem) is removed as the
  inferior fix: it invited cloning an entire batch — up to 256 MB of prompts —
  before every poll, purely so the copy would survive a failure that is rare.
  Removing it turns "which call sites need the new error?" into a compile
  error rather than an audit. Callers that cloned solely to enable retry
  should destructure `batch::Error` instead; callers that genuinely need a
  copy can clone the underlying prompts.

### Added

- **`Debug` for `batch::Pending<P>`, `batch::Meta`, `batch::Status` and
  `batch::Stats`.** `Pending`'s impl is unbounded — it does not require
  `P: Debug` and prints a placeholder for prompt bodies, matching the existing
  `Prompts<P>` impl. This is what lets `batch::Error<P>` implement
  `std::error::Error` for any `P`.


## [1.0.0-alpha.16] — 2026-08-10

### Deprecated

- **`Id::Opus41` and `Id::Opus41_20250805` are `#[deprecated]`.** Opus 4.1 was
  retired from Anthropic's first-party API (2026-08); CI's live
  `test_ids_are_valid` caught the 404. The variants stay in the enum rather
  than being deleted — third-party hosts (Bedrock, Vertex) may still serve
  the model under the same wire id, reachable via a custom
  [`Client::base_url`] — but referencing either now warns, and they'll be
  removed in 2.0.

[`Client::base_url`]: https://docs.rs/misanthropic/latest/misanthropic/struct.Client.html#method.base_url

## [1.0.0-alpha.15] — 2026-08-09

### Fixed

- **Derived schemas no longer send `$ref`/`$defs`.** `schemars` hoists a named
  type — a nested struct, a fieldless enum — into `$defs` and emits a `$ref` at
  the use site. Anthropic [documents both as supported][so-limits], but under
  [strict tool use] its grammar compiler mis-decodes them: `tool_use.input`
  carries a value the model did not choose, with HTTP 200, a schema-valid
  payload, and nothing downstream able to tell.

  Measured against a six-variant enum with the reasoning field declared first,
  so the contradiction is visible in the same tool call:

  | cell | model | n | anomalies |
  |---|---|---|---|
  | `strict` + `$ref` | claude-opus-4-6 | 30 | **6** (20%) |
  | `strict` + `$ref` | claude-haiku-4-5 | 61 | **26** (43%) |
  | `strict` + inlined | both | 26 | 0 |
  | non-strict + `$ref` | claude-haiku-4-5 | 26 | 0 |

  Sample failure: `{"reasoning": "A ripe banana is yellow.", "verdict":
  "purple"}`. One run wedged the decoder entirely — 400 output tokens consumed
  without ever emitting a `required` field. Both models fail at comparable
  rates, so it is the server-side grammar compiler, not sampling.
  `output_config.format` appears unaffected (50 samples, zero anomalies).

  The fix is the new default-on `schema-inline` feature: schemas derived from a
  Rust type get their subschemas inlined at generation, via `schemars`'
  `SchemaSettings::inline_subschemas`. Inlining is semantics-preserving for
  non-recursive schemas, so the only cost is duplicated bytes when one `$def`
  is referenced many times — turn the feature off to send `$ref`/`$defs` as
  generated, mindful of the usual Cargo additive-features caveat.

  This is an upstream bug, not a crate one. [Reproducer and raw captures][repro];
  [discussion][issue]. The workaround is expected to outlive the fix.

  [so-limits]: https://platform.claude.com/docs/en/build-with-claude/structured-outputs#json-schema-limitations
  [repro]: https://github.com/claudeopusagora/anthropic-strict-ref-repro
  [issue]: https://github.com/mdegans/misanthropic/issues/147

### Added

- **`prompt::output::schema_for::<T>()`** — the wire-ready schema for `T`:
  subschemas inlined, then sanitized by `sanitize_for_anthropic`. This is what
  `OutputConfig::for_type` and `ToolArgs::schema` now both call; reach for it
  directly when you want the schema itself. `sanitize_for_anthropic` is
  unchanged and remains the right tool for hand-built schemas — inlining
  happens at generation, sanitizing after, so its existing promise to leave
  `$ref`/`$defs` and key order untouched still holds verbatim.
- **`prompt::output::contains_ref()`** — whether a schema contains a `$ref` at
  any depth. With the `log` feature, `ToolArgs::definition()` uses it to warn
  when `STRICT` is set and a `$ref` survives: either the type is recursive
  (which cannot be inlined, and which Anthropic rejects outright with
  `400 Circular reference detected` when the cycle runs through `$defs`) or
  `schema-inline` is off.

## [1.0.0-alpha.14] — 2026-08-01

### Added

- **`strict` on typed tools** — [strict tool use] can now be enabled
  declaratively. `ToolArgs` gains a `STRICT` const (default `false`) that
  `definition()` carries onto the `CustomMethodDef`; the `#[tool]` impl macro
  accepts `#[tool(strict)]` (every method) and `#[method(strict)]` /
  `#[method(strict = false)]` (per-method override), and
  `#[derive(ToolArgs)]` accepts `#[tool(strict)]`. Paired with alpha.13's
  declaration-order schemas, this guarantees the constrained decoder emits an
  args struct's fields in declaration order — declare a reasoning field first
  and the model must reason before it commits. `strict` is not compatible
  with `allowed_callers` (programmatic tool calling); the API rejects that
  combination.

  [strict tool use]: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use

## [1.0.0-alpha.13] — 2026-07-25

### Fixed

- **JSON Schema `properties` now reach the wire in declaration order**, not
  alphabetically — the behaviour this crate's docs and structured-output
  examples have claimed all along. `schemars` builds `properties` in
  declaration order, but `serde_json::Map` was `BTreeMap`-backed, so every
  schema serialized sorted.

  This was not cosmetic. A live probe (`live_generation_order`, in
  `src/prompt/output.rs`) measured Anthropic's decoders against a struct
  whose fields are strictly reverse-alphabetical, reading emission order off
  raw stream deltas. Across Haiku 4.5, Sonnet 4.6, Sonnet 5 and Opus 5,
  24/24 constrained samples — structured output, and tool use with `strict`
  — followed **`properties`** order, deterministically. They ignore
  `required`, which schemars already emitted in declaration order. So the
  decoder was faithfully enforcing the sorted order: `structured_commit_
  classifier` shipped `body, breaking, category, summary`, exactly inverting
  the reasoning-before-conclusion order it documents, and `vote_intent` put
  `confidence` ahead of `rationale`. `strict: true` made it *worse*, since it
  hard-locks the inversion instead of leaving the model free to follow the
  schema text. After this change the same probe reports declaration order
  24/24.

  New default-on `schema-order` feature forwards to
  `schemars/preserve_order`. `sanitize_for_anthropic` was reworked to suit:
  it rebuilds each object rather than calling `serde_json::Map::remove`,
  which under `preserve_order` is `swap_remove` and moves the last entry
  into the removed slot. (The order-preserving `shift_remove` is itself
  gated on that feature, so it cannot be called unconditionally.) The
  `oneOf` → `anyOf` rewrite now happens in place instead of relocating the
  key to the end. `ToMarkdown for CustomMethodDef` switched to `Map::retain`
  for the same reason; its rendered key order is now the struct's rather
  than alphabetical.

  `#[serde(flatten)]`ed fields are excluded from the guarantee — schemars
  merges those subschemas through its own map surgery.

### Breaking

- **`schema-order` (default-on) enables `serde_json/preserve_order` for your
  whole dependency graph.** Cargo features are additive and global: turning
  it on here switches `serde_json::Map` to an `IndexMap` backend for *every*
  crate in your build, not just this one. Three things to check before
  upgrading:

  1. `serde_json::Value` object keys are no longer emitted in sorted order,
     anywhere — JSON snapshot tests in your own code may need regenerating.
  2. `serde_json::Map::remove` becomes `swap_remove` in **your** code too. If
     you remove keys from a `Value` and care about the order of what
     remains, switch to `shift_remove` or `retain`.
  3. The serialized bytes of every structured-output schema and tool
     definition change, so prompt caches keyed on them miss once on upgrade.

  `default-features = false` drops *our* edge to the feature but cannot
  guarantee it stays off: any other crate in your graph that enables
  `serde_json/preserve_order` re-enables the `IndexMap` backend for
  everyone. Our schema code is order-correct under either backend, so
  toggling is always safe — it may simply not have the effect you intend.

### Added

- **`Id::Sonnet5` and `Id::Opus5`** (#132). `Model::supports_system_role`
  gains Opus 5 only, verified live. Sonnet 5 is excluded and the gate
  matters there: unlike Sonnet 4.6, which rejects a mid-conversation
  `role: "system"` turn with a clean 400, Sonnet 5 returns 200 and silently
  ignores it, answering the preceding user message instead. There is no wire
  signal to detect that.

## [1.0.0-alpha.12] — 2026-07-22

### Added

- **`Transport` for `Arc<T>`** — a forwarding impl, so a type-erased
  transport still satisfies `T: Transport` and can be handed to anything
  generic over one. `Transport` was already dyn-compatible per prompt type,
  but `dyn Transport<…>` is unsized, so
  `Arc<dyn Transport<Prompt, Error = E>>` did not itself implement the trait
  and `Chat::new` rejected it. The motivating case is N chat loops sharing
  one endpoint, a clone apiece. All methods forward, the defaulted ones
  included — inheriting the defaults would silently downgrade an
  implementor's `send_batch`, `quirks`, or `max_concurrency` on erasure.

  `Arc` only, and deliberately: it is not `#[fundamental]`, so the orphan
  rule already forbids a downstream `impl Transport for Arc<TheirType>` and
  this addition can conflict with nothing. The same impl over `Box<T>`
  **would** be breaking — `Box` is `#[fundamental]`, so downstream
  `impl Transport for Box<TheirType>` is legal today and a blanket impl
  would collide with it.

## [1.0.0-alpha.11] — 2026-07-17

Re-tag of the unpublished alpha.9/alpha.10 (tags are immutable, so each
failed release burns a version: alpha.9 lacked the derive path-dep version
requirement publishing needs; alpha.10's tracked lockfile was stale against
the bumped member versions, tripping the image build's `--locked`).

### Added

- **`Transport` — the prompt→message call shape** (#126). Trait-level prompt
  generic (`Transport<P = Prompt>`, dyn-compatible per prompt type) with
  `send`, an order-preserving `send_batch` default bounded by
  `max_concurrency`, `models()`, and `quirks()`. `Client` implements it for
  `Prompt` and `CachedPrompt`. `Quirks` moves in from agentkit — endpoint
  behavior as data, `Default` is canonical Anthropic.
- **`Chat` promoted from the examples into the crate** (#104), behind the new
  `chat` feature — transport-generic, tokio-free (`futures::select!`), and
  independent of `client`. New opt-in quirk-aware cache placement
  (`Chat::cache`): canonical endpoints get `auto_cache` semantics,
  `breakpoint_after_assistant` transports a budget-aware rolling window
  re-marked per assistant turn, marker-ignoring endpoints nothing.
  Prompt-only for now — `CachedPrompt` genericity stays open on #104.
- **`response::Message::builder` + `TokenCounts::new`** (#134). The
  construction path for inference providers that synthesize responses rather
  than deserialize them; field-for-field equivalent to the deserialize path.
- **`Prompt::cache_windowed{,_1h,_with}`** — the budget-aware rolling
  breakpoint window, promoted from `CachedPrompt` (which now delegates).

## [1.0.0-alpha.5] — 2026-06-30

### Added

- **Client-side `tool_use`/`tool_result` adjacency validation** (#102).
  `Prompt` turn-order validation now models two more wire rules as constructive
  [`TurnOrderError`]s instead of deferring to a server 400:
  - `ToolResultNotLeading` — a turn's `tool_result` blocks must form a leading
    run (`[tool_result, text]` is accepted; `[text, tool_result]` is a 400).
  - `UnansweredToolUse` — every client `tool_use` must be answered by a matching
    leading `tool_result` in the immediately following user turn; the error
    names the unanswered ids. `server_tool_use` is excluded — the API answers
    those itself.

  [`TurnOrderError`]: https://docs.rs/misanthropic/latest/misanthropic/prompt/enum.TurnOrderError.html

### Breaking

- **`prompt::TurnOrderError` is `#[non_exhaustive]`.** Downstream `match` on it
  now needs a `_` arm. The wire turn-order grammar keeps growing (and shrinking
  — Anthropic relaxes rules too), so adding a variant must stay non-breaking
  (#102).

### Fixed

- **`bashd` release image now builds.** `Cargo.lock` was still excluded by
  `.dockerignore`, so the `--locked` build introduced in alpha.4 could not find
  the lockfile — the image build, and with it the whole release, failed.
  Un-ignore `Cargo.lock` so the release image builds reproducibly.

## [1.0.0-alpha.4] — 2026-06-30

> Tagged but never published: the `bashd` image build failed on the
> `.dockerignore` issue fixed in alpha.5, so `publish-crates` never ran. These
> changes ship in alpha.5.

### Added

- **`ModelInfo::satisfies`** for model/capability negotiation — compares a
  required `Model`/capability against an available one, ids compared
  `Model`-to-`Model` (#109).

### Breaking

- **`Model::name()` / `Id::name()` always return the canonical wire id.**
  `Id::name()` previously returned a short *display* form (`"opus-4.8"`); it now
  returns the wire id (`"claude-opus-4-8"`), identical to `Model::name()` and to
  the variant's `serde` rename. The short display form is removed — a
  human-readable label is the API's concern and lives on
  `ModelInfo::display_name`. This also fixes `Model`'s `PartialEq<Id>` /
  `PartialEq<&str>` impls, which compared against the display form and so
  returned `false` for a model's own wire id (#109).
- **Inbound wire structs are `#[non_exhaustive]`.** `response::Message`,
  `Container`, `StopDetails`, `Usage`, `TokenCounts`, `OutputTokensDetails`,
  `CacheCreation`, and `ServerToolUsage` can no longer be built with a struct
  literal or matched exhaustively by downstreams. Construct via
  `Default::default()` + field assignment (all fields are public); future wire
  fields are now non-breaking additions (#105).
- **`prompt::message::Block` (the enum) is `#[non_exhaustive]`.** Downstream
  `match` on a `Block` now needs a `_` arm, so future API-added variants (the
  wire grows these every few months) are non-breaking. The variants themselves
  are *not* sealed — `Block::Text { … }` literals, `Into<Block>` / `Into<Content>`,
  and `(Role, T)` construction are all unaffected (#105).
- **`tool::CustomMethodDef` is `#[non_exhaustive]`.** Build it via the `#[tool]`
  macro, `CustomMethodDef::builder()` / `MethodBuilder`, or
  `CustomMethodDef::simple()` — not a struct literal. This makes future tool
  fields non-breaking (it already grew `strict`, `defer_loading`,
  `allowed_callers`). No `Default` is derived, deliberately: an empty-schema
  default is an invalid tool (#106).

### Documentation

- Add this `CHANGELOG.md` (#108).
- `misan-messages-api` skill: correct the `Usage` response tree for the
  `Usage` → `TokenCounts` split (counters live on `usage.counts`; hold a
  `TokenCounts` for accumulation), and steer manual tool construction to the
  builder (#107, #106).

## Pre-1.0 breaking changes (through 1.0.0-alpha.3)

Reconstructed from the conventional-commit `!` history; first captured during
the downstream `agora` migration. These landed across `1.0.0-alpha.1` →
`1.0.0-alpha.3`.

### Breaking

- **Lifetimes removed from public types.** Drop `<'static>` / `<'_>` parameters
  and `.into_static()` calls.
- **`tool::Method` → `tool::CustomMethodDef`.** The hand-written schema struct
  was renamed; **`tool::Method` now names the typed-tool trait.** An old
  `use …tool::Method` keeps compiling but silently re-resolves to the trait,
  producing `expected struct, found trait` errors far from the import.
- **`tool::Choice::{Auto, Any}` are now struct variants** carrying
  `disable_parallel_tool_use`. Use `Choice::auto()` / `Choice::any()` to
  construct and match `{ .. }`.
- **`Content` is now `Content(Vec<Block>)`** — the `MultiPart` / `SinglePart`
  split is gone. `Block::Text` gained a `citations` field.
- **`Prompt.functions` → `Prompt.tools`** (`Vec<MethodDef>`).
- **`Usage` split into `Usage` + `TokenCounts`.** The `Copy` counters moved to
  `usage.counts` (a `TokenCounts`); `Usage` gained `service_tier` /
  `inference_geo`. Reads still work through `Usage`'s `Deref`.
- **Field additions on inbound types:** `response::Message` gained `kind` /
  `stop_details` / `container`; `AnthropicError::{RateLimit, Overloaded}` gained
  `retry_after` (with a `retry_after()` → `Duration` accessor); `tool::Use`
  gained `caller`.
- **`Client::with_base_url` → `Client::base_url`.**
- **The `json-schema` feature was removed** (always on now).
