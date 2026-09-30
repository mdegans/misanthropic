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

- **Turn order rejects abandoning an in-flight server tool** — new
  `TurnOrderError::UnfinishedServerToolUse`. A `server_tool_use` no result in
  its turn answers (a `pause_turn` turn) admits only an assistant
  continuation; a user or system turn after it was accepted client-side and
  400'd on the wire (live-probed). `check_turn_order`, `push_message` and
  `Prompt::seat` now refuse it (a system note buffers instead). A
  programmatic call's container awaiting client `tool_result`s is not in
  flight, and a system turn after a result still needs that result to answer
  every use in the turn.
- **`Chat::run` returns `chat::Error<State>`, handing the prompt and state
  back** (was a bare `BoxError`, which dropped both). `chat::Error { kind,
  prompt, state }` implements `std::error::Error`, so `?` into a `BoxError`
  still works; `kind` is the new `chat::Stop`: `Clipped` and `Unusable` (see
  *Fixed*), or `Transport` / `Beat` / `Tool` / `TurnOrder` wrapping what used
  to be the bare error. The returned prompt is legal to resend, and a `Chat`
  seeded with a prompt that awaits the model (ending in a user or system
  turn, or a paused one) now **answers it before asking for a beat** — so
  resuming is a loop:

  ```rust
  match chat.run((), &mut next_beat).await {
      Ok((prompt, ())) => { /* … */ }
      Err(chat::Error { kind: Stop::Clipped(_), prompt, state }) => {
          // Nothing was seated or run: raise max_tokens, resume.
      }
      Err(error) => return Err(error.into()),
  }
  ```

### Added

- **`blallama` feature and `just test-blallama <model>`: live `Chat`
  scenarios against a local Anthropic-compatible server** (drama_llama's
  `blallama`, on `localhost:11435`). Local runs only: the tests skip unless
  `BLALLAMA_URL` is set, with `BLALLAMA_MODEL` naming the model, which keeps
  them out of CI's `--all-features` builds. They run the scenario table's
  live-able rows — a plain turn, stop sequences, a clip mid-call, a forced
  tool call through a `FinalWord` wrap-up, system notes, a notification —
  and hold every response to Anthropic's shape, so a server deviation
  surfaces as a failure. The offline table (53 rows over `MockTransport`)
  asserts the requests, turn shapes, calls run, usage and wire legality of
  every request and hand-back.

- **`Message::unfinished_server_tool_uses()`**, **`Block::server_tool_result_id()`**
  and **`Caller::tool_id()`** — the pieces of the rule above: which server
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
  trailing one, and only on `stop_reason: tool_use`).

### Changed

- **`BudgetPolicy::FinalWord` asks for words.** The wrap-up call now goes out
  with `tool_choice: none` (the prompt's own `tool_choice` is restored after)
  instead of letting the model call tools only to answer them with synthetic
  errors. A transport with `Quirks::tool_choice_not_respected` gets the prompt
  unchanged, and calls a wrap-up makes anyway are still errored.

### Fixed

- **`Chat` no longer dispatches tool calls from a `max_tokens`-clipped turn**
  (#124). A clipped turn's `tool_use` can be valid JSON missing arguments the
  model never emitted; `Chat` seated it and ran the calls anyway. The loop now
  matches on `Disposition`: a `Clipped` turn (the `BudgetPolicy::FinalWord`
  wrap-up included) is never seated or dispatched, and `run` hands the
  un-advanced prompt back as `Stop::Clipped` — raising `max_tokens`,
  nudging the model, or giving up is the caller's policy.
- **`Chat` runs client tool calls only from a `tool_use` turn** (or a paused
  one). It dispatched by content, whatever the stop reason — but a
  `refusal` can cut a `tool_use` short. A finished (`Done`) turn that still
  calls client tools is now `Stop::Unusable`: none run, and the whole turn is
  dropped (with any paused turn it continued), never stripped — stripping
  could strand a `server_tool_use`. Calls an `on_assistant` hook seats still
  run.
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
- **An `on_assistant` return that breaks turn order is seated whole or not at
  all.** A hook returning, say, a `tool_use` turn followed by a user turn got
  `Stop::TurnOrder` with the `tool_use` turn already seated and unanswered —
  a hand-back no beat could legally follow. The seating now rolls back.

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
