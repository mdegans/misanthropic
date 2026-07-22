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

## [1.0.0-alpha.12] — 2026-07-22

### Added

- **`Transport` for `Arc<T>` and `Box<T>`** — forwarding impls, so a
  type-erased transport still satisfies `T: Transport` and can be handed to
  anything generic over one. `Transport` was already dyn-compatible per
  prompt type, but `dyn Transport<…>` is unsized, so
  `Arc<dyn Transport<Prompt, Error = E>>` did not itself implement the trait
  and `Chat::new` rejected it. `Arc` is the load-bearing case: N chat loops
  sharing one endpoint, a clone apiece. All methods forward, the defaulted
  ones included — inheriting the defaults would silently downgrade an
  implementor's `send_batch`, `quirks`, or `max_concurrency` on erasure.

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
