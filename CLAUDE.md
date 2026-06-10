# CLAUDE.md

Unofficial, ergonomic, async Rust client for the Anthropic Messages API.

## How we work

This is a collaboration, not a request queue. The best results here come from
designing together, so:

- **Chat before you plan; chat before you code.** When a task is open-ended —
  especially a new feature or an API-shape decision — start an *informal,
  two-directional* conversation. Say what you'd do and why, ask Michael what he
  would do given what we're trying to achieve, and bounce it back and forth.
  Surface the tension and your leanings in prose so he can add wider context.
  Do **not** jump straight to a plan, and do not reduce a design discussion to
  multiple choice.
- **`AskUserQuestion` is not for design.** Only use it *after a plan has been
  accepted* (it pings Michael's phone — that's its purpose). It is one-
  directional and a poor substitute for talking. Open design questions belong
  in the conversation, as questions, not as a menu.
- **Design the example first.** For a new feature, write the example you *wish*
  worked — the ideal, ergonomic API as a `Strawberry`-style program — and then
  build the code to make it real. Most of this crate's ergonomics and its many
  `From`/`Into` conversions came from exactly that: write the call site you
  want, then make it compile. Propose this up front rather than building
  bottom-up and fitting an example on at the end.

## Project structure

Cargo workspace with 4 members:

- `misanthropic/` — Main library crate (the API client)
- `chat/backend/` — Shuttle.rs + Axum web backend demo
- `chat/frontend/` — Dioxus web frontend demo
- `chat/model/` — Shared models for the chat demo

## Build & test

Prefer the `just` recipes — they mirror CI and are the source of truth for the
local gate. Run `just install-hooks` once per clone to enable the pre-commit
gate (`hooks/pre-commit` runs `just test` via `core.hooksPath`).

```sh
just                # list recipes
just test           # offline gate: fmt, clippy, all-features + no-default tests
just test-ignored   # live-API #[ignore]d tests (needs misanthropic/api.key)
just install-hooks  # enable the pre-commit gate (once per clone)
```

`just test` includes the `__skills` doc-tests, which compile the code blocks
in `.claude/skills/*/SKILL.md` so the skill docs can't drift from the API. It
also runs `just doc` (`RUSTDOCFLAGS="-D warnings" cargo doc … --no-deps
--examples`), so a broken intra-doc link or any rustdoc warning fails the gate —
keep doc links resolving (link to public items; for a private item use a plain
`` `code span` `` rather than `[`a link`]`). The gate is offline (free per
commit); only `just test-ignored` hits the API.

For the exact commands each recipe runs, read the `justfile` — it's the single
source of truth, so they aren't transcribed here where they'd drift. CI also
tests notable features individually; the authoritative set is the `[features]`
table in `misanthropic/Cargo.toml`.

## Code style

- Max line width: 80 characters (`rustfmt.toml`)
- `unsafe` code is forbidden (`#[forbid(unsafe_code)]`)
- Uses `rustls` by default (not OpenSSL)
- API keys are zeroized on drop; optionally encrypted in memory (`memsecurity`
  feature)
- Warnings are treated as errors

### Readability above all

Code should read like Python. Prefer expressive generics and trait bounds on
`Client` methods (e.g. `impl Into<Cow<'a, str>>`, `TryInto<Key>`) so that
*call sites* stay clean, even when signatures get verbose.

### Functional over imperative

Prefer iterator chains (`.map()`, `.filter()`, `.collect()`) and pattern
matching over mutable loops and temporary variables in library code. Examples
and tests can be more imperative when it aids clarity.

### Documentation style

Doc comments are terse — often a single sentence explaining where the item fits
relative to other types, linked with `[Type]` / `[Self::method]` /
`[crate::path]` syntax. Avoid restating signatures; focus on relationships and
intent. Module-level docs (`//!`) give a brief overview and point the reader to
key entry-point types.

### Naming choices

The crate deliberately picks domain-friendly names over HTTP jargon: `Prompt`
instead of `Request`, `Method` for individual tool function descriptors (to
avoid collision with the `Tool` trait). Enum variants are short and
self-describing (`StopReason::EndTurn`).

### Patterns in use

- **Builder-style fluent APIs** — `Prompt::default().add_tool(…).system(…)`
- **Borrowed-by-default with `into_static()`** — most public types carry a
  lifetime `'a`; call `.into_static()` when ownership is needed.
- **`From` / `Into` blanket conversions** — e.g. `(Role, &str) -> Message`,
  keeping construction ergonomic.
- **Feature-gated modules** — heavy or platform-specific deps hide behind
  Cargo features; `#[cfg(feature = "…")]` guards throughout.
- **Privacy-aware `Debug`** — `Prompt`'s `Debug` impl hides chat messages.
- **Cold-path hints** — error branches call `cold_path()` to guide the
  optimizer.
- **Conditional logging** — all `log::*` calls sit inside
  `#[cfg(feature = "log")]` blocks.
- **`thiserror` enums with a crate-level `Result<T>` alias.**

## Key features to know about

Default features: `rustls-tls`, `langsan`, `client`, `batch`.

Notable optional features: `prompt-caching`, `markdown`, `html`, `memsecurity`,
`dioxus`, `notepad`, `cot`.

The default tool suite should stay minimal.

The `batch` and `client` features don't build on wasm32.

## Testing notes

- Some tests are `#[ignore]`d and require an API key in `api.key` in the
  `misanthropic/` crate directory — i.e. `misanthropic/api.key`, which is the
  `CRATE_ROOT` that `load_api_key` reads, not the workspace root (CI provides
  this via secrets on push to main). Run them with e.g. `cargo test -p
  misanthropic --features client <name> -- --ignored`.

### Wire fixtures — capture, don't trust the docs

**Anthropic's API docs are guidelines, not rules.** They drift from the wire
repeatedly, and trusting them has cost real debugging time: the undocumented
`caller` field on result blocks, `web_fetch_tool_result_error` vs the documented
`web_fetch_tool_error`, `page_age` sent as explicit `null`, a no-citations fetch
that omits `citations`, `tool_search_requests` absent from the wire entirely
(#72). So:

- **When adding any feature, capture the real wire shape first — for *both* the
  non-streaming (`messages`) and streaming (SSE) paths — before writing types.**
  Capture on Haiku 4.5 (cheapest; server-tool shapes are model-independent) by
  `curl`ing the API directly (raw bytes — deserializing through our own types
  hides dropped fields). Then build types to match, not the docs.
- Save captures under `misanthropic/test/data/` (see its `README.md` for the
  discipline + per-fixture provenance) and replay them offline:
  - non-streaming: `crate::utils::roundtrip::<T>(include_str!(…))` — deserialize,
    re-serialize, assert value-equal to the captured bytes.
  - streaming: `stream::tests::mock_stream` (raw SSE) / `mock_stream_jsonl`
    (one `Event` per line).
- The round-trip assertion is load-bearing: **no response type uses
  `#[serde(deny_unknown_fields)]`**, so an undocumented/renamed/mis-tagged field
  is silently dropped on deserialize. Re-serializing and comparing to the
  captured bytes is the only offline guard that fails loudly when the wire
  drifts. Prefer a known/unknown `untagged` enum (à la `model::Model`,
  `tool::Caller`) for API-sourced unions so a future variant round-trips instead
  of failing to deserialize a live response.

## GitHub conventions

- When Claude files an issue or opens a PR on Michael's behalf (via `gh`),
  attribute it — even when Michael explicitly asked for it. This is about
  Claude getting credit for the work it does on the project, not only about
  avoiding misattribution to Michael. Commits already carry a
  `Co-Authored-By: Claude …` trailer; issues and PRs should close with a
  footer line crediting Claude Code (e.g. `🤖 Filed by Claude Code` on issues,
  the standard `🤖 Generated with Claude Code` footer on PRs).
