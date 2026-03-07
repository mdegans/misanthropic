# CLAUDE.md

Unofficial, ergonomic, async Rust client for the Anthropic Messages API.

## Project structure

Cargo workspace with 4 members:

- `misanthropic/` — Main library crate (the API client)
- `chat/backend/` — Shuttle.rs + Axum web backend demo
- `chat/frontend/` — Dioxus web frontend demo
- `chat/model/` — Shared models for the chat demo

## Build & test

```sh
# Build everything
cargo build --all-features

# Format check
cargo fmt --all -- --check

# Lint
cargo clippy --all-features

# Run all tests
cargo test --all-features

# Run tests without default features
cargo test --all-features --no-default-features

# Test individual features (CI tests each one separately)
cargo test --features <feature> --verbose
# Features: image, jpeg, png, gif, webp, prompt-caching, log, markdown,
#           partial-eq, langsan, memsecurity
```

## Code style

- Max line width: 80 characters (`rustfmt.toml`)
- `unsafe` code is forbidden (`#[forbid(unsafe_code)]`)
- Uses `rustls` by default (not OpenSSL)
- API keys are zeroized on drop; optionally encrypted in memory (`memsecurity`
  feature)

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

- **Builder-style fluent APIs** — `Prompt::default().add_tool(…).set_system(…)`
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

Default features: `rustls-tls`, `langsan`, `rate-limiting`, `client`, `batch`.

Notable optional features: `prompt-caching`, `markdown`, `html`, `memsecurity`,
`dioxus`, `memory-palace` (requires PostgreSQL), `notepad`, `cot`.

`memory-palace` is slated for removal and will move to a separate crate. Its
PostgreSQL/sqlx dependencies are too heavy for the core library. The default
tool suite should stay minimal.

The `batch` and `client` features don't build on wasm32.

## Testing notes

- Some tests are `#[ignore]`d and require an API key in `api.key` at the repo
  root (CI provides this via secrets on push to main).
- The `memory-palace` feature tests require a PostgreSQL instance. CI runs
  Postgres 17 on `localhost:5432` (see `.github/workflows/tests.yaml`).
