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
- API keys are zeroized on drop; optionally encrypted in memory (`memsecurity` feature)

## Key features to know about

Default features: `rustls-tls`, `langsan`, `rate-limiting`, `client`, `batch`.

Notable optional features: `prompt-caching`, `markdown`, `html`, `memsecurity`,
`dioxus`, `memory-palace` (requires PostgreSQL), `notepad`, `cot`.

The `batch` and `client` features don't build on wasm32.

## Testing notes

- Some tests are `#[ignore]`d and require an API key in `api.key` at the repo
  root (CI provides this via secrets on push to main).
- The `memory-palace` feature tests require a PostgreSQL instance. CI runs
  Postgres 17 on `localhost:5432` (see `.github/workflows/tests.yaml`).
