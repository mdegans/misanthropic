# Task runner for misanthropic. Install `just`: https://github.com/casey/just
# Run `just install-hooks` once per clone to enable the pre-commit gate.

# List available recipes.
default:
    @just --list

# Format the whole workspace in place.
fmt:
    cargo fmt --all

# Check formatting without writing (mirrors the first step of `test`).
fmt-check:
    cargo fmt --all -- --check

# Offline gate run by the pre-commit hook: fmt, clippy, all-features + no-default tests.
test:
    cargo fmt --all -- --check
    cargo clippy --all-features --all-targets
    cargo test --all-features
    cargo test --all-features --no-default-features

# Live-API #[ignore]d tests (needs misanthropic/api.key); not in the pre-commit hook.
test-ignored:
    cargo test -p misanthropic --all-features -- --ignored

# Cross-build the static linux-musl `bashd` into target-linux/ (via a container, no host musl toolchain needed).
build-bashd:
    docker run --rm -v "$PWD":/w -w /w -e CARGO_TARGET_DIR=/w/target-linux \
        rust:alpine sh -c 'apk add --no-cache musl-dev && cargo build -p bashd --release'
    @echo "built: target-linux/release/bashd  — use it via BASHD_PATH=$PWD/target-linux/release/bashd"

# Enable the pre-commit gate by pointing git at hooks/ (run once per clone).
install-hooks:
    git config core.hooksPath hooks
    @echo "Installed: core.hooksPath -> hooks/ (bypass a commit with --no-verify)"
