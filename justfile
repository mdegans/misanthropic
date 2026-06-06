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

# The sandbox image tag the DockerSandbox boots by default. Must match
# DEFAULT_IMAGE in misanthropic/src/tool/bash/docker.rs.
bashd_image := "misan-bashd:dev"

# Build the sandbox image (bashd baked into an immutable rootfs) and extract the
# static linux-musl binary to target-linux/ for the dev bind-mount path + live
# tests. See Dockerfile for the two-stage build.
build-bashd:
    docker build -t {{bashd_image}} -f Dockerfile .
    mkdir -p target-linux/release
    id=$(docker create {{bashd_image}}); \
        docker cp "$id:/usr/local/bin/bashd" target-linux/release/bashd; \
        docker rm "$id" >/dev/null
    @echo "built image {{bashd_image}}  +  target-linux/release/bashd"

# Enable the pre-commit gate by pointing git at hooks/ (run once per clone).
install-hooks:
    git config core.hooksPath hooks
    @echo "Installed: core.hooksPath -> hooks/ (bypass a commit with --no-verify)"
