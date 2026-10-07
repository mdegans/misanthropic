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

# Offline gate run by the pre-commit hook: fmt, clippy, doc, all-features +
# no-default tests.
test: && test-no-default
    cargo fmt --all -- --check
    cargo clippy --all-features --all-targets
    RUSTDOCFLAGS="-D warnings" cargo doc -p misanthropic --all-features --no-deps --examples
    cargo test --all-features

# The crate with its default features off (`--all-features` would override
# `--no-default-features`, so they're never combined). The lib-only clippy is
# the featureless build; test builds also get the self dev-dependency's `log`
# and `chat`. Examples needing more are skipped by their required-features.
# `json-schema` again without defaults: the one build where its impls meet
# `CowStr` as a plain `Cow` (no `langsan`) and no `batch` types.
test-no-default:
    cargo clippy -p misanthropic --no-default-features
    cargo test -p misanthropic --no-default-features
    cargo clippy -p misanthropic --no-default-features --features json-schema
    cargo test -p misanthropic --no-default-features --features json-schema

# Build the docs with broken intra-doc links (and any rustdoc warning) treated as
# errors — the doc half of the gate. Covers the lib (incl. the `__skills` skill
# files) and the examples. `--no-deps` so only our own docs are checked.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc -p misanthropic --all-features --no-deps --examples

# Live-API #[ignore]d tests (needs misanthropic/api.key); not in the pre-commit hook.
test-ignored:
    cargo test -p misanthropic --all-features -- --ignored

# The local blallama (drama_llama) the live recipes drive: `BLALLAMA_URL`, or
# port 11436 (VS Code's port forwarding can squat blallama's own 11435).
blallama_url := env_var_or_default("BLALLAMA_URL", "http://127.0.0.1:11436")

# `model` is a file name the server lists (`curl $BLALLAMA_URL/v1/models`).
# Single-threaded: one model, one set of weights. Never run in CI.
# Live Chat scenarios against a local blallama (drama_llama), one model.
test-blallama model="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf":
    BLALLAMA_URL='{{blallama_url}}' BLALLAMA_MODEL='{{model}}' \
        cargo test -p misanthropic --features blallama --lib \
        chat::scenarios::live -- --test-threads=1

# Runs a greedy multi-turn conversation with cache reuse, then replays each
# request cold (after evicting every prefix-cache slot, so don't run it beside
# other cache-sensitive work) and fails unless every reply matches its warm
# twin byte for byte, or a control explains the mismatch. A request that read
# back the previous turn's generated tokens (the tip) gets the matched-schedule
# control: request k-1 replayed cold must match warm k-1, then request k, sent
# with no flush on that fresh tip, must match warm k. Both match: tip schedule
# (explained), passed with a note; only k-1 matches: cache failure. k-1 doesn't:
# upstream nondeterminism, which fails when cold disagrees with itself and
# `slots` is 1, and warns otherwise. Any other request is replayed cold again:
# the replies agree (cache suspect) and it fails, or they don't
# (nondeterminism) and it warns. A tip longer than the previous turn's output
# (plus a stop sequence's slack) fails. Start blallama with `--no-penalty
# --cache-slots 1`, and pass `slots` if you didn't: a repetition penalty
# resumes warm but is rebuilt cold, and other sequences in the unified KV
# cache change logits, so either can split the replies with no KV corruption,
# and the server reports neither. The test also skips unless
# BLALLAMA_EQUIVALENCE=1, which only this recipe sets, so an exported
# BLALLAMA_URL never lets the pre-commit gate evict the slots. Never run in CI.
# Live warm-vs-cold KV-cache equivalence check against a local blallama.
test-equivalence model="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf" slots="1":
    BLALLAMA_URL='{{blallama_url}}' BLALLAMA_MODEL='{{model}}' \
        BLALLAMA_EQUIVALENCE=1 BLALLAMA_CACHE_SLOTS='{{slots}}' \
        cargo test -p misanthropic --features blallama --lib \
        chat::scenarios::equivalence::blallama -- --test-threads=1 \
        --nocapture

# Prints a table of each request's input / written / read / tip tokens and
# latency, and fails when the prefix isn't reused turn to turn, or a turn takes
# a re-prefill's time anyway. Never run in CI. A run that finds the server warm
# (no request prefills 1024 tokens to measure the rate on) is timed against
# BLALLAMA_PREFILL_RATE tokens/s instead: 420 by default, measured on Qwen3.6.
# `which` picks one run: canonical, after_assistant or long (the default runs
# all three; long needs a context of about 32k: set BLALLAMA_N_CTX to the
# server's --n-ctx if it isn't 32768, and every request must fit it).
# Live multi-turn prompt-caching check against a local blallama, one model.
test-cache model="Qwen3.6-35B-A3B-UD-Q4_K_S.gguf" which="":
    BLALLAMA_URL='{{blallama_url}}' BLALLAMA_MODEL='{{model}}' \
        cargo test -p misanthropic --features blallama --lib \
        chat::scenarios::cache::blallama::{{which}} -- --test-threads=1 \
        --nocapture

# The same caching check against Anthropic, for reference numbers: `which` is
# canonical (about 3 cents) or long (about 8). The tests are #[ignore]d and
# also skip unless MISANTHROPIC_PAID_CACHE=1, which only this recipe sets, so
# CI's live gate (every ignored test) never pays for them.
# PAID: claude-haiku-4-5, via misanthropic/api.key.
test-cache-anthropic which="canonical":
    MISANTHROPIC_PAID_CACHE=1 cargo test -p misanthropic --all-features \
        --lib chat::scenarios::cache::anthropic::{{which}} -- --ignored \
        --nocapture

# Run an example with every feature on (so logging and each example's tools are
# available). Extra args pass through to the example, and `RUST_LOG` works, e.g.
# `RUST_LOG=debug just run-example web_search "what did Anthropic announce?"`.
run-example name *args:
    cargo run -p misanthropic --all-features --example {{name}} -- {{args}}

# The sandbox image the DockerSandbox boots by default (DEFAULT_IMAGE in
# misanthropic/src/tool/bash/docker.rs): the published image tagged with the
# workspace version, both single-sourced from the root Cargo.toml. Building
# locally shadows the published tag for development.
version := `sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1`
bashd_image := "mdegans/misan-bashd:" + version

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
