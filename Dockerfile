# Two-stage build for the misanthropic bash sandbox image.
#
# Stage 1 builds the static linux-musl `bashd` daemon; stage 2 is the runtime
# image the `DockerSandbox` boots: an immutable rootfs with `bashd` baked in, a
# pinned non-root `agent` user, and login files so `bash -lc` picks up
# `~/.bashrc`. The host runs it `--read-only` with a writable tmpfs `/tmp` and a
# `$HOME` volume (see misanthropic::tool::bash::docker).
#
# Build + extract the binary with `just build-bashd`. Custom images should be
# built `FROM` this one so they keep an immutable rootfs at runtime.

# --- stage 1: build the daemon -------------------------------------------------
# rust:alpine is musl-hosted, so a plain release build is already static-musl.
# build-base + linux-headers are for bashd's rustls backend (aws-lc-rs): it
# compiles libcrypto from source and #includes <linux/random.h>.
FROM rust:alpine AS builder
RUN apk add --no-cache build-base linux-headers
WORKDIR /src
COPY . .
RUN cargo build -p bashd --release

# --- stage 2: the runtime sandbox image ----------------------------------------
FROM alpine:3
# bash is the shell bashd drives (`bash -lc`); coreutils for GNU userland.
RUN apk add --no-cache bash coreutils ca-certificates \
    && adduser -D -u 1000 agent \
    && mkdir -p /home/agent \
    && printf '%s\n' \
        '# Login shells source ~/.bashrc so interactive env setup applies.' \
        '[ -f ~/.bashrc ] && . ~/.bashrc' > /home/agent/.profile \
    && touch /home/agent/.bashrc \
    && chown -R agent:agent /home/agent
# Baked on the read-only rootfs; the host bind-mounts a dev build over this when
# `DockerSandbox::bashd_path` is set.
COPY --from=builder --chmod=0755 /src/target/release/bashd /usr/local/bin/bashd
