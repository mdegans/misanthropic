//! Client-side execution of the [bash tool] ([`ServerMethodDef::Bash`]).
//!
//! The bash tool is *predefined* (you add it by versioned name via
//! [`Bash::latest`], or as a richer custom def via [`Bash::rich`]) but
//! *client-executed*: the model emits an ordinary [`Use`] (`name: "bash"`) whose
//! [`input`](Use::input) is a [`Command`], and *you* run it — in a **sandbox**,
//! not a filesystem jail. Because `docker exec` per command loses the working
//! directory and environment, the sandbox runs a tiny **`bashd`** daemon inside
//! the container that owns a persistent session and serves the HTTP/SSE protocol
//! in this module, reached over a published `127.0.0.1` port.
//!
//! This module is sandbox-agnostic: it provides the typed [`Command`] vocabulary,
//! the wire protocol the daemon and host share, the [`BashSandbox`] trait, and
//! the [`BashTool`] adapter that drops into a [`ToolBox`](super::ToolBox). Enable
//! `bash-container` for the reference `DockerSandbox` executor.
//!
//! Like [`memory`](super::memory)/[`text_editor`](super::text_editor), it
//! *defines* like a server tool and *executes* like a custom one.
//!
//! [bash tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool>
//! [`ServerMethodDef::Bash`]: crate::tool::ServerMethodDef::Bash
//! [`Bash::latest`]: crate::tool::Bash::latest
//! [`Bash::rich`]: crate::tool::Bash::rich
//! [`Use`]: crate::tool::Use

use std::borrow::Cow;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

use super::{MethodDef, Tool, Use};

/// The reference [`BashSandbox`] backed by a Docker/Podman container (the
/// `bash-container` feature).
#[cfg(all(feature = "bash-container", not(target_arch = "wasm32")))]
pub mod docker;
#[cfg(all(feature = "bash-container", not(target_arch = "wasm32")))]
pub use docker::{DockerSandbox, HomeFs, Network};
/// Re-exported for [`DockerSandbox::home_id`].
#[cfg(all(feature = "bash-container", not(target_arch = "wasm32")))]
pub use uuid::Uuid;

/// Ephemeral per-container PKI for the [`DockerSandbox`] ↔ `bashd` mTLS channel.
#[cfg(all(feature = "bash-container", not(target_arch = "wasm32")))]
mod pki;

/// The `bashd` wire-protocol version. Bumped on a breaking change; the host
/// refuses a daemon whose [`Ready::protocol`] does not match.
pub const PROTOCOL_VERSION: u32 = 1;

/// A typed bash command, deserialized from a bash [`Use`]'s [`input`](Use::input)
/// — and the JSON body the host `POST`s to `bashd`'s `/run`.
///
/// A known/unknown union (à la [`model::Model`]/[`Caller`]): commands this crate
/// types land in [`Known`]; anything else round-trips through
/// [`Unknown`](Command::Unknown) rather than failing to deserialize a live
/// response.
///
/// [`model::Model`]: crate::model::Model
/// [`Caller`]: crate::tool::Caller
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(untagged)]
pub enum Command {
    /// A command with typed support.
    Known(Known),
    /// An unrecognized command, kept verbatim so it round-trips.
    Unknown {
        /// The raw fields of the command.
        #[serde(flatten)]
        rest: serde_json::Map<String, serde_json::Value>,
    },
}

/// The bash commands this crate models. **Untagged**, disambiguated by a
/// distinct required key per variant: only [`Run`](Known::Run) carries
/// `command`, so a `{"restart": …}` / `{"poll": …}` / `{"kill": …}` payload
/// falls through to its arm. The single-key variants are declared *first* so a
/// `Run` never shadows them.
///
/// The predefined `bash_20250124` schema ([`Bash::latest`]) only ever elicits
/// `Run`/`Restart`; the derived [`Bash::rich`] schema additionally advertises
/// `background`/`timeout_secs` and `Poll`/`Kill`.
///
/// [`Bash::latest`]: crate::tool::Bash::latest
/// [`Bash::rich`]: crate::tool::Bash::rich
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(untagged)]
pub enum Known {
    /// Reset the session — a fresh shell in the default working directory.
    /// `{"restart": true}`.
    Restart {
        /// Always `true`; restart the bash session.
        restart: bool,
    },
    /// Check on a background job started with `background: true`, returning its
    /// output so far and whether it is still running. `{"poll": <job id>}`.
    Poll {
        /// The job id to poll.
        poll: u64,
    },
    /// Stop a background job. `{"kill": <job id>}`.
    Kill {
        /// The job id to kill.
        kill: u64,
    },
    /// Run a shell command in the persistent session.
    Run {
        /// The shell command to run.
        command: Cow<'static, str>,
        /// Run detached and return a job id immediately instead of blocking;
        /// poll it with `poll` and stop it with `kill`. (Rich def only.)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        background: Option<bool>,
        /// Kill the command (and report a timeout) if it runs longer than this
        /// many seconds. (Rich def only.)
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_secs: Option<u64>,
    },
}

impl TryFrom<serde_json::Value> for Command {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

// ---------------------------------------------------------------------------
// `bashd` reply framing — *our* protocol, not Anthropic's. It is `bashd`'s
// internal reply representation, mapped onto SSE events (see [`event`]) and
// background-job files. Designed clean, round-trip tested.
// ---------------------------------------------------------------------------

/// `bashd`'s reply representation: zero or more [`Chunk`](Reply::Chunk)s of
/// streamed output (tagged by [`Stream`]) followed by exactly one terminal
/// [`Outcome`](Reply::Outcome). [`Ready`](Reply::Ready) is the (bare) `GET /`
/// handshake.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(untagged)]
pub enum Reply {
    /// The startup handshake (no request id). Disambiguated by its `ready` key.
    Ready {
        /// Handshake payload.
        ready: Ready,
    },
    /// A chunk of streamed output for a running command. Disambiguated by its
    /// `stream`/`data` keys.
    Chunk(Chunk),
    /// The terminal result of a command (exit status, flags). Disambiguated by
    /// its `id` + outcome keys.
    Outcome(Outcome),
}

/// The `bashd` startup handshake — the first [`Reply`] line.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Ready {
    /// The daemon's [`PROTOCOL_VERSION`]; the host refuses a mismatch.
    pub protocol: u32,
    /// The daemon's own version string (`CARGO_PKG_VERSION`).
    pub bashd: Cow<'static, str>,
    /// The shell the daemon drives, e.g. `/bin/bash`.
    pub shell: String,
    /// Whether the daemon persists the working directory across commands.
    pub persist_cwd: bool,
}

/// Which stream a [`Chunk`] of output came from.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
#[serde(rename_all = "snake_case")]
pub enum Stream {
    /// Standard output.
    Stdout,
    /// Standard error.
    Stderr,
}

/// A chunk of a command's streamed output.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Chunk {
    /// The [`Request::id`] this output belongs to.
    pub id: u64,
    /// Which stream produced it (tagged at the source — the host merges for the
    /// model if it likes).
    pub stream: Stream,
    /// The (UTF-8, lossy) output bytes.
    pub data: String,
}

/// The terminal result of a [`Request`].
#[derive(Clone, Debug, Default, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Outcome {
    /// The [`Request::id`] this is the outcome of.
    pub id: u64,
    /// The command's exit code, or `None` while a backgrounded job still runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exit: Option<i32>,
    /// Whether the command is still running (a backgrounded job).
    pub running: bool,
    /// Whether the command was killed for exceeding its `timeout_secs`.
    pub timed_out: bool,
    /// Whether output was truncated at the daemon's hard byte cap.
    pub truncated: bool,
    /// The job id, for backgrounded commands / `poll` / `kill`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<u64>,
    /// Model-facing steering, e.g. a hint not to busy-loop a `poll`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub advice: Option<String>,
    /// On a `poll` response (`GET /jobs/{id}`): the new read cursor (byte offset
    /// into the job's output) the host should send on its next poll. `None`
    /// outside of poll responses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cursor: Option<u64>,
    /// An op-level failure (e.g. an unsupported command), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ProtocolError>,
}

/// A `bashd` op-level failure, carried on an [`Outcome`].
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct ProtocolError {
    /// The kind of failure.
    pub kind: ErrorKind,
    /// A human/model-readable message.
    pub message: String,
}

/// The kind of a [`ProtocolError`]. A known/unknown union so a newer daemon's
/// error kind round-trips instead of failing to deserialize.
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The requested operation is not implemented by this daemon yet (e.g.
    /// `background`/`poll`/`kill` in the Phase-1 daemon).
    Unsupported,
    /// No such background job for a `poll`/`kill`.
    NoSuchJob,
    /// The command could not be spawned.
    Spawn,
    /// Anything else, including kinds a newer daemon introduced.
    #[serde(other)]
    Other,
}

// ---------------------------------------------------------------------------
// HTTP/SSE transport (host <-> bashd). bashd serves these over a TCP port the
// host reaches via a published `127.0.0.1` mapping (see [`docker`]). Still *our*
// protocol — designed clean, round-trip tested.
//
// - `GET /`               → a bare [`Ready`] (no [`Reply`] wrapper): the
//                           readiness/handshake the host polls with backoff.
// - `POST /run` (Command) → an SSE stream: [`event::CHUNK`] events carrying a
//                           [`Chunk`], then a terminal [`event::OUTCOME`]
//                           carrying an [`Outcome`]. A backgrounded run instead
//                           emits a single [`event::OUTCOME`] (`running: true`,
//                           `job` set) and closes.
// - `GET /jobs/{id}`      → a `poll`: the buffered output since `?cursor=` then
//                           an [`Outcome`] whose [`cursor`](Outcome::cursor)
//                           advances the host's read position.
// - `GET /jobs/{id}/wait` → like a poll, but the stream *follows* the job to
//                           completion (soft `?timeout=` → terminal
//                           `running: true`).
// - `POST /jobs/{id}/kill`→ signal the job's group (TERM→grace→KILL).
// ---------------------------------------------------------------------------

/// SSE `event:` names on the streaming endpoints. The host matches on these to
/// decode each event's `data` as a [`Chunk`] or an [`Outcome`].
pub mod event {
    /// A [`Chunk`](super::Chunk) of streamed output.
    pub const CHUNK: &str = "chunk";
    /// The terminal [`Outcome`](super::Outcome).
    pub const OUTCOME: &str = "outcome";
}

/// The mutual-TLS material `bashd` reads from **stdin** at startup — never argv,
/// env, or disk. The host (`DockerSandbox`) generates an ephemeral per-container
/// PKI, feeds this server half over the launch pipe (then closes it), and keeps
/// the matching client identity to itself. Serialized as one JSON object.
///
/// Stdin is the deliberate channel: argv shows in `ps`/`docker inspect`, and env
/// is logged *and* inherited by the very shell commands `bashd` runs as its
/// children — either would hand the sandboxed payload the keys. The client
/// private key never appears here, only the server identity and the CA public
/// cert `bashd` checks the host's client cert against, so reading this grants at
/// most the ability to *impersonate* `bashd`, never to drive it. Zeroized on
/// drop on both ends.
#[derive(Serialize, Deserialize, Zeroize, ZeroizeOnDrop)]
pub struct TlsServerMaterial {
    /// PEM: the server leaf certificate followed by the issuing CA — the chain
    /// `bashd` presents to the host.
    pub cert_chain_pem: String,
    /// PEM: the server leaf's PKCS#8 private key.
    pub key_pem: String,
    /// PEM: the CA certificate `bashd` verifies the host's client cert against.
    pub ca_pem: String,
}

// ---------------------------------------------------------------------------
// Host-side abstraction: the sandbox trait, the aggregated result, the tool.
// ---------------------------------------------------------------------------

/// A host-side aggregate of one command's result: streamed [`Chunk`]s merged
/// into `stdout`/`stderr` plus the terminal [`Outcome`]'s flags. This is what a
/// [`BashSandbox`] hands back; [`BashTool`] renders it for the model.
#[derive(Clone, Debug, Default)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct ExecResult {
    /// Merged standard output.
    pub stdout: String,
    /// Merged standard error.
    pub stderr: String,
    /// Exit code, or `None` while a backgrounded job still runs.
    pub exit: Option<i32>,
    /// Whether the command is still running.
    pub running: bool,
    /// Whether the command timed out.
    pub timed_out: bool,
    /// Whether output was truncated at the daemon's cap.
    pub truncated: bool,
    /// The job id, for backgrounded commands.
    pub job: Option<u64>,
    /// Model-facing steering from the daemon.
    pub advice: Option<String>,
    /// On a `poll`/`wait` result: the read cursor to send on the next poll
    /// (host bookkeeping — not rendered to the model).
    pub cursor: Option<u64>,
}

impl ExecResult {
    /// Render this result as the single model-facing string a bash `tool_use`
    /// expects: stdout, then stderr, then any timeout/exit/advice notes.
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.stdout);
        if !self.stderr.is_empty() {
            if !out.is_empty() && !out.ends_with('\n') {
                out.push('\n');
            }
            out.push_str(&self.stderr);
        }
        if self.timed_out {
            push_note(&mut out, "command timed out");
        }
        match self.exit {
            Some(code) if code != 0 => {
                push_note(&mut out, &format!("exit code: {code}"))
            }
            _ => {}
        }
        if let Some(job) = self.job {
            push_note(&mut out, &format!("started background job {job}"));
        }
        if let Some(advice) = &self.advice {
            push_note(&mut out, advice);
        }
        out
    }
}

/// Append a parenthesized note on its own line.
fn push_note(out: &mut String, note: &str) {
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    out.push('(');
    out.push_str(note);
    out.push(')');
}

/// Why a bash sandbox operation failed (host-side). [`BashTool`] turns these
/// into an error [`tool::Result`](crate::tool::Result) for the model.
#[derive(Debug, thiserror::Error)]
pub enum BashError {
    /// An I/O error talking to the sandbox or daemon.
    #[error("bash sandbox I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The `bashd` handshake failed (version mismatch, no `Ready`, …).
    #[error("bashd handshake failed: {0}")]
    Handshake(String),
    /// The daemon returned a malformed or unexpected protocol message.
    #[error("bashd protocol error: {0}")]
    Protocol(String),
    /// A command was issued before the sandbox was [`start`](BashSandbox::start)ed.
    #[error("bash sandbox not started")]
    NotStarted,
    /// A backend-specific failure (e.g. a `docker` invocation).
    #[error("{0}")]
    Backend(String),
}

/// A place to run bash commands: a persistent session backed by some sandbox
/// (the reference one is `DockerSandbox`, behind `bash-container`).
///
/// The lifecycle methods are where **host-side** concerns live — notably the
/// home backup: [`start`](Self::start) restores it and [`teardown`](Self::teardown)
/// snapshots it, and because that is *here* (not in `bashd`), a daemon crash
/// can't lose the backup.
#[async_trait::async_trait]
pub trait BashSandbox: Send {
    /// Provision/launch the sandbox, inject and start `bashd`, restore any home
    /// backup, and complete the [`Ready`] handshake.
    async fn start(&mut self) -> Result<Ready, BashError>;

    /// Run one [`Command`] in the session, returning the aggregated result. A
    /// `Run` with `background: true` returns immediately with the job id set on
    /// [`ExecResult::job`].
    async fn exec(&mut self, command: Command)
    -> Result<ExecResult, BashError>;

    /// Poll a background job: new output since the host's cursor, plus
    /// running/exit status.
    async fn poll(&mut self, job: u64) -> Result<ExecResult, BashError>;

    /// Block until a background job finishes (or the soft `timeout` elapses,
    /// after which it returns the output so far with `running: true`).
    async fn wait(
        &mut self,
        job: u64,
        timeout: Option<Duration>,
    ) -> Result<ExecResult, BashError>;

    /// Signal a background job's process group (TERM→grace→KILL).
    async fn kill(&mut self, job: u64) -> Result<(), BashError>;

    /// Reset the session (a fresh shell). At this layer `restart` may *also*
    /// drop a borked home volume for a clean start.
    async fn restart(&mut self) -> Result<(), BashError>;

    /// Snapshot the home backup, then tear the sandbox down. Runs host-side, so
    /// it happens regardless of the daemon's state.
    async fn teardown(&mut self) -> Result<(), BashError>;
}

/// The [`Tool`] adapter: wraps a [`BashSandbox`] and drives it from the
/// conversation. `on_init` → [`start`](BashSandbox::start), `on_teardown` →
/// [`teardown`](BashSandbox::teardown), and each bash `tool_use` →
/// [`exec`](BashSandbox::exec) (or [`restart`](BashSandbox::restart)).
///
/// Drop it into a [`ToolBox`](super::ToolBox) like any other tool: it contributes
/// the predefined [`Bash`](crate::tool::Bash) def (routed by the bare wire name
/// `"bash"`) and dispatches the resulting `tool_use` back to itself.
pub struct BashTool<S: BashSandbox> {
    sandbox: S,
    def: MethodDef,
}

impl<S: BashSandbox> BashTool<S> {
    /// A bash tool over `sandbox`, advertising the predefined `bash_20250124`
    /// def ([`Bash::latest`](crate::tool::Bash::latest)).
    pub fn new(sandbox: S) -> Self {
        Self {
            sandbox,
            def: crate::tool::Bash::latest().into(),
        }
    }

    /// A bash tool over `sandbox`, advertising the richer derived schema
    /// ([`Bash::rich`](crate::tool::Bash::rich)) so the model can drive
    /// background jobs and `poll`/`kill` them.
    pub fn rich(sandbox: S) -> Self {
        Self {
            sandbox,
            def: crate::tool::Bash::rich().into(),
        }
    }

    /// The wrapped sandbox.
    pub fn sandbox(&self) -> &S {
        &self.sandbox
    }
}

#[async_trait::async_trait]
impl<S: BashSandbox> Tool for BashTool<S> {
    fn name(&self) -> &str {
        "bash"
    }

    fn definitions(&self) -> Vec<MethodDef> {
        vec![self.def.clone()]
    }

    async fn on_init(
        &mut self,
        _prompt: &mut crate::Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.sandbox.start().await?;
        Ok(())
    }

    async fn on_teardown(
        &mut self,
        _prompt: &mut crate::Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.sandbox.teardown().await?;
        Ok(())
    }

    async fn call(&mut self, call: Use) -> crate::tool::Result {
        let id = call.id;
        let command = match Command::try_from(call.input) {
            Ok(command) => command,
            Err(e) => {
                return crate::tool::Result::new(
                    id,
                    format!("Error: could not parse bash command: {e}"),
                )
                .error();
            }
        };

        match command {
            // `restart` routes to the sandbox layer (it may drop a borked
            // home), not into the daemon as a command.
            Command::Known(Known::Restart { .. }) => {
                match self.sandbox.restart().await {
                    Ok(()) => {
                        crate::tool::Result::new(id, "bash session restarted")
                    }
                    Err(e) => {
                        crate::tool::Result::new(id, e.to_string()).error()
                    }
                }
            }
            // `poll`/`kill` address a background job by id, via their own
            // endpoints rather than the run stream.
            Command::Known(Known::Poll { poll }) => {
                render(id, self.sandbox.poll(poll).await)
            }
            Command::Known(Known::Kill { kill }) => {
                match self.sandbox.kill(kill).await {
                    Ok(()) => crate::tool::Result::new(
                        id,
                        format!("killed background job {kill}"),
                    ),
                    Err(e) => {
                        crate::tool::Result::new(id, e.to_string()).error()
                    }
                }
            }
            // `Run` (foreground or background) and anything unknown go to the
            // run endpoint.
            other => render(id, self.sandbox.exec(other).await),
        }
    }
}

/// Render an [`ExecResult`] (or error) into a model-facing
/// [`tool::Result`](crate::tool::Result); a timeout is surfaced as an error.
fn render(
    id: Cow<'static, str>,
    result: std::result::Result<ExecResult, BashError>,
) -> crate::tool::Result {
    match result {
        Ok(result) => {
            let reply = crate::tool::Result::new(id, result.render());
            if result.timed_out {
                reply.error()
            } else {
                reply
            }
        }
        Err(e) => crate::tool::Result::new(id, e.to_string()).error(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_known_variants_roundtrip() {
        for (raw, want_run, want_restart) in [
            (r#"{"command":"ls -la"}"#, true, false),
            (r#"{"command":"sleep 1","background":true}"#, true, false),
            (r#"{"command":"make","timeout_secs":30}"#, true, false),
            (r#"{"restart":true}"#, false, true),
        ] {
            let cmd: Command = serde_json::from_str(raw).unwrap();
            assert!(matches!(cmd, Command::Known(_)), "{raw}");
            if want_run {
                assert!(
                    matches!(cmd, Command::Known(Known::Run { .. })),
                    "expected Run for {raw}"
                );
            }
            if want_restart {
                assert!(
                    matches!(cmd, Command::Known(Known::Restart { .. })),
                    "expected Restart for {raw}"
                );
            }
        }
    }

    #[test]
    fn poll_and_kill_carry_job_ids() {
        let poll: Command = serde_json::from_str(r#"{"poll":7}"#).unwrap();
        assert!(matches!(poll, Command::Known(Known::Poll { poll: 7 })));
        let kill: Command = serde_json::from_str(r#"{"kill":7}"#).unwrap();
        assert!(matches!(kill, Command::Known(Known::Kill { kill: 7 })));
    }

    /// A `Run` with an unrelated stray key still parses as `Run` (the unique
    /// single-key variants come first, but they require *their* key, so a plain
    /// `command` with extra noise stays a `Run`).
    #[test]
    fn run_with_stray_key_stays_run() {
        let cmd: Command =
            serde_json::from_str(r#"{"command":"ls","unrelated":1}"#).unwrap();
        assert!(matches!(cmd, Command::Known(Known::Run { .. })));
    }

    #[test]
    fn command_roundtrips_byte_equal() {
        for raw in [
            r#"{"command":"ls -la"}"#,
            r#"{"restart":true}"#,
            r#"{"poll":3}"#,
            r#"{"kill":3}"#,
        ] {
            let cmd: Command = serde_json::from_str(raw).unwrap();
            assert_eq!(serde_json::to_string(&cmd).unwrap(), raw, "{raw}");
        }
    }

    #[test]
    fn reply_variants_discriminate() {
        let ready: Reply = serde_json::from_str(
            r#"{"ready":{"protocol":1,"bashd":"0.1.0","shell":"/bin/bash","persist_cwd":false}}"#,
        )
        .unwrap();
        assert!(matches!(ready, Reply::Ready { .. }));

        let chunk: Reply =
            serde_json::from_str(r#"{"id":1,"stream":"stdout","data":"hi\n"}"#)
                .unwrap();
        assert!(matches!(chunk, Reply::Chunk(_)));

        let outcome: Reply = serde_json::from_str(
            r#"{"id":1,"running":false,"timed_out":false,"truncated":false,"exit":0}"#,
        )
        .unwrap();
        assert!(matches!(outcome, Reply::Outcome(_)));
    }

    #[test]
    fn ready_handshake_roundtrips_bare() {
        // `GET /` returns a bare `Ready`, not wrapped in a `Reply`.
        let ready: Ready = crate::utils::roundtrip(
            r#"{"protocol":1,"bashd":"0.1.0","shell":"/bin/bash","persist_cwd":false}"#,
        );
        assert_eq!(ready.protocol, PROTOCOL_VERSION);
        assert!(!ready.persist_cwd);
    }

    #[test]
    fn outcome_carries_poll_cursor() {
        let o: Outcome = crate::utils::roundtrip(
            r#"{"id":0,"running":true,"timed_out":false,"truncated":false,"job":7,"cursor":4096}"#,
        );
        assert_eq!(o.job, Some(7));
        assert_eq!(o.cursor, Some(4096));
        assert!(o.running);
    }

    #[test]
    fn outcome_without_cursor_omits_it() {
        // Back-compat: an Outcome with no cursor round-trips unchanged.
        let o: Outcome = crate::utils::roundtrip(
            r#"{"id":1,"running":false,"timed_out":false,"truncated":false,"exit":0}"#,
        );
        assert_eq!(o.cursor, None);
    }

    #[test]
    fn sse_event_names() {
        assert_eq!(event::CHUNK, "chunk");
        assert_eq!(event::OUTCOME, "outcome");
    }

    #[test]
    fn tls_material_roundtrips() {
        let json = serde_json::to_string(&TlsServerMaterial {
            cert_chain_pem: "CERT\nCA".into(),
            key_pem: "KEY".into(),
            ca_pem: "CA".into(),
        })
        .unwrap();
        let back: TlsServerMaterial = serde_json::from_str(&json).unwrap();
        assert_eq!(back.cert_chain_pem, "CERT\nCA");
        assert_eq!(back.key_pem, "KEY");
        assert_eq!(back.ca_pem, "CA");
    }

    #[test]
    fn unsupported_error_kind_roundtrips() {
        let err: ProtocolError = serde_json::from_str(
            r#"{"kind":"unsupported","message":"poll not yet supported"}"#,
        )
        .unwrap();
        assert_eq!(err.kind, ErrorKind::Unsupported);
        // An unknown kind from a newer daemon falls into `Other`, not an error.
        let future: ProtocolError =
            serde_json::from_str(r#"{"kind":"from_the_future","message":"x"}"#)
                .unwrap();
        assert_eq!(future.kind, ErrorKind::Other);
    }

    #[test]
    fn exec_result_render_merges_and_notes() {
        let r = ExecResult {
            stdout: "out".into(),
            stderr: "err".into(),
            exit: Some(2),
            ..Default::default()
        };
        let s = r.render();
        assert!(s.contains("out"));
        assert!(s.contains("err"));
        assert!(s.contains("exit code: 2"));
    }

    /// A trivial in-process sandbox: `exec` echoes the command, so `BashTool`
    /// can be exercised end-to-end (dispatch, `on_init`, `on_teardown`) with no
    /// Docker.
    #[derive(Default)]
    struct MockSandbox {
        started: bool,
        torn_down: bool,
        restarts: usize,
        kills: usize,
    }

    #[async_trait::async_trait]
    impl BashSandbox for MockSandbox {
        async fn start(&mut self) -> Result<Ready, BashError> {
            self.started = true;
            Ok(Ready {
                protocol: PROTOCOL_VERSION,
                bashd: "mock".into(),
                shell: "/bin/bash".into(),
                persist_cwd: false,
            })
        }
        async fn exec(
            &mut self,
            command: Command,
        ) -> Result<ExecResult, BashError> {
            let echoed = match command {
                Command::Known(Known::Run { command, .. }) => {
                    command.into_owned()
                }
                other => format!("{other:?}"),
            };
            Ok(ExecResult {
                stdout: echoed,
                exit: Some(0),
                ..Default::default()
            })
        }
        async fn poll(&mut self, job: u64) -> Result<ExecResult, BashError> {
            Ok(ExecResult {
                stdout: format!("output of job {job}"),
                exit: Some(0),
                ..Default::default()
            })
        }
        async fn wait(
            &mut self,
            job: u64,
            _timeout: Option<Duration>,
        ) -> Result<ExecResult, BashError> {
            Ok(ExecResult {
                stdout: format!("job {job} done"),
                exit: Some(0),
                ..Default::default()
            })
        }
        async fn kill(&mut self, _job: u64) -> Result<(), BashError> {
            self.kills += 1;
            Ok(())
        }
        async fn restart(&mut self) -> Result<(), BashError> {
            self.restarts += 1;
            Ok(())
        }
        async fn teardown(&mut self) -> Result<(), BashError> {
            self.torn_down = true;
            Ok(())
        }
    }

    #[tokio::test]
    async fn bashtool_dispatches_and_lifecycles() {
        let mut tool = BashTool::new(MockSandbox::default());
        let mut prompt = crate::Prompt::default();

        // The advertised def is the predefined server tool, routed bare.
        let defs = tool.definitions();
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name(), "bash");

        tool.on_init(&mut prompt).await.unwrap();
        assert!(tool.sandbox().started);

        // A run command echoes through.
        let result = tool
            .call(
                Use::new("bash", serde_json::json!({ "command": "echo hi" }))
                    .with_id("u1"),
            )
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert!(result.content.to_string().contains("echo hi"));

        // `restart` routes to the sandbox layer.
        let restarted = tool
            .call(
                Use::new("bash", serde_json::json!({ "restart": true }))
                    .with_id("u2"),
            )
            .await;
        assert!(!restarted.is_error);
        assert_eq!(tool.sandbox().restarts, 1);

        // `poll` routes to the sandbox and renders its output.
        let polled = tool
            .call(
                Use::new("bash", serde_json::json!({ "poll": 7 }))
                    .with_id("u3"),
            )
            .await;
        assert!(!polled.is_error, "{}", polled.content);
        assert!(polled.content.to_string().contains("job 7"));

        // `kill` routes to the sandbox layer.
        let killed = tool
            .call(
                Use::new("bash", serde_json::json!({ "kill": 7 }))
                    .with_id("u4"),
            )
            .await;
        assert!(!killed.is_error);
        assert!(
            killed
                .content
                .to_string()
                .contains("killed background job 7")
        );
        assert_eq!(tool.sandbox().kills, 1);

        tool.on_teardown(&mut prompt).await.unwrap();
        assert!(tool.sandbox().torn_down);
    }

    #[test]
    fn bash_definition_roundtrips() {
        use crate::tool::{Bash, MethodDef, ServerMethodDef};
        // Request-side wire shape: a bare versioned `type` + `name`, no schema.
        let server: ServerMethodDef = crate::utils::roundtrip(
            r#"{"type":"bash_20250124","name":"bash"}"#,
        );
        assert!(matches!(server, ServerMethodDef::Bash(_)));
        // `add_tool(Bash::latest())` wraps it as a `MethodDef::Server`.
        let def: MethodDef = Bash::latest().into();
        assert_eq!(
            serde_json::to_value(&def).unwrap(),
            serde_json::json!({ "type": "bash_20250124", "name": "bash" }),
        );
    }

    #[test]
    fn bash_rich_schema_is_derived_and_sanitized() {
        let def = crate::tool::Bash::rich();
        assert_eq!(def.name, "bash");
        // The derived schema advertises the run/poll/kill vocabulary.
        let schema = serde_json::to_string(&def.schema).unwrap();
        assert!(schema.contains("command"), "{schema}");
        assert!(schema.contains("poll"), "{schema}");
        assert!(schema.contains("kill"), "{schema}");
        // `From<CustomMethodDef> for MethodDef` makes it addable.
        let _def: MethodDef = def.into();
    }
}
