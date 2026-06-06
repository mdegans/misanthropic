//! [`DockerSandbox`] — the reference [`BashSandbox`] that runs `bashd` inside a
//! Docker (or Podman) container, reached over HTTP/SSE.
//!
//! Lifecycle ([`start`](DockerSandbox::start)): optionally **provision** a custom
//! image (run a `setup` script *with* network, plus create the run user, then
//! `commit`), **run** a session container (`--init`, resource caps, bashd's port
//! published to `127.0.0.1`), and **launch** `bashd --http` (as the run user),
//! polling `GET /` until the [`Ready`] handshake validates. `bashd` is **baked
//! into the image** ([`DEFAULT_IMAGE`], `just build-bashd`), so there is no
//! runtime injection — a dev binary may be bind-mounted over it via
//! [`bashd_path`](DockerSandbox::bashd_path). Each [`exec`](DockerSandbox::exec)
//! POSTs a [`Command`](super::Command) and aggregates the SSE stream into an
//! [`ExecResult`]. [`teardown`](DockerSandbox::teardown) (and a blocking
//! [`Drop`] leak-guard) removes the container.
//!
//! Egress isolation (an internal network + a trusted `bashd relay` sidecar) is a
//! follow-up.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use uuid::Uuid;
use zeroize::Zeroizing;

use super::pki::Pki;
use super::{
    BashError, BashSandbox, Chunk, Command as BashCommand, ExecResult, Outcome,
    PROTOCOL_VERSION, Ready, Stream, event,
};

/// The container port `bashd --http` binds; published to an ephemeral
/// `127.0.0.1` host port the host then discovers via `docker port`.
const BASHD_PORT: u16 = 9099;

/// The sandbox image [`DockerSandbox::default`] boots: `bashd` baked into an
/// immutable rootfs with a pinned non-root `agent` user. Built (and the binary
/// extracted) by `just build-bashd`; must match `bashd_image` in the `justfile`.
const DEFAULT_IMAGE: &str = "misan-bashd:dev";

/// The home directory of the baked `agent` user — the default workdir and the
/// mount point for the `$HOME` volume.
const AGENT_HOME: &str = "/home/agent";

/// The uid/gid the baked `agent` user has (pinned in the Dockerfile) — used to
/// own a tmpfs `$HOME` (a fresh tmpfs is otherwise root-owned, unwritable).
const AGENT_UID: u32 = 1000;

/// The live HTTPS connection to `bashd`: a pinned mutual-TLS client and the host
/// base URL it reaches the published port at (e.g. `https://127.0.0.1:54321`),
/// plus the attached `docker exec` process running `bashd`.
struct Http {
    client: reqwest::Client,
    base: String,
    /// The attached (`docker exec -i`) host process running `bashd`. Its stdin
    /// carried the TLS material, then was closed; retained so teardown can reap
    /// it once `bashd` dies with the container.
    exec: tokio::process::Child,
}

/// How the sandbox container is networked, and therefore how the host reaches
/// `bashd`. **None of these isolate the agent's egress** — restricting what the
/// agent's commands can reach is the host environment's concern (most real
/// deployments already have one), or a future opt-in relay sidecar. They differ
/// only in reachability.
///
/// (`--network none` is intentionally absent: with no network there is no port
/// to publish and no route in, so `bashd` would be unreachable over HTTP.)
#[derive(Clone, Debug, Default)]
pub enum Network {
    /// A bridge network with `bashd`'s port published to `127.0.0.1` (an
    /// ephemeral host port). Works out of the box, including on Docker Desktop.
    #[default]
    Bridge,
    /// Share the host's network namespace (`--network host`); `bashd` is reached
    /// on host loopback directly, no published port. Linux-friendly — but the
    /// agent shares the host's network (so it is *less* isolated), and host
    /// networking is finicky on Docker Desktop.
    Host,
    /// Join a pre-existing docker network by name (the port is still published
    /// to `127.0.0.1`). The hook for your own topology or egress proxy.
    Named(String),
}

/// What backs the agent's `$HOME`. Mirrors [`Network`] — pass it to
/// [`DockerSandbox::home_fs`] (also accepts `"tmpfs"`/`"volume"`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum HomeFs {
    /// A Docker volume (disk-backed). Anonymous + ephemeral by default;
    /// persistent and named when [`DockerSandbox::home_id`] is set. The size cap
    /// ([`home_limit`](DockerSandbox::home_limit)) is **best-effort** — volume
    /// quotas aren't enforced on the common storage drivers.
    #[default]
    Volume,
    /// A tmpfs (RAM-backed, ephemeral). The size cap is **hard-enforced** but
    /// counts against `--memory`. Mutually exclusive with a persistent
    /// [`home_id`](DockerSandbox::home_id).
    Tmpfs,
}

impl From<&str> for HomeFs {
    /// `"tmpfs"`/`"ramfs"` → [`Tmpfs`](HomeFs::Tmpfs); anything else →
    /// [`Volume`](HomeFs::Volume).
    fn from(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "tmpfs" | "ramfs" => HomeFs::Tmpfs,
            _ => HomeFs::Volume,
        }
    }
}

/// A [`BashSandbox`] that runs [`bashd`] inside a Docker/Podman container.
///
/// Build it fluently, then hand it to a
/// [`BashTool`](super::BashTool)::[`new`](super::BashTool::new):
///
/// ```no_run
/// # #[cfg(all(feature = "bash-container", not(target_arch = "wasm32")))]
/// # fn f() {
/// use misanthropic::tool::bash::{BashTool, DockerSandbox};
///
/// // The default boots the baked `misan-bashd` image (`just build-bashd`):
/// // bashd is already on an immutable rootfs, as a non-root `agent` user.
/// let tool = BashTool::new(DockerSandbox::default());
/// # let _ = tool;
/// # }
/// ```
pub struct DockerSandbox {
    base_image: String,
    setup: Option<String>,
    user: Option<String>,
    workdir: String,
    persist_cwd: bool,
    tmp_limit: u64,
    home_limit: u64,
    home_id: Option<Uuid>,
    home_fs: HomeFs,
    memory_limit: Option<String>,
    pids_limit: Option<u64>,
    network: Network,
    runtime: String,
    bashd_path: Option<PathBuf>,
    // Runtime state, populated by `start`.
    container: Option<String>,
    /// A `misan-bashd-img-*` image we committed during provisioning, to `rmi` on
    /// teardown (so they don't accumulate). `None` when we ran a base image
    /// directly.
    provisioned: Option<String>,
    http: Option<Http>,
    /// Per-background-job read cursor (byte offset) for `poll`/`wait`.
    cursors: HashMap<u64, u64>,
}

impl Default for DockerSandbox {
    /// The happy path: boot the baked [`DEFAULT_IMAGE`] (built by
    /// `just build-bashd`) as its pinned non-root `agent` user. `bashd` is
    /// already on the rootfs, so no setup, user creation, or
    /// [`bashd_path`](Self::bashd_path) is needed.
    fn default() -> Self {
        Self {
            base_image: DEFAULT_IMAGE.to_string(),
            setup: None,
            user: Some("agent".to_string()),
            workdir: AGENT_HOME.to_string(),
            persist_cwd: false,
            tmp_limit: 1 << 30, // 1 GiB tmpfs /tmp (hard-enforced)
            home_limit: 10 << 30, // 10 GiB $HOME (hard for tmpfs, advisory else)
            home_id: None,
            home_fs: HomeFs::Volume,
            memory_limit: None,
            pids_limit: None,
            network: Network::default(),
            runtime: "docker".to_string(),
            bashd_path: None,
            container: None,
            provisioned: None,
            http: None,
            cursors: HashMap::new(),
        }
    }
}

impl DockerSandbox {
    /// A sandbox on a custom base `image` — which **must carry `bashd`** (build
    /// it `FROM` the [`DEFAULT_IMAGE`], or supply a dev binary via
    /// [`bashd_path`](Self::bashd_path)). Inherits the rest of
    /// [`Default`](Self::default) (the `agent` user, [`AGENT_HOME`] workdir).
    pub fn new(image: impl Into<String>) -> Self {
        let mut sandbox = Self::default();
        sandbox.base_image = image.into();
        sandbox
    }

    /// A provisioning script run **with network** in a build phase, then
    /// committed into the image the session runs from. Use it to `apk add`/`pip
    /// install` what the agent will need.
    pub fn setup(mut self, script: impl Into<String>) -> Self {
        self.setup = Some(script.into());
        self
    }

    /// Run commands as this (non-root) user. The user is created during
    /// provisioning if it does not already exist in the image.
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    /// The working directory commands start in (default [`AGENT_HOME`]).
    pub fn workdir(mut self, dir: impl Into<String>) -> Self {
        self.workdir = dir.into();
        self
    }

    /// Persist the working directory across commands (default off — see
    /// [`bashd`'s `--persist-cwd`](super::Ready::persist_cwd)).
    pub fn persist_cwd(mut self, persist: bool) -> Self {
        self.persist_cwd = persist;
        self
    }

    /// Size of the writable tmpfs `/tmp`, in bytes (default 1 GiB). The rootfs is
    /// read-only, so `/tmp` is the container's scratch — `bashd`'s job spool
    /// lives here too. Hard-enforced on every storage driver (it's a tmpfs), but
    /// **counts against `--memory`** (it's RAM-backed).
    pub fn tmp_limit(mut self, bytes: u64) -> Self {
        self.tmp_limit = bytes;
        self
    }

    /// Give the agent a **persistent** `$HOME`: a named Docker volume keyed by
    /// `id`, mounted at [`AGENT_HOME`] and surviving teardown — so a later
    /// session with the same `id` "boots the same computer back up", files
    /// intact. Without it, `$HOME` is ephemeral (see [`home_fs`](Self::home_fs)).
    /// Delete it with [`remove_home`](Self::remove_home). Mutually exclusive with
    /// `home_fs(`[`Tmpfs`](HomeFs::Tmpfs)`)` — [`start`](Self::start) errors.
    pub fn home_id(mut self, id: impl Into<Uuid>) -> Self {
        self.home_id = Some(id.into());
        self
    }

    /// What backs `$HOME` (default [`HomeFs::Volume`]). Accepts a [`HomeFs`] or a
    /// string (`"tmpfs"`/`"volume"`).
    pub fn home_fs(mut self, fs: impl Into<HomeFs>) -> Self {
        self.home_fs = fs.into();
        self
    }

    /// Size cap for `$HOME`, in bytes (default 10 GiB). **Hard-enforced** for a
    /// tmpfs home (and then counts against `--memory`); **advisory** for a volume
    /// (volume quotas aren't enforced on the common storage drivers).
    pub fn home_limit(mut self, bytes: u64) -> Self {
        self.home_limit = bytes;
        self
    }

    /// Cap container memory (e.g. `"512m"`, `"2g"`). Passed to `--memory`.
    pub fn memory(mut self, limit: impl Into<String>) -> Self {
        self.memory_limit = Some(limit.into());
        self
    }

    /// Cap the number of processes (`--pids-limit`).
    pub fn pids_limit(mut self, limit: u64) -> Self {
        self.pids_limit = Some(limit);
        self
    }

    /// How the container is networked (default [`Network::Bridge`]). See
    /// [`Network`] — note that **none** of the modes isolate the agent's egress.
    pub fn network(mut self, network: Network) -> Self {
        self.network = network;
        self
    }

    /// The container runtime binary (default `"docker"`; e.g. `"podman"`).
    pub fn runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = runtime.into();
        self
    }

    /// **Dev escape hatch:** bind-mount a freshly-built `bashd` (a static
    /// linux-musl binary, e.g. `target-linux/release/bashd` from
    /// `just build-bashd`) read-only over the one baked into the image, so you
    /// can iterate on the daemon without rebuilding the image. Unset (the
    /// default), the image's baked `bashd` is used.
    pub fn bashd_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.bashd_path = Some(path.into());
        self
    }

    /// The running container's name, once [`start`](Self::start)ed.
    pub fn container(&self) -> Option<&str> {
        self.container.as_deref()
    }

    /// Delete a persistent [`home_id`](Self::home_id) volume (`docker volume
    /// rm`). The `$HOME` named volume survives teardown by design, so this is the
    /// explicit way to reclaim it. Errors if it's still in use by a live sandbox.
    pub async fn remove_home(
        runtime: impl AsRef<str>,
        id: impl Into<Uuid>,
    ) -> Result<(), BashError> {
        let volume = format!("misan-bashd-home-{}", id.into());
        let out = capture(runtime.as_ref(), ["volume", "rm", &volume]).await?;
        if !out.status.success() {
            return Err(BashError::Backend(format!(
                "could not remove home volume {volume}: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(())
    }

    /// Start-time invariants the fluent builders can't enforce (they return
    /// `Self`, not `Result`). Pure — called at the top of [`start`](Self::start).
    fn validate(&self) -> Result<(), BashError> {
        if self.home_id.is_some() && self.home_fs == HomeFs::Tmpfs {
            return Err(BashError::Backend(
                "home_id is persistent but home_fs(tmpfs) is ephemeral — \
                 pick one"
                    .to_string(),
            ));
        }
        #[cfg(feature = "log")]
        if self.home_fs == HomeFs::Tmpfs && self.memory_limit.is_some() {
            log::warn!(
                "bash sandbox: a tmpfs $HOME and /tmp both count against \
                 --memory; size home_limit + tmp_limit to fit within it"
            );
        }
        Ok(())
    }

    /// Resolve (provisioning if needed) the image the session runs from.
    async fn provision(&self) -> Result<String, BashError> {
        // The baked `agent` exists in the default image, so don't provision just
        // to "create" it — that's the common path, and provisioning would commit
        // (and leak) an image needlessly. A custom base still gets the user.
        let agent_baked = self.base_image == DEFAULT_IMAGE
            && self.user.as_deref() == Some("agent");
        let creates_user =
            !agent_baked && self.user.as_deref().is_some_and(|u| u != "root");
        if self.setup.is_none() && !creates_user {
            return Ok(self.base_image.clone());
        }

        let prov = format!("misan-bashd-prov-{}", unique());
        let mut script = String::new();
        if let Some(user) = &self.user
            && user != "root"
        {
            // Best-effort across Alpine (adduser) and Debian (useradd).
            script.push_str(&format!(
                "(adduser -D {user} 2>/dev/null || useradd -m {user} \
                 2>/dev/null || true); "
            ));
        }
        if let Some(setup) = &self.setup {
            script.push_str(setup);
            script.push_str("; ");
        }
        // Ensure the working directory exists and is writable by the run user.
        script.push_str(&format!("mkdir -p {wd}; ", wd = self.workdir));
        if let Some(user) = &self.user
            && user != "root"
        {
            script.push_str(&format!(
                "chown -R {user} {wd} 2>/dev/null || true; ",
                wd = self.workdir
            ));
        }

        // Provision *with* network (no --network none here).
        let run = capture(
            &self.runtime,
            [
                "run",
                "--name",
                &prov,
                &self.base_image,
                "/bin/sh",
                "-c",
                &script,
            ],
        )
        .await?;
        if !run.status.success() {
            let _ = capture(&self.runtime, ["rm", "-f", &prov]).await;
            return Err(BashError::Backend(format!(
                "provisioning failed: {}",
                String::from_utf8_lossy(&run.stderr).trim()
            )));
        }

        let image = format!("misan-bashd-img-{}", unique());
        let commit = capture(&self.runtime, ["commit", &prov, &image]).await?;
        let _ = capture(&self.runtime, ["rm", "-f", &prov]).await;
        if !commit.status.success() {
            return Err(BashError::Backend(format!(
                "commit failed: {}",
                String::from_utf8_lossy(&commit.stderr).trim()
            )));
        }
        Ok(image)
    }

    /// `docker run -d` the locked-down session container (networked per
    /// [`Network`]): an immutable read-only rootfs, all capabilities dropped, no
    /// privilege escalation, a writable tmpfs `/tmp`, and an ephemeral `$HOME`
    /// volume. The only writable paths are `$HOME` (the volume) and `/tmp`.
    async fn run_container(&self, image: &str) -> Result<String, BashError> {
        let container = format!("misan-bashd-{}", unique());
        let pids = self.pids_limit.unwrap_or(512);
        let mut args: Vec<String> = vec![
            "run".into(),
            "-d".into(),
            // tini as PID 1 reaps orphaned grandchildren.
            "--init".into(),
            "--name".into(),
            container.clone(),
            // Hardening: immutable rootfs, no caps, no setuid escalation.
            "--read-only".into(),
            "--cap-drop".into(),
            "ALL".into(),
            "--security-opt".into(),
            "no-new-privileges".into(),
            // Writable scratch (bashd's job spool lives here); RAM-backed, so it
            // counts against --memory. nosuid/nodev: no setuid bins, no devices.
            "--tmpfs".into(),
            format!("/tmp:size={},mode=1777,nosuid,nodev", self.tmp_limit),
            "--pids-limit".into(),
            pids.to_string(),
        ];
        // Writable $HOME. Persistent named volume when `home_id` is set (survives
        // teardown); otherwise ephemeral — an anonymous volume (disk, reaped by
        // `rm -fv`) or a tmpfs (RAM, owned by the pinned agent uid since a fresh
        // tmpfs is root-owned). Volumes are seeded from the image, so dotfiles +
        // ownership carry over on first mount. (`validate` rejected id+tmpfs.)
        match (&self.home_id, &self.home_fs) {
            (Some(id), _) => {
                args.push("--mount".into());
                args.push(format!(
                    "type=volume,source=misan-bashd-home-{id},\
                     target={AGENT_HOME}"
                ));
            }
            (None, HomeFs::Tmpfs) => {
                args.push("--tmpfs".into());
                args.push(format!(
                    "{AGENT_HOME}:size={},uid={AGENT_UID},gid={AGENT_UID},\
                     mode=0700",
                    self.home_limit
                ));
            }
            (None, HomeFs::Volume) => {
                args.push("--mount".into());
                args.push(format!("type=volume,target={AGENT_HOME}"));
            }
        }
        // Networking. None of these isolate the agent's egress (see `Network`);
        // they differ only in how the host reaches bashd's port.
        match &self.network {
            Network::Bridge => {
                args.push("-p".into());
                args.push(format!("127.0.0.1::{BASHD_PORT}"));
            }
            Network::Host => {
                args.push("--network".into());
                args.push("host".into());
            }
            Network::Named(name) => {
                args.push("--network".into());
                args.push(name.clone());
                args.push("-p".into());
                args.push(format!("127.0.0.1::{BASHD_PORT}"));
            }
        }
        args.push("--workdir".into());
        args.push(self.workdir.clone());
        if let Some(mem) = &self.memory_limit {
            args.push("--memory".into());
            args.push(mem.clone());
        }
        // Dev escape hatch: bind-mount a freshly-built bashd read-only over the
        // baked one (the mount point exists on the image's rootfs).
        if let Some(path) = &self.bashd_path {
            let abs = std::fs::canonicalize(path).map_err(|e| {
                BashError::Backend(format!("bashd_path {path:?}: {e}"))
            })?;
            args.push("-v".into());
            args.push(format!("{}:/usr/local/bin/bashd:ro", abs.display()));
        }
        args.push(image.to_string());
        // Keep the container alive; we exec bashd into it separately.
        args.extend(["tail", "-f", "/dev/null"].map(String::from));

        let out = capture(&self.runtime, &args).await?;
        if !out.status.success() {
            return Err(BashError::Backend(format!(
                "could not start container: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        Ok(container)
    }

    /// Fail early with an actionable message if the baked [`DEFAULT_IMAGE`] is
    /// not built locally. Custom images get docker's own not-found error.
    async fn ensure_default_image(&self) -> Result<(), BashError> {
        if self.base_image != DEFAULT_IMAGE {
            return Ok(());
        }
        let ok = Command::new(&self.runtime)
            .args(["image", "inspect", DEFAULT_IMAGE])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
        ok.then_some(()).ok_or_else(|| {
            BashError::Backend(format!(
                "default sandbox image `{DEFAULT_IMAGE}` is not built — \
                 run `just build-bashd`"
            ))
        })
    }

    /// Launch `bashd --http` (attached, as the run user) over an ephemeral
    /// per-container mutual-TLS channel, discover its published host port, and
    /// poll `GET /` until it's ready. Returns the connection and the validated
    /// handshake.
    async fn launch(
        &self,
        container: &str,
    ) -> Result<(Http, Ready), BashError> {
        // Mint the per-container PKI host-side. The server half goes to bashd
        // over stdin below; the client half stays here, in the pinned client.
        let pki = Pki::generate()?;

        // bashd binds *inside* the container. Host networking shares the
        // namespace, so bind loopback (don't expose bashd on every host
        // interface). Bridge/named publish via DNAT to the container's eth0, so
        // bashd must bind 0.0.0.0 to be reachable through the mapping.
        let bind = match &self.network {
            Network::Host => format!("127.0.0.1:{BASHD_PORT}"),
            Network::Bridge | Network::Named(_) => {
                format!("0.0.0.0:{BASHD_PORT}")
            }
        };

        // Launch bashd *attached* (`-i`, not `-d`): we feed the TLS material to
        // its stdin and then close it. `docker exec` runs as root and does not
        // inherit `docker run --user`, so pass the run user explicitly — the
        // agent's commands then run unprivileged.
        let mut args: Vec<String> = vec!["exec".into(), "-i".into()];
        if let Some(user) = &self.user {
            args.push("--user".into());
            args.push(user.clone());
        }
        args.push(container.into());
        args.push("/usr/local/bin/bashd".into());
        args.push("--http".into());
        args.push(bind);
        args.push("--workdir".into());
        args.push(self.workdir.clone());
        if self.persist_cwd {
            args.push("--persist-cwd".into());
        }
        let mut exec = Command::new(&self.runtime)
            .args(&args)
            // Keep stderr so a startup failure (e.g. bashd can't write its
            // scratch dir on a misconfigured rootfs) surfaces instead of an
            // opaque readiness timeout.
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()?;

        // Hand bashd its TLS material over stdin (never argv/env/disk), then
        // EOF. A write failure means the exec never really started (bad
        // container, missing binary) — surface it rather than waiting out the
        // readiness deadline.
        let payload = Zeroizing::new(
            serde_json::to_string(&pki.server)
                .map_err(|e| BashError::Backend(e.to_string()))?,
        );
        let mut stdin = exec.stdin.take().ok_or_else(|| {
            BashError::Backend("could not open bashd stdin".into())
        })?;
        stdin.write_all(payload.as_bytes()).await.map_err(|e| {
            BashError::Backend(format!(
                "could not send bashd TLS material: {e}"
            ))
        })?;
        stdin.shutdown().await.ok();
        drop(stdin);

        // Where the host reaches bashd depends on the network mode.
        let base = match &self.network {
            // Host networking shares the host namespace: bashd's port is on the
            // host loopback directly (no published mapping to discover).
            Network::Host => format!("https://127.0.0.1:{BASHD_PORT}"),
            // Published modes: discover the ephemeral host port (127.0.0.1:NNNNN).
            Network::Bridge | Network::Named(_) => {
                let port = capture(
                    &self.runtime,
                    ["port", container, &format!("{BASHD_PORT}/tcp")],
                )
                .await?;
                let mapping = String::from_utf8_lossy(&port.stdout);
                let host_port = mapping
                    .lines()
                    .next()
                    .and_then(|l| l.trim().rsplit(':').next())
                    .ok_or_else(|| {
                        BashError::Backend(format!(
                            "could not read published bashd port: {mapping:?}"
                        ))
                    })?;
                format!("https://127.0.0.1:{host_port}")
            }
        };

        // A pinned mTLS client: present our client identity and trust *only* the
        // per-container CA (built-in roots off). bashd verifies our client cert
        // against the same CA, so nothing else on the host can drive it. `pki`
        // (and every PEM in it) drops at the end of this fn — reqwest has parsed
        // what it needs.
        let client = reqwest::Client::builder()
            .add_root_certificate(
                reqwest::Certificate::from_pem(pki.ca_pem.as_bytes())
                    .map_err(|e| BashError::Backend(e.to_string()))?,
            )
            .tls_built_in_root_certs(false)
            .identity(
                reqwest::Identity::from_pem(pki.client_identity_pem.as_bytes())
                    .map_err(|e| BashError::Backend(e.to_string()))?,
            )
            .build()
            .map_err(|e| BashError::Backend(e.to_string()))?;

        let ready = match await_ready(&client, &base).await {
            Ok(ready) => ready,
            Err(e) => {
                // bashd never answered — append whatever it logged, then reap it.
                let logs = drain_stderr(&mut exec).await;
                let _ = exec.start_kill();
                let _ = exec.wait().await;
                return Err(match e {
                    _ if logs.is_empty() => e,
                    BashError::Handshake(m) => {
                        BashError::Handshake(format!("{m}; bashd: {logs}"))
                    }
                    other => other,
                });
            }
        };
        if ready.protocol != PROTOCOL_VERSION {
            return Err(BashError::Handshake(format!(
                "protocol mismatch: daemon speaks {}, host speaks {}",
                ready.protocol, PROTOCOL_VERSION
            )));
        }
        Ok((Http { client, base, exec }, ready))
    }

    /// The HTTP client + base URL, cloned so callers needn't hold a `self`
    /// borrow across an `.await` (the cursor updates take `&mut self`).
    fn endpoint(&self) -> Result<(reqwest::Client, String), BashError> {
        let http = self.http.as_ref().ok_or(BashError::NotStarted)?;
        Ok((http.client.clone(), http.base.clone()))
    }

    /// Send one [`Command`] over HTTP and aggregate its SSE stream into an
    /// [`ExecResult`].
    async fn request(
        &self,
        command: BashCommand,
    ) -> Result<ExecResult, BashError> {
        let (client, base) = self.endpoint()?;
        let resp = client
            .post(format!("{base}/run"))
            .json(&command)
            .send()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        aggregate(resp).await
    }

    /// Remove the container (best-effort), forgetting it so [`Drop`] won't retry.
    async fn remove_container(&mut self) {
        // Drop the HTTP client; bashd dies with the container removed below.
        let http = self.http.take();
        if let Some(container) = self.container.take() {
            // `-v` reaps the anonymous $HOME volume; a named (persistent) one is
            // spared (only anonymous/attached volumes are removed).
            let _ = capture(&self.runtime, ["rm", "-fv", &container]).await;
        }
        // Reap the attached `docker exec` process now that bashd (and the
        // container) is gone, so it doesn't linger as a zombie.
        if let Some(mut http) = http {
            let _ = http.exec.start_kill();
            let _ = http.exec.wait().await;
        }
        // Drop the image we committed during provisioning (the container is gone
        // now, so nothing holds it). A named home volume, if any, survives.
        if let Some(image) = self.provisioned.take() {
            let _ = capture(&self.runtime, ["rmi", "-f", &image]).await;
        }
    }
}

#[async_trait::async_trait]
impl BashSandbox for DockerSandbox {
    async fn start(&mut self) -> Result<Ready, BashError> {
        if self.http.is_some() {
            return Err(BashError::Backend(
                "sandbox already started".to_string(),
            ));
        }
        self.validate()?;
        self.ensure_default_image().await?;
        let image = self.provision().await?;
        // A committed image (not the base) must be `rmi`'d at teardown.
        if image != self.base_image {
            self.provisioned = Some(image.clone());
        }
        let container = self.run_container(&image).await?;
        // From here on a failure must remove the container, not leak it. bashd
        // is baked into the rootfs (or bind-mounted via `bashd_path`), so there
        // is nothing to inject — launch reaches it directly.
        self.container = Some(container.clone());
        match self.launch(&container).await {
            Ok((http, ready)) => {
                self.http = Some(http);
                Ok(ready)
            }
            Err(e) => {
                self.remove_container().await;
                Err(e)
            }
        }
    }

    async fn exec(
        &mut self,
        command: BashCommand,
    ) -> Result<ExecResult, BashError> {
        self.request(command).await
    }

    async fn poll(&mut self, job: u64) -> Result<ExecResult, BashError> {
        let (client, base) = self.endpoint()?;
        let cursor = self.cursors.get(&job).copied().unwrap_or(0);
        let resp = client
            .get(format!("{base}/jobs/{job}?cursor={cursor}"))
            .send()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        let result = aggregate(resp).await?;
        if let Some(cursor) = result.cursor {
            self.cursors.insert(job, cursor);
        }
        Ok(result)
    }

    async fn wait(
        &mut self,
        job: u64,
        timeout: Option<Duration>,
    ) -> Result<ExecResult, BashError> {
        let (client, base) = self.endpoint()?;
        let cursor = self.cursors.get(&job).copied().unwrap_or(0);
        let mut url = format!("{base}/jobs/{job}/wait?cursor={cursor}");
        if let Some(timeout) = timeout {
            url.push_str(&format!("&timeout={}", timeout.as_secs()));
        }
        let resp = client
            .get(url)
            .send()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        let result = aggregate(resp).await?;
        if let Some(cursor) = result.cursor {
            self.cursors.insert(job, cursor);
        }
        Ok(result)
    }

    async fn kill(&mut self, job: u64) -> Result<(), BashError> {
        let (client, base) = self.endpoint()?;
        let resp = client
            .post(format!("{base}/jobs/{job}/kill"))
            .send()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        let outcome: Outcome = resp
            .json()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        if let Some(error) = outcome.error {
            return Err(BashError::Protocol(error.message));
        }
        self.cursors.remove(&job);
        Ok(())
    }

    async fn restart(&mut self) -> Result<(), BashError> {
        // Phase 1: reset the daemon session. Dropping a borked home volume is a
        // later phase.
        self.request(BashCommand::Known(super::Known::Restart {
            restart: true,
        }))
        .await
        .map(|_| ())
    }

    async fn teardown(&mut self) -> Result<(), BashError> {
        // Home backup snapshot is a later phase; for now just remove the box.
        self.remove_container().await;
        Ok(())
    }
}

impl Drop for DockerSandbox {
    /// Leak guard: if the container was never [`teardown`](BashSandbox::teardown)
    /// (e.g. a panic), remove it with a *blocking* `docker rm -fv` (best-effort;
    /// `-v` reaps the anonymous $HOME volume, spares a named one).
    fn drop(&mut self) {
        if let Some(container) = self.container.take() {
            let _ = std::process::Command::new(&self.runtime)
                .args(["rm", "-fv", &container])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
        // Container gone — drop the provisioned image too (best-effort).
        if let Some(image) = self.provisioned.take() {
            let _ = std::process::Command::new(&self.runtime)
                .args(["rmi", "-f", &image])
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status();
        }
    }
}

/// Aggregate a `POST /run` SSE response into an [`ExecResult`]: `chunk` events
/// append to stdout/stderr; the terminal `outcome` event sets the flags.
async fn aggregate(resp: reqwest::Response) -> Result<ExecResult, BashError> {
    let mut events = resp.bytes_stream().eventsource();
    let mut result = ExecResult::default();
    while let Some(event) = events.next().await {
        let event = event.map_err(|e| BashError::Protocol(e.to_string()))?;
        if event.event == event::CHUNK {
            if let Ok(chunk) = serde_json::from_str::<Chunk>(&event.data) {
                match chunk.stream {
                    Stream::Stdout => result.stdout.push_str(&chunk.data),
                    Stream::Stderr => result.stderr.push_str(&chunk.data),
                }
            }
        } else if event.event == event::OUTCOME {
            if let Ok(outcome) = serde_json::from_str::<Outcome>(&event.data) {
                if let Some(err) = outcome.error {
                    return Err(BashError::Protocol(err.message));
                }
                result.exit = outcome.exit;
                result.running = outcome.running;
                result.timed_out = outcome.timed_out;
                result.truncated = outcome.truncated;
                result.job = outcome.job;
                result.advice = outcome.advice;
                result.cursor = outcome.cursor;
            }
            break;
        }
    }
    Ok(result)
}

/// Poll `GET /` with bounded backoff until `bashd` answers a valid handshake —
/// the readiness gate, since `docker exec -d` returns before the port is bound.
async fn await_ready(
    client: &reqwest::Client,
    base: &str,
) -> Result<Ready, BashError> {
    let deadline = Instant::now() + Duration::from_secs(10);
    let mut delay = Duration::from_millis(50);
    loop {
        if let Ok(resp) = client.get(format!("{base}/")).send().await
            && let Ok(ready) = resp.json::<Ready>().await
        {
            return Ok(ready);
        }
        if Instant::now() >= deadline {
            return Err(BashError::Handshake(
                "bashd did not become ready in time".into(),
            ));
        }
        tokio::time::sleep(delay).await;
        delay = (delay * 2).min(Duration::from_millis(500));
    }
}

/// Drain a failed `bashd`'s stderr (briefly) for its diagnostic. A crashed bashd
/// has closed stderr so the read returns at once; the timeout only guards the
/// (unexpected) case of a live-but-unreachable daemon so teardown can't hang.
async fn drain_stderr(child: &mut tokio::process::Child) -> String {
    use tokio::io::AsyncReadExt;
    let Some(mut err) = child.stderr.take() else {
        return String::new();
    };
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(
        Duration::from_millis(300),
        err.read_to_end(&mut buf),
    )
    .await;
    String::from_utf8_lossy(&buf).trim().to_string()
}

/// Run `runtime args...`, capturing output and forwarding any stderr to
/// `log::warn!` (so docker's "cap not supported by driver" notes surface).
async fn capture<I, S>(
    runtime: &str,
    args: I,
) -> Result<std::process::Output, BashError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let out = Command::new(runtime).args(args).output().await?;
    #[cfg(feature = "log")]
    for line in String::from_utf8_lossy(&out.stderr).lines() {
        if !line.trim().is_empty() {
            log::warn!("{runtime}: {line}");
        }
    }
    Ok(out)
}

/// A process-unique suffix for container/image names (no RNG needed — `Drop`
/// and explicit teardown clean up, and a pid+counter never collides in-process).
fn unique() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_boot_the_baked_image() {
        let s = DockerSandbox::default();
        assert_eq!(s.base_image, DEFAULT_IMAGE);
        assert_eq!(s.user.as_deref(), Some("agent"));
        assert_eq!(s.workdir, AGENT_HOME);
        assert!(matches!(s.network, Network::Bridge));
        assert!(s.bashd_path.is_none());
    }

    #[test]
    fn builders_set_fields() {
        let s = DockerSandbox::new("custom:tag")
            .setup("apk add bash")
            .user("agent")
            .workdir("/work")
            .persist_cwd(true)
            .memory("512m")
            .pids_limit(128)
            .network(Network::Named("my-net".into()))
            .runtime("podman")
            .bashd_path("/tmp/bashd");
        assert_eq!(s.base_image, "custom:tag");
        assert!(matches!(&s.network, Network::Named(n) if n == "my-net"));
        assert_eq!(s.setup.as_deref(), Some("apk add bash"));
        assert_eq!(s.user.as_deref(), Some("agent"));
        assert_eq!(s.workdir, "/work");
        assert!(s.persist_cwd);
        assert_eq!(s.memory_limit.as_deref(), Some("512m"));
        assert_eq!(s.pids_limit, Some(128));
        assert_eq!(s.runtime, "podman");
        assert_eq!(
            s.bashd_path.as_deref(),
            Some(std::path::Path::new("/tmp/bashd"))
        );
        assert!(s.container().is_none());
    }

    #[test]
    fn unique_names_differ() {
        assert_ne!(unique(), unique());
    }

    #[test]
    fn home_fs_parses_from_str() {
        assert_eq!(HomeFs::from("tmpfs"), HomeFs::Tmpfs);
        assert_eq!(HomeFs::from("RamFS"), HomeFs::Tmpfs);
        assert_eq!(HomeFs::from("volume"), HomeFs::Volume);
        assert_eq!(HomeFs::from("anything-else"), HomeFs::Volume);
    }

    #[test]
    fn validate_rejects_persistent_tmpfs() {
        // A persistent id on an ephemeral tmpfs is a contradiction.
        let s = DockerSandbox::default()
            .home_id(Uuid::nil())
            .home_fs("tmpfs");
        assert!(s.validate().is_err());
        // Either alone is fine.
        assert!(
            DockerSandbox::default()
                .home_id(Uuid::nil())
                .validate()
                .is_ok()
        );
        assert!(DockerSandbox::default().home_fs("tmpfs").validate().is_ok());
        assert!(DockerSandbox::default().validate().is_ok());
    }

    /// An optional dev `bashd` to bind-mount over the baked one, from
    /// `BASHD_PATH` (exercises the override path). `None` → use the baked binary.
    fn bashd_override() -> Option<PathBuf> {
        std::env::var("BASHD_PATH")
            .ok()
            .map(PathBuf::from)
            .filter(|p| p.exists())
    }

    async fn docker_available() -> bool {
        Command::new("docker")
            .arg("version")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    /// Whether the baked [`DEFAULT_IMAGE`] is built locally (via `just
    /// build-bashd`). `false` → skip the live test.
    async fn default_image_built() -> bool {
        Command::new("docker")
            .args(["image", "inspect", DEFAULT_IMAGE])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false)
    }

    fn run(cmd: &str) -> crate::tool::bash::Command {
        crate::tool::bash::Command::Known(crate::tool::bash::Known::Run {
            command: cmd.to_string().into(),
            background: None,
            timeout_secs: None,
        })
    }

    fn run_bg(cmd: &str) -> crate::tool::bash::Command {
        crate::tool::bash::Command::Known(crate::tool::bash::Known::Run {
            command: cmd.to_string().into(),
            background: Some(true),
            timeout_secs: None,
        })
    }

    /// End-to-end: boot the baked default image, run real commands as the
    /// non-root agent, and tear it down. Deterministic (no model). Skips if
    /// Docker or the `misan-bashd` image (`just build-bashd`) is unavailable.
    /// Set `BASHD_PATH` to also exercise the dev bind-mount override.
    #[tokio::test]
    #[ignore = "requires Docker running and the misan-bashd image (just build-bashd)"]
    async fn live_docker_sandbox_runs_commands() {
        if !docker_available().await {
            eprintln!("skipping: docker not available");
            return;
        }
        if !default_image_built().await {
            eprintln!(
                "skipping: misan-bashd image not built (just build-bashd)"
            );
            return;
        }

        // The happy path: the baked image, no setup/user/workdir/bashd needed.
        let mut sandbox = DockerSandbox::default();
        if let Some(bashd) = bashd_override() {
            sandbox = sandbox.bashd_path(bashd);
        }

        let ready = sandbox.start().await.expect("start sandbox");
        assert_eq!(ready.protocol, crate::tool::bash::PROTOCOL_VERSION);
        assert!(sandbox.container().is_some());

        // Commands run in the agent's home, as the non-root agent.
        let out = sandbox.exec(run("echo hello && pwd")).await.unwrap();
        assert!(out.stdout.contains("hello"), "stdout: {:?}", out.stdout);
        assert!(out.stdout.contains(AGENT_HOME), "pwd: {:?}", out.stdout);
        assert_eq!(out.exit, Some(0));

        let out = sandbox.exec(run("whoami")).await.unwrap();
        assert!(out.stdout.contains("agent"), "whoami: {:?}", out.stdout);
        let out = sandbox.exec(run("id -u")).await.unwrap();
        assert_eq!(out.stdout.trim(), "1000", "uid: {:?}", out.stdout);

        // Hardening: rootfs is immutable, but $HOME and /tmp are writable.
        let out = sandbox.exec(run("touch /etc/x 2>&1")).await.unwrap();
        assert_ne!(out.exit, Some(0), "rootfs should be read-only");
        assert!(
            out.stdout.to_lowercase().contains("read-only"),
            "expected read-only error, got: {:?}",
            out.stdout
        );
        let out = sandbox
            .exec(run("echo hi > ~/state.txt && echo hi > /tmp/x && echo ok"))
            .await
            .unwrap();
        assert!(
            out.stdout.contains("ok"),
            "home/tmp write: {:?}",
            out.stdout
        );

        // The tmpfs /tmp is hard-capped: writing past it fails with ENOSPC.
        // Free it again afterwards (bashd's job spool lives on /tmp).
        let out = sandbox
            .exec(run("dd if=/dev/zero of=/tmp/big bs=1M count=2048 2>&1; \
                 rm -f /tmp/big"))
            .await
            .unwrap();
        assert!(
            out.stdout.to_lowercase().contains("no space"),
            "tmpfs cap (1 GiB) should stop a 2 GiB write: {:?}",
            out.stdout
        );

        // Exit codes are OS-authoritative.
        let out = sandbox.exec(run("exit 7")).await.unwrap();
        assert_eq!(out.exit, Some(7));

        // NB: the default bridge does not isolate the agent's egress — that is
        // the host environment's concern (see `Network`).

        // Background: launch a job, poll partial output, then wait it to done.
        let receipt = sandbox
            .exec(run_bg("for i in 1 2 3; do echo line $i; sleep 1; done"))
            .await
            .unwrap();
        let job = receipt.job.expect("background job id");
        assert!(receipt.running, "background job should report running");

        // A poll mid-flight returns some output and still-running status.
        tokio::time::sleep(std::time::Duration::from_millis(1500)).await;
        let partial = sandbox.poll(job).await.unwrap();
        assert!(partial.stdout.contains("line 1"), "{:?}", partial.stdout);

        // wait() blocks until completion; the soft timeout is well beyond ~3s.
        let done = sandbox
            .wait(job, Some(std::time::Duration::from_secs(15)))
            .await
            .unwrap();
        assert!(!done.running, "job should be finished");
        assert!(done.stdout.contains("line 3"), "{:?}", done.stdout);

        // Kill: start a long background job, then stop it.
        let job = sandbox
            .exec(run_bg("sleep 30"))
            .await
            .unwrap()
            .job
            .expect("job id");
        sandbox.kill(job).await.expect("kill");
        let after = sandbox
            .wait(job, Some(std::time::Duration::from_secs(5)))
            .await
            .unwrap();
        assert!(!after.running, "killed job should be finished");

        sandbox.restart().await.expect("restart");
        sandbox.teardown().await.expect("teardown");
        assert!(sandbox.container().is_none());
    }

    /// Count locally-built `misan-bashd-img-*` images (the provisioned ones).
    async fn provisioned_image_count() -> usize {
        let out = Command::new("docker")
            .args(["images", "--format", "{{.Repository}}"])
            .output()
            .await
            .expect("docker images");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|r| r.starts_with("misan-bashd-img-"))
            .count()
    }

    /// A `setup` script forces provisioning (a committed image); teardown must
    /// `rmi` it, so the count is unchanged. Guards the #85 image leak.
    #[tokio::test]
    #[ignore = "requires Docker running and the misan-bashd image (just build-bashd)"]
    async fn live_provisioned_image_is_cleaned() {
        if !docker_available().await || !default_image_built().await {
            eprintln!("skipping: docker / misan-bashd image unavailable");
            return;
        }
        let before = provisioned_image_count().await;

        let mut sandbox = DockerSandbox::default().setup("true");
        sandbox.start().await.expect("start");
        // It really did provision (committed an image to clean up later).
        assert!(sandbox.provisioned.is_some(), "setup should provision");
        let out = sandbox.exec(run("echo ok")).await.unwrap();
        assert!(out.stdout.contains("ok"));
        sandbox.teardown().await.expect("teardown");

        assert_eq!(
            provisioned_image_count().await,
            before,
            "the provisioned image must be rmi'd at teardown"
        );
    }

    /// A `home_id` volume persists across sessions; `remove_home` reclaims it,
    /// after which a fresh boot starts clean. Exercises #86 end to end.
    #[tokio::test]
    #[ignore = "requires Docker running and the misan-bashd image (just build-bashd)"]
    async fn live_home_id_persists_across_sessions() {
        if !docker_available().await || !default_image_built().await {
            eprintln!("skipping: docker / misan-bashd image unavailable");
            return;
        }
        let id = Uuid::new_v4();

        // Boot 1: write a file into the persistent $HOME, then tear down.
        let mut s = DockerSandbox::default().home_id(id);
        s.start().await.expect("start 1");
        let out = s
            .exec(run("echo persisted > ~/state.txt && cat ~/state.txt"))
            .await
            .unwrap();
        assert!(out.stdout.contains("persisted"), "{:?}", out.stdout);
        s.teardown().await.expect("teardown 1");

        // Boot 2: same id → the file is still there (the volume survived).
        let mut s = DockerSandbox::default().home_id(id);
        s.start().await.expect("start 2");
        let out = s.exec(run("cat ~/state.txt 2>&1")).await.unwrap();
        assert!(
            out.stdout.contains("persisted"),
            "home did not persist: {:?}",
            out.stdout
        );
        s.teardown().await.expect("teardown 2");

        // Reclaim the volume; a fresh boot then starts clean.
        DockerSandbox::remove_home("docker", id)
            .await
            .expect("remove_home");
        let mut s = DockerSandbox::default().home_id(id);
        s.start().await.expect("start 3");
        let out = s
            .exec(run("cat ~/state.txt 2>&1; echo done"))
            .await
            .unwrap();
        assert!(
            !out.stdout.contains("persisted"),
            "home was not cleared: {:?}",
            out.stdout
        );
        s.teardown().await.expect("teardown 3");
        DockerSandbox::remove_home("docker", id).await.ok();
    }
}
