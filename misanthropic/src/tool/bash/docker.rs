//! [`DockerSandbox`] — the reference [`BashSandbox`] that runs `bashd` inside a
//! Docker (or Podman) container, reached over HTTP/SSE.
//!
//! Lifecycle ([`start`](DockerSandbox::start)): optionally **provision** a custom
//! image (run a `setup` script *with* network, plus create the run user, then
//! `commit`), **run** a session container (`--init`, resource caps, bashd's port
//! published to `127.0.0.1`), **inject** the `bashd` binary (`docker cp`), and
//! **launch** `bashd --http` (detached, as the run user), polling `GET /` until
//! the [`Ready`] handshake validates. Each [`exec`](DockerSandbox::exec) POSTs a
//! [`Command`](super::Command) and aggregates the SSE stream into an
//! [`ExecResult`]. [`teardown`](DockerSandbox::teardown) (and a blocking
//! [`Drop`] leak-guard) removes the container.
//!
//! Egress isolation (an internal network + a trusted `bashd relay` sidecar) and
//! home backup/restore are follow-ups.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use eventsource_stream::Eventsource;
use futures::StreamExt;
use tokio::process::Command;

use super::{
    BashError, BashSandbox, Chunk, Command as BashCommand, ExecResult, Outcome,
    PROTOCOL_VERSION, Ready, Stream, event,
};

/// The container port `bashd --http` binds; published to an ephemeral
/// `127.0.0.1` host port the host then discovers via `docker port`.
const BASHD_PORT: u16 = 9099;

/// The live HTTP connection to `bashd`: a client and the host base URL it
/// reaches the published port at (e.g. `http://127.0.0.1:54321`).
struct Http {
    client: reqwest::Client,
    base: String,
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
/// let tool = BashTool::new(
///     DockerSandbox::alpine()
///         .setup("apk add --no-cache bash coreutils")
///         .user("agent")
///         .workdir("/work"),
/// );
/// # let _ = tool;
/// # }
/// ```
pub struct DockerSandbox {
    base_image: String,
    setup: Option<String>,
    user: Option<String>,
    workdir: String,
    persist_cwd: bool,
    storage_limit: Option<u64>,
    memory_limit: Option<String>,
    pids_limit: Option<u64>,
    runtime: String,
    bashd_path: Option<PathBuf>,
    // Runtime state, populated by `start`.
    container: Option<String>,
    http: Option<Http>,
}

impl DockerSandbox {
    /// A sandbox on a base `image` (e.g. `"alpine:3"`, `"debian:stable-slim"`).
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            base_image: image.into(),
            setup: None,
            user: None,
            workdir: "/".to_string(),
            persist_cwd: false,
            storage_limit: Some(10 << 30), // 10 GiB default (best-effort)
            memory_limit: None,
            pids_limit: None,
            runtime: "docker".to_string(),
            bashd_path: None,
            container: None,
            http: None,
        }
    }

    /// A sandbox on the latest Alpine image.
    pub fn alpine() -> Self {
        Self::new("alpine:3")
    }

    /// A provisioning script run **with network** in a build phase, then
    /// committed into the image the (network-isolated) session runs from. Use it
    /// to `apk add`/`pip install` what the agent will need.
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

    /// The working directory commands start in (default `/`).
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

    /// Cap the container's writable storage, in bytes (default 10 GiB).
    ///
    /// **Best-effort:** `--storage-opt size=` only works on storage drivers that
    /// support quotas (btrfs/zfs/devicemapper, or overlay2 on xfs+pquota). On
    /// other drivers the cap is skipped with a `log::warn!` rather than failing.
    pub fn storage_limit(mut self, bytes: u64) -> Self {
        self.storage_limit = Some(bytes);
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

    /// The container runtime binary (default `"docker"`; e.g. `"podman"`).
    pub fn runtime(mut self, runtime: impl Into<String>) -> Self {
        self.runtime = runtime.into();
        self
    }

    /// Path to a `bashd` binary built for the **container's** OS/arch (a static
    /// linux-musl binary). The dev escape hatch — CI publishes these per arch;
    /// without one (and with no download configured), [`start`](Self::start)
    /// errors. It is `docker cp`'d into the container at start.
    pub fn bashd_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.bashd_path = Some(path.into());
        self
    }

    /// The running container's name, once [`start`](Self::start)ed.
    pub fn container(&self) -> Option<&str> {
        self.container.as_deref()
    }

    /// Resolve (provisioning if needed) the image the session runs from.
    async fn provision(&self) -> Result<String, BashError> {
        let creates_user = self.user.as_deref().is_some_and(|u| u != "root");
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

    /// Whether the runtime's storage driver supports `--storage-opt size=`.
    async fn storage_quota_supported(&self) -> bool {
        let Ok(out) =
            capture(&self.runtime, ["info", "--format", "{{.Driver}}"]).await
        else {
            return false;
        };
        let driver = String::from_utf8_lossy(&out.stdout);
        matches!(driver.trim(), "btrfs" | "zfs" | "devicemapper")
    }

    /// `docker run -d` the network-isolated session container.
    async fn run_container(&self, image: &str) -> Result<String, BashError> {
        let container = format!("misan-bashd-{}", unique());
        let mut args: Vec<String> = vec![
            "run".into(),
            "-d".into(),
            // tini as PID 1 reaps orphaned grandchildren.
            "--init".into(),
            "--name".into(),
            container.clone(),
            // Publish bashd's port to an ephemeral 127.0.0.1 host port. (Egress
            // isolation — an internal network + a trusted `bashd relay` sidecar
            // — is a follow-up; for now the agent shares the default bridge.)
            "-p".into(),
            format!("127.0.0.1::{BASHD_PORT}"),
            "--workdir".into(),
            self.workdir.clone(),
        ];
        if let Some(mem) = &self.memory_limit {
            args.push("--memory".into());
            args.push(mem.clone());
        }
        if let Some(pids) = self.pids_limit {
            args.push("--pids-limit".into());
            args.push(pids.to_string());
        }
        if let Some(bytes) = self.storage_limit {
            if self.storage_quota_supported().await {
                args.push("--storage-opt".into());
                args.push(format!("size={bytes}"));
            } else {
                #[cfg(feature = "log")]
                log::warn!(
                    "bash sandbox: storage limit ({bytes} bytes) not enforced \
                     — this runtime's storage driver does not support \
                     `--storage-opt size`"
                );
            }
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

    /// `docker cp` the bashd binary into `container` and make it executable.
    async fn inject_bashd(&self, container: &str) -> Result<(), BashError> {
        let bashd = self.bashd_path.as_ref().ok_or_else(|| {
            BashError::Backend(
                "no bashd binary: set DockerSandbox::bashd_path(...) to a \
                 linux bashd built for the container's arch"
                    .to_string(),
            )
        })?;
        let dest = format!("{container}:/usr/local/bin/bashd");
        let cp = Command::new(&self.runtime)
            .arg("cp")
            .arg(bashd)
            .arg(&dest)
            .output()
            .await?;
        if !cp.status.success() {
            return Err(BashError::Backend(format!(
                "docker cp bashd failed: {}",
                String::from_utf8_lossy(&cp.stderr).trim()
            )));
        }
        let _ = capture(
            &self.runtime,
            ["exec", container, "chmod", "+x", "/usr/local/bin/bashd"],
        )
        .await?;
        Ok(())
    }

    /// Launch `bashd --http` (detached, as the run user), discover its published
    /// host port, and poll `GET /` until it's ready. Returns the connection and
    /// the validated handshake.
    async fn launch(
        &self,
        container: &str,
    ) -> Result<(Http, Ready), BashError> {
        // Start the HTTP daemon, detached. `docker exec` runs as root by default
        // and does *not* inherit `docker run --user`, so pass the run user
        // explicitly — the agent's commands then run unprivileged.
        let mut args: Vec<String> = vec!["exec".into(), "-d".into()];
        if let Some(user) = &self.user {
            args.push("--user".into());
            args.push(user.clone());
        }
        args.push(container.into());
        args.push("/usr/local/bin/bashd".into());
        args.push("--http".into());
        args.push(format!("0.0.0.0:{BASHD_PORT}"));
        args.push("--workdir".into());
        args.push(self.workdir.clone());
        if self.persist_cwd {
            args.push("--persist-cwd".into());
        }
        let out = capture(&self.runtime, &args).await?;
        if !out.status.success() {
            return Err(BashError::Backend(format!(
                "could not launch bashd: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }

        // Discover the published host port, e.g. `127.0.0.1:54321`.
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
        let base = format!("http://127.0.0.1:{host_port}");

        let client = reqwest::Client::new();
        let ready = await_ready(&client, &base).await?;
        if ready.protocol != PROTOCOL_VERSION {
            return Err(BashError::Handshake(format!(
                "protocol mismatch: daemon speaks {}, host speaks {}",
                ready.protocol, PROTOCOL_VERSION
            )));
        }
        Ok((Http { client, base }, ready))
    }

    /// Send one [`Command`] over HTTP and aggregate its SSE stream into an
    /// [`ExecResult`].
    async fn request(
        &self,
        command: BashCommand,
    ) -> Result<ExecResult, BashError> {
        let http = self.http.as_ref().ok_or(BashError::NotStarted)?;
        let resp = http
            .client
            .post(format!("{}/run", http.base))
            .json(&command)
            .send()
            .await
            .map_err(|e| BashError::Backend(e.to_string()))?;
        aggregate(resp).await
    }

    /// Remove the container (best-effort), forgetting it so [`Drop`] won't retry.
    async fn remove_container(&mut self) {
        // Drop the HTTP client; bashd dies with the container removed below.
        self.http = None;
        if let Some(container) = self.container.take() {
            let _ = capture(&self.runtime, ["rm", "-f", &container]).await;
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
        let image = self.provision().await?;
        let container = self.run_container(&image).await?;
        // From here on a failure must remove the container, not leak it.
        self.container = Some(container.clone());
        if let Err(e) = self.inject_bashd(&container).await {
            self.remove_container().await;
            return Err(e);
        }
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
    /// (e.g. a panic), remove it with a *blocking* `docker rm -f` (best-effort).
    fn drop(&mut self) {
        if let Some(container) = self.container.take() {
            let _ = std::process::Command::new(&self.runtime)
                .args(["rm", "-f", &container])
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
    fn builders_set_fields() {
        let s = DockerSandbox::alpine()
            .setup("apk add bash")
            .user("agent")
            .workdir("/work")
            .persist_cwd(true)
            .memory("512m")
            .pids_limit(128)
            .runtime("podman")
            .bashd_path("/tmp/bashd");
        assert_eq!(s.base_image, "alpine:3");
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

    /// A `bashd` binary built for the container's arch, from `BASHD_PATH` or the
    /// workspace's `target-linux/release/bashd`. `None` → skip the live test.
    fn bashd_binary() -> Option<PathBuf> {
        if let Ok(p) = std::env::var("BASHD_PATH") {
            let p = PathBuf::from(p);
            return p.exists().then_some(p);
        }
        let p = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../target-linux/release/bashd"
        ));
        p.exists().then_some(p)
    }

    async fn docker_available() -> bool {
        Command::new("docker")
            .arg("version")
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn run(cmd: &str) -> crate::tool::bash::Command {
        crate::tool::bash::Command::Known(crate::tool::bash::Known::Run {
            command: cmd.to_string().into(),
            background: None,
            timeout_secs: None,
        })
    }

    /// End-to-end: provision an Alpine image with a non-root user, run the
    /// network-isolated container, inject + exec bashd, run real commands, and
    /// tear it all down. Deterministic (no model). Skips if Docker or a linux
    /// bashd binary is unavailable.
    #[tokio::test]
    #[ignore = "requires Docker running and a linux bashd (BASHD_PATH)"]
    async fn live_docker_sandbox_runs_commands() {
        let Some(bashd) = bashd_binary() else {
            eprintln!("skipping: no linux bashd binary (set BASHD_PATH)");
            return;
        };
        if !docker_available().await {
            eprintln!("skipping: docker not available");
            return;
        }

        let mut sandbox = DockerSandbox::alpine()
            .setup("apk add --no-cache bash coreutils")
            .user("agent")
            .workdir("/work")
            .bashd_path(bashd);

        let ready = sandbox.start().await.expect("start sandbox");
        assert_eq!(ready.protocol, crate::tool::bash::PROTOCOL_VERSION);
        assert!(sandbox.container().is_some());

        // Commands run in the workdir, as the non-root agent.
        let out = sandbox.exec(run("echo hello && pwd")).await.unwrap();
        assert!(out.stdout.contains("hello"), "stdout: {:?}", out.stdout);
        assert!(out.stdout.contains("/work"), "pwd: {:?}", out.stdout);
        assert_eq!(out.exit, Some(0));

        let out = sandbox.exec(run("whoami")).await.unwrap();
        assert!(out.stdout.contains("agent"), "whoami: {:?}", out.stdout);

        // Exit codes are OS-authoritative.
        let out = sandbox.exec(run("exit 7")).await.unwrap();
        assert_eq!(out.exit, Some(7));

        // NB: egress is currently *open* (the container shares the default
        // bridge so its port can be published). Egress isolation — an internal
        // network + a trusted `bashd relay` sidecar — is a follow-up.

        sandbox.restart().await.expect("restart");
        sandbox.teardown().await.expect("teardown");
        assert!(sandbox.container().is_none());
    }
}
