//! [`DockerSandbox`] — the reference [`BashSandbox`] that runs [`bashd`] inside a
//! Docker (or Podman) container.
//!
//! Lifecycle ([`start`](DockerSandbox::start)): optionally **provision** a custom
//! image (run a `setup` script *with* network, plus create the run user, then
//! `commit`), **run** a session container with `--network none` and resource
//! caps, **inject** the `bashd` binary (`docker cp`), and **exec** it to open the
//! persistent stdio connection, validating the [`Ready`] handshake. Each
//! [`exec`](DockerSandbox::exec) writes one [`Request`] and aggregates the
//! [`Reply`]s back into an [`ExecResult`]. [`teardown`](DockerSandbox::teardown)
//! (and a blocking [`Drop`] leak-guard) removes the container.
//!
//! Phase-1 scope: the happy path. Home backup/restore, the disk-cap mechanism,
//! and `restart`-drops-a-borked-home are hardened in later phases.
//!
//! [`bashd`]: super::Request

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines};
use tokio::process::{ChildStdin, ChildStdout, Command};

use super::{
    BashError, BashSandbox, Command as BashCommand, ExecResult,
    PROTOCOL_VERSION, Ready, Reply, Request, Stream,
};

/// The live connection to a running `bashd`: the `docker exec -i bashd` child,
/// its stdio, and the next request id.
struct Daemon {
    child: tokio::process::Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    next_id: u64,
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
    daemon: Option<Daemon>,
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
            daemon: None,
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
            "--name".into(),
            container.clone(),
            "--network".into(),
            "none".into(),
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

    /// `docker exec -i bashd` and read its [`Ready`] handshake.
    async fn connect(&self, container: &str) -> Result<Daemon, BashError> {
        let persist = self.persist_cwd;
        let mut cmd = Command::new(&self.runtime);
        cmd.arg("exec").arg("-i");
        // `docker exec` runs as root by default — it does *not* inherit the
        // container's `docker run --user`. Pass the run user explicitly so the
        // agent's commands run unprivileged.
        if let Some(user) = &self.user {
            cmd.arg("--user").arg(user);
        }
        cmd.arg(container)
            .arg("/usr/local/bin/bashd")
            .arg("--workdir")
            .arg(&self.workdir);
        if persist {
            cmd.arg("--persist-cwd");
        }
        cmd.stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().expect("stdin piped");
        let stdout =
            BufReader::new(child.stdout.take().expect("stdout piped")).lines();
        let mut daemon = Daemon {
            child,
            stdin,
            stdout,
            next_id: 1,
        };

        let line = daemon.stdout.next_line().await?.ok_or_else(|| {
            BashError::Handshake("bashd sent no output".into())
        })?;
        let reply: Reply = serde_json::from_str(&line)
            .map_err(|e| BashError::Handshake(format!("{e}: {line}")))?;
        let ready = match reply {
            Reply::Ready { ready } => ready,
            other => {
                return Err(BashError::Handshake(format!(
                    "expected Ready, got {other:?}"
                )));
            }
        };
        if ready.protocol != PROTOCOL_VERSION {
            return Err(BashError::Handshake(format!(
                "protocol mismatch: daemon speaks {}, host speaks {}",
                ready.protocol, PROTOCOL_VERSION
            )));
        }
        Ok(daemon)
    }

    /// Send one [`Request`] and aggregate its [`Reply`]s into an [`ExecResult`].
    async fn request(
        &mut self,
        command: BashCommand,
    ) -> Result<ExecResult, BashError> {
        let daemon = self.daemon.as_mut().ok_or(BashError::NotStarted)?;
        let id = daemon.next_id;
        daemon.next_id += 1;

        let req = Request { id, command };
        let line = serde_json::to_string(&req)
            .map_err(|e| BashError::Protocol(e.to_string()))?;
        daemon.stdin.write_all(line.as_bytes()).await?;
        daemon.stdin.write_all(b"\n").await?;
        daemon.stdin.flush().await?;

        let mut result = ExecResult::default();
        loop {
            let line = daemon.stdout.next_line().await?.ok_or_else(|| {
                BashError::Protocol("bashd closed the connection".into())
            })?;
            let reply: Reply = serde_json::from_str(&line)
                .map_err(|e| BashError::Protocol(format!("{e}: {line}")))?;
            match reply {
                Reply::Chunk(c) if c.id == id => match c.stream {
                    Stream::Stdout => result.stdout.push_str(&c.data),
                    Stream::Stderr => result.stderr.push_str(&c.data),
                },
                Reply::Outcome(o) if o.id == id => {
                    if let Some(err) = o.error {
                        return Err(BashError::Protocol(err.message));
                    }
                    result.exit = o.exit;
                    result.running = o.running;
                    result.timed_out = o.timed_out;
                    result.truncated = o.truncated;
                    result.job = o.job;
                    result.advice = o.advice;
                    break;
                }
                // Stray replies (a different id, or a late Ready) are ignored.
                _ => {}
            }
        }
        Ok(result)
    }

    /// Remove the container (best-effort), forgetting it so [`Drop`] won't retry.
    async fn remove_container(&mut self) {
        if let Some(mut daemon) = self.daemon.take() {
            // End the `exec -i bashd` child explicitly (its `kill_on_drop` is a
            // backstop). bashd also exits on its own once `docker rm -f` below
            // tears the container out from under it.
            let _ = daemon.child.start_kill();
        }
        if let Some(container) = self.container.take() {
            let _ = capture(&self.runtime, ["rm", "-f", &container]).await;
        }
    }
}

#[async_trait::async_trait]
impl BashSandbox for DockerSandbox {
    async fn start(&mut self) -> Result<Ready, BashError> {
        if self.daemon.is_some() {
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
        match self.connect(&container).await {
            Ok(daemon) => {
                let ready = Ready {
                    protocol: PROTOCOL_VERSION,
                    bashd: env!("CARGO_PKG_VERSION").into(),
                    shell: "/bin/bash".into(),
                    persist_cwd: self.persist_cwd,
                };
                self.daemon = Some(daemon);
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

        // Network is isolated (--network none): a connect attempt fails.
        let out = sandbox
            .exec(run("ping -c1 -W1 1.1.1.1 >/dev/null 2>&1; echo $?"))
            .await
            .unwrap();
        assert!(
            out.stdout.trim() != "0",
            "network should be isolated, got: {:?}",
            out.stdout
        );

        sandbox.restart().await.expect("restart");
        sandbox.teardown().await.expect("teardown");
        assert!(sandbox.container().is_none());
    }
}
