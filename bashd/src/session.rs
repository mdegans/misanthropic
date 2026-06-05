//! The persistent bash session: turns [`Request`]s into child processes and
//! streams their output back as [`Reply`]s.
//!
//! Each command is its own child (`bash -lc <cmd>`, a *login* shell so
//! `~/.profile` env applies), in its own process group. Completion and the exit
//! code come from OS process-wait — never an in-band sentinel — so nothing the
//! command prints can fake "I'm done." Commands run **serially** (one at a time)
//! in Phase 1; `background`/`poll`/`kill` are recognized but reported
//! unsupported.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use misanthropic::tool::bash::{
    Chunk, Command, ErrorKind, Known, Outcome, ProtocolError, Reply, Request,
    Stream,
};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

/// A persistent bash session. Holds the working directory it carries across
/// commands (when `persist_cwd` is on) and the limits applied to each run.
pub struct Session {
    /// Where commands start when not persisting cwd, and the reset target.
    workdir: PathBuf,
    /// The current working directory (tracked only when `persist_cwd`).
    cwd: PathBuf,
    /// The shell binary to drive (run with `-lc`).
    shell: PathBuf,
    /// Whether to carry the working directory across commands.
    persist_cwd: bool,
    /// Hard per-command output cap before truncation.
    max_output_bytes: usize,
    /// Grace period after SIGTERM before SIGKILL on a timed-out command.
    grace: Duration,
    /// A random base for the private cwd-capture temp paths.
    cwd_base: String,
    /// Monotonic counter making each capture path unique.
    seq: u64,
}

impl Session {
    /// A new session rooted at `workdir`.
    pub fn new(
        workdir: PathBuf,
        shell: PathBuf,
        persist_cwd: bool,
        max_output_bytes: usize,
        grace: Duration,
    ) -> Self {
        Self {
            cwd: workdir.clone(),
            workdir,
            shell,
            persist_cwd,
            max_output_bytes,
            grace,
            cwd_base: random_base(),
            seq: 0,
        }
    }

    /// Dispatch one [`Request`], sending its [`Reply`]s on `tx`.
    pub async fn handle(&mut self, req: Request, tx: &UnboundedSender<Reply>) {
        let id = req.id;
        match req.command {
            Command::Known(Known::Restart { .. }) => {
                self.cwd = self.workdir.clone();
                let _ = tx.send(Reply::Outcome(Outcome {
                    id,
                    exit: Some(0),
                    ..Default::default()
                }));
            }
            Command::Known(Known::Run {
                command,
                background,
                timeout_secs,
            }) => {
                if background == Some(true) {
                    let _ = tx.send(unsupported(
                        id,
                        "background execution is not supported yet",
                    ));
                } else {
                    self.run(id, &command, timeout_secs, tx).await;
                }
            }
            Command::Known(Known::Poll { .. }) => {
                let _ = tx.send(unsupported(
                    id,
                    "poll is not supported yet (no background jobs)",
                ));
            }
            Command::Known(Known::Kill { .. }) => {
                let _ = tx.send(unsupported(
                    id,
                    "kill is not supported yet (no background jobs)",
                ));
            }
            Command::Unknown { .. } => {
                let _ = tx.send(unsupported(id, "unknown bash command"));
            }
        }
    }

    /// Run one foreground command to completion, streaming its output, then
    /// send the terminal [`Outcome`]. Delegates to the shared [`run_command`].
    async fn run(
        &mut self,
        id: u64,
        command: &str,
        timeout_secs: Option<u64>,
        tx: &UnboundedSender<Reply>,
    ) {
        self.seq += 1;
        // When persisting cwd, capture the command's final $PWD out-of-band on a
        // private temp path (preserving the command's own exit code). The path
        // is randomized; worst case an adversarial command corrupts its *own*
        // next cwd — the exit code stays OS-authoritative either way.
        let cwd_path = self.persist_cwd.then(|| {
            PathBuf::from(format!(
                "/tmp/bashd-cwd-{}-{}-{}",
                self.cwd_base,
                std::process::id(),
                self.seq
            ))
        });

        let done = run_command(
            RunParams {
                shell: &self.shell,
                cwd: &self.cwd,
                command,
                timeout_secs,
                max_output_bytes: self.max_output_bytes,
                grace: self.grace,
                cwd_capture: cwd_path.as_deref(),
            },
            id,
            tx,
            None,
        )
        .await;

        if let Some(error) = done.spawn_error {
            let _ = tx.send(Reply::Outcome(Outcome {
                id,
                error: Some(error),
                ..Default::default()
            }));
            return;
        }
        if let Some(cwd) = done.new_cwd {
            self.cwd = cwd;
        }
        let _ = tx.send(Reply::Outcome(Outcome {
            id,
            exit: done.exit,
            running: false,
            timed_out: done.timed_out,
            truncated: done.truncated,
            ..Default::default()
        }));
    }
}

/// Where and how to run one command, independent of session bookkeeping — so the
/// stdio loop ([`Session::run`]) and the HTTP server ([`crate::server`]) share
/// one runner.
pub(crate) struct RunParams<'a> {
    /// The shell to drive, run as a login shell (`-lc`).
    pub shell: &'a Path,
    /// The directory the command starts in.
    pub cwd: &'a Path,
    /// The shell command to run.
    pub command: &'a str,
    /// Kill (and report a timeout) after this many seconds, if `Some(>0)`.
    pub timeout_secs: Option<u64>,
    /// Hard per-command output cap before truncation.
    pub max_output_bytes: usize,
    /// Grace period after SIGTERM before SIGKILL.
    pub grace: Duration,
    /// When `Some`, capture the command's final `$PWD` to this private path.
    pub cwd_capture: Option<&'a Path>,
}

/// The terminal facts of a [`run_command`]; the caller composes the [`Outcome`]
/// (it owns the `job`/`cursor`/`advice` fields).
pub(crate) struct RunDone {
    /// The exit code, if the process exited normally.
    pub exit: Option<i32>,
    /// Whether the command was killed for exceeding its timeout.
    pub timed_out: bool,
    /// Whether output was truncated at the byte cap.
    pub truncated: bool,
    /// The captured final cwd, if `cwd_capture` was set and a valid directory.
    pub new_cwd: Option<PathBuf>,
    /// A spawn failure, if the child never started.
    pub spawn_error: Option<ProtocolError>,
}

/// Run one command to completion, streaming tagged [`Chunk`]s on `tx`. Returns
/// the terminal facts. If `cancel` fires (the consumer dropped — e.g. an SSE
/// client disconnected), the process group is signalled and reaped early.
pub(crate) async fn run_command(
    params: RunParams<'_>,
    id: u64,
    tx: &UnboundedSender<Reply>,
    cancel: Option<oneshot::Receiver<()>>,
) -> RunDone {
    let script = match params.cwd_capture {
        Some(p) => format!(
            "{}\n__bashd_ec=$?; pwd > {} 2>/dev/null; exit $__bashd_ec",
            params.command,
            single_quote(&p.to_string_lossy())
        ),
        None => params.command.to_string(),
    };

    // Build via std so we can set the process group with the *safe*
    // `process_group(0)` (no `pre_exec`/unsafe), then adopt into tokio.
    use std::os::unix::process::CommandExt;
    let mut std_cmd = std::process::Command::new(params.shell);
    std_cmd
        .arg("-lc")
        .arg(&script)
        .current_dir(params.cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    std_cmd.process_group(0);
    let mut cmd = tokio::process::Command::from(std_cmd);
    cmd.kill_on_drop(true);

    let mut child = match cmd.spawn() {
        Ok(child) => child,
        Err(e) => {
            return RunDone {
                exit: None,
                timed_out: false,
                truncated: false,
                new_cwd: None,
                spawn_error: Some(ProtocolError {
                    kind: ErrorKind::Spawn,
                    message: format!("failed to spawn {:?}: {e}", params.shell),
                }),
            };
        }
    };
    // The child leads its own group (pgid == pid), so signalling the pgid
    // reaches the whole tree it spawns.
    let pgid = child.id().map(|p| Pid::from_raw(p as i32));

    let budget = Arc::new(AtomicUsize::new(params.max_output_bytes));
    let stdout = child.stdout.take().expect("stdout is piped");
    let stderr = child.stderr.take().expect("stderr is piped");
    let out_task = tokio::spawn(read_stream(
        id,
        Stream::Stdout,
        stdout,
        tx.clone(),
        budget.clone(),
    ));
    let err_task = tokio::spawn(read_stream(
        id,
        Stream::Stderr,
        stderr,
        tx.clone(),
        budget.clone(),
    ));

    let mut timed_out = false;
    let status =
        wait_for(&mut child, pgid, &params, cancel, &mut timed_out).await;

    // Both read tasks finish once the pipes hit EOF (the child is reaped),
    // so all chunks are enqueued before the caller sends the terminal Outcome.
    let out_truncated = out_task.await.unwrap_or(false);
    let err_truncated = err_task.await.unwrap_or(false);
    let exit = status.ok().and_then(|s| s.code());

    let new_cwd = params.cwd_capture.and_then(|p| {
        let captured = std::fs::read_to_string(p).ok().and_then(|c| {
            let trimmed = c.trim();
            (!trimmed.is_empty())
                .then(|| PathBuf::from(trimmed))
                .filter(|d| d.is_dir())
        });
        let _ = std::fs::remove_file(p);
        captured
    });

    RunDone {
        exit,
        timed_out,
        truncated: out_truncated || err_truncated,
        new_cwd,
        spawn_error: None,
    }
}

/// Await the child, honoring an optional timeout and an optional cancel signal.
/// On either, signal the group SIGTERM→grace→SIGKILL and reap. Sets `timed_out`
/// when it was the timeout (not a cancel) that fired.
async fn wait_for(
    child: &mut tokio::process::Child,
    pgid: Option<Pid>,
    params: &RunParams<'_>,
    cancel: Option<oneshot::Receiver<()>>,
    timed_out: &mut bool,
) -> std::io::Result<std::process::ExitStatus> {
    let timeout = async {
        match params.timeout_secs {
            Some(secs) if secs > 0 => {
                tokio::time::sleep(Duration::from_secs(secs)).await
            }
            _ => std::future::pending::<()>().await,
        }
    };
    let cancelled = async {
        match cancel {
            Some(rx) => {
                let _ = rx.await;
            }
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(timeout, cancelled);

    tokio::select! {
        status = child.wait() => return status,
        _ = &mut timeout => *timed_out = true,
        _ = &mut cancelled => {}
    }

    // Timed out or cancelled: TERM, grace, KILL, reap.
    if let Some(pgid) = pgid {
        let _ = killpg(pgid, Signal::SIGTERM);
    }
    match tokio::time::timeout(params.grace, child.wait()).await {
        Ok(status) => status,
        Err(_) => {
            if let Some(pgid) = pgid {
                let _ = killpg(pgid, Signal::SIGKILL);
            }
            child.wait().await
        }
    }
}

/// An [`Outcome`] reporting an op the Phase-1 daemon does not implement.
fn unsupported(id: u64, message: &str) -> Reply {
    Reply::Outcome(Outcome {
        id,
        error: Some(ProtocolError {
            kind: ErrorKind::Unsupported,
            message: message.to_string(),
        }),
        ..Default::default()
    })
}

/// Drain a child stream, emitting [`Chunk`]s until EOF and stopping forwarding
/// once the shared `budget` is exhausted (but still draining, so the child never
/// blocks on a full pipe). Returns whether output was truncated.
async fn read_stream<R: AsyncRead + Unpin>(
    id: u64,
    stream: Stream,
    mut reader: R,
    tx: UnboundedSender<Reply>,
    budget: Arc<AtomicUsize>,
) -> bool {
    let mut buf = vec![0u8; 8192];
    let mut truncated = false;
    loop {
        match reader.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                let mut take = 0usize;
                let _ = budget.fetch_update(
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                    |remaining| {
                        take = n.min(remaining);
                        Some(remaining - take)
                    },
                );
                if take > 0 {
                    let data =
                        String::from_utf8_lossy(&buf[..take]).into_owned();
                    let _ = tx.send(Reply::Chunk(Chunk { id, stream, data }));
                }
                if take < n {
                    truncated = true;
                }
            }
        }
    }
    truncated
}

/// Single-quote a string for safe interpolation into a `bash -lc` script.
fn single_quote(s: &str) -> String {
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// 16 hex chars from `/dev/urandom` (zeros if unreadable — the path is not a
/// security boundary, only a private capture channel).
fn random_base() -> String {
    let mut buf = [0u8; 8];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        let _ = f.read_exact(&mut buf);
    }
    buf.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    fn session() -> Session {
        Session::new(
            std::env::temp_dir(),
            PathBuf::from("/bin/bash"),
            false,
            10 << 20,
            Duration::from_secs(5),
        )
    }

    fn run_req(id: u64, command: &str) -> Request {
        Request {
            id,
            command: Command::Known(Known::Run {
                command: command.to_string().into(),
                background: None,
                timeout_secs: None,
            }),
        }
    }

    /// Drive one request and collect every reply it produced.
    async fn drive(session: &mut Session, req: Request) -> Vec<Reply> {
        let (tx, mut rx) = mpsc::unbounded_channel();
        session.handle(req, &tx).await;
        drop(tx);
        let mut out = Vec::new();
        while let Ok(reply) = rx.try_recv() {
            out.push(reply);
        }
        out
    }

    fn stdout_text(replies: &[Reply]) -> String {
        replies
            .iter()
            .filter_map(|r| match r {
                Reply::Chunk(c) if matches!(c.stream, Stream::Stdout) => {
                    Some(c.data.as_str())
                }
                _ => None,
            })
            .collect()
    }

    fn outcome(replies: &[Reply]) -> &Outcome {
        replies
            .iter()
            .find_map(|r| match r {
                Reply::Outcome(o) => Some(o),
                _ => None,
            })
            .expect("an Outcome")
    }

    #[tokio::test]
    async fn run_echoes_stdout_and_exit_zero() {
        let mut s = session();
        let replies = drive(&mut s, run_req(1, "echo hello")).await;
        assert!(stdout_text(&replies).contains("hello"));
        let o = outcome(&replies);
        assert_eq!(o.id, 1);
        assert_eq!(o.exit, Some(0));
        assert!(!o.timed_out);
    }

    #[tokio::test]
    async fn nonzero_exit_is_reported() {
        let mut s = session();
        let replies = drive(&mut s, run_req(2, "exit 3")).await;
        assert_eq!(outcome(&replies).exit, Some(3));
    }

    #[tokio::test]
    async fn stderr_is_tagged_separately() {
        let mut s = session();
        let replies = drive(&mut s, run_req(3, "echo oops 1>&2")).await;
        let err: String = replies
            .iter()
            .filter_map(|r| match r {
                Reply::Chunk(c) if matches!(c.stream, Stream::Stderr) => {
                    Some(c.data.as_str())
                }
                _ => None,
            })
            .collect();
        assert!(err.contains("oops"), "stderr was: {err:?}");
    }

    #[tokio::test]
    async fn restart_resets_and_reports_ok() {
        let mut s = session();
        let replies = drive(&mut s, run_req(4, "true")).await;
        assert_eq!(outcome(&replies).exit, Some(0));
        let replies = drive(
            &mut s,
            Request {
                id: 5,
                command: Command::Known(Known::Restart { restart: true }),
            },
        )
        .await;
        let o = outcome(&replies);
        assert_eq!(o.exit, Some(0));
        assert!(o.error.is_none());
    }

    #[tokio::test]
    async fn poll_kill_background_are_unsupported() {
        let mut s = session();
        for command in [
            Command::Known(Known::Poll { poll: 1 }),
            Command::Known(Known::Kill { kill: 1 }),
            Command::Known(Known::Run {
                command: "sleep 1".into(),
                background: Some(true),
                timeout_secs: None,
            }),
        ] {
            let replies = drive(&mut s, Request { id: 9, command }).await;
            let o = outcome(&replies);
            assert!(
                matches!(
                    o.error.as_ref().map(|e| &e.kind),
                    Some(ErrorKind::Unsupported)
                ),
                "expected Unsupported, got {:?}",
                o.error
            );
        }
    }

    #[tokio::test]
    async fn output_is_capped() {
        let mut s = Session::new(
            std::env::temp_dir(),
            PathBuf::from("/bin/bash"),
            false,
            16, // tiny cap
            Duration::from_secs(5),
        );
        let replies = drive(
            &mut s,
            run_req(6, "head -c 100000 /dev/zero | tr '\\0' 'x'"),
        )
        .await;
        assert!(outcome(&replies).truncated, "should have truncated");
        // And we did not buffer the whole 100k.
        assert!(stdout_text(&replies).len() <= 16);
    }

    #[tokio::test]
    async fn times_out_long_command() {
        let mut s = session();
        let replies = drive(
            &mut s,
            Request {
                id: 7,
                command: Command::Known(Known::Run {
                    command: "sleep 30".into(),
                    background: None,
                    timeout_secs: Some(1),
                }),
            },
        )
        .await;
        assert!(outcome(&replies).timed_out, "sleep 30 should time out");
    }

    #[tokio::test]
    async fn persist_cwd_carries_directory() {
        let tmp = std::env::temp_dir();
        let mut s = Session::new(
            tmp.clone(),
            PathBuf::from("/bin/bash"),
            true, // persist
            10 << 20,
            Duration::from_secs(5),
        );
        // cd somewhere that exists, then a separate command sees it.
        drive(&mut s, run_req(10, "cd /")).await;
        let replies = drive(&mut s, run_req(11, "pwd")).await;
        let pwd = stdout_text(&replies);
        assert!(pwd.trim_end().ends_with('/'), "pwd after cd / was {pwd:?}");
    }
}
