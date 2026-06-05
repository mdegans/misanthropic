//! Executing one bash command: spawn `bash -lc <cmd>` in its own process group,
//! stream tagged output, and reap it OS-authoritatively.
//!
//! [`run_command`] is the shared runner the HTTP [`server`](crate::server)
//! drives — once per request (foreground) or per background job. Completion and
//! the exit code come from OS process-wait — never an in-band sentinel — so
//! nothing the command prints can fake "I'm done."

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use misanthropic::tool::bash::{
    Chunk, ErrorKind, ProtocolError, Reply, Stream,
};
use nix::sys::signal::{Signal, killpg};
use nix::unistd::Pid;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::sync::mpsc::UnboundedSender;
use tokio::sync::oneshot;

/// Where and how to run one command. The HTTP [`server`](crate::server) builds
/// this per request (foreground) or per background job.
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

/// The terminal facts of a [`run_command`]; the caller composes the
/// [`Outcome`](misanthropic::tool::bash::Outcome) (it owns the
/// `job`/`cursor`/`advice` fields).
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
/// client disconnected, or a background `kill`), the process group is signalled
/// and reaped early.
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

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc;

    /// Drive [`run_command`] for `command`, returning its [`Chunk`] replies and
    /// the terminal [`RunDone`].
    async fn run(
        command: &str,
        timeout_secs: Option<u64>,
        max_output_bytes: usize,
    ) -> (Vec<Reply>, RunDone) {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let done = run_command(
            RunParams {
                shell: Path::new("/bin/bash"),
                cwd: &std::env::temp_dir(),
                command,
                timeout_secs,
                max_output_bytes,
                grace: Duration::from_secs(5),
                cwd_capture: None,
            },
            0,
            &tx,
            None,
        )
        .await;
        drop(tx);
        let mut replies = Vec::new();
        while let Ok(reply) = rx.try_recv() {
            replies.push(reply);
        }
        (replies, done)
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

    #[tokio::test]
    async fn echoes_stdout_and_exit_zero() {
        let (replies, done) = run("echo hello", None, 10 << 20).await;
        assert!(stdout_text(&replies).contains("hello"));
        assert_eq!(done.exit, Some(0));
        assert!(!done.timed_out);
    }

    #[tokio::test]
    async fn nonzero_exit_is_reported() {
        let (_replies, done) = run("exit 3", None, 10 << 20).await;
        assert_eq!(done.exit, Some(3));
    }

    #[tokio::test]
    async fn stderr_is_tagged_separately() {
        let (replies, _done) = run("echo oops 1>&2", None, 10 << 20).await;
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
    async fn output_is_capped() {
        let (replies, done) =
            run("head -c 100000 /dev/zero | tr '\\0' 'x'", None, 16).await;
        assert!(done.truncated, "should have truncated");
        // And we did not buffer the whole 100k.
        assert!(stdout_text(&replies).len() <= 16);
    }

    #[tokio::test]
    async fn times_out_long_command() {
        let (_replies, done) = run("sleep 30", Some(1), 10 << 20).await;
        assert!(done.timed_out, "sleep 30 should time out");
    }

    #[tokio::test]
    async fn captures_final_cwd() {
        let capture = std::env::temp_dir()
            .join(format!("bashd-test-cwd-{}", std::process::id()));
        let _ = std::fs::remove_file(&capture);
        let (tx, _rx) = mpsc::unbounded_channel();
        let done = run_command(
            RunParams {
                shell: Path::new("/bin/bash"),
                cwd: &std::env::temp_dir(),
                command: "cd /",
                timeout_secs: None,
                max_output_bytes: 10 << 20,
                grace: Duration::from_secs(5),
                cwd_capture: Some(&capture),
            },
            0,
            &tx,
            None,
        )
        .await;
        assert_eq!(done.new_cwd.as_deref(), Some(Path::new("/")));
    }
}
