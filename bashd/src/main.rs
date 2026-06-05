//! `bashd` — a persistent-session bash daemon for the misanthropic bash tool.
//!
//! It runs *inside* a sandbox container, owns one persistent shell session, and
//! speaks the [`misanthropic::tool::bash`] newline-delimited JSON protocol over
//! stdio: it reads a [`Request`] per line on stdin and writes [`Reply`] lines on
//! stdout. **stdout is protocol-only** — all diagnostics go to stderr, so they
//! can never corrupt a frame.
//!
//! See [`session`] for how commands are executed.

mod session;

use std::path::PathBuf;
use std::time::Duration;

use clap::Parser;
use misanthropic::tool::bash::{PROTOCOL_VERSION, Ready, Reply, Request};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::mpsc;

use session::Session;

/// Command-line configuration. The host (`DockerSandbox`) sets these when it
/// launches the daemon inside the container.
#[derive(Parser, Debug)]
#[command(
    name = "bashd",
    version,
    about = "Persistent-session bash daemon for the misanthropic bash tool."
)]
struct Args {
    /// Persist the working directory across commands. Off (the default) starts
    /// each command in `--workdir`; on captures each command's final cwd and
    /// reuses it for the next. Off is race-safe under parallel calls.
    #[arg(long)]
    persist_cwd: bool,

    /// Default working directory — where commands start (and the `restart`
    /// reset target).
    #[arg(long, default_value = ".")]
    workdir: PathBuf,

    /// The shell to drive. Run as a login shell (`-lc`) so `~/.profile`
    /// environment setup applies.
    #[arg(long, default_value = "/bin/bash")]
    shell: PathBuf,

    /// Hard per-command output cap in bytes before output is truncated. A
    /// safety ceiling for the daemon; the host applies model-facing limits.
    #[arg(long, default_value_t = 10 << 20)]
    max_output_bytes: usize,

    /// Seconds to wait after SIGTERM before SIGKILL on a timed-out command.
    #[arg(long, default_value_t = 5)]
    grace_secs: u64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // A single writer task owns stdout, so the two per-command stream readers
    // (and the main loop) never interleave a half-written line.
    let (tx, mut rx) = mpsc::unbounded_channel::<Reply>();
    let writer = tokio::spawn(async move {
        let mut out = tokio::io::stdout();
        while let Some(reply) = rx.recv().await {
            match serde_json::to_string(&reply) {
                Ok(line) => {
                    if out.write_all(line.as_bytes()).await.is_err()
                        || out.write_all(b"\n").await.is_err()
                    {
                        break;
                    }
                    let _ = out.flush().await;
                }
                Err(e) => eprintln!("bashd: failed to serialize reply: {e}"),
            }
        }
    });

    // Handshake first, so the host can validate the protocol version.
    tx.send(Reply::Ready {
        ready: Ready {
            protocol: PROTOCOL_VERSION,
            bashd: env!("CARGO_PKG_VERSION").into(),
            shell: args.shell.display().to_string(),
            persist_cwd: args.persist_cwd,
        },
    })?;

    let mut session = Session::new(
        args.workdir,
        args.shell,
        args.persist_cwd,
        args.max_output_bytes,
        Duration::from_secs(args.grace_secs),
    );

    // Serial FIFO: handle each request fully before reading the next. (Parallel
    // / background execution is a later phase.)
    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match serde_json::from_str::<Request>(line) {
            Ok(req) => session.handle(req, &tx).await,
            Err(e) => eprintln!("bashd: ignoring unparseable request: {e}"),
        }
    }

    drop(tx);
    let _ = writer.await;
    Ok(())
}
