//! `bashd` — a persistent-session bash daemon for the misanthropic bash tool.
//!
//! It runs *inside* a sandbox container, owns the session, and serves the
//! [`misanthropic::tool::bash`] HTTP/SSE protocol (see [`server`]) on a TCP port
//! the host reaches via a published `127.0.0.1` mapping. See [`session`] for how
//! individual commands are executed.
//!
//! bashd is **unix-only** (it manages process groups and signals). On non-unix
//! it compiles to a stub `main` that exits with an error, so the workspace still
//! builds on those targets.

#[cfg(not(unix))]
fn main() {
    eprintln!(
        "bashd runs only on unix — it manages process groups and signals."
    );
    std::process::exit(1);
}

#[cfg(unix)]
mod server;
#[cfg(unix)]
mod session;

#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::time::Duration;

#[cfg(unix)]
use clap::Parser;

/// Command-line configuration. The host (`DockerSandbox`) sets these when it
/// launches the daemon inside the container.
#[cfg(unix)]
#[derive(Parser, Debug)]
#[command(
    name = "bashd",
    version,
    about = "Persistent-session bash daemon for the misanthropic bash tool."
)]
struct Args {
    /// Serve the HTTP/SSE front-end on this address (e.g. `0.0.0.0:9099`). The
    /// host reaches it via a published `127.0.0.1` port.
    #[arg(long)]
    http: std::net::SocketAddr,

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

#[cfg(unix)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let listener = tokio::net::TcpListener::bind(args.http).await?;
    eprintln!("bashd: serving HTTP on {}", listener.local_addr()?);
    server::serve(
        listener,
        server::ServeConfig {
            shell: args.shell,
            workdir: args.workdir,
            persist_cwd: args.persist_cwd,
            max_output_bytes: args.max_output_bytes,
            grace: Duration::from_secs(args.grace_secs),
        },
    )
    .await?;
    Ok(())
}
