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
mod tls;

#[cfg(unix)]
use std::path::PathBuf;
#[cfg(unix)]
use std::time::Duration;

#[cfg(unix)]
use clap::Parser;
#[cfg(unix)]
use misanthropic::tool::bash::TlsServerMaterial;
#[cfg(unix)]
use tokio::io::AsyncReadExt;
#[cfg(unix)]
use zeroize::Zeroize;

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
    /// Serve the HTTPS/SSE front-end on this address (e.g. `0.0.0.0:9099`). The
    /// host reaches it via a published `127.0.0.1` port, over mutual TLS. The
    /// TLS material is read from **stdin** at startup (never argv/env), then the
    /// pipe is closed.
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
impl From<Args> for server::ServeConfig {
    fn from(args: Args) -> Self {
        Self {
            shell: args.shell,
            workdir: args.workdir,
            persist_cwd: args.persist_cwd,
            max_output_bytes: args.max_output_bytes,
            grace: Duration::from_secs(args.grace_secs),
        }
    }
}

/// Read the mutual-TLS material from `input` (stdin: never argv/env/disk). The
/// host writes one JSON object and shuts the pipe; the raw bytes are wiped
/// once parsed.
#[cfg(unix)]
async fn read_material(
    mut input: impl tokio::io::AsyncRead + Unpin,
) -> Result<TlsServerMaterial, Box<dyn std::error::Error>> {
    let mut raw = String::new();
    input.read_to_string(&mut raw).await?;
    let material = serde_json::from_str(&raw);
    raw.zeroize();
    Ok(material?)
}

#[cfg(unix)]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let material = read_material(tokio::io::stdin()).await?;

    let listener = std::net::TcpListener::bind(args.http)?;
    listener.set_nonblocking(true)?;
    eprintln!("bashd: serving HTTPS on {}", listener.local_addr()?);
    server::serve(listener, args.into(), material).await?;
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;

    #[test]
    fn defaults_become_the_serve_config() {
        let args =
            Args::try_parse_from(["bashd", "--http", "0.0.0.0:9099"]).unwrap();
        assert_eq!(args.http.port(), 9099);

        let config = server::ServeConfig::from(args);
        assert_eq!(config.shell, PathBuf::from("/bin/bash"));
        assert_eq!(config.workdir, PathBuf::from("."));
        assert!(!config.persist_cwd);
        assert_eq!(config.max_output_bytes, 10 << 20);
        assert_eq!(config.grace, Duration::from_secs(5));
    }

    #[test]
    fn flags_override_the_defaults() {
        let args = Args::try_parse_from([
            "bashd",
            "--http",
            "127.0.0.1:1",
            "--persist-cwd",
            "--workdir",
            "/home/agent",
            "--shell",
            "/bin/sh",
            "--max-output-bytes",
            "64",
            "--grace-secs",
            "1",
        ])
        .unwrap();

        let config = server::ServeConfig::from(args);
        assert_eq!(config.shell, PathBuf::from("/bin/sh"));
        assert_eq!(config.workdir, PathBuf::from("/home/agent"));
        assert!(config.persist_cwd);
        assert_eq!(config.max_output_bytes, 64);
        assert_eq!(config.grace, Duration::from_secs(1));
    }

    #[test]
    fn http_is_required() {
        assert!(Args::try_parse_from(["bashd"]).is_err());
    }

    #[tokio::test]
    async fn material_is_read_from_input() {
        let json = br#"{"cert_chain_pem":"c","key_pem":"k","ca_pem":"a"}"#;
        let material = read_material(&json[..]).await.unwrap();
        assert_eq!(material.cert_chain_pem, "c");
        assert_eq!(material.key_pem, "k");
        assert_eq!(material.ca_pem, "a");
    }

    #[tokio::test]
    async fn malformed_material_is_an_error() {
        assert!(read_material(&b"not json"[..]).await.is_err());
    }
}
