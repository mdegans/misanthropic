//! The HTTP/SSE front-end: an [axum] router over a TCP listener that turns the
//! same [`Session`](crate::session) command logic into per-request streams and
//! background jobs.
//!
//! The host reaches this over a published `127.0.0.1` port (see the misanthropic
//! `tool::bash::docker` sandbox). Every command is its own request — *connection
//! = correlation* — so concurrency is just independent handlers and there is no
//! single-pipe demux.
//!
//! - `GET /`                → a bare [`Ready`] handshake.
//! - `POST /run` (Command)  → foreground: an SSE stream of [`event::CHUNK`]s then
//!   a terminal [`event::OUTCOME`]; background: spawns a job writing output to a
//!   container-side file and emits a single outcome (`job` set) immediately.
//! - `GET /jobs/{id}`       → `poll`: chunks since `?cursor=` then an outcome.
//! - `GET /jobs/{id}/wait`  → like poll, but follows the job to completion
//!   (soft `?timeout=` → terminal `running: true`).
//! - `POST /jobs/{id}/kill` → signal the job's group (TERM→grace→KILL).
//! - `POST /killall`        → signal every live job (teardown helper).

use std::collections::HashMap;
use std::convert::Infallible;
use std::io::SeekFrom;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use async_stream::stream;
use axum::extract::{Path as UrlPath, Query, State};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use misanthropic::tool::bash::{
    Command, ErrorKind, Known, Outcome, PROTOCOL_VERSION, ProtocolError, Ready,
    Reply, TlsServerMaterial, event,
};
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::{mpsc, oneshot, watch};

use crate::session::{RunParams, run_command};

/// Static configuration the server runs commands under (from the daemon's CLI).
pub struct ServeConfig {
    /// The shell to drive, run as a login shell (`-lc`).
    pub shell: PathBuf,
    /// Where commands start (and the `restart` reset target).
    pub workdir: PathBuf,
    /// Whether foreground commands carry the working directory across calls.
    pub persist_cwd: bool,
    /// Hard per-command output cap before truncation.
    pub max_output_bytes: usize,
    /// Grace period after SIGTERM before SIGKILL.
    pub grace: Duration,
}

/// Shared state behind every handler.
pub struct AppState {
    shell: PathBuf,
    workdir: PathBuf,
    persist_cwd: bool,
    max_output_bytes: usize,
    grace: Duration,
    /// The persist-cwd working directory, carried across foreground commands.
    /// Locked only to read/write the path, never across the command itself.
    cwd: Mutex<PathBuf>,
    /// Live background jobs, keyed by id.
    jobs: Mutex<HashMap<u64, JobEntry>>,
    /// Where background jobs stream their output, and cwd-capture temp files go.
    jobs_dir: PathBuf,
    /// Allocates background job ids (monotonic, starting at 1).
    next_job: AtomicU64,
    /// Makes cwd-capture temp paths unique.
    seq: AtomicU64,
}

impl AppState {
    /// A new state, creating a private (per-instance) jobs directory.
    pub fn new(config: ServeConfig) -> std::io::Result<Self> {
        static INSTANCE: AtomicU64 = AtomicU64::new(0);
        let jobs_dir = std::env::temp_dir().join(format!(
            "bashd-jobs-{}-{}",
            std::process::id(),
            INSTANCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&jobs_dir)?;
        Ok(Self {
            shell: config.shell,
            workdir: config.workdir.clone(),
            persist_cwd: config.persist_cwd,
            max_output_bytes: config.max_output_bytes,
            grace: config.grace,
            cwd: Mutex::new(config.workdir),
            jobs: Mutex::new(HashMap::new()),
            jobs_dir,
            next_job: AtomicU64::new(0),
            seq: AtomicU64::new(0),
        })
    }
}

/// The shared, lock-free handles a `poll`/`wait` reads after dropping the
/// registry lock.
struct JobShared {
    /// NDJSON [`Reply`] lines the background drainer appends to.
    out_path: PathBuf,
    /// `Some(terminal outcome)` once the job has finished, else `None`.
    done: watch::Receiver<Option<Outcome>>,
}

/// A live background job in the registry.
struct JobEntry {
    shared: Arc<JobShared>,
    /// Firing (or dropping) this signals the runner to TERM→grace→KILL the
    /// group — the same path an SSE disconnect uses. Taken by `kill`/`killall`.
    cancel: Option<oneshot::Sender<()>>,
    /// The drainer task; kept so it isn't detached (and is reaped on teardown).
    _handle: tokio::task::JoinHandle<()>,
}

/// Lock a [`Mutex`], recovering from poisoning (impossible under `panic=abort`,
/// but this keeps the handlers panic-free regardless).
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The router, with `state` installed.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(handshake))
        .route("/run", post(run))
        .route("/jobs/{id}", get(poll))
        .route("/jobs/{id}/wait", get(wait))
        .route("/jobs/{id}/kill", post(kill))
        .route("/killall", post(killall))
        .with_state(state)
}

/// Serve the router on `listener` over mutual TLS until the process ends. `tls`
/// is the per-container PKI the host fed in over stdin; `bashd` presents the
/// server cert and requires a client cert chaining to the same CA.
pub async fn serve(
    listener: std::net::TcpListener,
    config: ServeConfig,
    tls: TlsServerMaterial,
) -> std::io::Result<()> {
    let state = Arc::new(AppState::new(config)?);
    let tls_config = crate::tls::server_config(&tls)?;
    // rustls now owns parsed copies of the certs/key; wipe the source PEMs.
    drop(tls);
    let rustls = axum_server::tls_rustls::RustlsConfig::from_config(tls_config);
    axum_server::from_tcp_rustls(listener, rustls)?
        .serve(router(state).into_make_service())
        .await
}

/// `GET /` — the readiness/handshake the host polls with backoff.
async fn handshake(State(s): State<Arc<AppState>>) -> Json<Ready> {
    Json(Ready {
        protocol: PROTOCOL_VERSION,
        bashd: env!("CARGO_PKG_VERSION").into(),
        shell: s.shell.display().to_string(),
        persist_cwd: s.persist_cwd,
    })
}

/// `POST /run` — run a [`Command`] (foreground stream, background receipt, or a
/// `restart`).
async fn run(
    State(s): State<Arc<AppState>>,
    Json(command): Json<Command>,
) -> Response {
    match command {
        Command::Known(Known::Run {
            command,
            background,
            timeout_secs,
        }) => {
            let command = command.into_owned();
            if background == Some(true) {
                let id = spawn_job(&s, command, timeout_secs);
                sse_one(Outcome {
                    job: Some(id),
                    running: true,
                    ..Default::default()
                })
            } else {
                run_foreground(&s, command, timeout_secs)
            }
        }
        // `restart` clears any background jobs and resets the persist cwd.
        Command::Known(Known::Restart { .. }) => {
            signal_all(&s);
            *lock(&s.cwd) = s.workdir.clone();
            sse_one(Outcome {
                running: false,
                exit: Some(0),
                ..Default::default()
            })
        }
        // poll/kill have their own endpoints; the host never routes them here.
        other => sse_one(Outcome {
            running: false,
            error: Some(ProtocolError {
                kind: ErrorKind::Unsupported,
                message: format!("/run does not accept {other:?}"),
            }),
            ..Default::default()
        }),
    }
}

/// Run a foreground command, streaming its output as SSE. Dropping the response
/// (client disconnect) drops the held `cancel`, which reaps the process group.
fn run_foreground(
    s: &Arc<AppState>,
    command: String,
    timeout_secs: Option<u64>,
) -> Response {
    let (tx, mut rx) = mpsc::unbounded_channel::<Reply>();
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();

    let state = s.clone();
    let shell = s.shell.clone();
    let persist = s.persist_cwd;
    let max = s.max_output_bytes;
    let grace = s.grace;
    let start_cwd = if persist {
        lock(&s.cwd).clone()
    } else {
        s.workdir.clone()
    };
    let cwd_capture = persist.then(|| {
        let n = s.seq.fetch_add(1, Ordering::Relaxed);
        s.jobs_dir.join(format!("cwd-{}-{n}", std::process::id()))
    });

    tokio::spawn(async move {
        let done = run_command(
            RunParams {
                shell: &shell,
                cwd: &start_cwd,
                command: &command,
                timeout_secs,
                max_output_bytes: max,
                grace,
                cwd_capture: cwd_capture.as_deref(),
            },
            0,
            &tx,
            Some(cancel_rx),
        )
        .await;
        if persist && let Some(cwd) = &done.new_cwd {
            *lock(&state.cwd) = cwd.clone();
        }
        let _ = tx.send(Reply::Outcome(outcome_from(done)));
    });

    let body = stream! {
        // Hold `cancel`: dropping this generator (client disconnect or normal
        // end) drops it, which the runner observes to reap an in-flight child.
        let _cancel = cancel_tx;
        while let Some(reply) = rx.recv().await {
            yield Ok::<Event, Infallible>(reply_event(&reply));
        }
    };
    Sse::new(body).into_response()
}

/// Spawn a background job: it streams output to a container-side file and flips
/// its `done` watch when finished. Returns its id.
fn spawn_job(
    s: &Arc<AppState>,
    command: String,
    timeout_secs: Option<u64>,
) -> u64 {
    let id = s.next_job.fetch_add(1, Ordering::Relaxed) + 1;
    let out_path = s.jobs_dir.join(format!("{id}.ndjson"));
    let (done_tx, done_rx) = watch::channel(None::<Outcome>);
    let (cancel_tx, cancel_rx) = oneshot::channel::<()>();
    let shared = Arc::new(JobShared {
        out_path: out_path.clone(),
        done: done_rx,
    });

    let shell = s.shell.clone();
    // Background jobs always start at the workdir — they never touch the
    // foreground persist-cwd state (no cross-job cwd races).
    let cwd = s.workdir.clone();
    let max = s.max_output_bytes;
    let grace = s.grace;

    let handle = tokio::spawn(async move {
        let (tx, mut rx) = mpsc::unbounded_channel::<Reply>();
        // Writer: append each Reply as an NDJSON line to the job file.
        let writer = tokio::spawn(async move {
            let Ok(mut f) = tokio::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&out_path)
                .await
            else {
                return;
            };
            while let Some(reply) = rx.recv().await {
                let line = serde_json::to_string(&reply).unwrap_or_default();
                let _ = f.write_all(line.as_bytes()).await;
                let _ = f.write_all(b"\n").await;
                let _ = f.flush().await;
            }
        });

        let done = run_command(
            RunParams {
                shell: &shell,
                cwd: &cwd,
                command: &command,
                timeout_secs,
                max_output_bytes: max,
                grace,
                cwd_capture: None,
            },
            id,
            &tx,
            Some(cancel_rx),
        )
        .await;

        let mut outcome = outcome_from(done);
        outcome.job = Some(id);
        let _ = tx.send(Reply::Outcome(outcome.clone()));
        drop(tx);
        let _ = writer.await;
        let _ = done_tx.send(Some(outcome));
    });

    lock(&s.jobs).insert(
        id,
        JobEntry {
            shared,
            cancel: Some(cancel_tx),
            _handle: handle,
        },
    );
    id
}

/// `GET /jobs/{id}` — chunks since `?cursor=`, then a terminal outcome carrying
/// the advanced cursor.
async fn poll(
    State(s): State<Arc<AppState>>,
    UrlPath(id): UrlPath<u64>,
    Query(q): Query<CursorQuery>,
) -> Response {
    let Some(shared) = lookup(&s, id) else {
        return sse_one(no_such_job(id));
    };
    let (replies, cursor) =
        read_job_since(&shared.out_path, q.cursor.unwrap_or(0)).await;
    let mut term = shared.done.borrow().clone().unwrap_or(Outcome {
        running: true,
        ..Default::default()
    });
    term.job = Some(id);
    term.cursor = Some(cursor);

    let body = stream! {
        for reply in replies {
            if matches!(reply, Reply::Chunk(_)) {
                yield Ok::<Event, Infallible>(reply_event(&reply));
            }
        }
        yield Ok(reply_event(&Reply::Outcome(term)));
    };
    Sse::new(body).into_response()
}

/// `GET /jobs/{id}/wait` — follow the job to completion (or the soft timeout),
/// streaming chunks as they arrive then a terminal outcome.
async fn wait(
    State(s): State<Arc<AppState>>,
    UrlPath(id): UrlPath<u64>,
    Query(q): Query<WaitQuery>,
) -> Response {
    let Some(shared) = lookup(&s, id) else {
        return sse_one(no_such_job(id));
    };
    let mut cursor = q.cursor.unwrap_or(0);
    let deadline = q.timeout.map(|t| Instant::now() + Duration::from_secs(t));

    let body = stream! {
        loop {
            let (replies, nc) = read_job_since(&shared.out_path, cursor).await;
            cursor = nc;
            for reply in replies {
                if matches!(reply, Reply::Chunk(_)) {
                    yield Ok::<Event, Infallible>(reply_event(&reply));
                }
            }
            let done = shared.done.borrow().clone();
            if let Some(mut term) = done {
                term.job = Some(id);
                term.cursor = Some(cursor);
                yield Ok(reply_event(&Reply::Outcome(term)));
                break;
            }
            if deadline.is_some_and(|d| Instant::now() >= d) {
                yield Ok(reply_event(&Reply::Outcome(Outcome {
                    job: Some(id),
                    running: true,
                    cursor: Some(cursor),
                    ..Default::default()
                })));
                break;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    };
    Sse::new(body).into_response()
}

/// `POST /jobs/{id}/kill` — signal the job's group and acknowledge.
async fn kill(
    State(s): State<Arc<AppState>>,
    UrlPath(id): UrlPath<u64>,
) -> Json<Outcome> {
    let cancel = lock(&s.jobs).get_mut(&id).and_then(|e| e.cancel.take());
    match cancel {
        Some(tx) => {
            let _ = tx.send(());
            Json(Outcome {
                job: Some(id),
                running: false,
                ..Default::default()
            })
        }
        None => Json(no_such_job(id)),
    }
}

/// `POST /killall` — signal every live job; returns how many were signalled.
async fn killall(State(s): State<Arc<AppState>>) -> Json<usize> {
    Json(signal_all(&s))
}

/// Take and fire every job's cancel; returns the count signalled.
fn signal_all(s: &Arc<AppState>) -> usize {
    let cancels: Vec<_> = lock(&s.jobs)
        .values_mut()
        .filter_map(|e| e.cancel.take())
        .collect();
    let n = cancels.len();
    for tx in cancels {
        let _ = tx.send(());
    }
    n
}

/// Clone a job's shared handles out from under the registry lock.
fn lookup(s: &Arc<AppState>, id: u64) -> Option<Arc<JobShared>> {
    lock(&s.jobs).get(&id).map(|e| e.shared.clone())
}

/// Read a job's NDJSON output file from `cursor`, returning the complete
/// [`Reply`] lines and the new cursor (byte offset past the last full line).
async fn read_job_since(path: &Path, cursor: u64) -> (Vec<Reply>, u64) {
    let Ok(mut f) = tokio::fs::File::open(path).await else {
        return (Vec::new(), cursor);
    };
    if f.seek(SeekFrom::Start(cursor)).await.is_err() {
        return (Vec::new(), cursor);
    }
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).await.is_err() {
        return (Vec::new(), cursor);
    }
    let mut replies = Vec::new();
    let mut consumed = 0usize;
    for line in buf.split_inclusive(|&b| b == b'\n') {
        if line.last() != Some(&b'\n') {
            break; // incomplete trailing line — leave it for the next read
        }
        consumed += line.len();
        if let Ok(reply) =
            serde_json::from_slice::<Reply>(&line[..line.len() - 1])
        {
            replies.push(reply);
        }
    }
    (replies, cursor + consumed as u64)
}

/// Map a [`Reply`] to its SSE [`Event`] (named [`event::CHUNK`]/[`event::OUTCOME`]).
fn reply_event(reply: &Reply) -> Event {
    let (name, data) = match reply {
        Reply::Chunk(c) => (event::CHUNK, serde_json::to_string(c)),
        Reply::Outcome(o) => (event::OUTCOME, serde_json::to_string(o)),
        // Ready never appears on a stream; fold it into an outcome event.
        Reply::Ready { ready } => {
            (event::OUTCOME, serde_json::to_string(ready))
        }
    };
    Event::default()
        .event(name)
        .data(data.unwrap_or_else(|_| "{}".to_string()))
}

/// An SSE response that emits a single outcome event and closes (background
/// receipts, restart acks, errors).
fn sse_one(outcome: Outcome) -> Response {
    let body = stream! {
        yield Ok::<Event, Infallible>(reply_event(&Reply::Outcome(outcome)));
    };
    Sse::new(body).into_response()
}

/// Compose a terminal [`Outcome`] from a finished [`run_command`].
fn outcome_from(done: crate::session::RunDone) -> Outcome {
    match done.spawn_error {
        Some(error) => Outcome {
            running: false,
            error: Some(error),
            ..Default::default()
        },
        None => Outcome {
            exit: done.exit,
            running: false,
            timed_out: done.timed_out,
            truncated: done.truncated,
            ..Default::default()
        },
    }
}

/// A `no_such_job` error outcome.
fn no_such_job(id: u64) -> Outcome {
    Outcome {
        job: Some(id),
        running: false,
        error: Some(ProtocolError {
            kind: ErrorKind::NoSuchJob,
            message: format!("no such job: {id}"),
        }),
        ..Default::default()
    }
}

/// `?cursor=` for `poll`.
#[derive(Deserialize)]
struct CursorQuery {
    cursor: Option<u64>,
}

/// `?cursor=&timeout=` for `wait`.
#[derive(Deserialize)]
struct WaitQuery {
    cursor: Option<u64>,
    timeout: Option<u64>,
}

#[cfg(test)]
mod tests {
    use tokio::net::TcpListener;

    use super::*;

    /// Start a (plain-HTTP) server on an ephemeral 127.0.0.1 port; return its
    /// base URL. These exercise the router/handlers directly — the mutual-TLS
    /// serving path has its own end-to-end test in [`crate::tls`].
    async fn start() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let state = Arc::new(
            AppState::new(ServeConfig {
                shell: PathBuf::from("/bin/bash"),
                workdir: std::env::temp_dir(),
                persist_cwd: false,
                max_output_bytes: 1 << 20,
                grace: Duration::from_secs(2),
            })
            .unwrap(),
        );
        tokio::spawn(async move {
            axum::serve(listener, router(state)).await.unwrap();
        });
        format!("http://{addr}")
    }

    async fn post_run(
        base: &str,
        body: serde_json::Value,
    ) -> reqwest::Response {
        reqwest::Client::new()
            .post(format!("{base}/run"))
            .json(&body)
            .send()
            .await
            .unwrap()
    }

    /// Pull the `job` id out of an SSE response's outcome `data:` line.
    fn job_id(sse: &str) -> Option<u64> {
        sse.lines()
            .filter_map(|l| l.strip_prefix("data:"))
            .filter_map(|d| serde_json::from_str::<Outcome>(d.trim()).ok())
            .find_map(|o| o.job)
    }

    /// A unique sentinel path for "did the killed command keep running?" tests.
    fn sentinel(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let p = std::env::temp_dir().join(format!(
            "bashd-test-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_file(&p);
        p
    }

    #[tokio::test]
    async fn handshake_reports_protocol() {
        let base = start().await;
        let ready: Ready = reqwest::get(format!("{base}/"))
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(ready.protocol, PROTOCOL_VERSION);
        assert!(!ready.persist_cwd);
    }

    #[tokio::test]
    async fn foreground_streams_output_and_exit() {
        let base = start().await;
        let body =
            post_run(&base, serde_json::json!({"command": "echo hello"}))
                .await
                .text()
                .await
                .unwrap();
        assert!(body.contains("hello"), "{body}");
        assert!(body.contains(event::OUTCOME), "{body}");
        assert!(body.contains("\"exit\":0"), "{body}");
    }

    #[tokio::test]
    async fn foreground_reports_nonzero_exit() {
        let base = start().await;
        let body = post_run(&base, serde_json::json!({"command": "exit 3"}))
            .await
            .text()
            .await
            .unwrap();
        assert!(body.contains("\"exit\":3"), "{body}");
    }

    #[tokio::test]
    async fn background_runs_and_wait_collects() {
        let base = start().await;
        let receipt = post_run(
            &base,
            serde_json::json!({"command": "echo bg", "background": true}),
        )
        .await
        .text()
        .await
        .unwrap();
        let job =
            job_id(&receipt).expect("background receipt carries a job id");

        let waited = reqwest::get(format!("{base}/jobs/{job}/wait?timeout=10"))
            .await
            .unwrap()
            .text()
            .await
            .unwrap();
        assert!(waited.contains("bg"), "{waited}");
        assert!(waited.contains("\"exit\":0"), "{waited}");
    }

    #[tokio::test]
    async fn disconnect_reaps_child() {
        let base = start().await;
        let marker = sentinel("disconnect");
        let cmd = format!("sleep 2; echo done > {}", marker.display());

        // Start the request, then drop it without reading the body (disconnect).
        let resp = post_run(&base, serde_json::json!({ "command": cmd })).await;
        drop(resp);

        // Past when the sentinel would have been written had it survived.
        tokio::time::sleep(Duration::from_secs(3)).await;
        assert!(
            !marker.exists(),
            "disconnect should have reaped the child before it wrote the sentinel"
        );
    }

    #[tokio::test]
    async fn kill_stops_background_job() {
        let base = start().await;
        let marker = sentinel("kill");
        let cmd = format!("sleep 3; echo done > {}", marker.display());
        let receipt = post_run(
            &base,
            serde_json::json!({ "command": cmd, "background": true }),
        )
        .await
        .text()
        .await
        .unwrap();
        let job = job_id(&receipt).expect("job id");

        reqwest::Client::new()
            .post(format!("{base}/jobs/{job}/kill"))
            .send()
            .await
            .unwrap();

        tokio::time::sleep(Duration::from_secs(4)).await;
        assert!(
            !marker.exists(),
            "kill should stop the job before its sentinel"
        );
    }
}
