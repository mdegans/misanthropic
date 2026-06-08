//! [`RichBash`] — the bash tool as a *typed, multi-method* tool, the ergonomic
//! counterpart to the predefined [`BashTool`](crate::tool::bash::BashTool).
//!
//! Where [`BashTool`](crate::tool::bash::BashTool) exposes Anthropic's single
//! trained `bash` tool (one enum-shaped schema, `command`/`restart`),
//! `RichBash` uses this crate's Tool/Method split: one method per operation —
//! `run`/`restart`/`check_output`/`kill` — each with its own flat
//! `type: object` schema. The model sees them as distinct tools (`bash__run`,
//! `bash__check_output`, …). This sidesteps Anthropic's rule that a tool
//! `input_schema` may not be a top-level union (`anyOf`/`oneOf`/`allOf`), which
//! a single enum-shaped schema would be.
//!
//! Background jobs *call back*: `run` with `background: true` returns a job id
//! immediately and spawns a watcher ([`BashSandbox::watch`]) that pushes the
//! result as a [`User`](crate::prompt::message::Role::User) notification when
//! the job finishes — so the model is told, instead of polling. `check_output`
//! remains for peeking at a running job's output meanwhile. The watcher needs
//! the `tokio` feature (this whole module is gated on it).

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::Deserialize;
use tokio::task::JoinHandle;

use super::{BashSandbox, Command, ExecResult, Known};
use crate::{
    Prompt,
    prompt::message::{Content, Role},
    tool::{Mailbox, Notifications, tool},
};

/// A thread-safe boxed error — the [`Tool`](crate::tool::Tool) lifecycle-hook
/// error type.
type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Args for `run`: a shell command, optionally backgrounded / time-limited.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Run {
    /// The shell command to run.
    pub command: String,
    /// Run detached: return a job id immediately and notify you when it
    /// finishes (no need to poll). Use `check_output` to peek meanwhile.
    #[serde(default)]
    pub background: Option<bool>,
    /// Kill the command and report a timeout if it runs longer than this many
    /// seconds.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
}

/// Args for `restart`: none — the macro requires an `Args` type per method, so
/// this is an empty object (the model sends `{}`).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Restart {}

/// Args for `check_output`: which background job to inspect.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CheckOutput {
    /// The id of the background job whose output so far to return.
    pub job: u64,
}

/// Args for `kill`: which background job to stop.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Kill {
    /// The id of the background job to stop.
    pub job: u64,
}

/// The bash tool as a typed, multi-method tool with background-completion
/// callbacks. See the [module docs](self).
///
/// Holds the sandbox boxed (rather than a generic `S`) so the `#[tool]` macro's
/// generated impls stay monomorphic — bash work is network-bound, so the dyn
/// dispatch is free in practice.
pub struct RichBash {
    sandbox: Box<dyn BashSandbox>,
    /// Push channel for background-job completion notifications. When boxed in a
    /// [`ToolBox`](crate::tool::ToolBox), `connect` swaps this for the box's
    /// aggregate handle.
    mailbox: Mailbox,
    /// One completion-watcher task per in-flight background job, aborted on
    /// `kill` and at teardown.
    watchers: HashMap<u64, JoinHandle<()>>,
}

impl RichBash {
    /// A rich bash tool over `sandbox`.
    pub fn new(sandbox: impl BashSandbox + 'static) -> Self {
        Self {
            sandbox: Box::new(sandbox),
            mailbox: Mailbox::new("bash"),
            watchers: HashMap::new(),
        }
    }

    /// Spawn a completion-watcher for a freshly-started background `job`: it
    /// follows the job over an independent connection ([`BashSandbox::watch`])
    /// and pushes the result as a `User` notification when it finishes. A no-op
    /// if the sandbox can't watch out-of-band (jobs stay `check_output`-only).
    fn watch_job(&mut self, job: u64) {
        let Some(future) = self.sandbox.watch(job) else {
            return;
        };
        // A send-only handle on the same channel, owned by the watcher task.
        let mailbox = self.mailbox.derive(self.mailbox.source().to_string());
        let handle = tokio::spawn(async move {
            if let Ok(result) = future.await {
                let _ = mailbox
                    .send(completion_note(job, &result), vec![Role::User]);
            }
        });
        self.watchers.insert(job, handle);
    }
}

#[tool(name = "bash")]
impl RichBash {
    /// Run a shell command in the persistent session. For anything
    /// long-running, set `background: true` — you'll be notified when it
    /// finishes, so you don't have to wait or poll.
    #[method]
    async fn run(&mut self, args: Run) -> Result<Content, Content> {
        let command = Command::Known(Known::Run {
            command: args.command.into(),
            background: args.background,
            timeout_secs: args.timeout_secs,
        });
        match self.sandbox.exec(command).await {
            Ok(result) => {
                // A backgrounded run comes back still running with a job id —
                // watch it so its completion is pushed.
                if result.running
                    && let Some(job) = result.job
                {
                    self.watch_job(job);
                }
                Ok(result.render().into())
            }
            Err(e) => Err(e.to_string().into()),
        }
    }

    /// Reset the session — a fresh shell in the default working directory.
    #[method]
    async fn restart(&mut self, _args: Restart) -> Result<Content, Content> {
        match self.sandbox.restart().await {
            Ok(()) => Ok("bash session restarted".into()),
            Err(e) => Err(e.to_string().into()),
        }
    }

    /// Get a background job's output so far. You're notified automatically when
    /// the job finishes, so use this only to peek at a still-running job.
    #[method]
    async fn check_output(
        &mut self,
        args: CheckOutput,
    ) -> Result<Content, Content> {
        match self.sandbox.poll(args.job).await {
            Ok(result) => Ok(result.render().into()),
            Err(e) => Err(e.to_string().into()),
        }
    }

    /// Stop a background job (TERM→grace→KILL).
    #[method]
    async fn kill(&mut self, args: Kill) -> Result<Content, Content> {
        match self.sandbox.kill(args.job).await {
            Ok(()) => {
                if let Some(handle) = self.watchers.remove(&args.job) {
                    handle.abort();
                }
                Ok(format!("killed background job {}", args.job).into())
            }
            Err(e) => Err(e.to_string().into()),
        }
    }

    /// Boot the sandbox (launch the container + `bashd`).
    #[on_init]
    async fn start(&mut self, _prompt: &mut Prompt) -> Result<(), BoxError> {
        self.sandbox.start().await?;
        Ok(())
    }

    /// Abort outstanding watchers, then tear the sandbox down.
    #[on_teardown]
    async fn stop(&mut self, _prompt: &mut Prompt) -> Result<(), BoxError> {
        for (_, handle) in self.watchers.drain() {
            handle.abort();
        }
        self.sandbox.teardown().await?;
        Ok(())
    }

    /// Adopt a [`ToolBox`](crate::tool::ToolBox)'s aggregate mailbox.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.mailbox = mailbox;
    }

    /// Hand out the consumer end of our mailbox (standalone); `None` once boxed,
    /// where the box owns consumption.
    #[subscribe]
    fn subscribe(&mut self) -> Option<Notifications> {
        self.mailbox.subscribe()
    }
}

/// Format a finished background job's [`ExecResult`] as a notification body.
fn completion_note(job: u64, result: &ExecResult) -> String {
    let mut body = result.stdout.clone();
    if !result.stderr.is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(&result.stderr);
    }
    let exit = match result.exit {
        Some(code) => format!(" (exit {code})"),
        None => String::new(),
    };
    format!("background job {job} finished{exit}:\n{}", body.trim_end())
}
