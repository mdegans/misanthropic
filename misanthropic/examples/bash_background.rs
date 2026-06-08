//! Example: **background bash jobs that call back when done** — the bash tool as
//! a *typed, multi-method* tool ([`RichBash`]) in an interactive chat, where a
//! backgrounded command *pushes* a notification on completion instead of the
//! model polling for it.
//!
//! Where the `bash` example is an autonomous one-shot over Anthropic's
//! predefined `bash` def, this uses [`RichBash`] — the same sandbox re-expressed
//! through this crate's Tool/Method split, so the model gets `bash__run`,
//! `bash__check_output`, `bash__kill`, and `bash__restart` as distinct tools
//! (each a flat `type: object` schema, sidestepping the predefined tool's
//! enum-shaped one). `run` with `background: true` returns a job id immediately
//! and, when the job finishes, [`RichBash`] pushes the result as a [`User`]
//! notification — the examples' `Chat` driver seats it at the next turn
//! boundary, so a job that finishes *while you chat* is reported with nobody
//! polling.
//!
//! The chat closure is the **same** as `reminder` and `memory`: just read the
//! next user line. Lifecycle, tool dispatch, and notification interleaving are
//! all the driver's job.
//!
//! # Usage
//!
//! ```sh
//! just build-bashd   # build the misan-bashd sandbox image (once; needs Docker)
//! cargo run --features "client bash-container" --example bash_background
//! ```
//!
//! Enter your API key at the prompt (or it falls through from stdin), and have
//! the `misan-bashd` image built. Try: *"run `sleep 10; echo built` in the
//! background and tell me when it's done — meanwhile, what's the capital of
//! France?"* — the answer comes first, then the completion lands a few turns
//! later on its own.
//!
//! [`RichBash`]: misanthropic::tool::bash::RichBash
//! [`User`]: misanthropic::prompt::message::Role::User

mod utils;

use std::io::BufRead;

use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{
        ToolBox,
        bash::{DockerSandbox, RichBash},
    },
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(feature = "log")]
    env_logger::init();

    // Get the API key from stdin *before* the rustyline thread takes over stdin.
    println!("Enter your API key:");
    let key = std::io::stdin().lock().lines().next().unwrap()?;
    let client = Client::new(key)?;

    // The typed multi-method bash tool over the default Docker sandbox. It owns
    // a `Mailbox`; the box aggregates it and `Chat` subscribes to it for us, so
    // background-completion pushes flow into the driver's `select!`.
    let toolbox = ToolBox::new().add(RichBash::new(DockerSandbox::default()));

    let prompt = Prompt::default().set_system(
        "You are a helpful assistant with a sandboxed bash tool. For anything \
         long-running, start it in the background (`background: true`) — you \
         will be notified when it finishes, so do not poll. Keep chatting or \
         working while jobs run.",
    );

    println!("Starting sandbox (booting container, launching bashd)...");

    let (mut lines, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line(
        "Bash chat — background jobs call back when done. Ctrl-D quits.\n",
    );

    utils::Chat::new(client, prompt, toolbox)
        .on_assistant(move |_state: &mut (), msg| {
            if msg.tool_use().is_none() {
                printer.line(format!("\nclaude ▸ {}\n", msg.content));
            }
        })
        .run((), async move |_state: &mut ()| {
            Ok(lines
                .recv()
                .await
                .map(|line| vec![(Role::User, line).into()]))
        })
        .await?;

    println!("\nbye");
    Ok(())
}
