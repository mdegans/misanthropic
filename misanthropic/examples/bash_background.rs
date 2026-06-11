//! Example: **background bash jobs** via [`RichBash`] — the same Docker sandbox
//! as the `bash` example, re-expressed as a typed multi-method tool so the
//! model gets `bash__run`, `bash__check_output`, `bash__kill`, and
//! `bash__restart` as distinct flat-schema tools (avoiding the predefined
//! tool's enum-shaped schema). `run` with `background: true` returns a job id
//! immediately; when the job finishes [`RichBash`] pushes the result as a
//! [`User`] notification — the `Chat` driver seats it at the next turn boundary
//! with no polling. The chat closure is identical to `reminder` and `memory`.
//!
//! ```sh
//! # needs Docker; the published misan-bashd image is pulled on first run
//! # (`just build-bashd` builds the same tag locally to shadow it)
//! cargo run --features "client bash-container" --example bash_background
//! ```
//!
//! [`RichBash`]: misanthropic::tool::bash::RichBash
//! [`User`]: misanthropic::prompt::message::Role::User

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{
        ToolBox,
        bash::{DockerSandbox, RichBash},
    },
};

/// Chat with background bash jobs that push completion notifications.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // Get the API key from stdin *before* the rustyline thread takes over stdin.
    let client = Client::new(utils::api_key()?)?;

    // The box aggregates the mailbox; `Chat` subscribes so background-completion
    // pushes flow into the driver's `select!`.
    let toolbox = ToolBox::new().add(RichBash::new(DockerSandbox::default()));

    let prompt = cli.common.configure(Prompt::default().system(
        "You are a helpful assistant with a sandboxed bash tool. For anything \
         long-running, start it in the background (`background: true`) — you \
         will be notified when it finishes, so do not poll. Keep chatting or \
         working while jobs run.",
    ));

    println!("Starting sandbox (booting container, launching bashd)...");

    let (mut lines, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line(
        "Bash chat — background jobs call back when done. Ctrl-D quits.\n",
    );

    cli.chat
        .configure(utils::Chat::new(client, prompt, toolbox))
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
