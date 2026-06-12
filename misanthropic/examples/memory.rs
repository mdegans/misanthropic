//! Example: the **`memory` tool** ([`Memory`]) — a *client-side predefined*
//! tool driven as a multi-session REPL. Anthropic defines the schema (add by
//! versioned name, no schema of your own); the model emits a [`tool_use`] that
//! *you* execute with a [`tool::Result`] — defines like a server tool, executes
//! like a custom one. [`FsMemoryBackend`] deserializes each call into a typed
//! [`memory::Command`] and runs it against a disk directory jailed to
//! `./memories`. Quit and rerun to see persistence across sessions.
//!
//! ```sh
//! cargo run --features "client memory-fs" --example memory
//! ```
//!
//! Notes accumulate in `./memories` between runs — delete it to start fresh.
//!
//! [`Memory`]: misanthropic::tool::Memory
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`tool_use`]: misanthropic::tool::Use
//! [`tool::Result`]: misanthropic::tool::Result
//! [`FsMemoryBackend`]: misanthropic::tool::memory::FsMemoryBackend
//! [`memory::Command`]: misanthropic::tool::memory::Command

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{ToolBox, memory::FsMemoryBackend},
};

/// Persistent memory chat: notes survive across sessions in ./memories.
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

    // Path-traversal attempts (`../`, absolute paths) are rejected by the
    // backend. The box installs the predefined definition and routes the bare
    // `"memory"` tool_use back to it.
    let toolbox = ToolBox::new().add(FsMemoryBackend::new("./memories").await?);

    // The memory protocol ("ALWAYS VIEW YOUR MEMORY DIRECTORY FIRST …") is
    // injected server-side when the tool is enabled.
    let prompt = cli.common.configure(Prompt::default().system(
        "You are a helpful assistant with a persistent memory. Record \
             durable facts, decisions, and progress so you can resume in a \
             later session, and keep your notes tidy — prune what's stale.",
    ));

    let (mut lines, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line("Memory chat — notes persist in ./memories across runs.");
    printer.line(
        "Talk, then Ctrl-D to quit and run again to watch it remember.\n",
    );

    // The hook fires on every turn; print only the tool-free final answer.
    cli.chat
        .configure(utils::Chat::new(client, prompt, toolbox))
        .on_assistant(move |_state: &mut (), msg| {
            if msg.tool_use().is_none() {
                printer.line(format!("\nclaude ▸ {}\n", msg.content));
            }
            [msg.into()] // seat the turn unchanged
        })
        .run((), async move |_state: &mut ()| {
            Ok(lines
                .recv()
                .await
                .map(|line| vec![(Role::User, line).into()]))
        })
        .await?;

    println!("\nbye — your memory is saved in ./memories");
    Ok(())
}
