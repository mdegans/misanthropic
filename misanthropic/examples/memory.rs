//! Example: the **`memory` tool** ([`Memory`]) — a *client-side* predefined
//! tool, driven as a tiny multi-session REPL.
//!
//! Three flavours of tool live in this crate, and memory is the third:
//!
//! - A **custom tool** (see `strawberry`) — *you* define the schema and *you*
//!   execute it via [`Tool::call`].
//! - A **server tool** (see `web_search`) — Anthropic defines *and* executes
//!   it; the call and result come back in the response and you never run a
//!   thing.
//! - A **client-side predefined tool** (memory, here) — Anthropic *defines* the
//!   schema (you add it by versioned name, no schema of your own), but the
//!   model emits an ordinary [`tool_use`] that *you* execute and answer with a
//!   [`tool::Result`], exactly like a custom tool.
//!
//! So memory *defines* like a server tool and *executes* like a custom one.
//! [`FsMemoryBackend`] is the executor: it deserializes each call's input into
//! a typed [`memory::Command`] and runs it against a directory on disk —
//! jailed to `./memories` and, by default, to markdown files.
//!
//! ## Why a REPL?
//!
//! Persistence *across sessions* is the whole point of the tool, and a process
//! you can quit and restart is the most honest way to show it: hold a
//! conversation, `Ctrl-D` to quit, run it again, and the model `view`s
//! `./memories` and picks up where it left off. (Browser/localStorage demos
//! fight this; a CLI writing real files does not.)
//!
//! # Usage
//!
//! ```sh
//! cargo run --features "client memory-fs" --example memory
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin. Your
//! notes accumulate in `./memories` between runs — delete it to start fresh.
//!
//! [`Memory`]: misanthropic::tool::Memory
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`tool_use`]: misanthropic::tool::Use
//! [`tool::Result`]: misanthropic::tool::Result
//! [`FsMemoryBackend`]: misanthropic::tool::memory::FsMemoryBackend
//! [`memory::Command`]: misanthropic::tool::memory::Command

mod utils;

use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{ToolBox, memory::FsMemoryBackend},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    utils::log_init(false);

    // Get the API key from stdin *before* the rustyline thread takes over stdin.
    let client = Client::new(utils::api_key()?)?;

    // The client-side executor. Every memory operation is confined to
    // `./memories` (created if missing) and, by default, to `.md` files;
    // path-traversal attempts (`../`, absolute escapes) are rejected. It drops
    // into a `ToolBox` like any other tool — the box installs its predefined
    // definition and routes the bare `"memory"` `tool_use` back to it. Add a
    // custom tool here and the same driver dispatches both.
    let toolbox = ToolBox::new().add(FsMemoryBackend::new("./memories").await?);

    // The memory *protocol* ("ALWAYS VIEW YOUR MEMORY DIRECTORY FIRST …") is
    // injected server-side when the tool is enabled, so we don't repeat it.
    let prompt = Prompt::default().set_system(
        "You are a helpful assistant with a persistent memory. Record \
             durable facts, decisions, and progress so you can resume in a \
             later session, and keep your notes tidy — prune what's stale.",
    );

    let (mut lines, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line("Memory chat — notes persist in ./memories across runs.");
    printer.line(
        "Talk, then Ctrl-D to quit and run again to watch it remember.\n",
    );

    // `Chat` runs the model to quiescence: before answering, the model may
    // `view` its memory, then `create`/`str_replace`/`insert`/… across several
    // turns. Each memory `tool_use` is routed through the one box — the bare
    // `"memory"` call, dispatched to the backend, which runs the typed
    // `memory::Command` against `./memories` and feeds the canonical result
    // back — until a turn arrives with no tool call. We print only that final,
    // tool-free answer (the `on_assistant` hook fires on every turn).
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

    println!("\nbye — your memory is saved in ./memories");
    Ok(())
}
