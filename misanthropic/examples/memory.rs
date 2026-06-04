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
//! cargo run --features "client memory" --example memory
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

use std::io::{BufRead, stdin};

use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{Memory, Tool, memory::FsMemoryBackend},
};
use rustyline::{DefaultEditor, error::ReadlineError};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let key = std::env::var("ANTHROPIC_API_KEY").or_else(|_| {
        eprintln!("ANTHROPIC_API_KEY not set. Enter your API key:");
        stdin()
            .lock()
            .lines()
            .next()
            .ok_or("no input")?
            .map_err(|e| e.to_string())
    })?;
    let client = Client::new(key)?;

    // The client-side executor. Every memory operation is confined to
    // `./memories` (created if missing) and, by default, to `.md` files;
    // path-traversal attempts (`../`, absolute escapes) are rejected.
    let mut memory = FsMemoryBackend::new("./memories").await?;

    // The memory *protocol* ("ALWAYS VIEW YOUR MEMORY DIRECTORY FIRST …") is
    // injected server-side when the tool is enabled, so we don't repeat it —
    // we just add the predefined definition. `add_tool` takes anything
    // `Into<ToolDef>`, so the schema-less `Memory::latest()` drops in next to
    // any custom tool.
    let mut chat = Prompt::default().add_tool(Memory::latest()).set_system(
        "You are a helpful assistant with a persistent memory. Record \
             durable facts, decisions, and progress so you can resume in a \
             later session, and keep your notes tidy — prune what's stale.",
    );

    println!("Memory chat — notes persist in ./memories across runs.");
    println!("Talk, then Ctrl-D to quit and run again to watch it remember.\n");

    let mut rl = DefaultEditor::new()?;
    loop {
        let line = match rl.readline("you ▸ ") {
            Ok(line) if line.trim().is_empty() => continue,
            Ok(line) => line,
            Err(ReadlineError::Interrupted) => continue, // Ctrl-C: ignore
            Err(ReadlineError::Eof) => break,            // Ctrl-D: quit
            Err(e) => return Err(e.into()),
        };
        rl.add_history_entry(&line).ok();
        chat.push_message((Role::User, line))?;

        // Drive the tool loop. Before answering, the model may `view` its
        // memory, then `create`/`str_replace`/`insert`/… across several turns.
        // Each memory `tool_use` is executed locally and fed back, until a turn
        // arrives with no tool call — that one is the answer.
        let answer = loop {
            let message = client.message(&chat).await?;
            let Some(call) = message.tool_use() else {
                break message;
            };
            // Own the call so we can append the assistant turn first.
            let call = call.clone();
            chat.push_message(message)?;
            // Typed dispatch: `call.input` -> `memory::Command`, executed
            // against `./memories`, with the canonical (line-numbered, etc.)
            // string handed back to the model.
            let result = memory.call(call).await;
            chat.push_message(result)?;
        };

        println!("\nclaude ▸ {}\n", answer.inner.content);
        chat.push_message(answer)?;
    }

    println!("\nbye — your memory is saved in ./memories");
    Ok(())
}
