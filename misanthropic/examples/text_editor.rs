//! Example: the **text editor tool** ([`TextEditor`]) — Anthropic's predefined
//! `str_replace_based_edit_tool`, driven as a one-shot coding assistant.
//!
//! Like `memory`, the text editor is a *client-side predefined* tool: Anthropic
//! defines the schema (you add it by versioned name via [`TextEditor::latest`],
//! no schema of your own), the model emits an ordinary [`tool_use`] carrying a
//! typed [`text_editor::Command`] (`view` / `create` / `str_replace` /
//! `insert`), and *you* execute it and answer with a [`tool::Result`]. So it
//! *defines* like a server tool and *executes* like a custom one.
//! [`FsEditorBackend`] is the executor: it edits real files under a working
//! directory it is jailed to.
//!
//! This example plants a `primes.py` with a syntax error in a temp workspace,
//! asks Claude to fix it, drives the tool loop, and prints the repaired file —
//! the documentation's canonical demo, made real (and verified on disk).
//!
//! The loop is **bounded** (`for _ in 0..MAX_TURNS`): a one-shot, autonomous
//! drive must terminate, unlike a human-gated CLI where each turn is driven by
//! a person.
//!
//! # Usage
//!
//! ```sh
//! cargo run --features "client text-editor-fs" --example text_editor
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment.
//!
//! [`TextEditor`]: misanthropic::tool::TextEditor
//! [`TextEditor::latest`]: misanthropic::tool::TextEditor::latest
//! [`tool_use`]: misanthropic::tool::Use
//! [`tool::Result`]: misanthropic::tool::Result
//! [`FsEditorBackend`]: misanthropic::tool::text_editor::FsEditorBackend
//! [`text_editor::Command`]: misanthropic::tool::text_editor::Command

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{Tool, ToolBox, text_editor::FsEditorBackend},
};

/// Drive the text editor tool to fix a syntax error in a temp workspace.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
}

/// A `primes.py` missing the colon on its `for` loop — the exact bug from the
/// text-editor tool documentation.
const BUGGY: &str = "\
def get_primes(limit):
    primes = []
    for num in range(2, limit + 1)
        if is_prime(num):
            primes.append(num)
    return primes
";

/// Cap on autonomous turns so a confused model can't loop forever. A view +
/// str_replace + final answer is three; this leaves generous headroom.
const MAX_TURNS: usize = 8;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    // A workspace the backend is jailed to: the model only ever sees paths
    // under it, and `../`/absolute escapes are rejected.
    let dir = tempfile::tempdir()?;
    tokio::fs::write(dir.path().join("primes.py"), BUGGY).await?;
    println!("--- primes.py before ---\n{BUGGY}");

    // The client-side executor drops into a `ToolBox` like any other tool: the
    // box installs the predefined definition and routes the bare
    // `"str_replace_based_edit_tool"` `tool_use` back to it. Add a custom tool
    // here and the same `tools.call(..)` below would dispatch both.
    let mut tools = ToolBox::new().add(FsEditorBackend::new(dir.path()).await?);

    let mut chat = cli.common.configure(Prompt::default()).add_message((
        Role::User,
        "There's a syntax error in primes.py. Please fix it.",
    ))?;

    // Install the toolbox: writes the editor def onto the prompt and runs each
    // tool's `on_init`. One call wires every tool the box owns.
    tools.prepare(&mut chat).await?;

    // Drive the tool loop, bounded. The model `view`s the file, then
    // `str_replace`s the fix; each editor `tool_use` is executed locally
    // against the workspace and fed back, until a turn arrives with no tool
    // call — that one is the answer.
    let mut answer = None;
    for _ in 0..MAX_TURNS {
        let message = client.message(&chat).await?;
        match message.tool_use() {
            None => {
                answer = Some(message);
                break;
            }
            Some(call) => {
                let call = call.clone();
                chat.push_message(message)?;
                let result = tools.call(call).await;
                chat.push_message(result)?;
            }
        }
    }
    let answer = answer.ok_or("text editor loop did not converge in time")?;

    println!("\nclaude ▸ {}\n", answer.inner.content);
    println!(
        "--- primes.py after ---\n{}",
        tokio::fs::read_to_string(dir.path().join("primes.py")).await?
    );
    Ok(())
}
