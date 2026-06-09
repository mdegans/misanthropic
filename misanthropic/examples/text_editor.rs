//! Example: the **text editor tool** ([`TextEditor`]) — Anthropic's predefined
//! `str_replace_based_edit_tool`. Anthropic defines the schema (add by versioned
//! name via [`TextEditor::latest`]); the model emits a [`tool_use`] carrying a
//! typed [`text_editor::Command`] (`view`/`create`/`str_replace`/`insert`); you
//! execute it and answer with a [`tool::Result`]. [`FsEditorBackend`] jails edits
//! to a working directory. Drive loop is **bounded** — autonomous one-shots must
//! terminate.
//!
//! ```sh
//! cargo run --features "client text-editor-fs" --example text_editor
//! ```
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

/// `primes.py` missing the colon on its `for` loop.
const BUGGY: &str = "\
def get_primes(limit):
    primes = []
    for num in range(2, limit + 1)
        if is_prime(num):
            primes.append(num)
    return primes
";

/// Cap on autonomous turns. A view + str_replace + answer is three; 8 is
/// generous headroom.
const MAX_TURNS: usize = 8;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    // `../` and absolute-path escapes are rejected by the backend.
    let dir = tempfile::tempdir()?;
    tokio::fs::write(dir.path().join("primes.py"), BUGGY).await?;
    println!("--- primes.py before ---\n{BUGGY}");

    let mut tools = ToolBox::new().add(FsEditorBackend::new(dir.path()).await?);

    let mut chat = cli.common.configure(Prompt::default()).add_message((
        Role::User,
        "There's a syntax error in primes.py. Please fix it.",
    ))?;

    tools.prepare(&mut chat).await?;

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
