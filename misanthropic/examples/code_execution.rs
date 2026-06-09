//! Example: the **`code_execution` server tool**
//! ([`ServerMethodDef::code_execution`]) — Anthropic runs a sandboxed Linux
//! container with `bash` and `text_editor` sub-tools and returns
//! [`BashCodeExecutionToolResult`] / [`TextEditorCodeExecutionToolResult`]
//! blocks, which the loop below walks. A long run can pause the turn
//! ([`StopReason::PauseTurn`]); we resume by sending it back. To let the
//! container call *your own* tools, see the [`programmatic_tool_calling`]
//! example.
//!
//! ```sh
//! cargo run --features client --example code_execution -- \
//!     "Write a CSV of three rows to /tmp/d.csv, then sum the second column."
//! ```
//!
//! [`ServerMethodDef::code_execution`]: misanthropic::tool::ServerMethodDef::code_execution
//! [`BashCodeExecutionToolResult`]: misanthropic::prompt::message::Block::BashCodeExecutionToolResult
//! [`TextEditorCodeExecutionToolResult`]: misanthropic::prompt::message::Block::TextEditorCodeExecutionToolResult
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`programmatic_tool_calling`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/programmatic_tool_calling.rs

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt,
    prompt::message::{
        BashCodeExecutionResultContent as Bash, Block, Role,
        TextEditorCodeExecutionResultContent as Edit,
    },
    response::StopReason,
    tool::ServerMethodDef,
};

/// Demonstrate the `code_execution` server tool.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The task for the model to accomplish with code execution.
    task: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let task = cli.task.unwrap_or_else(|| {
        "Create /tmp/nums.txt with the numbers 1 through 5, one per line, \
         then use bash to sum them."
            .to_string()
    });

    // No `call()` to write — Anthropic runs it; needs a model with
    // `code_execution` support (Opus / Sonnet 4.5+).
    let mut prompt = cli
        .common
        .configure(Prompt::default().model(Id::Sonnet46))
        .add_message((Role::User, task))?
        .add_tool(ServerMethodDef::code_execution());

    // Resume on `pause_turn`; otherwise the turn is done.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if matches!(response.stop_reason, Some(StopReason::PauseTurn)) {
            prompt.push_message(response)?;
            continue;
        }

        break response;
    };

    for block in answer.inner.content.iter() {
        match block {
            Block::BashCodeExecutionToolResult { content, .. } => match content
            {
                Bash::Result {
                    stdout,
                    stderr,
                    return_code,
                    content: files,
                } => {
                    println!("$ (exit {return_code})");
                    if !stdout.is_empty() {
                        print!("{stdout}");
                    }
                    if !stderr.is_empty() {
                        eprint!("{stderr}");
                    }
                    // Files land in `$OUTPUT_DIR` as `file_id`s — fetch bytes
                    // via the Files API (#32).
                    for file in files {
                        println!("[emitted file {}]", file.file_id);
                    }
                }
                Bash::Error { error_code, .. } => {
                    eprintln!("[bash tool error: {error_code}]");
                }
            },
            Block::TextEditorCodeExecutionToolResult { content, .. } => {
                match content {
                    Edit::Create { is_file_update } => println!(
                        "[{} a file]",
                        if *is_file_update {
                            "updated"
                        } else {
                            "created"
                        }
                    ),
                    Edit::View { content, .. } => {
                        println!("[viewed]\n{content}")
                    }
                    Edit::StrReplace { lines, .. } => {
                        println!("[edited]");
                        for line in lines {
                            println!("  {line}");
                        }
                    }
                    Edit::Error { error_code, .. } => {
                        eprintln!("[editor tool error: {error_code}]")
                    }
                }
            }
            Block::Text { text, .. } => println!("\n{text}"),
            _ => {}
        }
    }

    // The container id is reusable — pass it via `Prompt::container` later.
    if let Some(container) = &answer.container {
        eprintln!("\n[container {} — reusable]", container.id);
    }

    Ok(())
}
