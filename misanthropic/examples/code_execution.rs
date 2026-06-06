//! Example: the **`code_execution` server tool**
//! ([`ServerMethodDef::code_execution`]).
//!
//! Anthropic runs this one: you add it to the prompt and the model gets a
//! sandboxed Linux container with two sub-tools — `bash_code_execution` (run
//! shell commands) and `text_editor_code_execution` (view / create / edit
//! files). Their results come back *in the response* as
//! [`BashCodeExecutionToolResult`] and [`TextEditorCodeExecutionToolResult`]
//! blocks; this example walks them and prints what the container did.
//!
//! For *programmatic* tool calling — letting the container call your own tools
//! from inside the sandbox — see the [`programmatic_tool_calling`] example.
//!
//! ## `pause_turn`
//!
//! A long container run can make the API yield mid-turn with
//! [`StopReason::PauseTurn`]; send the paused assistant turn straight back to
//! let it continue. The loop below does exactly that.
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example code_execution -- \
//!     "Write a CSV of three rows to /tmp/d.csv, then sum the second column."
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [`ServerMethodDef::code_execution`]: misanthropic::tool::ServerMethodDef::code_execution
//! [`BashCodeExecutionToolResult`]: misanthropic::prompt::message::Block::BashCodeExecutionToolResult
//! [`TextEditorCodeExecutionToolResult`]: misanthropic::prompt::message::Block::TextEditorCodeExecutionToolResult
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`programmatic_tool_calling`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/programmatic_tool_calling.rs

use std::io::{BufRead, stdin};

use misanthropic::{
    AnthropicModel, Client, Prompt,
    prompt::message::{
        BashCodeExecutionResultContent as Bash, Block, Role,
        TextEditorCodeExecutionResultContent as Edit,
    },
    response::StopReason,
    tool::ServerMethodDef,
};

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

    let task = std::env::args().nth(1).unwrap_or_else(|| {
        "Create /tmp/nums.txt with the numbers 1 through 5, one per line, \
         then use bash to sum them."
            .to_string()
    });

    // `code_execution` is a pure declaration — no `call()` to write. The
    // container (and both sub-tools) come for free. It needs a model that
    // supports `code_execution_20260120` (Opus/Sonnet 4.5+).
    let mut prompt = Prompt::default()
        .model(AnthropicModel::Sonnet46)
        .add_message((Role::User, task))?
        .add_tool(ServerMethodDef::code_execution());

    // Drive the server-side loop to completion, resuming on `pause_turn`.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if matches!(response.stop_reason, Some(StopReason::PauseTurn)) {
            prompt.push_message(response)?;
            continue;
        }

        break response;
    };

    // Walk the result blocks the container produced, in order.
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
                    // Files written to `$OUTPUT_DIR` come back as `file_id`s;
                    // fetch their bytes via the Files API (#32).
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
            // The model's own narration; everything else is skipped here.
            Block::Text { text, .. } => println!("\n{text}"),
            _ => {}
        }
    }

    // The container persists for reuse: feed `answer.container` (its id) back
    // via `Prompt::container` on a later request to keep the same workspace.
    if let Some(container) = &answer.container {
        eprintln!("\n[container {} — reusable]", container.id);
    }

    Ok(())
}
