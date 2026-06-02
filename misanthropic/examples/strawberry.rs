//! An example of *typed* tool use. Language models are sometimes unreasonably
//! mocked since they cannot count letters within tokens (because they do not
//! see words as humans do). This example gives the assistant an assistive
//! device — a `count_letters` tool — built with the [`tool`] macro.
//!
//! The win over hand-written tools: declare an `Args` struct and an annotated
//! `async fn`, and the JSON schema, argument deserialization, and validation
//! are all generated. No hand-written schema, no `call.input["letter"]` fishing.
//!
//! [`tool`]: misanthropic::tool::tool

// Note: This example uses blocking calls for simplicity such as `println!()`
// and `stdin().lock()`. In a real application, these should *usually* be
// replaced with async alternatives.
use std::io::BufRead;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    markdown::ToMarkdown,
    prompt::message::{Content, Role},
    tool::{Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Count the number of letters in a word (or any string). An example of tool
/// use and tool results.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// User prompt.
    #[arg(
        short,
        long,
        default_value = "Count the number of r's in 'strawberry'"
    )]
    prompt: String,
    /// Show tool use.
    #[arg(long)]
    verbose: bool,
}

/// Arguments for the `count_letters` method. The field docs become the schema
/// property descriptions the model sees (via `schemars`).
#[derive(Debug, Deserialize, JsonSchema)]
struct CountLetters {
    /// The letter to count.
    letter: char,
    /// The string to count letters in.
    string: String,
}

/// A stateless tool that counts letters. The [`tool`] macro generates the
/// `Method`/`ToolArgs`/`Methods` wiring *and* a concrete `impl Tool` from the
/// annotated method below — so `Strawberry` is usable as a tool directly, no
/// wrapper. Each tagged fn stays a real inherent method it delegates to.
///
/// [`tool`]: misanthropic::tool::tool
struct Strawberry;

#[tool]
impl Strawberry {
    /// Count the occurrences of a letter in a string.
    #[method]
    async fn count_letters(
        &mut self,
        args: CountLetters,
    ) -> Result<Content<'static>, Content<'static>> {
        let letter = args.letter.to_ascii_lowercase();
        let count = args
            .string
            .chars()
            .filter(|c| c.to_ascii_lowercase() == letter)
            .count();

        Ok(count.to_string().into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the command line arguments.
    let args = Args::parse();

    // Get API key from stdin.
    println!("Enter your API key:");
    let key = std::io::stdin().lock().lines().next().unwrap()?;

    // Create a client. `key` will be consumed and zeroized.
    let client = Client::new(key)?;

    // Our typed tool, used directly — the `#[tool]` macro gave `Strawberry` a
    // concrete `impl Tool`. Its `definitions()` are the wire schemas — derived
    // from `CountLetters` — that we hand to the model.
    let mut strawberry = Strawberry;

    let mut chat = Prompt::default()
        // Inform the assistant about their limitations.
        .set_system("You are a helpful assistant. You cannot count letters in a word by yourself because you see in tokens, not letters. Use the `count_letters` tool to overcome this limitation.")
        // Add user input.
        .add_message((Role::User, args.prompt))?;

    // Register the tool's generated definition(s) with the prompt. The method
    // is namespaced by the tool's name, e.g. `Strawberry__count_letters`.
    for definition in strawberry.definitions() {
        chat = chat.add_tool(definition);
    }

    // Generate the next message in the chat.
    let message = client.message(&chat).await?;

    // Check if the Assistant called the Tool. The `stop_reason` must be
    // `ToolUse` and the last `Content` `Block` must be `ToolUse`.
    if let Some(call) = message.tool_use() {
        // Own the call so we can append the assistant's message first.
        let call = call.clone().into_static();
        chat.push_message(message)?;

        // Typed dispatch: `Use.input` is deserialized into `CountLetters` and
        // validated for us — bad arguments become a helpful, model-facing
        // error automatically, no hand-parsing required.
        let result = strawberry.call(call).await;
        chat.push_message(result)?;
    } else {
        // The Assistant did not call the tool. This may not be an error if the
        // user did not ask for the tool to be used, in which case it could be
        // handled as a normal message.
        return Err("Tool was not called".into());
    }

    let message = client.message(&chat).await?;

    if args.verbose {
        // Append the message and print the entire conversation as Markdown. The
        // default display also renders markdown, but without system prompt and
        // tool use information.
        chat.push_message(message)?;
        println!("{}", chat.markdown_verbose());
    } else {
        // Just print the message content. The response `Message` contains the
        // `request::Message` with a `Role` and `Content`. The message can also
        // be printed directly, but this will include the `Role` header.
        println!("{}", message.inner.content);
    }

    Ok(())
}
