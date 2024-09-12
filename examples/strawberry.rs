//! An example of tool use and tool results. Language models are sometimes
//! unreasonably mocked since they cannot count letters within tokens (because
//! they do not see words as humans do). This example demonstrates how easy it
//! is to overcome this with an assistive device in the form of a tool.

use std::io::BufRead;

use clap::Parser;
use misanthropic::{
    json,
    markdown::{self, ToMarkdown},
    request::{
        message::{Block, Role},
        Message,
    },
    response, tool, Client, Request, Tool,
};

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

/// Things that can go wrong.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Assistant did not call the tool.
    #[error("Assistant did not call the tool. Response: {message}")]
    NoToolCall { message: response::Message },
    /// Assistant called the wrong tool.
    #[error("Assistant called the tool incorrectly. Call: {call}")]
    MalformedToolCall { call: tool::Use },
}

/// Count the number of letters in a word (or any string).
pub fn count_letters(letter: char, string: String) -> usize {
    let letter = letter.to_ascii_lowercase();
    let string = string.to_ascii_lowercase();

    string.chars().filter(|c| *c == letter).count()
}

/// Handle the tool call. Returns a [`User`] message with the result.
///
/// [`User`]: Role::User
pub fn handle_tool_call(call: &tool::Use) -> Result<Message, Error> {
    if call.name != "count_letters" {
        return Err(Error::MalformedToolCall { call: call.clone() });
    }

    if let (Some(letter), Some(string)) = (
        call.input["letter"].as_str().and_then(|s| s.chars().next()),
        call.input["string"].as_str(),
    ) {
        let count = count_letters(letter, string.into());

        Ok(Message {
            role: Role::User,
            content: Block::ToolResult {
                tool_use_id: call.id.clone(),
                content: count.to_string().into(),
                is_error: false,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
            // A Content Block is always convertable into Content.
            .into(),
        })
    } else {
        // Optionally, we could always return a Message and inform the Assistant
        // that they called the tool incorrectly so they can try again.
        Err(Error::MalformedToolCall { call: call.clone() })
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

    // Craft our chat request, providing a Tool definition to call
    // `count_letters`. In the future this will be derivable from the function
    // signature and docstring. Like many things in our API, `Tool` is also
    // convertable from a `serde_json::Value`.
    let mut chat = Request::default().add_tool(Tool {
        name: "count_letters".into(),
        description: "Count the number of letters in a word.".into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "The letter to count",
                },
                "string": {
                    "type": "string",
                    "description": "The string to count letters in",
                },
            },
            "required": ["letter", "string"],
        }),
        #[cfg(feature = "prompt-caching")]
        cache_control: None,
    // Inform the assistant about their limitations.
    }).system("You are a helpful assistant. You cannot count letters in a word by yourself because you see in tokens, not letters. Use the `count_letters` tool to overcome this limitation.")
    // Add user input.
    .add_message(Message {
        role: Role::User,
        content: args.prompt.into(),
    });

    // Generate the next message in the chat.
    let message = client.message(&chat).await?;

    // Check if the Assistant called the Tool. The `stop_reason` must be
    // `ToolUse` and the last `Content` `Block` must be `ToolUse`.
    if let Some(call) = message.tool_use() {
        let result = handle_tool_call(call)?;
        // Append the tool request and result messages to the chat.
        chat.messages.push(message.into());
        chat.messages.push(result);
    } else {
        // The Assistant did not call the tool. This may not be an error if the
        // user did not ask for the tool to be used, in which case it could be
        // handled as a normal message.
        return Err(Error::NoToolCall { message }.into());
    }

    let message = client.message(&chat).await?;

    if args.verbose {
        // Append the message and print the entire conversation as Markdown. The
        // default display also renders markdown, but without system prompt and
        // tool use information.
        chat.messages.push(message.into());
        println!(
            "{}",
            chat.markdown_custom(
                &markdown::Options::default()
                    .with_system()
                    .with_tool_use()
                    .with_tool_results()
            )
        );
    } else {
        // Just print the message content. The response `Message` contains the
        // `request::Message` with a `Role` and `Content`. The message can also
        // be printed directly, but this will include the `Role` header.
        println!("{}", message.message.content);
    }

    Ok(())
}
