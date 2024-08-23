//! See `source` for an example of [`Client::message`] using the "neologism
//! creator" prompt. For a streaming example, see the `website_wizard` example.

// Note: This example uses blocking calls for simplicity such as `print`
// `read_to_string`, `stdin().lock()`, and `write`. In a real application, these
// should usually be replaced with async alternatives.

use clap::Parser;
use misanthropic::{
    request::{message::Role, Message},
    Client, Model, Request,
};
use std::io::{stdin, BufRead};

/// Invent new words and provide their definitions based on user-provided
/// concepts or ideas.
///
/// https://docs.anthropic.com/en/prompt-library/neologism-creator
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Prompt
    #[arg(
        short,
        long,
        default_value = "Can you help me create a new word for the act of pretending to understand something in order to avoid looking ignorant or uninformed?"
    )]
    prompt: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    // Read the command line arguments.
    let args = Args::parse();

    // Get API key from stdin.
    println!("Enter your API key:");
    let key = stdin().lock().lines().next().unwrap()?;

    // Create a client. `key` will be consumed and zeroized.
    let client = Client::new(key)?;

    // Request a completion. `json!` can be used, `Request` or a combination of
    // strings and types like `Model`. Client request methods accept anything
    // serializable for maximum flexibility.
    let message = client
        .message(Request {
            model: Model::Sonnet35,
            messages: vec![Message {
                role: Role::User,
                content: args.prompt.into(),
            }],
            max_tokens: 1000.try_into().unwrap(),
            metadata: serde_json::Value::Null,
            stop_sequences: None,
            stream: None,
            system: None,
            temperature: Some(1.0),
            tool_choice: None,
            tools: None,
            top_k: None,
            top_p: None,
        })
        .await?;

    println!("{}", message);

    Ok(())
}
