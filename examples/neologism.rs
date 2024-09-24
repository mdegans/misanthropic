//! See `source` for an example of [`Client::message`] using the "neologism
//! creator" prompt. For a streaming example, see the `website_wizard` example.

// Note: This example uses blocking calls for simplicity such as `println!()`
// and `stdin().lock()`. In a real application, these should *usually* be
// replaced with async alternatives.
use clap::Parser;
use misanthropic::{prompt::message::Role, Client, Prompt};
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

    // Create a client. The key is encrypted in memory and source string is
    // zeroed. When requests are made, the key header is marked as sensitive.
    let client = Client::new(key)?;

    // Request a completion. `json!` can be used, the `Request` builder pattern,
    // or anything serializable. Many common usage patterns are supported out of
    // the box for building `Request`s, such as messages from a list of tuples
    // of `Role` and `String`.
    let message = client
        .message(Prompt::default().messages([(Role::User, args.prompt)]))
        .await?;

    println!("{}", message);

    Ok(())
}
