//! See `source` for an example of [`Client::stream`] using the "website wizard"
//! prompt. For a non-streaming example, see the `neologism` example.

// Note: This example uses blocking calls for simplicity such as `print`
// `read_to_string`, `stdin().lock()`, and `write`. In a real application, these
// should usually be replaced with async alternatives.

use clap::Parser;
use futures::TryStreamExt;
use misanthropic::{
    json, request::message::Role, stream::FilterExt, Client, Model,
};
use std::{
    io::{stdin, BufRead},
    path::PathBuf,
};

/// Generate a one-page website based on the given specifications. This is
/// adapted from the "Prompt library" example:
///
/// https://docs.anthropic.com/en/prompt-library/website-wizard
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Specification text.
    #[arg(short, long)]
    specs: PathBuf,
    /// Output file.
    #[arg(short, long)]
    output: PathBuf,
    /// Maximum tokens.
    #[arg(short, long, default_value = "4000")]
    max_tokens: u64,
    /// System prompt
    #[arg(
        long,
        default_value = "Your task is to create a one-page website based on the given specifications, delivered as an HTML file with embedded JavaScript and CSS. The website should incorporate a variety of engaging and interactive design features, such as drop-down menus, dynamic text and content, clickable buttons, and more. Ensure that the design is visually appealing, responsive, and user-friendly. The HTML, CSS, and JavaScript code should be well-structured, efficiently organized, and properly commented for readability and maintainability."
    )]
    system: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    // Read the command line arguments.
    let args = Args::parse();

    // Read the specification text.
    let specs = std::fs::read_to_string(args.specs)?;

    // Get API key from stdin.
    println!("Enter your API key:");
    let key = stdin().lock().lines().next().unwrap()?;

    // Create a client. `key` will be consumed and zeroized.
    let client = Client::new(key)?;

    // Request a streaming completion. `json!` can be used, the concrete type,
    // `Request` or a combination of strings and concrete types like `Model`.
    // Client request methods accept anything serializable.
    let stream = client
        .stream(json!({
          "model": Model::Sonnet35,
          "max_tokens": args.max_tokens,
          "temperature": 0,
          "system": args.system,
          "messages": [
            {
              "role": Role::User,
              "content": specs,
            }
          ],
        }))
        .await?
        // Filter out rate limit and overloaded errors. This is optional but
        // recommended for most use cases. The stream will continue when the
        // server is ready. Otherwise the stream will include these errors.
        .filter_rate_limit()
        // Filter out everything but text pieces (and errors).
        .text();

    println!("Generating website...\n");
    // Collect the stream into a single string.
    let content: String = stream
        .map_ok(|piece| {
            print!("{}", &piece);
            piece
        })
        .try_collect()
        .await?;
    println!("\nWebsite generated.");

    // Write the output to a file.
    std::fs::write(args.output, content)?;

    Ok(())
}
