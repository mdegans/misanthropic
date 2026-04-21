//! Example: classify a unified diff into a structured commit message.
//!
//! Reads a diff from `--diff PATH` (or stdin if omitted), sends it to
//! Claude with [`Prompt::structured_output::<CommitClassification>()`], and
//! prints the result as a conventional-commit-style message.
//!
//! Dogfoods the [`Prompt::structured_output`] / [`Message::json`] pair
//! added in the `json-schema` feature: the [`CommitClassification`] struct
//! below is the same type the API sees (via [`schemars::JsonSchema`]) and
//! the same type we deserialize the response into (via
//! [`serde::Deserialize`]).
//!
//! # Usage
//!
//! ```sh
//! # Against the last commit
//! git diff HEAD~1 | cargo run --features json-schema \
//!     --example structured_commit_classifier
//!
//! # Against a file
//! cargo run --features json-schema --example structured_commit_classifier \
//!     -- --diff changes.patch
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output
//! [`Message::json`]: misanthropic::response::Message::json

use std::io::{BufRead, Read, stdin};

use clap::Parser;
use misanthropic::{AnthropicModel, Client, Prompt, prompt::message::Role};
use schemars::JsonSchema;
use serde::Deserialize;

/// Conventional-commit category for a diff.
///
/// Renamed to `snake_case` on the wire — Anthropic's schema subset
/// supports string enums, which schemars emits as an `anyOf` of `const`
/// variants once [`OutputConfig::for_type`] sanitizes the schema.
///
/// [`OutputConfig::for_type`]: misanthropic::prompt::OutputConfig::for_type
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Category {
    /// New feature or a new capability added to an existing feature.
    Feat,
    /// Bug fix — restores intended behavior.
    Fix,
    /// Internal rework with no behavioral change.
    Refactor,
    /// Documentation-only change.
    Docs,
    /// Test-only change.
    Test,
    /// Build system, tooling, or dependency change.
    Build,
    /// CI configuration change.
    Ci,
    /// Performance improvement with no behavioral change.
    Perf,
    /// Whitespace, formatting, or lint-only change.
    Style,
    /// Catch-all for anything that doesn't fit above.
    Chore,
}

/// Structured classification of a diff into commit-message components.
///
/// `#[derive(JsonSchema)]` propagates the `///` doc comments above each
/// field into the JSON Schema's `description` slots, which the model
/// uses as part of its constrained-decoding guidance.
#[derive(Debug, Deserialize, JsonSchema)]
struct CommitClassification {
    /// Conventional-commit category that best describes this diff.
    category: Category,
    /// Imperative one-line summary of the change, 70 characters or
    /// fewer, with no trailing period. Example: "Add cache_1h variant
    /// for 1-hour TTL". Do NOT include the category prefix.
    summary: String,
    /// True if this change likely alters the public API in a
    /// backwards-incompatible way (removed/renamed public items,
    /// changed signatures, new required fields on public structs).
    breaking: bool,
    /// Optional body paragraph explaining the "why" of the change.
    /// Wrap at roughly 72 columns. Empty string if no body needed.
    body: String,
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Classify a unified diff into a structured commit message using Claude."
)]
struct Args {
    /// Path to a diff file. If omitted, reads the diff from stdin.
    #[arg(short, long)]
    diff: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let args = Args::parse();

    let diff = match args.diff {
        Some(path) => std::fs::read_to_string(&path)?,
        None => {
            let mut buf = String::new();
            stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if diff.trim().is_empty() {
        return Err(
            "No diff provided. Pipe a diff to stdin or pass --diff PATH."
                .into(),
        );
    }

    // API key: env var first, then prompt. Matches the other examples'
    // posture of not assuming a particular secret store.
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

    let system = "You are an experienced code reviewer classifying a diff \
        into a conventional-commit-style message. Base the category on \
        what the diff actually does, not on the filenames. Keep the \
        summary short, imperative, and specific. Only mark `breaking` \
        true if the public API is altered incompatibly.";

    let prompt = Prompt::default()
        .model(AnthropicModel::Haiku45)
        .structured_output::<CommitClassification>()
        .set_system(system)
        .add_message((
            Role::User,
            format!("Classify this diff:\n\n```diff\n{diff}\n```"),
        ))?;

    let response = client.message(&prompt).await?;
    let classification: CommitClassification = response.json()?;

    // Render as a conventional-commit message.
    let prefix = match classification.category {
        Category::Feat => "feat",
        Category::Fix => "fix",
        Category::Refactor => "refactor",
        Category::Docs => "docs",
        Category::Test => "test",
        Category::Build => "build",
        Category::Ci => "ci",
        Category::Perf => "perf",
        Category::Style => "style",
        Category::Chore => "chore",
    };
    let bang = if classification.breaking { "!" } else { "" };
    println!("{prefix}{bang}: {}", classification.summary);
    if !classification.body.is_empty() {
        println!();
        println!("{}", classification.body);
    }

    Ok(())
}
