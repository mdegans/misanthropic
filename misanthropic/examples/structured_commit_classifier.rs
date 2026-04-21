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
//! # Field order and chain-of-thought
//!
//! schemars preserves source-code order for struct fields, and
//! Anthropic's constrained decoding emits required fields in schema
//! order. That means field order *is* the generation order, which in
//! turn acts as inline chain-of-thought for the model.
//!
//! [`summary`] is declared before [`category`] so the model describes
//! what the diff does before committing to a conventional-commit
//! category label — otherwise `category` gets picked first and the
//! summary becomes post-hoc justification. The effect is most visible
//! on smaller models.
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
//! [`summary`]: CommitClassification::summary
//! [`category`]: CommitClassification::category

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
///
/// Field order is deliberate: the model generates [`summary`] first (a
/// description of what the diff does), then [`category`] (a label
/// informed by that description), then the remaining fields. See the
/// module-level docs for the reasoning.
///
/// [`summary`]: CommitClassification::summary
/// [`category`]: CommitClassification::category
#[derive(Debug, Deserialize, JsonSchema)]
struct CommitClassification {
    /// Imperative one-line summary of the change, 70 characters or
    /// fewer, with no trailing period. Example: "Add cache_1h variant
    /// for 1-hour TTL". Do NOT include the category prefix. Generated
    /// first so the model describes the change before labeling it.
    summary: String,
    /// Conventional-commit category that best describes this diff,
    /// chosen after articulating the summary above.
    category: Category,
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
