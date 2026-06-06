//! Example: analyze a social-network post and produce a structured
//! [`VoteIntent`].
//!
//! Reads a post body from `--post PATH` (or stdin), sends it to Claude
//! with [`Prompt::structured_output::<VoteIntent>()`], and prints the
//! parsed result. Demonstrates structured output with an enum, a
//! bounded-feeling `f32`, and a `Vec<String>` — the common shape of an
//! agent decision in an [Agora]-style governed social network.
//!
//! # Field order and chain-of-thought
//!
//! schemars preserves source-code order for struct fields, and
//! Anthropic's constrained decoding emits required fields in schema
//! order. That means field order *is* the generation order, which in
//! turn acts as inline chain-of-thought for the model.
//!
//! We deliberately declare [`rationale`] and [`concerns`] before
//! [`stance`] and [`confidence`] so the model reasons out loud before
//! committing to a decision — otherwise `stance` gets decided first and
//! `rationale` becomes post-hoc justification. The effect is biggest on
//! smaller models like Haiku; larger models tend to think ahead
//! regardless.
//!
//! # Usage
//!
//! ```sh
//! echo "The proposal would rename `Method` to `Function` for clarity." | \
//!     cargo run --features json-schema --example vote_intent
//!
//! cargo run --features json-schema --example vote_intent \
//!     -- --post post.md
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [Agora]: https://subliminal.technology/agora/hello-world
//! [`Prompt::structured_output::<VoteIntent>()`]:
//!     misanthropic::Prompt::structured_output
//! [`rationale`]: VoteIntent::rationale
//! [`concerns`]: VoteIntent::concerns
//! [`stance`]: VoteIntent::stance
//! [`confidence`]: VoteIntent::confidence

use std::io::{BufRead, Read, stdin};

use clap::Parser;
use misanthropic::{Client, Id, Prompt, prompt::message::Role};
use schemars::JsonSchema;
use serde::Deserialize;

/// How an agent decides to vote on a post or proposal.
///
/// `Approve` / `Reject` / `Abstain` mirrors the three-way vote common in
/// governance systems where abstention is distinct from non-participation.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Stance {
    /// Vote in favor. Pick this only if the post's claims hold up and
    /// the action it proposes is on balance good.
    Approve,
    /// Vote against. Pick this if the post is factually wrong, harmful,
    /// or the proposed action has serious downsides.
    Reject,
    /// Decline to vote. Pick this when you genuinely can't decide — not
    /// as a hedge for a weak opinion.
    Abstain,
}

/// Structured vote intent produced by an agent reasoning about a post.
///
/// Field order is deliberate: the model reasons in [`rationale`] and
/// [`concerns`] before committing to [`stance`] and [`confidence`].
/// See the module-level docs for why this matters.
///
/// [`rationale`]: VoteIntent::rationale
/// [`concerns`]: VoteIntent::concerns
/// [`stance`]: VoteIntent::stance
/// [`confidence`]: VoteIntent::confidence
#[derive(Debug, Deserialize, JsonSchema)]
struct VoteIntent {
    /// One-paragraph rationale, 2–4 sentences, written as if explaining
    /// your vote to another thoughtful agent. No hedging phrases like
    /// "as an AI"; just the reasoning. Generated first so the model
    /// thinks before deciding.
    rationale: String,
    /// Concrete concerns you'd want addressed even if the vote passes.
    /// Each entry is a single short sentence. Empty if no concerns.
    concerns: Vec<String>,
    /// How to vote, after weighing the rationale and concerns above.
    stance: Stance,
    /// Confidence in the stance, from 0.0 (coin flip) to 1.0 (certain).
    /// Pick numbers deliberately: 0.5 means you're on the fence, 0.9
    /// means you're highly confident, don't just emit 1.0 by default.
    confidence: f32,
}

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Analyze a post and produce a structured VoteIntent using Claude."
)]
struct Args {
    /// Path to a post body. If omitted, reads from stdin.
    #[arg(short, long)]
    post: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let args = Args::parse();

    let post = match args.post {
        Some(path) => std::fs::read_to_string(&path)?,
        None => {
            let mut buf = String::new();
            stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    if post.trim().is_empty() {
        return Err(
            "No post provided. Pipe text to stdin or pass --post PATH.".into(),
        );
    }

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

    let system = "You are a thoughtful agent participating in a governed \
        social network. Read the user-provided post and produce a \
        VoteIntent. Be willing to Reject if the post is poorly argued or \
        harmful; be willing to Abstain if you genuinely can't tell. \
        Don't default to Approve. Keep the rationale short and concrete.";

    let prompt = Prompt::default()
        .model(Id::Haiku45)
        .structured_output::<VoteIntent>()
        .set_system(system)
        .add_message((Role::User, format!("POST:\n\n{post}")))?;

    let response = client.message(&prompt).await?;
    let intent: VoteIntent = response.json()?;

    // Pretty-print for humans; machine consumers would just reuse the
    // struct directly.
    println!(
        "stance:     {}",
        match intent.stance {
            Stance::Approve => "approve",
            Stance::Reject => "reject",
            Stance::Abstain => "abstain",
        }
    );
    println!("confidence: {:.2}", intent.confidence);
    println!("rationale:  {}", intent.rationale);
    if !intent.concerns.is_empty() {
        println!("concerns:");
        for c in &intent.concerns {
            println!("  - {c}");
        }
    }

    Ok(())
}
