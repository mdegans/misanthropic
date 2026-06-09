//! Shared command-line flags for the examples.
//!
//! Composed with clap's `#[command(flatten)]`: an example derives its own
//! `Parser` and flattens [`CommonArgs`] (and [`ChatArgs`] for chat-loop
//! examples), adding only its own arguments. Examples that need nothing beyond
//! the common shape can parse [`Args`] directly.

use std::num::NonZeroU32;

use clap::{Args as ClapArgs, Parser};
use misanthropic::{
    Prompt,
    model::{Id, Model},
};

/// Flags shared by most examples. Flatten into an example's `Parser` and apply
/// with [`configure`](CommonArgs::configure).
#[derive(ClapArgs, Debug, Clone)]
pub struct CommonArgs {
    /// Model: a wire id (`claude-opus-4-8`) or an alias (`opus`/`sonnet`/
    /// `haiku`). Unknown strings pass through as a custom model.
    #[arg(short, long)]
    pub model: Option<String>,

    /// Override `max_tokens` (default: the crate's 4096).
    #[arg(long)]
    pub max_tokens: Option<NonZeroU32>,

    /// Override the example's built-in system prompt.
    #[arg(long)]
    pub system: Option<String>,

    /// Verbose output (the example decides what that means).
    #[arg(long)]
    pub verbose: bool,
}

impl CommonArgs {
    /// Apply whichever of `--model` / `--max-tokens` / `--system` the user set
    /// onto `prompt`, leaving the example's own defaults for the rest.
    pub fn configure(&self, mut prompt: Prompt) -> Prompt {
        if let Some(model) = &self.model {
            prompt = prompt.model(resolve_model(model));
        }
        if let Some(max_tokens) = self.max_tokens {
            prompt = prompt.max_tokens(max_tokens);
        }
        if let Some(system) = &self.system {
            prompt = prompt.set_system(system.clone());
        }
        prompt
    }
}

/// Map the friendly aliases to current wire ids; anything else passes straight
/// through to [`Model`] (an unknown string becomes a custom model).
fn resolve_model(s: &str) -> Model {
    match s.to_ascii_lowercase().as_str() {
        "opus" => Id::Opus48.into(),
        "sonnet" => Id::Sonnet46.into(),
        "haiku" => Id::Haiku45.into(),
        other => Model::from(other.to_owned()),
    }
}

/// Extra flag for chat-loop examples: the consecutive-tool-call cap.
#[derive(ClapArgs, Debug, Clone)]
pub struct ChatArgs {
    /// Cap consecutive tool-call rounds within one user beat (default: 8).
    #[arg(long)]
    pub max_tool_calls: Option<usize>,
}

#[cfg(feature = "client")]
impl ChatArgs {
    /// Apply `--max-tool-calls` onto a [`Chat`](super::Chat) if set, else leave
    /// the driver's default.
    pub fn configure<S>(&self, chat: super::Chat<S>) -> super::Chat<S> {
        match self.max_tool_calls {
            Some(max) => chat.max_consecutive_tool_calls(max),
            None => chat,
        }
    }
}

/// The common chat-example shape — [`CommonArgs`] + [`ChatArgs`] + a prompt.
/// Examples needing nothing more can `Args::parse()` directly.
#[derive(Parser, Debug)]
pub struct Args {
    #[command(flatten)]
    pub common: CommonArgs,

    #[command(flatten)]
    pub chat: ChatArgs,

    /// The initial user prompt / question.
    #[arg(short, long)]
    pub prompt: Option<String>,
}
