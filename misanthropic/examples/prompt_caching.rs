//! Example: **prompt caching** with [`CachedPrompt`] — pay once to write a
//! large prefix into the cache, then re-use it at a fraction of the price on
//! every following turn.
//!
//! The cache is keyed on the request prefix (`tools` → `system` →
//! `messages`), so mutating any prefix field after the first request turns
//! every later request back into a full-price cache *write*. [`CachedPrompt`]
//! makes that mistake a compile error: [`CachedPrompt::cached`] adds a
//! 5-minute breakpoint after the system prompt and freezes the prefix behind
//! the wrapper, leaving only cache-safe operations like
//! [`CachedPrompt::push_message`].
//!
//! This is the *manual placement* side of caching: a big fixed prefix, one
//! deliberate breakpoint, mutation forbidden. Its sibling is
//! [`Prompt::auto_cache`], where the API re-places the breakpoint at the end
//! of the prompt on every request — right for growing conversations (the
//! `swarm` example uses it); manual placement is right when the breakpoint
//! must sit at a *specific* position, like the fixed-prefix/varying-suffix
//! shape here.
//!
//! The demo is self-referential: the system prompt embeds the crate README
//! *and the `prompt::cached` module source* (large enough to clear every
//! model's minimum cacheable prefix length), then asks two questions about
//! caching. Watch the usage line flip between turns: turn one reports a
//! large `cache write`, turn two reports the same tokens as `cache read`
//! (billed at a fraction of the input price).
//!
//! ```sh
//! cargo run --features client --example prompt_caching
//!
//! # Or ask your own questions about the crate:
//! cargo run --features client --example prompt_caching -- \
//!     "How do I stream a response?" "What does into_static do?"
//! ```
//!
//! [`CachedPrompt`]: misanthropic::CachedPrompt
//! [`CachedPrompt::cached`]: misanthropic::CachedPrompt::cached
//! [`CachedPrompt::push_message`]: misanthropic::CachedPrompt::push_message
//! [`Prompt::auto_cache`]: misanthropic::Prompt::auto_cache

mod utils;

use clap::Parser;
use misanthropic::{CachedPrompt, Client, Id, Prompt, prompt::message::Role};

/// The large, stable prefix worth caching: instructions plus the documents
/// they refer to. `concat!` + `include_str!` bakes the crate's own docs in at
/// compile time — roughly ten thousand tokens, comfortably past the cache's
/// model-dependent minimum prefix length.
const SYSTEM: &str = concat!(
    "You are an expert on the `misanthropic` Rust crate. Answer questions \
     using the crate README and the `prompt::cached` module source below. \
     Be concise and refer to items by name.\n\n# README.md\n\n",
    include_str!("../README.md"),
    "\n\n# src/prompt/cached.rs\n\n```rust\n",
    include_str!("../src/prompt/cached.rs"),
    "\n```",
);

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Multi-turn Q&A over the crate's own docs with the large prefix \
             cached: full-price cache write on turn one, cheap cache reads \
             after."
)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Questions to ask, one turn each. At least two turns are needed to
    /// see a cache read.
    #[arg(default_values_t = [
        "What bug does CachedPrompt turn into a compile error, and how?"
            .to_string(),
        "Which operations stay available on a CachedPrompt, and why are \
         they cache-safe?"
            .to_string(),
    ])]
    questions: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    let client = Client::new(utils::api_key()?)?;

    // Wrap and drop the default 5-minute breakpoint after the system prompt.
    // From here on the prefix is immutable — `prompt.system(…)` or
    // `prompt.model(…)` simply don't exist on the wrapper.
    let mut prompt = CachedPrompt::cached(
        args.common
            .configure(Prompt::default().model(Id::Haiku45).system(SYSTEM)),
    );

    for (i, question) in args.questions.iter().enumerate() {
        println!("\n## Q{}: {question}\n", i + 1);

        prompt.push_message((Role::User, question.clone()))?;
        let message = client.message(&prompt).await?;
        println!("{message}");

        // On turn one, `cache write` covers the whole system prefix; on
        // later turns (within the TTL) the same tokens show up as `cache
        // read` instead.
        let usage = &message.usage;
        println!(
            "\n(input: {} | cache write: {} | cache read: {} | output: {})",
            usage.input_tokens,
            usage.cache_creation_input_tokens.unwrap_or(0),
            usage.cache_read_input_tokens.unwrap_or(0),
            usage.output_tokens,
        );

        // The response converts straight back into a prompt message.
        prompt.push_message(message)?;
    }

    Ok(())
}
