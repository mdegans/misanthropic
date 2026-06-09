//! Example: the **`web_fetch` server tool** ([`ServerMethodDef::WebFetch`])
//! paired with [`web_search`]. Anthropic fetches the URL for the model, which
//! receives the page as a [`WebFetchToolResult`] block; with
//! [`WebFetch::citations`] enabled the model cites passages on its [`Text`]
//! blocks. The model may only fetch a URL already in the conversation — pair
//! with `web_search` so it can find one first. A fetch can yield
//! [`StopReason::PauseTurn`] mid-turn; send the paused turn back to continue.
//!
//! ```sh
//! cargo run --features client --example web_fetch -- \
//!     "https://www.rust-lang.org and summarize what Rust is"
//! ```
//!
//! [`ServerMethodDef::WebFetch`]: misanthropic::tool::ServerMethodDef::WebFetch
//! [`WebFetch::citations`]: misanthropic::tool::WebFetch::citations
//! [`web_search`]: misanthropic::tool::ServerMethodDef::web_search
//! [`WebFetchToolResult`]: misanthropic::prompt::message::Block::WebFetchToolResult
//! [`Text`]: misanthropic::prompt::message::Block::Text
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt,
    prompt::message::{CitationsConfig, Role},
    response::StopReason,
    tool::{ServerMethodDef, WebFetch, WebSearch},
};

/// Demonstrate the `web_fetch` server tool paired with `web_search`.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The question / URL to fetch and summarize.
    question: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let question = cli.question.unwrap_or_else(|| {
        "Fetch https://www.rust-lang.org and summarize what Rust is, \
         with citations."
            .to_string()
    });

    // Citations are off by default for `web_fetch` — opt in explicitly.
    let mut prompt = cli
        .common
        .configure(Prompt::default().model(Id::Haiku45))
        .add_message((Role::User, question))?
        .add_tool(ServerMethodDef::web_search(WebSearch {
            max_uses: Some(3),
            ..Default::default()
        }))
        .add_tool(ServerMethodDef::web_fetch(WebFetch {
            max_uses: Some(5),
            citations: Some(CitationsConfig { enabled: true }),
            ..Default::default()
        }));

    // Resume on `pause_turn`; otherwise the turn is done.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if matches!(response.stop_reason, Some(StopReason::PauseTurn)) {
            prompt.push_message(response)?;
            continue;
        }

        break response;
    };

    println!("{}", answer.inner.content);

    if let Some(usage) = answer.usage.server_tool_use {
        eprintln!(
            "\n[{} web searches, {} web fetches]",
            usage.web_search_requests, usage.web_fetch_requests
        );
    }

    Ok(())
}
