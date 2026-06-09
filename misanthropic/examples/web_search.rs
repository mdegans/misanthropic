//! Example: the **`web_search` server tool** ([`ServerMethodDef::WebSearch`]).
//! Anthropic runs the search; the [`ServerToolUse`] call and its
//! [`WebSearchToolResult`] come back in the response, and citations appear on
//! the response [`Text`] blocks — you never call [`Tool::call`] or return a
//! [`tool::Result`]. A long search can yield [`StopReason::PauseTurn`]
//! mid-turn; send the paused assistant turn back (it carries a server-tool-use
//! block, so [`push_message`] / [`TurnOrderError`] accept it) and the model
//! continues.
//!
//! ```sh
//! cargo run --features client --example web_search -- \
//!     "What did Anthropic announce most recently?"
//! ```
//!
//! [`ServerMethodDef::WebSearch`]: misanthropic::tool::ServerMethodDef::WebSearch
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`ServerToolUse`]: misanthropic::prompt::message::Block::ServerToolUse
//! [`WebSearchToolResult`]: misanthropic::prompt::message::Block::WebSearchToolResult
//! [`Text`]: misanthropic::prompt::message::Block::Text
//! [`tool::Result`]: misanthropic::tool::Result
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`push_message`]: misanthropic::Prompt::push_message
//! [`TurnOrderError`]: misanthropic::prompt::TurnOrderError

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt,
    prompt::message::Role,
    response::StopReason,
    tool::{ServerMethodDef, WebSearch},
};

/// Demonstrate the `web_search` server tool.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The question to answer via web search.
    question: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let question = cli.question.unwrap_or_else(|| {
        "What did Anthropic announce most recently, and when?".to_string()
    });

    // The model decides whether to search (capped by `max_uses`).
    let mut prompt = cli
        .common
        .configure(Prompt::default().model(Id::Haiku45))
        .add_message((Role::User, question))?
        .add_tool(ServerMethodDef::web_search(WebSearch {
            max_uses: Some(5),
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
        eprintln!("\n[{} web searches]", usage.web_search_requests);
    }

    Ok(())
}
