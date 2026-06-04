//! Example: the **`web_fetch` server tool** ([`ServerTool::WebFetch`]).
//!
//! Like [`web_search`], `web_fetch` is run by Anthropic: you add it to the
//! prompt and the model fetches a URL itself, receiving the page (or PDF) as a
//! [`WebFetchToolResult`] block *in the response*. With
//! [`citations`](WebFetch::citations) enabled the model cites passages from the
//! fetched document on its response [`Text`] blocks.
//!
//! For security the model may only fetch a URL that already appeared in the
//! conversation. Pass one in your message — or pair `web_fetch` with
//! [`web_search`] so the model can *find* a page and then fetch it, which is
//! what this example does.
//!
//! ## `pause_turn`
//!
//! A fetch (or search) can make the API yield mid-turn with
//! [`StopReason::PauseTurn`]; send the paused assistant turn back to continue.
//! See the [`web_search`] example for the same loop.
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example web_fetch -- \
//!     "https://www.rust-lang.org and summarize what Rust is"
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [`ServerTool::WebFetch`]: misanthropic::tool::ServerTool::WebFetch
//! [`WebFetch::citations`]: misanthropic::tool::WebFetch::citations
//! [`web_search`]: misanthropic::tool::ServerTool::web_search
//! [`WebFetchToolResult`]: misanthropic::prompt::message::Block::WebFetchToolResult
//! [`Text`]: misanthropic::prompt::message::Block::Text
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn

use std::io::{BufRead, stdin};

use misanthropic::{
    AnthropicModel, Client, Prompt,
    prompt::message::{CitationsConfig, Role},
    response::StopReason,
    tool::{ServerTool, WebFetch, WebSearch},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

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

    let question = std::env::args().nth(1).unwrap_or_else(|| {
        "Fetch https://www.rust-lang.org and summarize what Rust is, \
         with citations."
            .to_string()
    });

    // Add both server tools: `web_search` to locate pages and `web_fetch` to
    // read them. Citations are off by default for `web_fetch`, so opt in.
    let mut prompt = Prompt::default()
        .model(AnthropicModel::Haiku45)
        .add_message((Role::User, question))?
        .add_server_tool(ServerTool::web_search(WebSearch {
            max_uses: Some(3),
            ..Default::default()
        }))
        .add_server_tool(ServerTool::web_fetch(WebFetch {
            max_uses: Some(5),
            citations: Some(CitationsConfig { enabled: true }),
            ..Default::default()
        }));

    // Drive the server-side loop to completion, resuming on `pause_turn`.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if matches!(response.stop_reason, Some(StopReason::PauseTurn)) {
            prompt.push_message(response)?;
            continue;
        }

        break response;
    };

    // The model's answer, with citations rendered inline by `Display`.
    println!("{}", answer.inner.content);

    if let Some(usage) = answer.usage.server_tool_use {
        eprintln!(
            "\n[{} web searches, {} web fetches]",
            usage.web_search_requests, usage.web_fetch_requests
        );
    }

    Ok(())
}
