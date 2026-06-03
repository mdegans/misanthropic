//! Example: the **`web_search` server tool** ([`ServerTool::WebSearch`]).
//!
//! Unlike a custom tool (which you execute via [`Tool::call`]), a *server tool*
//! is run by Anthropic. You add it to the prompt and the model issues search
//! queries internally; the [`ServerToolUse`] call and its
//! [`WebSearchToolResult`] come back *in the response*, and the model cites its
//! sources on the response [`Text`] blocks. You never handle execution and
//! never return a [`tool::Result`].
//!
//! ## `pause_turn`
//!
//! A long-running search can make the API yield mid-turn with
//! [`StopReason::PauseTurn`]. To continue, you send the paused assistant turn
//! back — keeping the same tools — and the model picks up where it left off.
//! Across *several* pauses this produces consecutive assistant turns, which the
//! crate's user/assistant alternation check rejects, so the loop below appends
//! the continuation straight to [`Prompt::messages`] (the escape hatch) rather
//! than via [`push_message`].
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example web_search -- \
//!     "What did Anthropic announce most recently?"
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [`ServerTool::WebSearch`]: misanthropic::tool::ServerTool::WebSearch
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`ServerToolUse`]: misanthropic::prompt::message::Block::ServerToolUse
//! [`WebSearchToolResult`]: misanthropic::prompt::message::Block::WebSearchToolResult
//! [`Text`]: misanthropic::prompt::message::Block::Text
//! [`tool::Result`]: misanthropic::tool::Result
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`Prompt::messages`]: misanthropic::Prompt::messages
//! [`push_message`]: misanthropic::Prompt::push_message

use std::io::{BufRead, stdin};

use misanthropic::{
    AnthropicModel, Client, Prompt,
    prompt::message::Role,
    response::StopReason,
    tool::{ServerTool, WebSearch},
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
        "What did Anthropic announce most recently, and when?".to_string()
    });

    // Add the web_search server tool. The model decides whether and how often
    // to search (capped by `max_uses`); we never run anything ourselves.
    let mut prompt = Prompt::default()
        .model(AnthropicModel::Opus48)
        .add_message((Role::User, question))?
        .add_server_tool(ServerTool::web_search(WebSearch {
            max_uses: Some(5),
            ..Default::default()
        }));

    // Drive the server-side loop to completion, resuming on `pause_turn`.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if matches!(response.stop_reason, Some(StopReason::PauseTurn)) {
            // The API paused mid-turn after a search. Append the partial
            // assistant turn and resend to let it continue. A second pause
            // would make two assistant turns adjacent — which `push_message`
            // rejects — so push to `messages` directly.
            prompt.messages.push(response.into());
            continue;
        }

        break response;
    };

    // The model's answer, with citations rendered inline by `Display`.
    println!("{}", answer.inner.content);

    if let Some(usage) = answer.usage.server_tool_use {
        eprintln!("\n[{} web searches]", usage.web_search_requests);
    }

    Ok(())
}
