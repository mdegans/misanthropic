//! Example: **programmatic tool calling** (PTC).
//!
//! PTC lets the model call *your* tools from inside Anthropic's
//! [`code_execution`] container instead of one model round-trip per call. You
//!
//! 1. add the [`code_execution`] server tool, and
//! 2. mark a custom [`MethodDef`] callable from it with
//!    [`programmatic`](misanthropic::tool::MethodBuilder::programmatic)
//!    (`allowed_callers: ["code_execution_20260120"]`).
//!
//! The model writes Python that calls the tool in a loop. Each call **pauses
//! the turn** and hands you a [`tool_use`] with a
//! [`caller`](misanthropic::tool::Use::caller) of `code_execution_20260120`
//! (vs. a direct call) — you run it and answer exactly as you would any
//! client-side tool. The intermediate results never enter the model's context;
//! only the container's final stdout (a [`CodeExecutionToolResult`]) does.
//!
//! Two things differ from ordinary tool use:
//! - **Container reuse is required.** The paused container is mid-run, so each
//!   resume must target the *same* one: copy [`response.container`] into
//!   [`Prompt::container`].
//! - **The answering user turn must contain *only* `tool_result` blocks** — no
//!   trailing text — while a programmatic call is pending. (The API enforces
//!   this; here we simply send one result and nothing else.)
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example programmatic_tool_calling
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin. PTC
//! requires a `code_execution_20260120`-capable model (Opus/Sonnet 4.5+); it is
//! not available on Haiku.
//!
//! [`code_execution`]: misanthropic::tool::ServerTool::code_execution
//! [`MethodDef`]: misanthropic::tool::MethodDef
//! [`tool_use`]: misanthropic::prompt::message::Block::ToolUse
//! [`CodeExecutionToolResult`]: misanthropic::prompt::message::Block::CodeExecutionToolResult
//! [`response.container`]: misanthropic::response::Message::container
//! [`Prompt::container`]: misanthropic::Prompt::container

use std::io::{BufRead, stdin};

use misanthropic::{
    AnthropicModel, Client, Prompt, json,
    prompt::message::{Block, Role},
    response::StopReason,
    tool::{self, Caller, KnownCaller, MethodDef, ServerTool},
};

/// Stand-in for a real data source: revenue per region. A real tool would hit a
/// database; the point is that this runs on *your* side, called from the
/// model's container.
fn query_sales(region: &str) -> String {
    let revenue = match region {
        "West" => 98_000,
        "East" => 72_000,
        "Central" => 65_000,
        "North" => 54_000,
        "South" => 81_000,
        _ => 0,
    };
    json!({ "region": region, "revenue": revenue }).to_string()
}

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

    // A custom tool the model may call *only* from code execution.
    let query_sales_tool = MethodDef::builder("query_sales")
        .description(
            "Look up sales revenue for a region. Returns a JSON object like \
             {\"region\": \"West\", \"revenue\": 12345}.",
        )
        .schema(json!({
            "type": "object",
            "properties": {
                "region": { "type": "string", "description": "Region name" }
            },
            "required": ["region"],
        }))
        .programmatic()
        .build()?;

    let mut prompt = Prompt::default()
        .model(AnthropicModel::Sonnet46)
        .add_server_tool(ServerTool::code_execution())
        .add_tool(query_sales_tool)
        .add_message((
            Role::User,
            "For the West, East, Central, North, and South regions, call \
             query_sales to get each revenue, then tell me which region had \
             the highest. Do all the lookups in code.",
        ))?;

    // Drive the container loop: each programmatic call pauses the turn; we run
    // the tool and resume the same container until the code finishes.
    let answer = loop {
        let response = client.message(&prompt).await?;

        // Resume the *same* container on the next request (required: it is
        // paused mid-run waiting for our result).
        if let Some(container) = &response.container {
            prompt.container = Some(container.id.clone());
        }

        if !matches!(response.stop_reason, Some(StopReason::ToolUse)) {
            break response;
        }

        // Pull the pending programmatic tool calls out before we hand the
        // assistant turn back to the prompt.
        let calls: Vec<tool::Use> = response
            .inner
            .content
            .iter()
            .filter_map(|block| match block {
                Block::ToolUse { call }
                    if matches!(
                        &call.caller,
                        Some(Caller::Known(
                            KnownCaller::CodeExecution20260120 { .. }
                        ))
                    ) =>
                {
                    Some(call.clone())
                }
                _ => None,
            })
            .collect();

        // Echo the paused assistant turn, then answer each call with a turn of
        // *only* tool_result blocks.
        prompt.push_message(response)?;
        for call in calls {
            let region =
                call.input["region"].as_str().unwrap_or_default().to_owned();
            eprintln!("[query_sales({region})]");
            prompt.push_message(tool::Result::new(
                call.id,
                query_sales(&region),
            ))?;
        }
    };

    // Only the container's final stdout reached the model's context — the
    // per-region results were filtered in code.
    println!("{}", answer.inner.content);

    Ok(())
}
