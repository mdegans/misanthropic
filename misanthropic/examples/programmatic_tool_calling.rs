//! Example: **programmatic tool calling** (PTC) — calling client-side tools
//! from inside Anthropic's [`code_execution`] container. Mark a
//! [`CustomMethodDef`] with
//! [`.programmatic()`](misanthropic::tool::MethodBuilder::programmatic); the
//! model writes Python that calls it in a loop, each call pauses the turn and
//! delivers a [`tool_use`] with
//! [`caller`](misanthropic::tool::Use::caller)`=code_execution_20260120`.
//! Intermediate results stay inside the container; only the final stdout
//! ([`CodeExecutionToolResult`]) enters the model's context. Two differences
//! from ordinary tool use: each resume must target the same container
//! ([`response.container`] → [`Prompt::container`]); the answering user turn
//! must contain *only* `tool_result` blocks. Needs Opus/Sonnet 4.5+ (not
//! Haiku).
//!
//! ```sh
//! cargo run --features client --example programmatic_tool_calling
//! ```
//!
//! [`code_execution`]: misanthropic::tool::ServerMethodDef::code_execution
//! [`CustomMethodDef`]: misanthropic::tool::CustomMethodDef
//! [`tool_use`]: misanthropic::prompt::message::Block::ToolUse
//! [`CodeExecutionToolResult`]: misanthropic::prompt::message::Block::CodeExecutionToolResult
//! [`response.container`]: misanthropic::response::Message::container
//! [`Prompt::container`]: misanthropic::Prompt::container

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt, json,
    prompt::message::{Block, Role},
    response::StopReason,
    tool::{self, Caller, CustomMethodDef, KnownCaller, ServerMethodDef},
};

/// Run a sales-query task using programmatic tool calling from code execution.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
}

/// Stand-in for a real data source; a real tool would hit a database.
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
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    // Only callable from code execution (`allowed_callers`).
    let query_sales_tool = CustomMethodDef::builder("query_sales")
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

    let mut prompt = cli
        .common
        .configure(Prompt::default().model(Id::Sonnet46))
        .add_tool(ServerMethodDef::code_execution())
        .add_tool(query_sales_tool)
        .add_message((
            Role::User,
            "For the West, East, Central, North, and South regions, call \
             query_sales to get each revenue, then tell me which region had \
             the highest. Do all the lookups in code.",
        ))?;

    // Each programmatic call pauses the turn; resume the *same* container.
    let answer = loop {
        let response = client.message(&prompt).await?;

        if let Some(container) = &response.container {
            prompt.container = Some(container.id.clone());
        }

        if !matches!(response.stop_reason, Some(StopReason::ToolUse)) {
            break response;
        }

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

        // Answer with *only* tool_result blocks (API requirement).
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

    println!("{}", answer.inner.content);

    Ok(())
}
