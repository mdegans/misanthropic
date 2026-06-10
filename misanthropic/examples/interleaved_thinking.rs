//! Example: *adaptive* extended thinking with *interleaved* thinking around
//! client-side tool use ([`Thinking::adaptive`], [`Block::Thought`],
//! [`Prompt`]). Adaptive thinking is the recommended mode on Claude 4 and the
//! only mode Opus 4.7+ accepts; with it the model thinks *between* tool calls
//! automatically. Thought blocks carry signatures and must round-trip back
//! unmodified — pushing the whole assistant message into [`Prompt`] handles
//! this. Pass `--effort` to control thinking depth. Requires Sonnet 4.6+ or
//! Opus 4.6+.
//!
//! ```sh
//! cargo run --features client,derive --example interleaved_thinking
//! ```
//!
//! [`Thinking::adaptive`]: misanthropic::prompt::Thinking::adaptive
//! [`Block::Thought`]: misanthropic::prompt::message::Block::Thought
//! [`Prompt`]: misanthropic::Prompt

mod utils;

use clap::{Parser, ValueEnum};
use misanthropic::{
    Client, Id, Prompt,
    prompt::{
        Effort, Thinking,
        message::{Block, Content, Role},
    },
    tool::{Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Watch the model reason between tool calls with adaptive interleaved
/// thinking.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// User prompt. The default needs several chained operations, so the model
    /// has to think, compute, then think about the intermediate result before
    /// the next step.
    #[arg(
        short,
        long,
        default_value = "Compute (12 + 30) * 7 - 100, one operation at a time."
    )]
    prompt: String,
    /// How hard the model works — text, tool calls, and thinking depth. The
    /// recommended thinking-depth control when paired with adaptive thinking.
    /// Omit to use the API default (`high`).
    #[arg(long)]
    effort: Option<EffortArg>,
    /// Stop after this many assistant turns, in case the loop never settles.
    #[arg(long, default_value_t = 10)]
    max_turns: usize,
}

/// CLI mirror of [`Effort`] — [`Effort`] isn't a `clap::ValueEnum`.
#[derive(Copy, Clone, Debug, ValueEnum)]
enum EffortArg {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl From<EffortArg> for Effort {
    fn from(arg: EffortArg) -> Self {
        match arg {
            EffortArg::Low => Effort::Low,
            EffortArg::Medium => Effort::Medium,
            EffortArg::High => Effort::High,
            EffortArg::XHigh => Effort::XHigh,
            EffortArg::Max => Effort::Max,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
struct Compute {
    /// Left operand.
    a: f64,
    /// Operator: one of `+`, `-`, `*`, `/`.
    op: String,
    /// Right operand.
    b: f64,
}

/// One binary operation per call — forces the model to chain and reason
/// about each intermediate result.
struct Calculator;

#[tool]
impl Calculator {
    /// Perform a single binary arithmetic operation and return the result.
    #[method]
    async fn compute(&mut self, args: Compute) -> Result<Content, Content> {
        let result = match args.op.as_str() {
            "+" => args.a + args.b,
            "-" => args.a - args.b,
            "*" => args.a * args.b,
            "/" if args.b != 0.0 => args.a / args.b,
            "/" => return Err("division by zero".into()),
            other => return Err(format!("unknown operator: {other}").into()),
        };

        Ok(result.to_string().into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let mut calc = Calculator;

    let mut chat = args
        .common
        .configure(
            Prompt::default()
                .model(Id::Sonnet46)
                // Adaptive: model decides depth; interleaves with tool use
                // automatically. No `budget_tokens` to set.
                .thinking(Thinking::adaptive())
                .system(
                    "You are a careful calculator's assistant. You cannot do \
                 arithmetic reliably in your head, so use the `compute` tool \
                 for every operation, one operation per call, and reason about \
                 each intermediate result before the next step.",
                ),
        )
        .add_message((Role::User, args.prompt))?;

    // Method is namespaced: `Calculator__compute`.
    for definition in calc.definitions() {
        chat = chat.add_tool(definition);
    }

    if let Some(effort) = args.effort {
        chat = chat.effort(effort.into());
    }

    // On Sonnet 4.6 the default `display` is summarized, so the thought we
    // print is the model's own summary.
    for _ in 0..args.max_turns {
        let message = client.message(&chat).await?;

        for block in message.inner.content.iter() {
            match block {
                Block::Thought { thought, .. } => {
                    println!("\n🧠 {thought}");
                }
                Block::ToolUse { call } if args.common.verbose => {
                    println!("🔧 {}({})", call.name, call.input);
                }
                _ => {}
            }
        }

        match message.tool_use() {
            // Append the whole turn (thoughts included) so signatures round-trip.
            Some(call) => {
                let call = call.clone();
                chat.push_message(message)?;
                let result = calc.call(call).await;
                if args.common.verbose {
                    println!("   = {}", result.content);
                }
                chat.push_message(result)?;
            }
            None => {
                println!("\n✅ {}", message.inner.content);
                return Ok(());
            }
        }
    }

    Err("model did not finish within max_turns".into())
}
