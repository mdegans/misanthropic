//! An example of *adaptive* extended thinking with *interleaved* thinking
//! around client-side tool use.
//!
//! [`Thinking::adaptive`] lets the model decide how much to reason per request
//! — the recommended mode on Claude 4 and the only mode Opus 4.7+ accept. With
//! adaptive thinking the model also thinks *between* tool calls automatically:
//! it reasons, calls a tool, sees the result, reasons again about that result,
//! then calls the next tool — instead of planning everything blind up front.
//!
//! For *client-side* tools (like the calculator below) the tool result returns
//! in a new user message, so the interleaving plays out across turns: each new
//! assistant turn opens with a fresh [`Block::Thought`]. Those thought blocks
//! carry signatures and must round-trip back unmodified, which is exactly what
//! pushing the whole assistant message back into the [`Prompt`] does.
//!
//! Pass `--effort low|medium|high|xhigh|max` to control how hard the model
//! works (text, tool calls, *and* thinking depth) — the recommended
//! thinking-depth knob when paired with adaptive thinking.
//!
//! Adaptive + interleaved thinking requires Sonnet 4.6 or an Opus 4.6+ model;
//! this example uses Sonnet 4.6 as the cheapest option.
//!
//! [`Thinking::adaptive`]: misanthropic::prompt::Thinking::adaptive
//! [`Block::Thought`]: misanthropic::prompt::message::Block::Thought
//! [`Prompt`]: misanthropic::Prompt

// Note: This example uses blocking calls for simplicity such as `println!()`
// and `stdin().lock()`. In a real application, these should *usually* be
// replaced with async alternatives.
use std::io::BufRead;

use clap::{Parser, ValueEnum};
use misanthropic::{
    AnthropicModel, Client, Prompt,
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
    /// User prompt. The default needs several chained operations, so the model
    /// has to think, compute, then think about the intermediate result before
    /// the next step.
    #[arg(
        short,
        long,
        default_value = "Compute (12 + 30) * 7 - 100, one operation at a time."
    )]
    prompt: String,
    /// Show each tool call and its result, not just the thinking.
    #[arg(long)]
    verbose: bool,
    /// How hard the model works — text, tool calls, and thinking depth. The
    /// recommended thinking-depth control when paired with adaptive thinking.
    /// Omit to use the API default (`high`).
    #[arg(long)]
    effort: Option<EffortArg>,
    /// Stop after this many assistant turns, in case the loop never settles.
    #[arg(long, default_value_t = 10)]
    max_turns: usize,
}

/// CLI mirror of [`Effort`] — the library type isn't a `clap::ValueEnum`, so
/// we map across at the boundary.
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

/// Arguments for the `compute` method. The field docs become the schema
/// property descriptions the model sees (via `schemars`).
#[derive(Debug, Deserialize, JsonSchema)]
struct Compute {
    /// Left operand.
    a: f64,
    /// Operator: one of `+`, `-`, `*`, `/`.
    op: String,
    /// Right operand.
    b: f64,
}

/// A stateless calculator that does *one* binary operation per call — forcing
/// the model to chain calls and reason about each intermediate result.
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
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let args = Args::parse();

    // Get API key from stdin.
    println!("Enter your API key:");
    let key = std::io::stdin().lock().lines().next().unwrap()?;

    // Create a client. `key` will be consumed and zeroized.
    let client = Client::new(key)?;

    let mut calc = Calculator;

    let mut chat = Prompt::default()
        // Sonnet 4.6 is the cheapest model with adaptive + interleaved support.
        .model(AnthropicModel::Sonnet46)
        // Adaptive: the model decides how much to think, and interleaves
        // thinking with tool use automatically. No `budget_tokens` to set.
        .thinking(Thinking::adaptive())
        .set_system(
            "You are a careful calculator's assistant. You cannot do \
             arithmetic reliably in your head, so use the `compute` tool for \
             every operation, one operation per call, and reason about each \
             intermediate result before the next step.",
        )
        .add_message((Role::User, args.prompt))?;

    // Register the tool's generated definition(s). The method is namespaced by
    // the tool's name, e.g. `Calculator__compute`.
    for definition in calc.definitions() {
        chat = chat.add_tool(definition);
    }

    // Effort composes with structured output and thinking — set it here and
    // any output_config the prompt already carries is preserved.
    if let Some(effort) = args.effort {
        chat = chat.effort(effort.into());
    }

    // Agentic loop: think -> tool_use -> (tool result) -> think -> ... -> text.
    // Each iteration is one assistant turn; on Sonnet 4.6 the default `display`
    // is summarized, so the thought we print is the model's own summary.
    for _ in 0..args.max_turns {
        let message = client.message(&chat).await?;

        for block in message.inner.content.iter() {
            match block {
                Block::Thought { thought, .. } => {
                    println!("\n🧠 {thought}");
                }
                Block::ToolUse { call } if args.verbose => {
                    println!("🔧 {}({})", call.name, call.input);
                }
                _ => {}
            }
        }

        match message.tool_use() {
            // The model wants to compute. Own the call, append the assistant
            // turn (thoughts included, so signatures round-trip), then push the
            // typed-dispatched result back as the next user message.
            Some(call) => {
                let call = call.clone();
                chat.push_message(message)?;
                let result = calc.call(call).await;
                if args.verbose {
                    println!("   = {}", result.content);
                }
                chat.push_message(result)?;
            }
            // No tool call: this turn is the final answer.
            None => {
                println!("\n✅ {}", message.inner.content);
                return Ok(());
            }
        }
    }

    Err("model did not finish within max_turns".into())
}
