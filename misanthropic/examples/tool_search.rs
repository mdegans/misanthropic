//! Example: the **tool-search tool** ([`ServerTool::tool_search_regex`]) over a
//! catalog of [`defer_loading`] tools.
//!
//! When you have many tools, loading every schema into the prompt up front
//! bloats the cached prefix and hurts tool-selection accuracy. Instead you mark
//! the tools [`defer_loading`] (here, in bulk with [`Prompt::defer_tools`]) and
//! add a *tool-search* server tool. The model then sees only the search tool,
//! searches your catalog by name/description/argument when it needs something,
//! and the API expands the matching [`tool_reference`] into a full definition on
//! demand — keeping context small while the catalog can grow to thousands.
//!
//! The catalog here is a single [`tool`]-macro toolkit with several
//! pure-function methods (like the [`strawberry`] example, but many methods).
//! Each generated [`MethodDef`] is its own deferred tool the search can find.
//!
//! ## The loop
//!
//! Two kinds of model action drive the loop:
//! - The **tool-search tool is server-side**: the model searches and the API
//!   runs it internally, sometimes yielding [`StopReason::PauseTurn`] mid-turn.
//!   We resume by sending the paused turn back (see the [`web_search`] example).
//! - The **discovered tools are custom**: once the model calls one, we execute
//!   it ourselves via [`Tool::call`] and return a [`tool::Result`], exactly as
//!   in [`strawberry`].
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client,derive --example tool_search -- \
//!     "What is 2024 in Roman numerals?"
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, or prompts on stdin.
//!
//! [`ServerTool::tool_search_regex`]: misanthropic::tool::ServerTool::tool_search_regex
//! [`defer_loading`]: misanthropic::tool::MethodDef::defer_loading
//! [`Prompt::defer_tools`]: misanthropic::Prompt::defer_tools
//! [`tool_reference`]: misanthropic::prompt::message::Block::ToolReference
//! [`tool`]: misanthropic::tool::tool
//! [`MethodDef`]: misanthropic::tool::MethodDef
//! [`strawberry`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/strawberry.rs
//! [`web_search`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/web_search.rs
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`tool::Result`]: misanthropic::tool::Result

use std::io::{BufRead, stdin};

use misanthropic::{
    AnthropicModel, Client, Prompt,
    prompt::message::{Block, Content, Role, ToolSearchToolResultContent},
    response::StopReason,
    tool::{ServerTool, Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Method handlers return `Ok(content)` on success or `Err(content)` for a
/// model-facing error; [`Content`] is `Into`-constructed from a `&str`/`String`.
type MethodResult = Result<Content, Content>;

/// A grab-bag of small, deterministic utilities. The [`tool`] macro turns each
/// `#[method]` into a deferred-able [`MethodDef`](misanthropic::tool::MethodDef)
/// the search tool can discover by its name, description, and argument names —
/// so write those to read like the questions a user would ask.
///
/// [`tool`]: misanthropic::tool::tool
struct Toolkit;

/// Arguments for [`Toolkit::roman_numeral`].
#[derive(Debug, Deserialize, JsonSchema)]
struct RomanNumeral {
    /// The integer to convert (1..=3999).
    number: u16,
}

/// Arguments for [`Toolkit::convert_temperature`].
#[derive(Debug, Deserialize, JsonSchema)]
struct ConvertTemperature {
    /// The temperature value to convert.
    value: f64,
    /// The unit of `value`: one of "C", "F", or "K".
    from: char,
    /// The unit to convert into: one of "C", "F", or "K".
    to: char,
}

/// Arguments for [`Toolkit::nth_prime`].
#[derive(Debug, Deserialize, JsonSchema)]
struct NthPrime {
    /// Which prime to return (1 yields 2, 2 yields 3, …).
    n: u32,
}

/// Arguments for [`Toolkit::count_letters`].
#[derive(Debug, Deserialize, JsonSchema)]
struct CountLetters {
    /// The letter to count.
    letter: char,
    /// The string to count letters in.
    string: String,
}

#[tool]
impl Toolkit {
    /// Convert an integer to a Roman numeral (e.g. 2024 -> MMXXIV).
    #[method]
    async fn roman_numeral(&mut self, args: RomanNumeral) -> MethodResult {
        const TABLE: [(u16, &str); 13] = [
            (1000, "M"),
            (900, "CM"),
            (500, "D"),
            (400, "CD"),
            (100, "C"),
            (90, "XC"),
            (50, "L"),
            (40, "XL"),
            (10, "X"),
            (9, "IX"),
            (5, "V"),
            (4, "IV"),
            (1, "I"),
        ];
        if !(1..=3999).contains(&args.number) {
            return Err("number must be between 1 and 3999".into());
        }
        let mut n = args.number;
        let roman: String = TABLE
            .iter()
            .flat_map(|&(value, sym)| {
                let count = (n / value) as usize;
                n %= value;
                std::iter::repeat_n(sym, count)
            })
            .collect();
        Ok(roman.into())
    }

    /// Convert a temperature between Celsius (C), Fahrenheit (F), and Kelvin (K).
    #[method]
    async fn convert_temperature(
        &mut self,
        args: ConvertTemperature,
    ) -> MethodResult {
        // Normalize to Celsius, then to the target unit.
        let celsius = match args.from.to_ascii_uppercase() {
            'C' => args.value,
            'F' => (args.value - 32.0) * 5.0 / 9.0,
            'K' => args.value - 273.15,
            other => return Err(format!("unknown `from` unit: {other}").into()),
        };
        let out = match args.to.to_ascii_uppercase() {
            'C' => celsius,
            'F' => celsius * 9.0 / 5.0 + 32.0,
            'K' => celsius + 273.15,
            other => return Err(format!("unknown `to` unit: {other}").into()),
        };
        Ok(format!("{out:.2}").into())
    }

    /// Return the nth prime number (the 1st prime is 2).
    #[method]
    async fn nth_prime(&mut self, args: NthPrime) -> MethodResult {
        if args.n == 0 {
            return Err("n must be at least 1".into());
        }
        let is_prime = |x: u64| {
            (2..x)
                .take_while(|d| d * d <= x)
                .all(|d| !x.is_multiple_of(d))
        };
        let prime = (2u64..)
            .filter(|&x| is_prime(x))
            .nth(args.n as usize - 1)
            .expect("there are infinitely many primes");
        Ok(prime.to_string().into())
    }

    /// Count the occurrences of a letter in a string.
    #[method]
    async fn count_letters(&mut self, args: CountLetters) -> MethodResult {
        let letter = args.letter.to_ascii_lowercase();
        let count = args
            .string
            .chars()
            .filter(|c| c.to_ascii_lowercase() == letter)
            .count();
        Ok(count.to_string().into())
    }
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

    let question = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "What is the 100th prime number?".to_string());

    let mut toolkit = Toolkit;

    // Nudge the model to reach for tools rather than computing by hand — like
    // the `strawberry` example, but here the tools are deferred, so it must
    // *search* the catalog to find them first.
    let system = "You are a careful assistant with a set of exact computation \
        tools. You cannot reliably compute things like large primes, Roman \
        numerals, or temperature conversions in your head, so you must search \
        for and use the appropriate tool. Do not guess.";

    // Register every method as a custom tool, then `defer_tools()` flips them
    // all to `defer_loading: true` so none of their schemas sit in the prompt
    // prefix. The regex tool-search tool stays non-deferred (server tools never
    // defer) — it is the one tool the model sees up front, and how it finds the
    // rest. (Per-method `#[method(defer_loading)]` is the alternative when you
    // want only *some* methods deferred.)
    let mut prompt = Prompt::default()
        .model(AnthropicModel::Haiku45)
        .set_system(system)
        .add_message((Role::User, question))?;
    for definition in toolkit.definitions() {
        prompt = prompt.add_tool(definition);
    }
    prompt = prompt
        .add_server_tool(ServerTool::tool_search_regex())
        .defer_tools();

    // Drive the conversation: resume on `pause_turn` (a server-side search in
    // progress), execute any custom tool the model discovered and called, and
    // stop once it answers in plain text.
    let mut searches = 0;
    let answer = loop {
        let response = client.message(&prompt).await?;

        // Report each tool-search the model ran and what it turned up — these
        // `ToolSearchToolResult` blocks come back in the assistant turn, and
        // the API has already expanded the references into full definitions.
        for block in response.inner.content.iter() {
            if let Block::ToolSearchToolResult { content, .. } = block {
                searches += 1;
                if let ToolSearchToolResultContent::Results {
                    tool_references,
                } = content
                {
                    let found: Vec<_> = tool_references
                        .iter()
                        .map(|r| r.tool_name.as_ref())
                        .collect();
                    eprintln!("[search] found: {}", found.join(", "));
                }
            }
        }

        match response.stop_reason {
            // The server paused mid-turn while searching; send the partial
            // assistant turn back so it can continue.
            Some(StopReason::PauseTurn) => {
                prompt.push_message(response)?;
                continue;
            }
            // The model called one or more discovered tools. Run each and
            // return the results in a single user turn.
            Some(StopReason::ToolUse) => {
                let calls: Vec<_> = response
                    .inner
                    .content
                    .iter()
                    .filter_map(Block::tool_use)
                    .map(|call| call.clone())
                    .collect();

                for call in &calls {
                    eprintln!("[tool] {}({})", call.name, call.input);
                }

                prompt.push_message(response)?;

                let mut results = Vec::new();
                for call in calls {
                    let result = toolkit.call(call).await;
                    results.push(Block::ToolResult { result });
                }
                prompt.push_message((Role::User, results))?;
                continue;
            }
            // Done — the model answered in text.
            _ => break response,
        }
    };

    println!("{}", answer.inner.content);
    eprintln!("\n[{searches} tool searches]");

    Ok(())
}
