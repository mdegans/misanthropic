//! Example: the **tool-search tool** ([`ServerMethodDef::tool_search_regex`])
//! over a catalog of [`defer_loading`] tools. Mark tools deferred (here in
//! bulk via [`Prompt::defer_tools`]); the model sees only the search tool,
//! searches by name/description/argument, and the API expands the matching
//! [`tool_reference`] on demand — context stays small as the catalog grows.
//! The search is server-side (yields [`StopReason::PauseTurn`]; resume like
//! [`web_search`]); discovered [`CustomMethodDef`]s are client-side (execute
//! via [`Tool::call`] / [`tool::Result`], like [`strawberry`]).
//!
//! ```sh
//! cargo run --features client,derive --example tool_search -- \
//!     "What is 2024 in Roman numerals?"
//! ```
//!
//! [`ServerMethodDef::tool_search_regex`]: misanthropic::tool::ServerMethodDef::tool_search_regex
//! [`defer_loading`]: misanthropic::tool::CustomMethodDef::defer_loading
//! [`Prompt::defer_tools`]: misanthropic::Prompt::defer_tools
//! [`tool_reference`]: misanthropic::prompt::message::Block::ToolReference
//! [`tool`]: misanthropic::tool::tool
//! [`CustomMethodDef`]: misanthropic::tool::CustomMethodDef
//! [`strawberry`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/strawberry.rs
//! [`web_search`]: https://github.com/mdegans/misanthropic/blob/main/misanthropic/examples/web_search.rs
//! [`StopReason::PauseTurn`]: misanthropic::response::StopReason::PauseTurn
//! [`Tool::call`]: misanthropic::tool::Tool::call
//! [`tool::Result`]: misanthropic::tool::Result

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt,
    prompt::message::{Block, Content, Role, ToolSearchToolResultContent},
    response::StopReason,
    tool::{ServerMethodDef, Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

/// Method handlers return `Ok(content)` on success or `Err(content)` for a
/// model-facing error; [`Content`] is `Into`-constructed from a `&str`/`String`.
type MethodResult = Result<Content, Content>;

/// A grab-bag of small, deterministic utilities. Each `#[method]` becomes a
/// [`CustomMethodDef`](misanthropic::tool::CustomMethodDef) the search tool can
/// discover by name, description, and argument names.
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

/// Demonstrate the tool-search server tool over a deferred catalog.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The question to answer using the deferred tool catalog.
    question: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let question = cli
        .question
        .unwrap_or_else(|| "What is the 100th prime number?".to_string());

    let mut toolkit = Toolkit;

    let system = "You are a careful assistant with a set of exact computation \
        tools. You cannot reliably compute things like large primes, Roman \
        numerals, or temperature conversions in your head, so you must search \
        for and use the appropriate tool. Do not guess.";

    // `defer_tools()` flips all custom methods to `defer_loading: true` so
    // none of their schemas sit in the prompt prefix. The regex search tool
    // stays non-deferred (server tools never defer) — it is the only tool the
    // model sees up front. Use per-method `#[method(defer_loading)]` to defer
    // only some methods.
    let mut prompt = cli
        .common
        .configure(Prompt::default().model(Id::Haiku45).system(system))
        .add_message((Role::User, question))?;
    for definition in toolkit.definitions() {
        prompt = prompt.add_tool(definition);
    }
    prompt = prompt
        .add_tool(ServerMethodDef::tool_search_regex())
        .defer_tools();

    let mut searches = 0;
    let answer = loop {
        let response = client.message(&prompt).await?;

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
            // Server paused mid-search; resume.
            Some(StopReason::PauseTurn) => {
                prompt.push_message(response)?;
                continue;
            }
            // Model called a discovered custom tool; execute and return results.
            Some(StopReason::ToolUse) => {
                let calls: Vec<_> = response
                    .inner
                    .content
                    .iter()
                    .filter_map(Block::tool_use)
                    .cloned()
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
            _ => break response,
        }
    };

    println!("{}", answer.inner.content);
    eprintln!("\n[{searches} tool searches]");

    Ok(())
}
