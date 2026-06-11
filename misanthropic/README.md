# `misanthropic`

![Build Status](https://github.com/mdegans/misanthropic/actions/workflows/tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/mdegans/misanthropic/branch/main/graph/badge.svg)](https://codecov.io/gh/mdegans/misanthropic)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Is an unofficial simple, ergonomic, async client for the Anthropic Messages
API.

- [Documentation](https://docs.rs/misanthropic)
- [Examples](https://github.com/mdegans/misanthropic/tree/main/misanthropic/examples)
- [Agent skills](https://github.com/mdegans/misanthropic/tree/main/.claude/skills)
  for writing code against the crate (doc-tested in CI, so they can't drift)

This README is also the crate front page, and every code block below compiles
as a doc-test.

## Usage

```toml
[dependencies]
misanthropic = "1.0.0-alpha.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
# Tool argument and structured output structs derive these:
schemars = "0.8"
serde = { version = "1", features = ["derive"] }
```

### Streaming

`Client::stream` returns a `futures::Stream` of events. The `FilterExt`
combinators reduce it to what you care about — here, text tokens as they
arrive:

```rust,no_run
use futures::TryStreamExt;
use misanthropic::{
    Client, Id, Prompt, prompt::message::Role, stream::FilterExt,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

    let stream = client
        .stream(
            Prompt::default()
                .model(Id::Sonnet46)
                .system("You are a helpful assistant.")
                .add_message((Role::User, "Write a haiku about Rust."))?,
        )
        .await?
        // Just the text pieces, as owned `String`s.
        .text();

    // Print each token as it arrives, collecting the full reply.
    let haiku: String = stream
        .map_ok(|piece| {
            print!("{piece}");
            piece
        })
        .try_collect()
        .await?;

    Ok(())
}
```

### Tool use

The `#[tool]` macro turns an `impl` block into a typed tool: your argument
struct's `JsonSchema` becomes the wire definition (field docs become the
property descriptions the model reads), and dispatch is deserialized and
validated for you — no hand-parsing `serde_json::Value`:

```rust,no_run
use misanthropic::{
    Client, Id, Prompt,
    prompt::message::{Content, Role},
    tool::{Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct GetWeather {
    /// City name.
    city: String,
}

struct Weather;

#[tool]
impl Weather {
    /// Get the weather for a city.
    #[method]
    async fn get_weather(
        &mut self,
        args: GetWeather,
    ) -> Result<Content, Content> {
        Ok(format!("Sunny, 22C in {}", args.city).into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
    let mut weather = Weather;

    let mut chat = Prompt::default()
        .model(Id::Sonnet46)
        .tools(weather.definitions())
        .add_message((Role::User, "What's the weather in Paris?"))?;

    let message = client.message(&chat).await?;

    if let Some(call) = message.tool_use() {
        let call = call.clone();
        chat.push_message(message)?;
        // Typed dispatch — bad arguments become a model-facing error.
        chat.push_message(weather.call(call).await)?;

        println!("{}", client.message(&chat).await?);
    }

    Ok(())
}
```

### Structured output

`Prompt::structured_output::<T>()` constrains generation (grammar-based
decoding, not prompting) to JSON matching `T`'s schema. Parse the reply with
`response.json::<T>()`:

```rust,no_run
use misanthropic::{Client, Id, Prompt, prompt::message::Role};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Triage {
    /// Doc comments become schema descriptions — the model reads them.
    summary: String,
    severity: String,
    is_regression: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

    let prompt = Prompt::default()
        .model(Id::Haiku45)
        .structured_output::<Triage>()
        .add_message((Role::User, "Checkout shows $0.00 on mobile Safari."))?;

    let triage: Triage = client.message(&prompt).await?.json()?;
    println!("{triage:?}");

    Ok(())
}
```

`Prompt::add_examples` adds few-shot exemplars that share the same schema, so
the constraint can't drift from the examples. For lists, `prompt::Items<T>`
pairs with the streaming `json_items::<T>()` combinator to consume each
element as it's generated.

## Features

- [x] Async but does not _directly_ depend on tokio
- [x] Typed tool use — the `#[tool]` macro generates schemas and dispatch
- [x] Server tools — web search, web fetch, code execution, tool search, and
  programmatic tool calling
- [x] Anthropic's client-executed tools — memory, text editor, and bash,
  with batteries included: filesystem backends and a hardened Docker
  sandbox (`mdegans/misan-bashd`)
- [x] Structured output (grammar-constrained JSON), few-shot examples, and
  streaming JSON (consume list elements as they're generated)
- [x] Streaming responses with composable `FilterExt` combinators
- [x] Extended thinking, including adaptive thinking and effort control
- [x] Documents and citations
- [x] Prompt caching
- [x] Batch API support
- [x] Image support with or without the `image` crate
- [x] Markdown and HTML formatting of messages, including images
- [x] Dioxus support
- [x] [Sanitization](https://crates.io/crates/langsan) of input and output to mitigate [injection attacks](https://arstechnica.com/security/2024/10/ai-chatbots-can-read-and-write-invisible-text-creating-an-ideal-covert-channel/)
- [x] Wasm support (without the Client itself, just the data structures and stream types, extensions, tools, and so on)
- [x] Custom request and endpoint support
- [x] API keys zeroized on drop, optionally encrypted in memory
  (`memsecurity` feature)
- [ ] Amazon Bedrock support
- [ ] Vertex AI support

## FAQ

- **Why is it called `misanthropic`?** No reason, really. I just like the word.
  Anthropic is both a company and a word meaning "relating to mankind". This
  crate is neither official or related to mankind so, `misanthropic` it is.
- **Did you know Elon Musk called Anthropic misanthropic?** Not until recently
  and this crate predates that asshole's utterances on the topic.
- **Doesn't `reqwest` depend on `tokio`?** On some platforms, yes.
- **Can i use `misanthropic` with Amazon or Vertex?** Not yet, but it's on the
  roadmap. for now the `Client` does support custom endpoints and the inner
  `reqwest::Client` can be accessed directly to make necessary adjustments to
  headers, etc.
- **Has this crate been audited?** No, but auditing is welcome. A best effort
  has been made to ensure security and privacy. The API key is encrypted in
  memory when using the `memsecurity` feature and any headers containing copies
  marked as sensitive. `rustls` is an optional feature and is recommended for
  security. It is on by default.
