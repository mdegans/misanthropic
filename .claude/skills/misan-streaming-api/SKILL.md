---
name: misan-streaming-api
description: >-
  Write Rust code against the `misanthropic` crate's streaming
  (`Client::stream`) path — SSE event streams, the `FilterExt` combinators
  (`text`, `deltas`, `with_message`, `with_tool_use`), incremental token
  output, and streaming tool-call assembly. Use when writing or editing Rust
  that streams from the Anthropic Messages API via `misanthropic`, or when the
  user mentions `Client::stream`, `FilterExt`, streaming deltas, or
  token-by-token output. For single-shot requests, see the misan-messages-api
  skill instead.
---

# `misanthropic` — Streaming API

`misanthropic` is an unofficial, ergonomic, async Rust client for the
Anthropic Messages API. This skill covers the streaming `Client::stream`
path. For single-message (non-streaming) usage, see the **misan-messages-api**
skill.

- Crate: [`misanthropic`](https://crates.io/crates/misanthropic)
- Repository: <https://github.com/mdegans/misanthropic>
- Docs: <https://docs.rs/misanthropic>
- License: MIT

## Cargo.toml

```toml
[dependencies]
misanthropic = "1.0.0-alpha.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"  # needed for TryStreamExt, StreamExt
```

## Quick start — streaming text

```no_run
use futures::TryStreamExt;
use misanthropic::{
    Id, Client, Prompt,
    prompt::message::Role,
    stream::FilterExt,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

    // `Client::stream` forces `stream = true` and returns a `Stream`
    // (implements `futures::Stream<Item = Result<Event, Error>>`).
    let stream = client
        .stream(
            Prompt::default()
                .model(Id::Sonnet46)
                .system("You are a helpful assistant.")
                .add_message((Role::User, "Write a haiku about Rust."))?,
        )
        .await?
        // Extract only text deltas as owned `String`s.
        .text();

    // Collect the entire response.
    let response: String = stream.try_collect().await?;
    println!("{response}");

    Ok(())
}
```

> **Note:** there is no `filter_rate_limit` combinator. Rate limiting is
> applied *before* the request is sent (a token bucket inside the `Client`),
> so the stream you receive does not normally carry transient
> `RateLimit`/`Overloaded` events to filter out. Handle any error item the
> stream yields like any other `Result::Err`.

## Print tokens as they arrive

```no_run
use futures::TryStreamExt;
use misanthropic::{Id, Client, Prompt, prompt::message::Role, stream::FilterExt};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(
        Prompt::default()
            .model(Id::Sonnet46)
            .add_message((Role::User, "Tell me a joke."))?,
    )
    .await?
    .text();

// Use `map_ok` to print each piece as it arrives, then collect.
let full: String = stream
    .map_ok(|piece| {
        print!("{piece}");
        piece
    })
    .try_collect()
    .await?;
println!(); // final newline
# Ok(())
# }
```

## Using `json!` for the request

`Client::stream` accepts anything `Serialize`:

```no_run
use futures::TryStreamExt;
use misanthropic::{Client, json, prompt::message::Role, stream::FilterExt};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "temperature": 0,
        "system": "You are a website generator.",
        "messages": [{
            "role": Role::User,
            "content": "Build me a landing page for a coffee shop.",
        }],
    }))
    .await?
    .text();

let html: String = stream.try_collect().await?;
std::fs::write("output.html", &html)?;
# Ok(())
# }
```

## Stream events and `FilterExt`

The raw `Stream` yields `Result<Event, stream::Error>`. The `FilterExt` trait
(import from `misanthropic::stream::FilterExt`) provides ergonomic combinators.
It is auto-implemented for any `Stream<Item = Result<Event, Error>> + Send`, so
the methods compose by chaining.

| Method | Returns | Description |
|--------|---------|-------------|
| `.deltas()` | `Stream<Result<Delta, Error>>` | Only `ContentBlockDelta` events, as `Delta` values. |
| `.text()` | `Stream<Result<String, Error>>` | Only text deltas, as owned `String`s (built on `.deltas()`). |
| `.with_message()` | `Stream<Result<Event, Error>>` | Assembles a complete `response::Message` and yields it as `Event::Message` at stream end (implies `with_tool_use`). |
| `.with_message_ip(&mut Option<Message>)` | `Stream<Result<Event, Error>>` | Same, but assembles *in place* so you can break early and keep the partial message. |
| `.with_tool_use()` | `Stream<Result<Event, Error>>` | Assembles complete `tool::Use` from JSON deltas and yields it as `Event::ToolUse` (skips raw `input_json_delta` events). |
| `.with_json()` | `Stream<Result<Event, Error>>` | Incrementally scans text / tool-input JSON and yields each completed element of the outermost array as `Event::JsonObject`, as its bytes arrive. Apply *upstream* of `with_tool_use` / `with_message`. |
| `.json_items::<T>()` | `Stream<Result<T, Error>>` | Typed `with_json`: each element deserialized to `T`. Pair with `structured_output::<Items<T>>()`. |

### Chaining

```rust
# use misanthropic::stream::FilterExt;
# fn f(stream: impl misanthropic::stream::FilterExt) {
// Only text pieces:
let text_stream = stream.text();
# }
```

```rust
# use misanthropic::stream::FilterExt;
# fn f(stream: impl misanthropic::stream::FilterExt) {
// Assembled tool-use events:
let events = stream.with_tool_use();
# }
```

## Raw event types (`stream::Event`)

Every `Event` variant, shown as an exhaustive `match` — this block is a
**drift guard**: it is doc-tested, so adding an `Event` variant upstream (a new
server event, say) fails the build until this list is updated.

```rust
use misanthropic::stream::Event;

# #[allow(unused_variables)]
# fn document(event: Event) {
match event {
    Event::Ping => {}                                  // periodic keep-alive
    Event::MessageStart { message } => {}              // response::Message, empty content
    Event::ContentBlockStart { index, content_block } => {}
    Event::ContentBlockDelta { index, delta } => {}    // delta: stream::Delta
    Event::ContentBlockStop { index } => {}
    Event::MessageDelta { delta, usage } => {}         // usage: Option<Usage>
    Event::MessageStop => {}
    // Synthetic — assembled by FilterExt, never sent by the API:
    Event::Message { message } => {}                   // via with_message()
    Event::ToolUse { tool_use } => {}                  // via with_tool_use(); tool::Use
    Event::ServerToolUse { tool_use } => {}            // via with_tool_use(); server tool (e.g. web_search)
    Event::JsonObject { index, value } => {}           // via with_json(); element of the outermost array
}
# }
```

## Delta types (`stream::Delta`)

Exhaustive `match` over every `Delta` kind — also a doc-tested drift guard, so
a new delta kind (Anthropic adds these over time) breaks the build until it is
documented here.

```rust
use misanthropic::stream::Delta;

# #[allow(unused_variables)]
# fn document(delta: Delta) {
match delta {
    Delta::Text { text } => {}                    // text content
    Delta::Json { partial_json } => {}            // tool input JSON fragment
    Delta::Thought { thinking, signature } => {}  // extended thinking (Sonnet 3.7+)
    Delta::RedactedThought { signature } => {}
    Delta::Signature { signature } => {}          // signature of a complete thought
    Delta::CitationsDelta { citation } => {}      // when document citations are enabled
}
# }
```

## Stream error types (`stream::Error`)

Errors carry the offending SSE event for context where applicable. Exhaustive
`match` (doc-tested drift guard — a new variant breaks the build):

```rust
use misanthropic::stream::Error;

# #[allow(unused_variables)]
# fn document(error: Error) {
match error {
    Error::Stream { error } => {}                   // SSE transport error
    Error::Parse { error, event } => {}             // JSON parse failed (raw event kept)
    Error::Anthropic { error, event } => {}         // API error over SSE (raw event kept)
    Error::MessageAssembly { message, delta } => {} // e.g. a delta before MessageStart
    Error::Delta { error } => {}                    // a delta could not be applied
    Error::JsonAssembly { message, index } => {}    // with_json(): bad element or truncated block
}
# }
```

## Streaming with tool use

Use the `#[tool]` macro (default `derive` feature) to declare a typed tool,
then `with_tool_use()` to receive a fully assembled `Event::ToolUse` instead
of raw JSON deltas. See the misan-messages-api skill for the full macro walk-
through; the runnable example is `misanthropic/examples/strawberry.rs`.

```no_run
use futures::TryStreamExt;
use misanthropic::{
    Id, Client, Prompt,
    prompt::message::{Content, Role},
    stream::{Event, FilterExt},
    tool::{Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Calculate {
    /// Math expression to evaluate.
    expression: String,
}

struct Calculator;

#[tool]
impl Calculator {
    /// Evaluate a math expression.
    #[method]
    async fn calculate(
        &mut self,
        args: Calculate,
    ) -> Result<Content, Content> {
        let answer = "56088"; // your evaluation logic
        let _ = args.expression;
        Ok(answer.into())
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
let mut calculator = Calculator;

let mut chat = Prompt::default()
    .model(Id::Sonnet46)
    .add_message((Role::User, "What is 123 * 456?"))?;

for definition in calculator.definitions() {
    chat = chat.add_tool(definition);
}

let mut stream = Box::pin(
    client
        .stream(&chat)
        .await?
        .with_tool_use(),
);

while let Some(event) = stream.try_next().await? {
    match event {
        Event::ToolUse { tool_use } => {
            // Complete tool::Use (already `'static`), assembled from the JSON
            // deltas. Dispatch through the typed tool (validates args for you):
            let result = calculator.call(tool_use).await;
            // For multi-turn tool use: drop the stream, push the assistant
            // message + `result`, then start a new stream. Here we just stop.
            let _ = result;
            break;
        }
        Event::ContentBlockDelta {
            delta: misanthropic::stream::Delta::Text { text }, ..
        } => {
            print!("{text}");
        }
        _ => {} // other events
    }
}
# Ok(())
# }
```

## `with_message` — full message assembly

Use `with_message()` to get a complete `response::Message` at the end of the
stream while still processing events as they arrive:

```no_run
use futures::TryStreamExt;
use misanthropic::{stream::{Event, FilterExt}, Client, Prompt, prompt::message::Role};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(Prompt::default().add_message((Role::User, "Hello!"))?)
    .await?
    .with_message();

futures::pin_mut!(stream);

while let Some(event) = stream.try_next().await? {
    if let Event::Message { message } = event {
        // Complete assembled message at stream end.
        println!("Complete: {message}");
        println!("Tokens used: {}", message.usage.output_tokens);
    }
}
# Ok(())
# }
```

Need to interrupt early and keep what was assembled so far? Use
`with_message_ip(&mut your_option)` instead and read your `Option<Message>`
after breaking out of the loop.

## `json_items` — incremental structured output

With `Prompt::structured_output` the response text block is schema-guaranteed
JSON. For a *list*, wrap the element type in `prompt::Items<T>` (the API
requires a top-level object schema) and consume each element the moment its
bytes arrive — no waiting for the message to finish. The runnable example is
`misanthropic/examples/triage_stream.rs`.

```no_run
use futures::TryStreamExt;
use misanthropic::{
    Client, Id, Prompt,
    prompt::{Items, message::Role},
    stream::FilterExt,
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Fact {
    claim: String,
    confidence: String,
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let prompt = Prompt::default()
    .model(Id::Haiku45)
    .structured_output::<Items<Fact>>()
    .add_message((Role::User, "Three facts about crabs."))?;

let facts = client.stream(&prompt).await?.json_items::<Fact>();
futures::pin_mut!(facts);

while let Some(fact) = facts.try_next().await? {
    println!("{fact:?}"); // printed as each element completes
}
# Ok(())
# }
```

The untyped layer is `with_json()`, which yields `Event::JsonObject { index,
value }` per completed element of the *outermost* array (the `Items` shape)
while passing all other events through — it also scans tool-input
`input_json_delta`s, so a list-shaped tool argument streams the same way.
Apply it upstream of `with_tool_use()` / `with_message()` (those consume the
input JSON deltas). A block that ends mid-value (e.g. `max_tokens`) yields
`Error::JsonAssembly`.

## Server-Sent Events (SSE) backend

Under the hood, `Client::stream` sends a POST with `stream: true` and wraps the
response `bytes_stream()` with `eventsource_stream`. The `Stream` type
implements `futures::Stream` and handles SSE parsing, per-event JSON
deserialization, and error extraction automatically.

## Key design notes

- **`Stream` is `Send`** — it can be moved across tasks.
- **Rate limiting** is applied *before* the request (a client-side token
  bucket); there is no in-stream rate-limit filter to apply.
- **`FilterExt`** is auto-implemented for any
  `Stream<Item = Result<Event, Error>> + Sized + Send`, so combinators chain.
- **Error items in the stream** are Anthropic API errors delivered via SSE,
  carrying the raw `eventsource_stream::Event`. HTTP-level errors are returned
  from `client.stream()` itself, before the stream begins.
- **Owned data, no lifetimes** — `Delta`, assembled `Message`/`tool::Use`, and
  the other public types own their data and carry no lifetime parameter, so
  they are already `'static`. There is no `.into_static()` to call.
