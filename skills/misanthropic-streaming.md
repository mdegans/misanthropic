# Using `misanthropic` — Streaming API

`misanthropic` is an unofficial, ergonomic, async Rust client for the
Anthropic Messages API. This skill covers the streaming `Client::stream`
path. For single-message (non-streaming) usage, see the message skill.

- Crate: [`misanthropic`](https://crates.io/crates/misanthropic)
- Repository: <https://github.com/mdegans/misanthropic>
- License: MIT

## Cargo.toml

```toml
[dependencies]
misanthropic = "1.0.0-alpha"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
futures = "0.3"  # needed for TryStreamExt, StreamExt
```

## Quick start — streaming text

```rust
use futures::TryStreamExt;
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    stream::FilterExt,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

    // `Client::stream` forces `stream = true` and returns a
    // `Stream` (implements `futures::Stream<Item = Result<Event, Error>>`).
    let stream = client
        .stream(
            Prompt::default()
                .model(misanthropic::AnthropicModel::Sonnet35)
                .set_system("You are a helpful assistant.")
                .add_message((Role::User, "Write a haiku about Rust."))?
        )
        .await?
        // Filter out RateLimit and Overloaded errors — the stream
        // continues when the server is ready. Recommended.
        .filter_rate_limit()
        // Extract only text deltas as Strings.
        .text();

    // Collect the entire response.
    let response: String = stream.try_collect().await?;
    println!("{response}");

    Ok(())
}
```

## Print tokens as they arrive

```rust
use futures::TryStreamExt;
use misanthropic::{Client, Prompt, prompt::message::Role, stream::FilterExt};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(
        Prompt::default()
            .model(misanthropic::AnthropicModel::Sonnet35)
            .add_message((Role::User, "Tell me a joke."))?
    )
    .await?
    .filter_rate_limit()
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
```

## Using `json!` for the request

`Client::stream` accepts anything `Serialize`:

```rust
use futures::TryStreamExt;
use misanthropic::{Client, json, prompt::message::Role, stream::FilterExt};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 4096,
        "temperature": 0,
        "system": "You are a website generator.",
        "messages": [{
            "role": Role::User,
            "content": "Build me a landing page for a coffee shop.",
        }],
    }))
    .await?
    .filter_rate_limit()
    .text();

let html: String = stream.try_collect().await?;
std::fs::write("output.html", &html)?;
```

## Stream events and `FilterExt`

The raw `Stream` yields `Result<Event, stream::Error>`. The `FilterExt`
trait (import from `misanthropic::stream::FilterExt`) provides
ergonomic filters:

| Method | Returns | Description |
|--------|---------|-------------|
| `.filter_rate_limit()` | `Stream<Result<Event, Error>>` | Drops `RateLimit` and `Overloaded` errors silently. |
| `.deltas()` | `Stream<Result<Delta, Error>>` | Only `ContentBlockDelta` events, as `Delta` values. |
| `.text()` | `Stream<Result<String, Error>>` | Only text deltas as owned `String`s. |
| `.with_message()` | `Stream<Result<Event, Error>>` | Assembles a complete `response::Message` and yields it as `Event::Message` at stream end. |
| `.with_tool_use()` | `Stream<Result<Event, Error>>` | Assembles complete `tool::Use` from JSON deltas and yields as `Event::ToolUse`. |

### Chaining filters

Filters compose because each returns a new `Stream`. Common pattern:

```rust
let text_stream = stream
    .filter_rate_limit()  // drop transient errors
    .text();              // only text pieces
```

Or for tool use assembly:

```rust
let events = stream
    .filter_rate_limit()
    .with_tool_use();  // yields Event::ToolUse when complete
```

## Raw event types (`stream::Event`)

```rust
pub enum Event {
    Ping,
    MessageStart { message: response::Message },
    ContentBlockStart { index: usize, content_block: Block },
    ContentBlockDelta { index: usize, delta: Delta },
    ContentBlockStop { index: usize },
    MessageDelta { delta: MessageDelta, usage: Option<Usage> },
    MessageStop,
    // Synthetic events (from FilterExt, not the API):
    Message { message: response::Message },
    ToolUse { tool_use: tool::Use },
}
```

## Delta types (`stream::Delta`)

```rust
pub enum Delta {
    Text { text: Cow<str> },          // text content
    Json { partial_json: Cow<str> },  // tool input JSON fragment
    Thought { thinking: Cow<str>, signature: Option<Cow<str>> },
    RedactedThought { signature: Cow<str> },
    Signature { signature: Cow<str> },
}
```

## Stream error types (`stream::Error`)

```rust
pub enum Error {
    Anthropic { error: AnthropicError },  // API error in stream
    Stream { error: String },             // SSE transport error
    Parse { error: String },              // JSON parse error
    Delta { error: DeltaError },          // Delta application error
    MessageAssembly { message: &'static str },
}
```

## Streaming with tool use

When streaming with tools, use `with_tool_use()` to receive assembled
`Event::ToolUse` events instead of raw JSON deltas:

```rust
use futures::{StreamExt, TryStreamExt};
use misanthropic::{
    Client, Prompt, json,
    prompt::{Message, message::Role},
    stream::{Event, FilterExt},
    tool::{self, Method},
};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let mut chat = Prompt::default()
    .model(misanthropic::AnthropicModel::Sonnet35)
    .add_tool(Method {
        name: "calculate".into(),
        description: "Evaluate a math expression.".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate",
                }
            },
            "required": ["expression"],
        }),
        #[cfg(feature = "prompt-caching")]
        cache_control: None,
    })
    .add_message((Role::User, "What is 123 * 456?"))?;

let mut stream = Box::pin(
    client
        .stream(&chat)
        .await?
        .filter_rate_limit()
        .with_tool_use()
);

let mut text = String::new();

while let Some(event) = stream.try_next().await? {
    match event {
        Event::ToolUse { tool_use } => {
            // Complete tool::Use is assembled from JSON deltas.
            let expr = tool_use.input["expression"]
                .as_str().unwrap();
            let answer = "56088"; // your evaluation logic

            let result: Message = tool::Result {
                tool_use_id: tool_use.id.to_string().into(),
                content: answer.into(),
                is_error: false,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
            .into();

            // For multi-turn tool use, you would break here,
            // push messages, and start a new stream.
            break;
        }
        Event::ContentBlockDelta {
            delta: misanthropic::stream::Delta::Text { text: t }, ..
        } => {
            print!("{t}");
            text.push_str(&t);
        }
        _ => {} // other events
    }
}
```

## `with_message` — full message assembly

Use `with_message()` to get a complete `response::Message` at the end
of the stream while still processing events as they arrive:

```rust
use futures::TryStreamExt;
use misanthropic::{stream::{Event, FilterExt}, Client, Prompt, prompt::message::Role};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let stream = client
    .stream(
        Prompt::default()
            .add_message((Role::User, "Hello!"))?
    )
    .await?
    .filter_rate_limit()
    .with_message();

futures::pin_mut!(stream);

while let Some(event) = stream.try_next().await? {
    match event {
        Event::Message { message } => {
            // Complete assembled message at stream end.
            println!("Complete: {message}");
            println!("Tokens used: {}", message.usage.output_tokens);
        }
        _ => {} // intermediate events
    }
}
```

## Server-Sent Events (SSE) backend

Under the hood, `Client::stream` sends a POST with `stream: true` and
wraps the response `bytes_stream()` with `eventsource_stream`. The
`Stream` type implements `futures::Stream` and handles SSE parsing,
JSON deserialization of each event, and error extraction automatically.

## Key design notes

- **`Stream` is `Send`** — it can be moved across tasks.
- **Rate limiting** is applied before the request. The stream itself
  may contain `RateLimit`/`Overloaded` error events (use
  `filter_rate_limit()` to handle them).
- **`FilterExt`** is automatically implemented for any
  `Stream<Item = Result<Event, Error>> + Sized + Send`.
- **Error events in the stream** are Anthropic API errors delivered
  via SSE, not HTTP errors. HTTP errors are returned from
  `client.stream()` itself.
