# Using `misanthropic` — Non-Streaming (Message) API

`misanthropic` is an unofficial, ergonomic, async Rust client for the
Anthropic Messages API. This skill covers the non-streaming `Client::message`
path. For streaming, see the streaming skill.

- Crate: [`misanthropic`](https://crates.io/crates/misanthropic)
- Repository: <https://github.com/mdegans/misanthropic>
- Docs: <https://docs.rs/misanthropic>
- License: MIT

## Cargo.toml

```toml
[dependencies]
misanthropic = "1.0.0-alpha"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

### Feature flags (selected)

| Flag | Default | Purpose |
|------|---------|---------|
| `client` | yes | Enables `Client` (HTTP). Disable for wasm data-only. |
| `rustls-tls` | yes | Use rustls instead of system OpenSSL. |
| `rate-limiting` | yes | Client-side rate limiting via `governor`. |
| `prompt-caching` | no | Anthropic prompt-caching beta headers. |
| `markdown` | no | `ToMarkdown` trait, markdown rendering. |
| `image` / `png` / `jpeg` / `gif` / `webp` | no | Image support via the `image` crate. |
| `langsan` | yes | Output sanitization. |
| `memsecurity` | no | Encrypt the API key in memory. |

## Quick start — single message

```rust
use misanthropic::{Client, Prompt, prompt::message::Role};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a client. The key String is consumed, zeroized,
    // and stored encrypted (with `memsecurity`) or zeroed on
    // drop. The key header is marked sensitive on requests.
    let key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Client::new(key)?;

    // Build a Prompt (the request type) and send it.
    // `Client::message` forces `stream = false` and returns
    // a `response::Message` directly.
    let message = client
        .message(
            Prompt::default()
                .set_messages([(Role::User, "What is 2+2?")])
        )
        .await?;

    // `response::Message` implements `Display` (prints content).
    println!("{message}");

    // Access fields:
    //   message.inner          — the prompt::AssistantMessage
    //   message.inner.content  — Content (Display, iterable)
    //   message.model          — model::Id
    //   message.stop_reason    — Option<StopReason>
    //   message.usage          — Usage { input_tokens, output_tokens, .. }

    Ok(())
}
```

## The `Prompt` builder

`Prompt` is the request type. It defaults to model `Haiku30` with
`max_tokens = 4096`. All fields are public; the builder methods are
convenience helpers that return `Self`.

```rust
use std::num::NonZeroU32;
use misanthropic::{
    AnthropicModel, Prompt,
    prompt::message::Role,
};

let prompt = Prompt::default()
    // Set model — accepts AnthropicModel or any string.
    .model(AnthropicModel::Sonnet35)
    // System prompt — accepts &str, String, or Content.
    .set_system("You are a helpful assistant.")
    // Append to system prompt.
    .add_system("Respond concisely.")
    // Set max tokens.
    .max_tokens(NonZeroU32::new(1024).unwrap())
    // Set temperature (0.0–1.0).
    .temperature(Some(0.7))
    // Add a user message — returns Result because turn order
    // is validated (must alternate User/Assistant).
    .add_message((Role::User, "Hello!"))?;
```

### Available models (`AnthropicModel` enum)

| Variant | Serialized ID |
|---------|---------------|
| `Haiku30` (default) | `claude-3-haiku-20240307` |
| `Haiku35` | `claude-3-5-haiku-latest` |
| `Sonnet30` | `claude-3-sonnet-20240229` |
| `Sonnet35` | `claude-3-5-sonnet-latest` |
| `Sonnet37` | `claude-3-7-sonnet-latest` |
| `Opus30` | `claude-3-opus-latest` |

Use `model::Id::Custom("your-model-id".into())` for custom or
newer models not yet in the enum.

### Messages from tuples

Messages can be created from `(Role, impl Into<String>)` tuples:

```rust
use misanthropic::{Prompt, prompt::message::Role};

// set_messages replaces all messages; add_message appends one.
let prompt = Prompt::default().set_messages([
    (Role::User, "What is Rust?"),
    (Role::Assistant, "Rust is a systems programming language."),
    (Role::User, "What are its key features?"),
])?;
```

### Multi-turn conversation

```rust
use misanthropic::{Client, Prompt, prompt::message::Role};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let mut chat = Prompt::default()
    .model(misanthropic::AnthropicModel::Sonnet35)
    .set_system("You are a helpful assistant.")
    .add_message((Role::User, "What is Rust?"))?;

let reply = client.message(&chat).await?;
println!("Assistant: {reply}");

// Append assistant reply, then user follow-up.
// `push_message` is the in-place version of `add_message`.
chat.push_message(reply)?;
chat.push_message((Role::User, "What about memory safety?"))?;

let reply = client.message(&chat).await?;
println!("Assistant: {reply}");
```

## Tool use

Define tools with `tool::Method`, check for tool calls on the response,
and return results via `tool::Result`.

```rust
use misanthropic::{
    Client, Prompt, json,
    prompt::{Message, message::Role},
    tool::{self, Method},
};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

// Define a tool the model can call.
let mut chat = Prompt::default()
    .model(misanthropic::AnthropicModel::Sonnet35)
    .add_tool(Method {
        name: "get_weather".into(),
        description: "Get the weather for a city.".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name",
                }
            },
            "required": ["city"],
        }),
        #[cfg(feature = "prompt-caching")]
        cache_control: None,
    })
    .set_system("Use tools when appropriate.")
    .add_message((Role::User, "What's the weather in Paris?"))?;

let message = client.message(&chat).await?;

// Check if the model wants to use a tool.
if let Some(call) = message.tool_use() {
    // call.name  — tool name ("get_weather")
    // call.id    — unique ID for this call
    // call.input — serde_json::Value with arguments

    let city = call.input["city"].as_str().unwrap();
    let weather = format!("Sunny, 22C in {city}"); // your logic

    // Build a tool result message (always Role::User).
    let result: Message = tool::Result {
        tool_use_id: call.id.to_string().into(),
        content: weather.into(),
        is_error: false,
        #[cfg(feature = "prompt-caching")]
        cache_control: None,
    }
    .into();

    // Append the assistant's tool-call message and the result.
    chat.push_message(message)?;
    chat.push_message(result)?;

    // Get the final response incorporating the tool result.
    let final_reply = client.message(&chat).await?;
    println!("{final_reply}");
}
```

## Using `json!` instead of `Prompt`

`Client::message` accepts anything `Serialize`. You can use raw JSON:

```rust
use misanthropic::{Client, json, prompt::message::Role};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let message = client
    .message(json!({
        "model": "claude-3-5-sonnet-latest",
        "max_tokens": 1024,
        "system": "You are a pirate.",
        "messages": [{
            "role": Role::User,
            "content": "Ahoy!",
        }],
    }))
    .await?;

println!("{message}");
```

## Error handling

`Client` methods return `client::Result<T>` which wraps `client::Error`:

```rust
use misanthropic::client::{Error, AnthropicError};

match client.message(&prompt).await {
    Ok(msg) => println!("{msg}"),
    Err(Error::Anthropic(AnthropicError::RateLimit { message })) => {
        eprintln!("Rate limited: {message}");
    }
    Err(Error::Anthropic(AnthropicError::Authentication { message })) => {
        eprintln!("Auth error: {message}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
```

### Error variants

| Variant | Description |
|---------|-------------|
| `Error::HTTP` | Network / reqwest error |
| `Error::Parse` | JSON deserialization failed |
| `Error::Anthropic(AnthropicError::*)` | API error (see below) |
| `Error::UnexpectedResponse` | Wrong response type (should not happen) |

**`AnthropicError` variants:** `InvalidRequest` (400),
`Authentication` (401), `Permission` (403), `NotFound` (404),
`RequestTooLarge` (413), `RateLimit` (429), `API` (500),
`Overloaded` (529), `Timeout`, `Billing`, `Unknown`.

## Response structure

```
response::Message
├── id: Cow<str>              — unique message ID
├── inner: AssistantMessage
│   └── inner: prompt::Message
│       ├── role: Role::Assistant
│       └── content: Content  — Display, iterable over Blocks
├── model: model::Id
├── stop_reason: Option<StopReason>
│   └── EndTurn | MaxTokens | StopSequence | ToolUse
├── stop_sequence: Option<Cow<str>>
└── usage: Usage
    ├── input_tokens: u64
    └── output_tokens: u64
```

## Key design notes

- **API keys** are zeroized on drop. With `memsecurity`, they are
  encrypted in memory. The `x-api-key` header is marked sensitive.
- **No `unsafe` code** — the crate uses `#[forbid(unsafe_code)]`.
- **Rate limiting** is built in (default: 50 req/min, tier 1).
  Adjust with `client.set_rate_limit(quota)`.
- **`Client` is cheap to clone** — it wraps `Arc`s internally.
- **Turn order is enforced** — messages must alternate User/Assistant.
  The first message must be User. Methods return `TurnOrderError` on
  violation.
