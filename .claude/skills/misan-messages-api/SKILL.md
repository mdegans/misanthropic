---
name: misan-messages-api
description: >-
  Write Rust code against the `misanthropic` crate's non-streaming
  (`Client::message`) path — single messages, multi-turn chats, system
  prompts, and tool use (the `#[tool]` macro and manual `CustomMethodDef`s). Use
  when writing or editing Rust that calls the Anthropic Messages API through
  the `misanthropic` crate, or when the user mentions `misanthropic`,
  `Client::message`, or `Prompt`. For token-by-token streaming, see the
  misan-streaming-api skill instead.
---

# `misanthropic` — Non-Streaming (Message) API

`misanthropic` is an unofficial, ergonomic, async Rust client for the
Anthropic Messages API. This skill covers the non-streaming `Client::message`
path. For streaming, see the **misan-streaming-api** skill.

- Crate: [`misanthropic`](https://crates.io/crates/misanthropic)
- Repository: <https://github.com/mdegans/misanthropic>
- Docs: <https://docs.rs/misanthropic>
- License: MIT

## Cargo.toml

```toml
[dependencies]
misanthropic = "1.0.0-alpha.1"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
# For the `#[tool]` macro (below). Already pulled in by the default `derive`
# feature, but tool *argument* structs need these directly:
schemars = "0.8"
serde = { version = "1", features = ["derive"] }
```

### Feature flags (selected)

Default features: `rustls-tls`, `langsan`, `client`, `batch`, `derive`.

| Flag | Default | Purpose |
|------|---------|---------|
| `client` | yes | Enables `Client` (HTTP). Disable for wasm data-only. |
| `rustls-tls` | yes | Use rustls instead of system OpenSSL. |
| `langsan` | yes | Output sanitization (allow-list of benign Unicode). |
| `derive` | yes | The `#[tool]` / `#[derive(ToolArgs)]` macros. |
| `batch` | yes | Message Batches API. Does not build on wasm32. |
| `prompt-caching` | no | Anthropic prompt-caching beta headers. |
| `markdown` | no | `ToMarkdown` trait, markdown rendering. |
| `image` / `png` / `jpeg` / `gif` / `webp` | no | Image support via the `image` crate. |
| `memsecurity` | no | Encrypt the API key in memory. |

## Quick start — single message

```no_run
use misanthropic::{Client, Prompt, prompt::message::Role};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The key String is consumed, zeroized, and stored encrypted (with
    // `memsecurity`) or zeroed on drop. `Client::new` takes anything that is
    // `TryInto<Key>` (e.g. `String`). The `x-api-key` header is marked
    // sensitive on requests.
    let key = std::env::var("ANTHROPIC_API_KEY")?;
    let client = Client::new(key)?;

    // Build a Prompt (the request type) and send it. `Client::message`
    // forces `stream = false` and returns a `response::Message` directly.
    let message = client
        .message(
            Prompt::default()
                .set_messages([(Role::User, "What is 2+2?")]),
        )
        .await?;

    // `response::Message` implements `Display` (prints content).
    println!("{message}");

    // Access fields:
    //   message.inner          — the prompt::AssistantMessage
    //   message.inner.content  — Content (Display, iterable over Blocks)
    //   message.model          — model::Id
    //   message.stop_reason    — Option<StopReason>
    //   message.usage          — Usage { input_tokens, output_tokens, .. }

    Ok(())
}
```

## The `Prompt` builder

`Prompt` is the request type. It defaults to model **`Haiku45`** (Claude
Haiku 4.5) with `max_tokens = 4096`. All fields are public; the builder
methods are convenience helpers that return `Self`.

```rust
use std::num::NonZeroU32;
use misanthropic::{AnthropicModel, Prompt, prompt::message::Role};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let prompt = Prompt::default()
    // Set model — accepts AnthropicModel, model::Id, or any string.
    .model(AnthropicModel::Sonnet46)
    // System prompt — accepts &str, String, or Content.
    .set_system("You are a helpful assistant.")
    // Append to the system prompt.
    .add_system("Respond concisely.")
    // Set max tokens.
    .max_tokens(NonZeroU32::new(1024).unwrap())
    // Set temperature (0.0–1.0).
    .temperature(Some(0.7))
    // Add a user message — returns Result because turn order is validated
    // (must alternate User/Assistant; first message must be User).
    .add_message((Role::User, "Hello!"))?;
# let _ = prompt;
# Ok(())
# }
```

### Available models (`AnthropicModel` enum)

`.model(...)` takes an `AnthropicModel`, a `model::Id`, or any string. The
`AnthropicModel` enum tracks current and historical models (each variant
serializes to its wire ID, e.g. `Sonnet46` → `claude-sonnet-4-6`), with both
"latest"-style and pinned/dated variants. The **default** is `Haiku45`
(`claude-haiku-4-5`).

Rather than reproduce the list here (it drifts as models ship), see the
[`AnthropicModel` docs](https://docs.rs/misanthropic/latest/misanthropic/model/enum.AnthropicModel.html)
for the authoritative set — the variant↔wire-ID mapping is verified against the
live `/v1/models` endpoint by `test_client_models`. For a model not in the enum
yet, pass `model::Id::Custom("your-model-id".into())` or just the string.

### Messages — generic conversions

The crate leans **heavily on generics and `From`/`Into`** so call sites stay
clean. Every message method (`add_message`, `push_message`, `set_messages`,
`add_messages`, …) is bounded on `M: Into<Message>`, and the tuple
conversion is `(Role, T) where T: Into<Content>` — not just `&str`. A
`Message`/`Content`/`Block` can be built from a `&str`, a `String`, a
`tool::Use`, a `tool::Result`, an `Image`, a `DocumentSource`, a slice of
`&str`, a `response::Message`, and more. So a tuple's second element is
whatever converts into `Content` — a string is just the common case.

```rust
use misanthropic::{Prompt, prompt::message::Role};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
// set_messages replaces all messages; add_message appends one.
let prompt = Prompt::default().set_messages([
    (Role::User, "What is Rust?"),
    (Role::Assistant, "Rust is a systems programming language."),
    (Role::User, "What are its key features?"),
])?;
# let _ = prompt;
# Ok(())
# }
```

> **The crate is meant to be extended.** If something *logically* should
> convert and doesn't — e.g. `prompt.push_message(some_pushable_thing)` won't
> compile — that's usually a missing `From`/`Into` impl, not a design wall.
> Adding the conversion in the crate (it's a small, local change) is the
> idiomatic fix and a stated goal of the project; reach for that before
> contorting the call site.

### Multi-turn conversation

```no_run
use misanthropic::{AnthropicModel, Client, Prompt, prompt::message::Role};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let mut chat = Prompt::default()
    .model(AnthropicModel::Sonnet46)
    .set_system("You are a helpful assistant.")
    .add_message((Role::User, "What is Rust?"))?;

let reply = client.message(&chat).await?;
println!("Assistant: {reply}");

// Append the assistant reply, then a user follow-up. `push_message` is the
// in-place version of `add_message` (both validate turn order).
chat.push_message(reply)?;
chat.push_message((Role::User, "What about memory safety?"))?;

let reply = client.message(&chat).await?;
println!("Assistant: {reply}");
# Ok(())
# }
```

## Tool use — the `#[tool]` macro (preferred)

The `#[tool]` macro (default `derive` feature) turns an `impl` block into a
typed tool: it generates the wire definitions from your argument struct's
`JsonSchema`, and gives the type a concrete `impl Tool` so dispatch is typed
and validated — no hand-parsing of `serde_json::Value`.

```no_run
use misanthropic::{
    Client, Prompt,
    prompt::message::{Content, Role},
    tool::{Tool, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

// Argument struct. Field docs become the schema property descriptions the
// model sees (via `schemars`). Must be `Deserialize + JsonSchema`.
#[derive(Debug, Deserialize, JsonSchema)]
struct GetWeather {
    /// City name.
    city: String,
}

// A (possibly stateful) tool. Methods tagged `#[method]` must be
// `async fn(&mut self, args: ArgsTy) -> Result<Content, Content>`.
// `Ok` is the tool result; `Err` is a model-facing error.
struct Weather;

#[tool]
impl Weather {
    /// Get the weather for a city.
    #[method]
    async fn get_weather(
        &mut self,
        args: GetWeather,
    ) -> Result<Content, Content> {
        let report = format!("Sunny, 22C in {}", args.city); // your logic
        Ok(report.into())
    }
}

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
let mut weather = Weather;

let mut chat = Prompt::default()
    .model(misanthropic::AnthropicModel::Sonnet46)
    .set_system("Use tools when appropriate.")
    .add_message((Role::User, "What's the weather in Paris?"))?;

// Register the generated definition(s). Methods are namespaced by the tool
// type, e.g. `Weather__get_weather` (the `__` separator).
for definition in weather.definitions() {
    chat = chat.add_tool(definition);
}

let message = client.message(&chat).await?;

// `tool_use()` is `Some` when stop_reason is ToolUse and the last block is a
// tool call.
if let Some(call) = message.tool_use() {
    let call = call.clone();
    chat.push_message(message)?;

    // Typed dispatch: `call.input` is deserialized into `GetWeather` and
    // validated. Bad arguments become a helpful, model-facing error
    // automatically. Returns a `tool::Result` ready to push.
    let result = weather.call(call).await;
    chat.push_message(result)?;

    let final_reply = client.message(&chat).await?;
    println!("{final_reply}");
}
# Ok(())
# }
```

Notes on the macro:

- Each `#[method]` becomes a real inherent method you can still call directly.
- One `#[tool]` block can hold several `#[method]`s; each is namespaced
  `TypeName__method_name`.
- `#[method(defer_loading)]` marks a method's schema as deferrable for use
  with the tool-search server tool (large tool sets).
- The `Tool` trait also has `definitions()`, `call()`, plus optional
  `on_init` / `on_turn` lifecycle hooks and `save_json` / `load_json` for
  state persistence.

See the runnable example: `misanthropic/examples/strawberry.rs`.

## Tool use — manual `CustomMethodDef` (no macro)

When you want a hand-written schema, build a `tool::CustomMethodDef` and hand it
to `add_tool`. (This struct was previously named `MethodDef`, and before that
`Method`.)

```no_run
use misanthropic::{
    Client, Prompt, json,
    prompt::{Message, message::Role},
    tool::CustomMethodDef,
};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let mut chat = Prompt::default()
    .model(misanthropic::AnthropicModel::Sonnet46)
    .add_tool(CustomMethodDef {
        name: "get_weather".into(),
        description: "Get the weather for a city.".into(),
        schema: json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"],
        }),
        // These four are plain `Option`s — NOT feature-gated. Leave `None`
        // unless you need them:
        cache_control: None,    // prompt-caching breakpoint
        strict: None,           // Some(true) = grammar-constrained decoding
        defer_loading: None,    // Some(true) = defer schema (tool-search)
        allowed_callers: None,  // Some(...) = programmatic tool calling
    })
    .set_system("Use tools when appropriate.")
    .add_message((Role::User, "What's the weather in Paris?"))?;

let message = client.message(&chat).await?;

if let Some(call) = message.tool_use() {
    // call.name  — tool name ("get_weather")
    // call.id    — unique ID for this call
    // call.input — serde_json::Value with arguments
    let city = call.input["city"].as_str().unwrap();
    let weather = format!("Sunny, 22C in {city}"); // your logic

    // Build a tool result message (always Role::User under the hood).
    let result: Message = misanthropic::tool::Result {
        tool_use_id: call.id.to_string().into(),
        content: weather.into(),
        is_error: false,
        cache_control: None,
    }
    .into();

    chat.push_message(message)?;
    chat.push_message(result)?;

    let final_reply = client.message(&chat).await?;
    println!("{final_reply}");
}
# Ok(())
# }
```

`add_tool` accepts anything `Into<MethodDef>` — a `CustomMethodDef`, a
`ServerMethodDef` (Anthropic-executed, e.g. `web_search`), or a `Memory` (the
client-executed memory tool). `try_add_tool` accepts `TryInto<CustomMethodDef>`
(e.g. a `MethodBuilder` with validation).

## Predefined tools — server tools & client-executed tools

Predefined tools are added by versioned name with no schema of your own. Most
are **server-executed**: Anthropic runs them and the result blocks arrive in
the response (you never call anything) — e.g. `web_search` (see
`examples/web_search.rs`), `web_fetch`, `code_execution`, the tool-search
tools.

`code_execution` is the richest of these: enabling it gives the model a
sandboxed container with two sub-tools, whose outcomes come back as
`BashCodeExecutionToolResult` and `TextEditorCodeExecutionToolResult` blocks
(a *failed* bash command is still a `Result` with a non-zero `return_code`; the
`Error` variants are for the sub-tool itself failing):

```no_run
use misanthropic::{Prompt, prompt::message::{
    Block, BashCodeExecutionResultContent as Bash,
    TextEditorCodeExecutionResultContent as Edit, Role,
}, tool::ServerMethodDef};

# fn f(response: misanthropic::response::Message) -> Result<(), Box<dyn std::error::Error>> {
let _prompt = Prompt::default()
    .add_message((Role::User, "Sum 1..5 in a file with bash."))?
    .add_tool(ServerMethodDef::code_execution());

// Read the result blocks the container produced (in a response).
for block in response.inner.content.iter() {
    match block {
        Block::BashCodeExecutionToolResult { content, .. } => match content {
            Bash::Result { stdout, return_code, .. } => {
                println!("exit {return_code}: {stdout}");
            }
            Bash::Error { error_code, .. } => eprintln!("bash: {error_code}"),
        },
        Block::TextEditorCodeExecutionToolResult { content, .. } => match content {
            Edit::Create { is_file_update } => println!("created (update={is_file_update})"),
            Edit::View { content, .. } => println!("viewed: {content}"),
            Edit::StrReplace { lines, .. } => println!("edited: {} diff lines", lines.len()),
            Edit::Error { error_code, .. } => eprintln!("editor: {error_code}"),
        },
        _ => {}
    }
}
# Ok(())
# }
```

A few are **client-executed**: Anthropic defines the schema, but the model
emits an ordinary `tool_use` that *you* run against storage you control, just
like a custom tool — so they *define* like a server tool and *execute* like a
custom one. Each has a focused guide; load the one you need:

- **memory** — persistent notes across sessions, filesystem-backed
  (`FsMemoryBackend`). See [MEMORY.md](MEMORY.md).
- **text editor** (`str_replace_based_edit_tool`) — view/edit files in a
  working tree (`FsEditorBackend`). See [TEXT_EDITOR.md](TEXT_EDITOR.md).

Both share the same shape: `add_tool(Memory::latest())` /
`add_tool(TextEditor::latest())`, then drive a tool loop that executes each
`tool_use` locally and feeds the `tool::Result` back. Their backends drop into a
`ToolBox` and route by their fixed bare name with no special-casing.

## Using `json!` instead of `Prompt`

> **For quick experiments and tests only.** In production, prefer `Prompt`:
> you get turn-order validation, the typed builder, model/feature helpers, and
> compile-time checking that raw JSON skips. Reach for `json!` to reproduce a
> payload quickly or in a throwaway test, not in real code.

`Client::message` accepts anything `Serialize`. You can use raw JSON:

```no_run
use misanthropic::{Client, json, prompt::message::Role};

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

let message = client
    .message(json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1024,
        "system": "You are a pirate.",
        "messages": [{
            "role": Role::User,
            "content": "Ahoy!",
        }],
    }))
    .await?;

println!("{message}");
# Ok(())
# }
```

## Error handling

`Client` methods return `client::Result<T>` which wraps `client::Error`:

```no_run
use misanthropic::client::{Error, AnthropicError};

# async fn run(client: &misanthropic::Client, prompt: &misanthropic::Prompt) {
match client.message(prompt).await {
    Ok(msg) => println!("{msg}"),
    Err(Error::Anthropic(AnthropicError::RateLimit { message, .. })) => {
        eprintln!("Rate limited: {message}");
    }
    Err(Error::Anthropic(AnthropicError::Authentication { message })) => {
        eprintln!("Auth error: {message}");
    }
    Err(e) => eprintln!("Error: {e}"),
}
# }
```

### Error variants

| Variant | Description |
|---------|-------------|
| `Error::HTTP` | Network / reqwest error |
| `Error::Parse` | JSON deserialization failed |
| `Error::Anthropic(AnthropicError::*)` | API error (see below) |
| `Error::UnexpectedResponse` | Wrong response type (should not happen) |

**`AnthropicError` variants:** `InvalidRequest` (400), `Authentication`
(401), `Permission` (403), `NotFound` (404), `RequestTooLarge` (413),
`RateLimit` (429), `API` (500), `Overloaded` (529), `Timeout`, `Billing`,
`Unknown`. Exhaustive `match` (doc-tested drift guard for the prose above):

```rust
use misanthropic::client::AnthropicError;

# #[allow(unused_variables)]
# fn document(err: AnthropicError) {
match err {
    AnthropicError::InvalidRequest { .. } => {}    // 400
    AnthropicError::Authentication { .. } => {}    // 401
    AnthropicError::Permission { .. } => {}        // 403
    AnthropicError::NotFound { .. } => {}          // 404
    AnthropicError::RequestTooLarge { .. } => {}   // 413
    AnthropicError::RateLimit { .. } => {}         // 429 (has retry_after)
    AnthropicError::API { .. } => {}               // 500
    AnthropicError::Overloaded { .. } => {}        // 529 (has retry_after)
    AnthropicError::Timeout { .. } => {}
    AnthropicError::Billing { .. } => {}
    AnthropicError::Unknown { .. } => {}
}
# }
```

`RateLimit` and `Overloaded` carry a `retry_after: Option<u64>` field
populated from the `retry-after` response header. Prefer the
`AnthropicError::retry_after()` accessor, which returns it as an
`Option<std::time::Duration>`:

```rust
use misanthropic::client::{Error, AnthropicError};

# fn handle(e: &Error) {
if let Error::Anthropic(api_err) = e {
    if let Some(wait) = api_err.retry_after() {
        eprintln!("retry after {wait:?}");
    }
}
# }
```

## Response structure

```text
response::Message
├── id: Cow<str>              — unique message ID
├── inner: AssistantMessage
│   └── inner: prompt::Message
│       ├── role: Role::Assistant
│       └── content: Content  — Display, iterable over Blocks
├── model: model::Id
├── stop_reason: Option<StopReason>
│   └── EndTurn | MaxTokens | StopSequence | ToolUse | PauseTurn | Refusal
├── stop_sequence: Option<Cow<str>>
└── usage: Usage
    ├── input_tokens: u64
    └── output_tokens: u64
```

`StopReason` in full, as an exhaustive `match` (doc-tested drift guard — a new
variant breaks the build, since the tree above is prose and can't be):

```rust
use misanthropic::response::StopReason;

# #[allow(unused_variables)]
# fn document(reason: StopReason) {
match reason {
    StopReason::EndTurn => {}       // natural stopping point
    StopReason::MaxTokens => {}     // hit max_tokens
    StopReason::StopSequence => {}  // a stop sequence was generated
    StopReason::ToolUse => {}       // wants a tool call — see `tool_use()`
    StopReason::PauseTurn => {}     // server tool paused; resend to continue
    StopReason::Refusal => {}       // model declined (safety)
}
# }
```

## Examples

Runnable examples live in `misanthropic/examples/` (run with
`cargo run --example <name> --all-features`). Read the one whose shape matches
your task — they're the most current, compiler-checked usage.

| Example | Covers |
|---------|--------|
| `strawberry.rs` | **Typed tool use** via the `#[tool]` macro (`count_letters`) — the canonical tool example. |
| `python.rs` | Tool use where the assistant calls a `python` tool to compute an answer. |
| `few_shot_triage.rs` | **Few-shot prompting** + structured output — triage a free-text bug report into a structured form. |
| `structured_commit_classifier.rs` | Structured output — classify a unified diff into a commit message. |
| `vote_intent.rs` | Structured output — analyze a social-network post into a typed result. |
| `mid_conversation_system.rs` | Mid-conversation `Role::System` message (Opus 4.8+). |
| `interleaved_thinking.rs` | Adaptive extended thinking with interleaved thinking. |
| `tool_search.rs` | The tool-search server tool over a large, `defer_loading` tool set. |
| `web_search.rs` | The `web_search` server tool. |
| `web_fetch.rs` | The `web_fetch` server tool, paired with `web_search`. |
| `code_execution.rs` | The `code_execution` server tool — bash + file editing in a sandbox container. |
| `programmatic_tool_calling.rs` | `code_execution` calling a `.programmatic()` custom tool from inside the container (PTC). |
| `memory.rs` | The client-executed `memory` tool (`FsMemoryBackend`) — see [MEMORY.md](MEMORY.md). |
| `text_editor.rs` | The client-executed `text_editor` tool (`FsEditorBackend`) — see [TEXT_EDITOR.md](TEXT_EDITOR.md). |
| `neologism.rs` | `Client::message` with a custom system prompt. |
| `website_wizard.rs` | **Streaming** (`Client::stream`) — see the misan-streaming-api skill. |

## Key design notes

- **API keys** are zeroized on drop. With `memsecurity`, they are encrypted in
  memory. The `x-api-key` header is marked sensitive.
- **No `unsafe` code** — the crate uses `#[forbid(unsafe_code)]`.
- **Rate limiting** is built in (default: 50 req/min, tier 1). Adjust with
  `client.set_rate_limit(quota)`.
- **`Client` is cheap to clone** — it wraps `Arc`s internally.
- **Turn order is enforced (client-side)** — messages must alternate
  User/Assistant and the first must be User. Methods return `TurnOrderError`
  on violation. Two exceptions: Opus 4.8+ supports an authoritative
  mid-conversation `Role::System` turn (must follow a user turn and either end
  the array or precede an assistant turn); and **two adjacent assistant turns
  are allowed when the first contains a `ServerToolUse` block** — the API
  pauses a long-running server tool with `StopReason::PauseTurn`, and you
  resume by appending the paused turn back. Note Anthropic itself **no longer
  enforces** strict alternation server-side, but many Anthropic-compatible
  backends still do, so the crate keeps the check (in `prompt.rs`,
  `check_turn_order` / `Message::may_precede`).
- **Owned data, no lifetimes** — public types own their string data
  (`Cow<'static, str>` under the hood, sanitized when `langsan` is on) and
  carry **no lifetime parameter**. You can freely store a `Use`/`Message`/etc.
  with no `.into_static()` dance — they are already `'static`. (The crate used
  to thread a pervasive `'a`; it was removed because it bought no real
  zero-copy — without `#[serde(borrow)]` every deserialized string allocates
  anyway — at a large ergonomic cost.)
