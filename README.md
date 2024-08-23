# `misanthropic`

Is an unofficial simple, ergonomic, client for the Anthropic Messages API.

## Usage

### Streaming

```rust
// Create a client. `key` will be consumed, zeroized, and stored securely.
let client = Client::new(key)?;

// Request a stream of events or errors. `json!` can be used, a `Request`, or a
// combination of strings and concrete types like `Model`. All Client request
// methods accept anything serializable for maximum flexibility.
let stream = client
    // Forces `stream=true` in the request.
    .stream(json!({
      "model": Model::Sonnet35,
      "max_tokens": args.max_tokens,
      "temperature": 0,
      "system": args.system,
      "messages": [
        {
          "role": Role::User,
          "content": specs,
        }
      ],
    }))
    .await?
    // Filter out rate limit and overloaded errors. This is optional but
    // recommended for most use cases. The stream will continue when the
    // server is ready. Otherwise the stream will include these errors.
    .filter_rate_limit()
    // Filter out everything but text pieces (and errors).
    .text();

// Collect the stream into a single string.
let content: String = stream
    .try_collect()
    .await?;
```

### Single Message

```rust
// Create a client. `key` will be consumed and zeroized.
let client = Client::new(key)?;

// Request a single message. The parameters are the same as the streaming
// example above. If a value is `None` it will be omitted from the request.
// This is less flexible than json! but some may prefer it. A Builder pattern
// is not yet available but is planned to reduce the verbosity.
let message = client
    .message(Request {
        model: Model::Sonnet35,
        messages: vec![Message {
            role: Role::User,
            content: args.prompt.into(),
        }],
        max_tokens: 1000.try_into().unwrap(),
        metadata: serde_json::Value::Null,
        stop_sequences: None,
        stream: None,
        system: None,
        temperature: Some(1.0),
        tool_choice: None,
        tools: None,
        top_k: None,
        top_p: None,
    })
    .await?;

println!("{}", message);
```

## Features

- [x] Async but does not _directly_ depend on tokio
- [x] Streaming responses
- [x] Message responses
- [x] Image support with or without the `image` crate
- [x] Markdown formatting of messages, including images
- [x] Prompt caching support
- [x] Custom request and endpoint support
- [ ] Amazon Bedrock support
- [ ] Vertex AI support

[reqwest]: https://docs.rs/reqwest

## FAQ

- **Why is it called `misanthropic`?** No reason, really. I just like the word.
  Anthropic is both a company and a word meaning "relating to mankind". This
  crate is neither official or related to mankind so, `misanthropic` it is.
- **Doesn't `reqwest` depend on `tokio`?** On some platforms, yes.
- **Can i use `misanthropic` with Amazon or Vertex?** Not yet, but it's on the
  roadmap. for now the `Client` does support custom endpoints and the inner
  `reqwest::Client` can be accessed directly to make necessary adjustments to
  headers, etc.
- **Has this crate been audited?** No, but auditing is welcome. A best effort
  has been made to ensure security and privacy. The API key is encrypted in
  memory using the `memsecurity` crate and any headers containing copies marked
  as sensitive. `rustls` is an optional feature and is recommended for security.
  It is on by default.
