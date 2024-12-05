# `misanthropic`

![Build Status](https://github.com/mdegans/misanthropic/actions/workflows/tests.yaml/badge.svg)
[![codecov](https://codecov.io/gh/mdegans/misanthropic/branch/main/graph/badge.svg)](https://codecov.io/gh/mdegans/misanthropic)

Is an unofficial simple, ergonomic, client for the Anthropic Messages API.

## Usage

### Streaming

```rust
let client = Client::new(key)?;

// Request a stream of events or errors. `json!` can be used, the `Prompt`
// builder pattern (shown in the `Single Message` example below), or anything
// serializable.
let stream = client
    // Forces `stream=true` in the request.
    .stream(json!({
      "model": "claude-3-5-sonnet-latest",
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
let client = Client::new(key)?;

// Many common usage patterns are supported out of the box for building
// `Prompt`s, such as messages from an iterable of tuples of `Role` and
// `String`.
let message = client
    .message(Prompt::default().messages([(Role::User, args.prompt)]))
    .await?;

println!("{}", message);
```

## Features

- [x] Async but does not _directly_ depend on tokio
- [x] Tool use,
- [x] Streaming responses
- [x] Message responses
- [x] Image support with or without the `image` crate
- [x] Markdown formatting of messages, including images
- [x] HTML formatting of messages\*.
- [x] Prompt caching support
- [x] Custom request and endpoint support
- [x] Zero-copy where possible
- [x] [Sanitization](https://crates.io/crates/langsan) of input and output to mitigate [injection attacks](https://arstechnica.com/security/2024/10/ai-chatbots-can-read-and-write-invisible-text-creating-an-ideal-covert-channel/)
- [ ] Amazon Bedrock support
- [ ] Vertex AI support

\* _Base64 encoded images are currently not implemented for HTML but this is a planned feature._

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
  memory when using the `memsecurity` feature and any headers containing copies
  marked as sensitive. `rustls` is an optional feature and is recommended for
  security. It is on by default.
