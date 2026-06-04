# Examples

Runnable examples for the `misanthropic` crate. Each reads an API key (most
prompt for it on stdin) and calls the live API, so running one costs a few
tokens.

Run with `cargo run --example <name>` plus the features it needs (listed
below), or just enable everything:

```sh
cargo run --example strawberry --all-features
# or, with only what the example needs:
cargo run --example strawberry --features "client,markdown,derive"
```

Pass `-- --help` to see an example's own arguments, e.g.
`cargo run --example strawberry --all-features -- --help`.

| Example | What it shows | Features |
|---------|---------------|----------|
| `strawberry` | **Typed tool use** via the `#[tool]` macro — a `count_letters` tool. The canonical tool example. | `client, markdown, derive` |
| `python` | Tool use where the assistant calls a `python` tool to compute an answer. | `client, markdown` |
| `few_shot_triage` | **Few-shot prompting** + structured output — triage a free-text bug report into a structured form. | `client` |
| `structured_commit_classifier` | Structured output — classify a unified diff into a commit message. | `client` |
| `vote_intent` | Structured output — analyze a social-network post into a typed result. | `client` |
| `mid_conversation_system` | A mid-conversation `Role::System` message (Opus 4.8+). | `client` |
| `interleaved_thinking` | Adaptive extended thinking with interleaved thinking. | `client, derive` |
| `tool_search` | The tool-search server tool over a large, `defer_loading` tool set. | `client, derive` |
| `web_search` | The `web_search` server tool. | `client` |
| `neologism` | A non-streaming `Client::message` call with a custom system prompt. | `client` |
| `website_wizard` | **Streaming** with `Client::stream` — collects a generated HTML page. | `client` |

For prose walkthroughs of the message and streaming APIs, see the agent skills
under [`.claude/skills/`](../../.claude/skills/), which are doc-tested so they
stay in sync with the crate.
