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

Or use the `just run-example` recipe, which turns on every feature for you and
passes the rest through — `just run-example web_search "what's new?"`. The
crate's `log` feature is always on for examples, so set `RUST_LOG` (or pass
`--verbose` where an example supports it) to see the client's internal logs and
the `Chat` loop's tracing: `RUST_LOG=debug just run-example bash_background`.

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
| `web_fetch` | The `web_fetch` server tool, paired with `web_search`. | `client` |
| `code_execution` | The `code_execution` server tool — bash + file editing in a sandbox container. | `client` |
| `programmatic_tool_calling` | The `code_execution` tool calling a custom `.programmatic()` tool from inside the container (PTC). | `client` |
| `neologism` | A non-streaming `Client::message` call with a custom system prompt. | `client` |
| `batch_haiku` | **The Batch API** — submit many prompts at half price via `Client::tagged_batch`, poll with `Client::batch_poll`, match results back by id. | `batch` |
| `website_wizard` | **Streaming** with `Client::stream` — collects a generated HTML page. | `client` |
| `swarm` | **A multi-agent swarm** — a dev team in miniature: boss coordinates, ant designs, wasp critiques, bee implements, moth QAs; one concurrent `Chat` loop each, wired by a `#[tool]` mail tool with a postage ledger only the human refills (`/grant`). Needs Docker. | `client, bash-container, derive` |

## Shared helpers (`utils/`)

`examples/utils/` is a module pulled into each example with `mod utils;`. It
is not an example target (no `main.rs`) but its helpers are copy-pasteable into
real projects.

- **`api_key()`** (requires `client` feature) — acquires the Anthropic API key:
  tries `ANTHROPIC_API_KEY` first (with a privacy warning), then prompts and
  reads one line from stdin, then best-effort clears the key from the system
  clipboard if it looks like one (`sk-ant…`). Call this *before*
  `spawn_readline_loop` hands stdin to the line editor.

- **`Chat<State>` event loop + `spawn_readline_loop` / `Printer`** (requires
  `client` feature) — `Chat` drives the model to quiescence on each user beat,
  dispatches tool calls, and races tool-pushed notifications against user input.
  `spawn_readline_loop` runs `rustyline` on a dedicated thread so async output
  can print *above* the live prompt via the returned `Printer`.

- **`CommonArgs` / `ChatArgs` / `Args`** — shared clap flag groups. Flatten
  `CommonArgs` into an example's `Parser` and call `common.configure(prompt)` to
  apply `--model`, `--max-tokens`, and `--system` overrides while keeping the
  example's own defaults. `ChatArgs` adds `--max-tool-calls` and wires into
  `Chat` via `chat.configure(Chat::new(…))`. `Args` bundles both plus `--prompt`
  for examples that need nothing more.

For prose walkthroughs of the message and streaming APIs, see the agent skills
under [`.claude/skills/`](../../.claude/skills/), which are doc-tested so they
stay in sync with the crate.
