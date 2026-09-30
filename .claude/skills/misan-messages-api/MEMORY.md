# `memory` tool — client-executed, filesystem-backed

The `memory` tool (feature `memory` for the typed `Command` + definition) is a
**client-executed predefined tool**: Anthropic defines the schema (you add it by
versioned name, no schema of your own), but the model emits an ordinary
`tool_use` that *you* run against storage you control — just like a custom tool.
So it *defines* like a server tool and *executes* like a custom one.

`tool::memory::FsMemoryBackend` (feature `memory-fs`, which adds an async tokio
executor) is a ready-made filesystem backend, jailed to one directory and (by
default) to markdown files:

```no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use misanthropic::{
    Client, Prompt,
    prompt::message::{Role, UserMessage},
    tool::{Memory, Tool, memory::FsMemoryBackend},
};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
let mut memory = FsMemoryBackend::new("./memories").await?; // markdown-jailed

let mut chat = Prompt::default()
    .add_tool(Memory::latest())                       // predefined, no schema
    .add_message((Role::User, "Check your notes, then help me."))?;

// Drive the tool loop: execute every memory `tool_use` locally and feed the
// result back, until a turn arrives with no tool call — that one is the answer.
let answer = loop {
    let message = client.message(&chat).await?;
    // Every call in the turn; empty unless it stopped for `tool_use`.
    let calls: Vec<_> = message.tool_uses().cloned().collect();
    if calls.is_empty() { break message }
    chat.push_message(message)?;
    let mut results = Vec::new();
    for call in calls {
        results.push(memory.call(call).await);        // typed dispatch
    }
    // All results in one user turn.
    chat.push_message(results.into_iter().collect::<UserMessage>())?;
};
println!("{}", answer.inner.content);
# Ok(())
# }
```

The model's `tool_use` input deserializes into a typed `memory::Command`
(`view`/`create`/`str_replace`/`insert`/`delete`/`rename`, plus an `Unknown`
catch-all so a newer memory version still round-trips). `FsMemoryBackend`
handles all of that for you; implement `Tool` yourself for a different store.
Drop the backend into a `ToolBox` and it installs its own predefined definition
and routes the bare `"memory"` `tool_use` back to itself — no special-casing.

See the runnable example: `misanthropic/examples/memory.rs` (a persistent-notes
CLI). The sibling client-executed tool is the **text editor** — see
[TEXT_EDITOR.md](TEXT_EDITOR.md).
