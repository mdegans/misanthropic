# `text_editor` tool — client-executed, filesystem-backed

The `text_editor` tool (`str_replace_based_edit_tool`, feature `text-editor` for
the typed `Command` + definition) is the sibling of [`memory`](MEMORY.md):
another **client-executed predefined tool**. Anthropic defines the schema (added
by versioned name via `TextEditor::latest()` — `text_editor_20250728`, the
Claude-4 line); the model emits an ordinary `tool_use` that *you* execute and
answer with a `tool::Result`.

`tool::text_editor::FsEditorBackend` (feature `text-editor-fs`, async on tokio)
is a ready-made filesystem backend, jailed to a working directory — any file
type (no extension allowlist, unlike the markdown-jailed memory backend):

```no_run
# async fn run() -> Result<(), Box<dyn std::error::Error>> {
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{TextEditor, Tool, text_editor::FsEditorBackend},
};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
let mut editor = FsEditorBackend::new("./workspace").await?; // jailed to dir

let mut chat = Prompt::default()
    .add_tool(TextEditor::latest())                   // predefined, no schema
    .add_message((Role::User, "Fix the syntax error in primes.py."))?;

// Drive the tool loop: execute each editor `tool_use` locally and feed the
// result back, until a turn arrives with no tool call — that one is the answer.
let answer = loop {
    let message = client.message(&chat).await?;
    let Some(call) = message.tool_use() else { break message };
    let call = call.clone();
    chat.push_message(message)?;
    let result = editor.call(call).await;             // typed dispatch
    chat.push_message(result)?;
};
println!("{}", answer.inner.content);
# Ok(())
# }
```

The model's `tool_use` input deserializes into a typed `text_editor::Command`
(`view`/`create`/`str_replace`/`insert`, plus an `Unknown` catch-all — e.g. the
`undo_edit` of an older version, dropped from `text_editor_20250728`, still
round-trips). `FsEditorBackend` handles dispatch for you; implement `Tool`
yourself to edit a different store. Optional truncation of large `view`s:
`FsEditorBackend::new(dir).await?.with_max_characters(10_000)` (also advertised
on the definition as `max_characters`).

Like the memory backend, it drops into a `ToolBox`: the box installs the
predefined def and routes the bare `"str_replace_based_edit_tool"` `tool_use`
back to it, with no per-tool special-casing.

See the runnable example: `misanthropic/examples/text_editor.rs` (a one-shot,
bounded-loop bug fix that verifies the repair on disk).
