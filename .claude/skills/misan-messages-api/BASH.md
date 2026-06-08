# `bash` tool — client-executed, sandbox-backed

The `bash` tool (feature `bash` for the typed `Command` + `BashSandbox` trait +
`BashTool` adapter) is the third **client-executed predefined tool**, alongside
[`memory`](MEMORY.md) and [`text_editor`](TEXT_EDITOR.md): Anthropic defines the
schema (added by versioned name via `Bash::latest()` — `bash_20250124`), the
model emits an ordinary `tool_use` (`name: "bash"`) whose input is a typed
`bash::Command`, and *you* run it. The difference from its siblings: bash
executes in a **sandbox**, not a filesystem jail — an untrusted shell needs
isolation, not just a working directory.

Don't confuse it with the `code_execution` *server* tool: that one runs in
**Anthropic's** container (you only read result blocks). This bash tool runs on
**your** host, in a sandbox you control and tear down.

`tool::bash::DockerSandbox` (feature `bash-container`) is the reference executor:
it boots the baked `misan-bashd` image — `bashd` (a tiny session daemon) on an
immutable read-only rootfs — as a non-root user, and reaches it over an
ephemeral per-container mutual-TLS channel. A `docker exec` per command would
lose the working directory and environment, so `bashd` owns one persistent
shell session for the life of the sandbox.

```no_run
# async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{Tool, ToolBox, bash::{BashTool, DockerSandbox}},
};

let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

// The default sandbox: the baked `misan-bashd` image (read-only rootfs, bashd
// already inside), booted as a non-root user and torn down at the end.
let mut tools = ToolBox::new().add(BashTool::new(DockerSandbox::default()));

let mut chat = Prompt::default().add_message((
    Role::User,
    "Write a shell script that prints the 10th prime, run it, report the number.",
))?;

// `prepare` installs the bash def and runs `on_init` — booting the container
// and launching bashd inside it.
tools.prepare(&mut chat).await?;

// Drive the tool loop, bounded — an autonomous run must terminate.
let mut answer = None;
for _ in 0..10 {
    let message = client.message(&chat).await?;
    let Some(call) = message.tool_use() else { answer = Some(message); break };
    let call = call.clone();
    chat.push_message(message)?;
    let result = tools.call(call).await;          // runs in the container
    chat.push_message(result)?;
}

// `on_teardown` removes the container (a blocking `Drop` guard backstops it).
tools.teardown_tools(&mut chat).await?;
println!("{}", answer.ok_or("did not converge")?.inner.content);
# Ok(())
# }
```

The model's input deserializes into a typed `bash::Command` — a known/unknown
union so a newer command still round-trips. The known variants are `Run`
(`command`, optionally `background`/`timeout_secs`), `Restart` (`{"restart":
true}` — a fresh shell), and, for background jobs, `Poll` and `Kill` by job id.
`Bash::latest()` advertises the predefined `bash_20250124` schema, which only
elicits `Run`/`Restart`; `BashTool` routes each: `restart` resets the session
(and may drop a borked home), everything else runs. For a richer surface, the
typed `RichBash` tool (the `#[tool]` Tool/Method split, `derive` + `tokio`)
exposes `bash__run` (with `background`/`timeout_secs`), `bash__restart`,
`bash__check_output`, and `bash__kill` as separate flat-schema methods — and a
backgrounded `run` *calls back*, pushing the result as a `User` notification via
its `Mailbox` when the job finishes (no polling). See
`misanthropic/examples/bash_background.rs`.

Implement `BashSandbox` yourself to back bash with a different isolation
mechanism — it's the extension point; `DockerSandbox` is one impl. The trait's
lifecycle methods (`start`/`exec`/`poll`/`wait`/`kill`/`restart`/`teardown`) are
where **host-side** concerns live: notably the `$HOME` backup is restored on
`start` and snapshotted on `teardown`, *outside* `bashd`, so a daemon crash
can't lose it.

`DockerSandbox` is a fluent builder over that default: `.network(Network::…)`
(egress is off by default), `.workdir`/`.user`/`.setup(script)`,
`.memory`/`.pids_limit`/`.tmp_limit` resource caps, and a persistent named
`$HOME` volume via `.home_id(uuid)` (+ `.home_fs`/`.home_limit`) so a session's
home survives teardown and restores on the next `start`.

See the runnable example: `misanthropic/examples/bash.rs` (a bounded one-shot
that writes and runs a script in the container). It needs the image built first
(`just build-bashd`) and the `bash-container` feature.
