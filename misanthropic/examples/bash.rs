//! Example: the **bash tool** ([`Bash`]) — Anthropic's predefined
//! `bash_20250124`, executed in a locked-down Docker sandbox.
//!
//! Like `memory`/`text_editor`, bash is a *client-side predefined* tool:
//! Anthropic defines the schema (you add it by versioned name via
//! [`Bash::latest`]), the model emits a [`tool_use`] carrying a typed
//! [`bash::Command`], and *you* execute it — here in a container, via
//! [`BashTool`] over a [`DockerSandbox`]. The sandbox provisions an image *with*
//! network (so `apk add` works), then runs the session with `--network none` as
//! a non-root user, and is torn down (its container removed) at the end.
//!
//! The drive loop is **bounded** — an autonomous one-shot must terminate.
//!
//! # Usage
//!
//! ```sh
//! # Needs Docker running and a linux bashd built for the container's arch:
//! docker run --rm -v "$PWD":/w -w /w -e CARGO_TARGET_DIR=/w/target-linux \
//!     rust:alpine sh -c 'apk add --no-cache musl-dev && cargo build -p bashd --release'
//! BASHD_PATH=target-linux/release/bashd \
//!     cargo run --features "client bash-container" --example bash
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment, and `BASHD_PATH` pointing at a
//! `bashd` binary built for the container (defaults to `target-linux/release/bashd`).
//!
//! [`Bash`]: misanthropic::tool::Bash
//! [`Bash::latest`]: misanthropic::tool::Bash::latest
//! [`tool_use`]: misanthropic::tool::Use
//! [`BashTool`]: misanthropic::tool::bash::BashTool
//! [`DockerSandbox`]: misanthropic::tool::bash::DockerSandbox
//! [`bash::Command`]: misanthropic::tool::bash::Command

use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{
        Tool, ToolBox,
        bash::{BashTool, DockerSandbox},
    },
};

/// Cap on autonomous turns so a confused model can't loop forever.
const MAX_TURNS: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;
    let bashd = std::env::var("BASHD_PATH")
        .unwrap_or_else(|_| "target-linux/release/bashd".to_string());

    // Provisioned WITH network (apk), then run with `--network none`, non-root,
    // and torn down at the end. The sandbox is explicit: `BashTool` wraps it.
    let mut tools = ToolBox::new().add(BashTool::new(
        DockerSandbox::alpine()
            .setup("apk add --no-cache bash coreutils")
            .user("agent")
            .workdir("/work")
            .persist_cwd(false)
            .bashd_path(bashd),
    ));

    let mut chat = Prompt::default().add_message((
        Role::User,
        "Write a shell script that prints the 10th prime number, then run it. \
         Report the number it prints.",
    ))?;

    // `prepare` installs the bash def and runs `on_init` — which provisions the
    // image and launches the container + bashd.
    println!("Starting sandbox (provisioning image, launching bashd)...");
    tools.prepare(&mut chat).await?;

    // Drive the tool loop, bounded. Each bash `tool_use` runs in the container.
    let mut answer = None;
    for _ in 0..MAX_TURNS {
        let message = client.message(&chat).await?;
        match message.tool_use() {
            None => {
                answer = Some(message);
                break;
            }
            Some(call) => {
                let call = call.clone();
                println!("bash ▸ {}", call.input);
                chat.push_message(message)?;
                let result = tools.call(call).await;
                chat.push_message(result)?;
            }
        }
    }

    // Tear down every tool — `on_teardown` removes the container. Best-effort,
    // and the `DockerSandbox` also has a blocking `Drop` guard as a backstop.
    tools.teardown_tools(&mut chat).await?;

    let answer = answer.ok_or("bash loop did not converge in time")?;
    println!("\nclaude ▸ {}", answer.inner.content);
    Ok(())
}
