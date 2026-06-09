//! Example: the **bash tool** ([`Bash`]) — Anthropic's predefined
//! `bash_20250124`, executed in a locked-down Docker sandbox.
//!
//! Like `memory`/`text_editor`, bash is a *client-side predefined* tool:
//! Anthropic defines the schema (you add it by versioned name via
//! [`Bash::latest`]), the model emits a [`tool_use`] carrying a typed
//! [`bash::Command`], and *you* execute it — here in a container, via
//! [`BashTool`] over a [`DockerSandbox`]. The default sandbox boots the baked
//! `misan-bashd` image (`bashd` on an immutable read-only rootfs) as a non-root
//! user, and is torn down (its container removed) at the end.
//!
//! The drive loop is **bounded** — an autonomous one-shot must terminate.
//!
//! # Usage
//!
//! ```sh
//! just build-bashd   # build the misan-bashd sandbox image (once; needs Docker)
//! cargo run --features "client bash-container" --example bash
//! ```
//!
//! Expects `ANTHROPIC_API_KEY` in the environment and the `misan-bashd` image
//! built (`just build-bashd`).
//!
//! [`Bash`]: misanthropic::tool::Bash
//! [`Bash::latest`]: misanthropic::tool::Bash::latest
//! [`tool_use`]: misanthropic::tool::Use
//! [`BashTool`]: misanthropic::tool::bash::BashTool
//! [`DockerSandbox`]: misanthropic::tool::bash::DockerSandbox
//! [`bash::Command`]: misanthropic::tool::bash::Command

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    prompt::message::Role,
    tool::{
        Tool, ToolBox,
        bash::{BashTool, DockerSandbox},
    },
};

/// Run a bounded autonomous bash session in a Docker sandbox.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
}

/// Cap on autonomous turns so a confused model can't loop forever.
const MAX_TURNS: usize = 10;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    // The default sandbox: the baked `misan-bashd` image (read-only rootfs,
    // bashd already inside) booted as the non-root `agent`, torn down at the
    // end. The sandbox is explicit: `BashTool` wraps it.
    let mut tools = ToolBox::new().add(BashTool::new(DockerSandbox::default()));

    let mut chat = cli.common.configure(Prompt::default()).add_message((
        Role::User,
        "Write a shell script that prints the 10th prime number, then run it. \
         Report the number it prints.",
    ))?;

    // `prepare` installs the bash def and runs `on_init` — which boots the
    // container and launches bashd inside it.
    println!("Starting sandbox (booting container, launching bashd)...");
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
