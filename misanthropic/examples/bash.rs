//! Example: the **bash tool** ([`Bash`]) — Anthropic's predefined
//! `bash_20250124` executed in a locked-down Docker sandbox. Anthropic defines
//! the schema (add by versioned name via [`Bash::latest`]); the model emits a
//! [`tool_use`] carrying a typed [`bash::Command`]; you execute it via
//! [`BashTool`] over a [`DockerSandbox`]. The default sandbox boots the baked
//! `misan-bashd` image (immutable read-only rootfs, non-root user) and tears
//! it down at the end. The drive loop is **bounded** — an autonomous one-shot
//! must terminate.
//!
//! ```sh
//! just build-bashd   # build the misan-bashd sandbox image (once; needs Docker)
//! cargo run --features "client bash-container" --example bash
//! ```
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

    let mut tools = ToolBox::new().add(BashTool::new(DockerSandbox::default()));

    let mut chat = cli.common.configure(Prompt::default()).add_message((
        Role::User,
        "Write a shell script that prints the 10th prime number, then run it. \
         Report the number it prints.",
    ))?;

    // `prepare` installs the bash def and boots the container via `on_init`.
    println!("Starting sandbox (booting container, launching bashd)...");
    tools.prepare(&mut chat).await?;

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

    // `DockerSandbox` also has a blocking `Drop` guard as a backstop.
    tools.teardown_tools(&mut chat).await?;

    let answer = answer.ok_or("bash loop did not converge in time")?;
    println!("\nclaude ▸ {}", answer.inner.content);
    Ok(())
}
