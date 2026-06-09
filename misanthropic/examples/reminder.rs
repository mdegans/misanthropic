//! Example: tool *callbacks* via a [`Mailbox`] — a push-only [`Tool`] that
//! drops a [`Content`] reminder into the chat every few turns without ever
//! being called. [`tool::Use`]/[`tool::Result`] pairs handle request/response;
//! a [`Mailbox`] handles free-standing pushes. The tool hands out
//! [`Notifications`] via [`Tool::subscribe`]; a [`ToolBox`] aggregates the
//! mailbox; the `Chat` driver races the notification [`Stream`] against user
//! input and seats turns append-only. [`Notification::preferred_roles`] is a
//! preference (`[System, User]`); the driver resolves it per model via
//! [`Prompt::resolve_role`] — Haiku lands as user, Opus 4.8+ as system.
//!
//! ```sh
//! cargo run --features client --example reminder
//! ```
//!
//! [`Content`]: misanthropic::prompt::message::Content
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notifications`]: misanthropic::tool::Notifications
//! [`Notification::preferred_roles`]: misanthropic::tool::Notification::preferred_roles
//! [`Prompt::resolve_role`]: misanthropic::Prompt::resolve_role
//! [`Stream`]: futures::Stream
//! [`tool::Result`]: misanthropic::tool::Result
//! [`Tool::subscribe`]: misanthropic::tool::Tool::subscribe
//! [`tool::Use`]: misanthropic::tool::Use
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolBox`]: misanthropic::tool::ToolBox

mod utils;

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    prompt::message::{Content, Role},
    tool::{Mailbox, Notifications, ToolBox, tool},
};

/// Interactive chat with periodic push-only reminders from a background tool.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
}

const REMINDER_INSTRUCTIONS: &str = r#"<reminder_instructions>
Ever few turns you will get a [reminder] message from the runtime. This reminder does not come from the user.
</reminder_instructions>"#;

/// Push-only tool: no `#[method]`, so the model never sees it in the tools
/// array. Pushes a reminder every `every` turns without being called.
struct Reminder {
    /// Push a reminder every this many turns.
    every: u32,
    /// Turns seen so far (counted in `#[on_turn]`).
    turns: u32,
    /// Our channel pair wrapper
    mailbox: Mailbox,
}

impl Reminder {
    fn new(every: u32) -> Self {
        Self {
            every,
            turns: 0,
            mailbox: Mailbox::new("reminder"),
        }
    }
}

#[tool]
impl Reminder {
    /// A [`ToolBox`] replaces our mailbox with its aggregate send handle.
    ///
    /// [`ToolBox`]: misanthropic::tool::ToolBox
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.mailbox = mailbox;
    }

    /// Yields `None` once boxed (the box owns consumption).
    #[subscribe]
    fn subscribe(&mut self) -> Option<Notifications> {
        self.mailbox.subscribe()
    }

    /// Injects reminder instructions. Production tools should append, not
    /// overwrite, and be idempotent.
    #[on_init]
    async fn set_system(
        &mut self,
        prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let system: Content = [
            "This is an example from the `misanthropic` Rust client crate.",
            REMINDER_INSTRUCTIONS,
        ]
        .into_iter()
        .collect();

        prompt.system = Some(system);

        Ok(())
    }

    /// Counts the turn; every `every` turns pushes a reminder. Dropped reminders
    /// are fine — dropped job completions would not be.
    #[on_turn]
    async fn nudge(
        &mut self,
        _prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.turns += 1;
        if self.turns.is_multiple_of(self.every) {
            let _ = self.mailbox.send(
                "[reminder] Keep answers concise and on-task.",
                vec![Role::System, Role::User],
            );
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // Get the API key from stdin *before* the rustyline thread takes over stdin.
    let client = Client::new(utils::api_key()?)?;

    let toolbox = ToolBox::new().add(Reminder::new(3));

    let (mut lines, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line(
        "Chat with the model; a reminder lands every 3rd turn. Ctrl-D quits.\n",
    );

    cli.chat
        .configure(utils::Chat::new(
            client,
            cli.common.configure(Prompt::default()),
            toolbox,
        ))
        .on_assistant(move |_state: &mut (), msg| {
            printer.line(format!("claude ▸ {}\n", msg.content))
        })
        .run((), async move |_state: &mut ()| {
            Ok(lines
                .recv()
                .await
                .map(|line| vec![(Role::User, line).into()]))
        })
        .await?;

    println!("bye");
    Ok(())
}
