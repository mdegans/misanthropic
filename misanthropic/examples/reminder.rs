//! Example: tool *callbacks* via a [`Mailbox`], shown as a small CLI chat where
//! a single tool — **no [`ToolBox`] in sight** — drops a conversational
//! reminder into the conversation every few turns.
//!
//! # The idea
//!
//! Tool use is a *pair*: a [`tool::Use`] is answered by exactly one
//! [`tool::Result`] in the next message. That covers request/response, but not
//! a tool that wants to drop free-standing [`Content`] into the chat without
//! being called — a backgrounded job reporting in, or (here) a periodic nudge.
//! Those are **pushes**, not replies.
//!
//! A [`Tool`] owns a [`Mailbox`] (channel pair), `send`s through it, and hands
//! out [`Notifications`] via [`Tool::subscribe`]. The driver drains that
//! [`Stream`] and seats each beat. A [`ToolBox`] is only needed to *group and
//! aggregate* tools — it isn't mandatory — so here we drive a lone [`Tool`]
//! directly: `subscribe`, then call its lifecycle hooks
//! (`on_init`/`on_turn`/`on_teardown`) by hand.
//!
//! # Role is a *preference*, resolved by the driver
//!
//! [`Notification::preferred_roles`] is a `Vec<Role>`, not a baked role: a
//! reminder wants [`System`] where the model supports in-message system turns
//! and [`User`] where it doesn't. The driver picks the first the current model
//! supports — [`Prompt::resolve_role`]. Under the default model (Haiku) this
//! lands as a **user** turn; on Opus 4.8+ it would be a **system** turn.
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example reminder
//! ```
//!
//! Chat at the prompt; every third turn a reminder lands as its own turn and the
//! model takes it into account. `Ctrl-D` quits.
//!
//! [`Content`]: misanthropic::prompt::message::Content
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notification::preferred_roles`]: misanthropic::tool::Notification::preferred_roles
//! [`Prompt::resolve_role`]: misanthropic::Prompt::resolve_role
//! [`Stream`]: futures::Stream
//! [`System`]: misanthropic::prompt::message::Role::System
//! [`tool::Result`]: misanthropic::tool::Result
//! [`Tool::subscribe`]: misanthropic::tool::Tool::subscribe
//! [`tool::Use`]: misanthropic::tool::Use
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolBox`]: misanthropic::tool::ToolBox
//! [`User`]: misanthropic::prompt::message::Role::User

mod utils;

use std::io::BufRead;

use misanthropic::{
    Client, Prompt,
    prompt::{
        UserMessage,
        message::{Content, Role},
    },
    tool::{Mailbox, Notifications, Tool, tool},
};

/// Demo cap on total turns — the chat ends after this many exchanges, some
/// driven by the human, some by the reminder tool.
const MAX_TURNS: usize = 20;

/// In a real chat you should probably instruct the Assistant not to mention the
/// reminder and arrange for it to be joined with the User's message or appended
/// after as a System message.
const REMINDER_INSTRUCTIONS: &str = r#"<reminder_instructions>
Ever few turns you will get a [reminder] message from the runtime. This reminder does not come from the user.
</reminder_instructions>"#;

/// A **push-only** tool: it exposes no callable method (the `#[tool]` impl below
/// has no `#[method]`), so the model never sees it in the tools array and can't
/// call it. It only pushes a reminder every `every` turns — the simplest
/// demonstration of "[`Content`] without a [`Use`]".
///
/// [`Use`]: misanthropic::tool::Use
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

// The `#[tool]` macro builds a concrete `impl Tool` from the markers below.
// `#[connect]`/`#[subscribe]` are the new siblings of `#[on_init]`/`#[on_turn]`/
// `#[on_teardown]`; a tool with no `#[method]` is push-only.
#[tool]
impl Reminder {
    /// A [`ToolBox`] hands us a send-only handle on its aggregate channel,
    /// replacing our own. Unused in this standalone example, but it makes the
    /// tool box-ready (next session: bash-background + this, aggregated).
    ///
    /// [`ToolBox`]: misanthropic::tool::ToolBox
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.mailbox = mailbox;
    }

    /// Hand out our consumer end. Standalone, the driver takes it directly; once
    /// boxed, our mailbox is the box's send-only handle so this yields `None`
    /// and the box owns consumption.
    #[subscribe]
    fn subscribe(&mut self) -> Option<Notifications> {
        self.mailbox.subscribe()
    }

    /// Inject Reminder instructions into the system prompt.
    ///
    /// ## Note
    /// - Production tools should generally not overwrite the system prompt.
    ///   This is for brevity. Tools, on_init, should append and be idempotent.
    #[on_init]
    async fn set_system(
        &mut self,
        prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Content collects from an iterable of T where T: Into<Block>
        let system: Content = [
            "This is an example from the `misanthropic` Rust client crate.",
            REMINDER_INSTRUCTIONS,
        ]
        .into_iter()
        .collect();

        prompt.system = Some(system);

        Ok(())
    }

    /// Count the turn and, every `every`, push a reminder. The `send` stamps the
    /// source (`"reminder"`) — we can't fake it — and we ignore the result: a
    /// dropped reminder is fine (a dropped *job completion* would not be).
    #[on_turn]
    async fn nudge(
        &mut self,
        _prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.turns += 1;
        if self.turns.is_multiple_of(self.every) {
            let _ = self.mailbox.send(
                // Content to send (impl Into<Content>)
                "[reminder] Keep answers concise and on-task.",
                // Preferred roles for the message (System role is another
                // option but requires more careful handling of turn order.)
                vec![Role::System, Role::User],
            );
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(feature = "log")]
    env_logger::init();

    // Get the API key from stdin *before* the rustyline thread takes over stdin.
    println!("Enter your API key:");
    let key = std::io::stdin().lock().lines().next().unwrap()?;
    let client = Client::new(key)?;

    let mut reminder = Reminder::new(3);
    let mut chat = Prompt::default();
    let mut notifications = reminder.subscribe().expect("the reminder pushes");

    // A reminder has no #[method]s so no method definitions need to be set in
    // Prompt::methods.
    assert!(reminder.definitions().is_empty());

    // Calls `set_system` above, adding the tool's instructions.
    reminder.on_init(&mut chat).await?;

    // Stdin/stdout handling helper
    let (mut line_rx, mut printer) = utils::spawn_readline_loop("you ▸ ")?;
    printer.line(
        "Chat with the model; a reminder lands every 3rd turn. Ctrl-D quits.\n",
    );

    // The conversation is a strict alternation of user and assistant turns —
    // except the "user" beat sometimes comes from the reminder tool instead of
    // the human. Note that the official Anthropic API no longer requires this
    // strict alternation but many third party implementations do.
    for _ in 0..MAX_TURNS {
        reminder.on_turn(&mut chat).await?;

        // It is the User's turn
        tokio::select! {
            line = line_rx.recv() => match line {
                None => break, // reader thread ended (Ctrl-D)
                Some(line) => chat.push_message((Role::User, line))?,
            },
            Some(note) = notifications.recv() => {
                chat.push_message(
                    // For brevity, we always use User role since System role
                    // requires Opus 4.8 as of writing and more careful handing
                    // of turn order. In real code use Prompt::resolve_role to
                    // find the preferred role, but this may require message
                    // reordering.
                    UserMessage::from(note.content)
                )?;
            }
        }
        // It is the Assistant's turn

        let message = client.message(&chat).await?;
        printer.line(format!("claude ▸ {}\n", message.inner.content));
        chat.push_message(message)?;
    }

    reminder.on_teardown(&mut chat).await?;
    printer.line("bye");
    Ok(())
}
