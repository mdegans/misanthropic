//! Example: tool *callbacks* via a [`Mailbox`], shown as a small CLI chat where
//! a single tool — **no [`ToolBox`] in sight** — drops a conversational reminder
//! into the conversation every few turns. This is the draft of the agent reactor
//! we'll eventually encapsulate.
//!
//! # The idea
//!
//! Tool use is a *pair*: a [`tool_use`] is answered by exactly one
//! [`tool::Result`] in the next message. That covers request/response, but not a
//! tool that wants to drop free-standing [`Content`] into the chat without being
//! called — a backgrounded job reporting in, or (here) a periodic nudge. Those
//! are **pushes**, not replies.
//!
//! A [`Tool`] owns a [`Mailbox`], `send`s through it, and hands out the consumer
//! end via [`Tool::subscribe`]. The driver drains that stream and seats each
//! beat. A [`ToolBox`] is only needed to *group and aggregate* tools — it isn't
//! mandatory — so here we drive a lone [`Tool`] directly: `subscribe`, then call
//! its lifecycle hooks (`on_init`/`on_turn`/`on_teardown`) by hand.
//!
//! ## Blocking input under `select!` — the cancel-safety trick
//!
//! `rustyline` is blocking and has no async mode. Putting
//! `spawn_blocking(|| rl.readline())` *inside* a `select!` branch would be a bug:
//! when the other branch wins, `select!` drops that future, but dropping a
//! `spawn_blocking` handle does **not** cancel the blocking thread — the typed
//! line is lost and the next iteration spawns a *second* `readline` racing the
//! first on the tty. Instead we keep the blocking call **out of the future
//! tree**: one long-lived [`std::thread`] owns stdin and forwards finished lines
//! over a [`tokio::mpsc`] channel. `select!` then only ever polls `mpsc` `recv()`
//! and [`Notifications::recv`], both **cancel-safe** — a buffered item survives a
//! lost branch.
//!
//! ## Role is a *preference*, resolved by the driver
//!
//! [`Notification::preferred_roles`] is a `Vec<Role>`, not a baked role: a
//! reminder wants [`System`] where the model supports in-message system turns and
//! [`User`] where it doesn't. The driver picks the first the current model
//! supports — [`Prompt::resolve_role`]. Under the default model (Haiku) this
//! lands as a **user** turn; on Opus 4.8+ it would be a **system** turn.
//!
//! [`tokio::mpsc`]: https://docs.rs/tokio/latest/tokio/sync/mpsc/index.html
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
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notification::preferred_roles`]: misanthropic::tool::Notification::preferred_roles
//! [`Notifications::recv`]: misanthropic::tool::Notifications::recv
//! [`Prompt::resolve_role`]: misanthropic::Prompt::resolve_role
//! [`Tool`]: misanthropic::tool::Tool
//! [`Tool::subscribe`]: misanthropic::tool::Tool::subscribe
//! [`ToolBox`]: misanthropic::tool::ToolBox
//! [`Content`]: misanthropic::prompt::message::Content
//! [`tool_use`]: misanthropic::tool::Use
//! [`tool::Result`]: misanthropic::tool::Result
//! [`User`]: misanthropic::prompt::message::Role::User
//! [`System`]: misanthropic::prompt::message::Role::System

use misanthropic::{
    Client, Prompt,
    prompt::message::{Content, Role},
    tool::{Mailbox, Notifications, Tool, tool},
};
use rustyline::{DefaultEditor, error::ReadlineError};

/// Demo cap on total turns — the chat ends after this many exchanges, some
/// driven by the human, some by the reminder tool.
const MAX_TURNS: usize = 20;

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
    /// Owns its own channel when standalone (as here); a [`ToolBox`] swaps in a
    /// send-only handle via `#[connect]` when this tool is boxed for
    /// aggregation.
    ///
    /// [`ToolBox`]: misanthropic::tool::ToolBox
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
                Content::text("[reminder] Keep answers concise and on-task."),
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

    let client = Client::new(std::env::var("ANTHROPIC_API_KEY")?)?;

    // No ToolBox — a single Tool, driven directly. Take its push stream and run
    // its setup; `subscribe`/`on_init` are Tool-trait methods.
    let mut reminder = Reminder::new(3);
    let mut chat = Prompt::default();
    let mut notifications = reminder.subscribe().expect("the reminder pushes");
    reminder.on_init(&mut chat).await?;

    // A dedicated blocking thread owns stdin for the program's life and forwards
    // finished lines over a cancel-safe channel — see the module docs.
    let (line_tx, mut line_rx) = tokio::sync::mpsc::channel::<String>(16);
    std::thread::spawn(move || {
        let Ok(mut rl) = DefaultEditor::new() else {
            return;
        };
        loop {
            match rl.readline("you ▸ ") {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => {
                    rl.add_history_entry(&line).ok();
                    // Receiver gone (driver exited): stop reading.
                    if line_tx.blocking_send(line).is_err() {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => continue, // Ctrl-C
                Err(ReadlineError::Eof) => break,            // Ctrl-D
                Err(_) => break,
            }
        }
        // Dropping `line_tx` here signals EOF to the driver's `select!`.
    });

    println!(
        "Chat with the model; a reminder lands every 3rd turn. Ctrl-D quits.\n"
    );

    // The conversation is a bounded alternation of user and assistant turns —
    // except the "user" beat sometimes comes from the reminder tool instead of
    // the human. The reminder has a null schema (it adds nothing to
    // `Prompt::tools`), so the model has no tool to call and there is never a
    // `tool_use` to dispatch: each turn is just "seat a beat, answer it".
    for _ in 0..MAX_TURNS {
        // The next user-side beat: the human, or the tool waking us. Both
        // branches await cancel-safe `recv()`; see the module docs.
        tokio::select! {
            line = line_rx.recv() => match line {
                None => break, // reader thread ended (Ctrl-D)
                Some(line) => chat.push_message((Role::User, line))?,
            },
            Some(note) = notifications.recv() => {
                // Seat the push at the role the current model supports. On Haiku
                // this resolves to User; on Opus 4.8+ it would be System.
                let role = chat.resolve_role(&note.preferred_roles);
                println!("⏰ [{}] delivered as {role}", note.source);
                chat.push_message((role, note.content))?;
            }
        }

        // `on_turn`: the reminder counts this turn and, every third, enqueues a
        // nudge the next `select!` pass will seat.
        reminder.on_turn(&mut chat).await?;

        let message = client.message(&chat).await?;
        println!("claude ▸ {}\n", message.inner.content);
        chat.push_message(message)?;
    }

    reminder.on_teardown(&mut chat).await?;
    println!("bye");
    Ok(())
}
