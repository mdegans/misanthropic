//! Example: tool *callbacks* via a [`Mailbox`] sink, shown as a small CLI chat
//! where a tool drops a conversational reminder into the conversation every few
//! turns. This is the draft of the agent reactor we'll eventually encapsulate.
//!
//! # The idea
//!
//! Tool use is a *pair*: a [`tool_use`] is answered by exactly one
//! [`tool::Result`] in the next message. That covers request/response, but not a
//! tool that wants to drop free-standing [`Content`] into the chat without being
//! called — a backgrounded job reporting in, or (here) a periodic nudge. Those
//! are **pushes**, not replies.
//!
//! So a [`Tool`] is handed a [`Mailbox`] — an outbox the [`ToolBox`] gives it on
//! `add` — and pushes a [`Notification`] whenever it likes. The box owns the
//! single `mpsc` receiver; the driver `select!`s [`Notifications::recv`] against
//! user input and, when a push arrives, seats it and takes a turn. A tool can
//! thus drive a turn with **no user input at all** — the whole point.
//!
//! ## Why a sink (and not "return your stream")
//!
//! Because the box owns the channel, a tool **added mid-session** just gets a
//! fresh [`Mailbox`] clone and pushes into the receiver the driver has held
//! since turn zero — no re-subscribe, no busted cache. A
//! `Tool::subscribe() -> Stream` design can't: once the aggregate is handed out,
//! there's nothing left to push new branches into.
//!
//! ## Blocking input under `select!` — the cancel-safety trick
//!
//! `rustyline` is blocking and has no async mode. Putting
//! `spawn_blocking(|| rl.readline())` *inside* a `select!` branch would be a bug:
//! when the other branch wins, `select!` drops that future, but dropping a
//! `spawn_blocking` handle does **not** cancel the blocking thread — the typed
//! line is lost and the next iteration spawns a *second* `readline` racing the
//! first on the tty. Instead we keep the blocking call **out of the future tree**:
//! one long-lived [`std::thread`] owns stdin and forwards finished lines over a
//! [`tokio::mpsc`] channel. `select!` then only ever polls `mpsc` `recv()`,
//! which is **cancel-safe** — a buffered line survives a lost branch.
//!
//! ## Role is a *preference*, resolved by the driver
//!
//! [`Notification::preferred_roles`] is a `Vec<Role>`, not a baked role: a
//! reminder wants [`System`] where the model supports in-message system turns and
//! [`User`] where it doesn't. The driver picks the first the current model
//! supports — [`Prompt::resolve_role`]. Under the default model (Haiku) this
//! lands as a **user** turn; on Opus 4.8+ it would be a **system** turn. Seated
//! after the assistant's answer, the beat is `Assistant → User` (or
//! `Assistant → System`), both legal today — so this example does not depend on
//! the `may_precede` loosening that the feature ships for out-of-band arrivals.
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
//! model acknowledges it. `Ctrl-D` quits.
//!
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notification`]: misanthropic::tool::Notification
//! [`Notification::preferred_roles`]: misanthropic::tool::Notification::preferred_roles
//! [`Notifications::recv`]: misanthropic::tool::Notifications::recv
//! [`Prompt::resolve_role`]: misanthropic::Prompt::resolve_role
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolBox`]: misanthropic::tool::ToolBox
//! [`Content`]: misanthropic::prompt::message::Content
//! [`tool_use`]: misanthropic::tool::Use
//! [`tool::Result`]: misanthropic::tool::Result
//! [`User`]: misanthropic::prompt::message::Role::User
//! [`System`]: misanthropic::prompt::message::Role::System

use misanthropic::{
    Client, Prompt,
    prompt::message::{Content, Role},
    tool::{Mailbox, ToolBox, tool},
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
    /// The outbox, handed over by the `#[connect]` marker. `None` until then.
    mailbox: Option<Mailbox>,
}

impl Reminder {
    fn new(every: u32) -> Self {
        Self {
            every,
            turns: 0,
            mailbox: None,
        }
    }
}

// The `#[tool]` macro builds a concrete `impl Tool` from the markers below.
// `#[connect]` is the new sibling of the existing `#[on_init]`/`#[on_turn]`/
// `#[on_teardown]` markers; a tool with no `#[method]` is push-only.
#[tool]
impl Reminder {
    /// NEW marker: the [`ToolBox`] hands every tool its [`Mailbox`] on `add`. A
    /// pusher stores it; a tool that never pushes simply omits `#[connect]`.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.mailbox = Some(mailbox);
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
        if self.turns.is_multiple_of(self.every)
            && let Some(mailbox) = &self.mailbox
        {
            let _ = mailbox.send(
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

    // One push-only tool, reminding every 3rd turn. `prepare` installs its
    // (empty) defs and runs `on_init` — but only after `add` has `connect`ed a
    // Mailbox.
    let mut tools = ToolBox::new().add(Reminder::new(3));
    let mut chat = Prompt::default();
    tools.prepare(&mut chat).await?;

    // The single consumer end. Every tool's pushes (now and any added later)
    // arrive here.
    let mut notifications = tools.subscribe();

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
        // The next user-side beat: the human, or a tool waking us. Both branches
        // await cancel-safe `recv()`; see the module docs.
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
        tools.update_turn_context(&mut chat).await?;

        let message = client.message(&chat).await?;
        println!("claude ▸ {}\n", message.inner.content);
        chat.push_message(message)?;
    }

    tools.teardown_tools(&mut chat).await?;
    println!("bye");
    Ok(())
}
