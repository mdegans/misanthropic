//! Example: a **multi-agent swarm** — one `Chat` loop per agent, wired
//! together by a mail tool built with the [`tool`] macro. You chat with
//! `boss`, who plans and delegates; three workers (`ant`, `bee`, `moth`)
//! each carry a `Mail` clone plus their own sandboxed [`RichBash`], and
//! run *headless*: their `next_beat` pends on shutdown, so incoming letters
//! are their only stimulus.
//!
//! A letter is pushed into the recipient's [`ToolBox`] channel as a
//! [`Notification`] preferring the [`User`] role — the same push path
//! [`RichBash`] uses for background-job completions — so mail wakes the
//! recipient's driver and drives a model round, and a worker's report races
//! *your* typing at the boss's prompt exactly like any other notification.
//! The sender's identity is stamped by the tool, not the model, so a
//! `From:` line cannot be forged. Postage is the brake: each agent has
//! finite stamps, and an empty book turns `send` into an `is_error` result
//! telling the agent to wrap up with what it has.
//!
//! The point: the `Chat` driver is unchanged. N concurrent loops,
//! agent-to-agent communication, and budget brakes all compose from
//! [`Tool`], [`ToolBox`], and [`Mailbox`] as they already are.
//!
//! ```sh
//! # needs Docker; the published misan-bashd image is pulled on first run
//! # (`just build-bashd` builds the same tag locally to shadow it)
//! cargo run --features "client bash-container derive" --example swarm
//! ```
//!
//! Try: "have each worker count the lines in a different section of
//! /etc/services, then total them." `--verbose` shows the workers thinking;
//! `--model` steers the boss (workers stay on the default model).
//!
//! [`Mailbox`]: misanthropic::tool::Mailbox
//! [`Notification`]: misanthropic::tool::Notification
//! [`RichBash`]: misanthropic::tool::bash::RichBash
//! [`Tool`]: misanthropic::tool::Tool
//! [`ToolBox`]: misanthropic::tool::ToolBox
//! [`User`]: misanthropic::prompt::message::Role::User
//! [`tool`]: misanthropic::tool::tool

mod utils;

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use clap::Parser;
use misanthropic::{
    Client, Prompt,
    model::Id,
    prompt::message::{Content, Role},
    tool::{
        Mailbox, ToolBox,
        bash::{DockerSandbox, RichBash},
        tool,
    },
};
use schemars::JsonSchema;
use serde::Deserialize;
use utils::{BoxError, BudgetPolicy, Printer};

/// The boss's address — the agent whose `Chat` loop is your readline seat.
const BOSS: &str = "boss";
/// The headless workers. Each gets a [`Mail`] clone and its own sandbox.
const WORKERS: [&str; 3] = ["ant", "bee", "moth"];

/// A boss/worker agent swarm wired together by a mail tool.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,
    #[command(flatten)]
    chat: utils::ChatArgs,
    /// Stamps per worker; the boss gets one book per worker (3x).
    #[arg(long, default_value_t = 8)]
    stamps: u32,
}

/// All async output rides the single rustyline [`Printer`] (it is not
/// `Clone`, and printing takes `&mut`), shared behind a mutex. No lock is
/// ever held across an `await`.
type SharedPrinter = Arc<Mutex<Printer>>;

/// The swarm's address book: agent name → a send-only clone of that agent's
/// `ToolBox` mailbox. Registered by [`Mail::connect`], which a `ToolBox`
/// calls *inside* `add` — so building every toolbox before spawning any
/// `Chat` guarantees a complete roster before the first letter.
type Registry = Arc<Mutex<HashMap<String, Mailbox>>>;

/// Hands out [`Mail`] clones sharing one [`Registry`] and one printer.
struct PostOffice {
    registry: Registry,
    printer: SharedPrinter,
}

impl PostOffice {
    fn new(printer: SharedPrinter) -> Self {
        Self {
            registry: Registry::default(),
            printer,
        }
    }

    /// A [`Mail`] clone for `name`, with a book of `stamps`.
    fn address(&self, name: impl Into<String>, stamps: u32) -> Mail {
        Mail {
            name: name.into(),
            stamps,
            registry: Arc::clone(&self.registry),
            printer: Arc::clone(&self.printer),
        }
    }
}

/// One agent's mail client. `send` looks the recipient up in the shared
/// [`Registry`] and pushes the letter into *their* `Chat` loop as a
/// `[User]`-preferred notification; the envelope's `From:` is stamped from
/// this clone's own identity, so the model cannot forge a sender.
struct Mail {
    name: String,
    stamps: u32,
    registry: Registry,
    printer: SharedPrinter,
}

impl Mail {
    /// Everyone reachable from this desk, sorted for stable prompts.
    fn roster(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .registry
            .lock()
            .expect("registry poisoned")
            .keys()
            .filter(|name| **name != self.name)
            .cloned()
            .collect();
        names.sort();
        names
    }
}

/// A letter for [`Mail::send`]. The field docs become the JSON-schema
/// property descriptions the model sees (via `schemars`).
#[derive(Debug, Deserialize, JsonSchema)]
struct Letter {
    /// The recipient agent's name.
    to: String,
    /// A one-line subject.
    subject: String,
    /// The letter itself. Include everything the recipient needs — they
    /// see only what you write here, never your conversation.
    body: String,
}

#[tool(name = "mail")]
impl Mail {
    /// Register this agent's address. The handed mailbox is a send-only
    /// handle onto *our own* box's notification channel — exactly what
    /// other agents need in order to reach us — and it fires during
    /// `ToolBox::add`, before any model call, so no letter can beat it.
    #[connect]
    fn connect(&mut self, mailbox: Mailbox) {
        self.registry
            .lock()
            .expect("registry poisoned")
            .insert(self.name.clone(), mailbox);
    }

    /// Brief the agent: identity, roster, and postage. Appends (never
    /// overwrites) so it composes with the example's persona and any other
    /// tool's contribution, in any `on_init` order.
    #[on_init]
    async fn brief(&mut self, prompt: &mut Prompt) -> Result<(), BoxError> {
        let briefing = format!(
            "<mail>\nYou are `{}`. You can write to: {}. Mail is your only \
             channel to the other agents; results travel in the letter \
             body. You have {} stamps and each letter costs one, so make \
             letters count.\n</mail>",
            self.name,
            self.roster().join(", "),
            self.stamps,
        );
        match prompt.system.as_mut() {
            Some(system) => {
                system.push(briefing);
            }
            None => prompt.system = Some(briefing.into()),
        }
        Ok(())
    }

    /// Send a letter to another agent. Costs one stamp.
    #[method]
    async fn send(&mut self, letter: Letter) -> Result<Content, Content> {
        if letter.to == self.name {
            return Err("you cannot mail yourself".into());
        }
        if self.stamps == 0 {
            return Err("out of stamps — no more letters. Wrap up and \
                        report what you have through channels you already \
                        opened (or your own reply, if you talk to the \
                        human directly)."
                .into());
        }
        let recipient = self
            .registry
            .lock()
            .expect("registry poisoned")
            .get(&letter.to)
            .cloned();
        let Some(recipient) = recipient else {
            return Err(format!(
                "no agent named `{}` — the roster: {}",
                letter.to,
                self.roster().join(", "),
            )
            .into());
        };

        // The tool composes the envelope, so `From:` is authoritative.
        let envelope = format!(
            "From: {}\nSubject: {}\n\n{}",
            self.name, letter.subject, letter.body,
        );
        if recipient.send(envelope, vec![Role::User]).is_err() {
            return Err(
                format!("`{}` is gone (their loop ended)", letter.to).into()
            );
        }

        self.stamps -= 1;
        self.printer.lock().expect("printer poisoned").line(format!(
            "{} ✉ {} ▸ {}",
            self.name, letter.to, letter.subject
        ));
        Ok(format!("sent ({} stamps left)", self.stamps).into())
    }
}

/// A worker's persona. The mail briefing (identity, roster, postage) is
/// appended by [`Mail::brief`].
fn worker_prompt(name: &str) -> Prompt {
    Prompt::default().system(format!(
        "You are `{name}`, a worker agent in a tiny company run by `boss`. \
         Work arrives as letters. Do the work with your sandboxed bash \
         tool, then mail the results back to whoever wrote to you — \
         usually the boss. You never speak to a human; anything not sent \
         by mail is lost. For long-running commands use `background: \
         true` — you'll be notified when the job finishes. Be concise: \
         letters cost stamps."
    ))
}

/// The boss's persona; the human side of the swarm.
const BOSS_SYSTEM: &str = "You are `boss`, the coordinator of a tiny agent company. The human \
     you are chatting with is your client. Your workers each have a \
     sandboxed bash environment; you do not — you plan, split the work, \
     and mail each worker a clear, self-contained assignment (a worker \
     sees only what you write in the letter). Their reports arrive \
     between the human's messages. Synthesize and answer the human in \
     chat — that costs no stamps. Your roster and postage are in your \
     mail briefing.";

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);

    // Get the API key from stdin *before* the rustyline thread takes over
    // stdin.
    let client = Client::new(utils::api_key()?)?;

    let (mut lines, printer) = utils::spawn_readline_loop("you ▸ ")?;
    let printer: SharedPrinter = Arc::new(Mutex::new(printer));
    let office = PostOffice::new(Arc::clone(&printer));

    // Build every toolbox before any `Chat` runs: `add` fires `connect`,
    // which registers the agent's address — so the roster is complete
    // before the first `on_init` briefing or letter.
    let worker_boxes: Vec<(&str, ToolBox)> = WORKERS
        .iter()
        .map(|&name| {
            let toolbox = ToolBox::new()
                .add(office.address(name, cli.stamps))
                .add(RichBash::new(DockerSandbox::default()));
            (name, toolbox)
        })
        .collect();
    let boss_box = ToolBox::new()
        .add(office.address(BOSS, cli.stamps * WORKERS.len() as u32));

    printer.lock().expect("printer poisoned").line(format!(
        "The office is open: you ↔ boss; workers {} are booting their \
         sandboxes. Ctrl-D folds the company.\n",
        WORKERS.join(", "),
    ));

    // The workers: headless Chats. Mail is their only stimulus, so the
    // beat closure just pends until shutdown. `FinalWord` instead of the
    // default hand-back: a worker that silently hands back would sit idle
    // until the next letter, so let it wrap up (and mail the boss) instead.
    let (quit, _) = tokio::sync::watch::channel(false);
    let mut swarm = tokio::task::JoinSet::new();
    for (name, toolbox) in worker_boxes {
        let client = client.clone();
        let printer = Arc::clone(&printer);
        let mut done = quit.subscribe();
        swarm.spawn(async move {
            let outcome =
                utils::Chat::new(client, worker_prompt(name), toolbox)
                    .max_consecutive_tool_calls(16)
                    .on_budget_exhausted(BudgetPolicy::FinalWord)
                    .on_assistant(move |_state: &mut (), msg| {
                        // The workers' side of the story, under `--verbose`.
                        log::debug!("{name} ▸ {}", msg.content);
                        [msg.into()] // seat the turn unchanged
                    })
                    .run((), async move |_state: &mut ()| {
                        // Mail drives everything; the only beat is shutdown
                        // (a closed channel counts).
                        done.changed().await.ok();
                        Ok(None)
                    })
                    .await;
            if let Err(error) = outcome {
                printer
                    .lock()
                    .expect("printer poisoned")
                    .line(format!("☠ {name}: {error}"));
            }
        });
    }

    // The boss: your seat — the same loop as every chat example. Worker
    // reports race your typing in the driver's `select!`.
    let boss_prompt = cli
        .common
        .configure(Prompt::default().model(Id::Sonnet46).system(BOSS_SYSTEM));
    let boss_printer = Arc::clone(&printer);
    let outcome = cli
        .chat
        .configure(utils::Chat::new(client, boss_prompt, boss_box))
        .on_assistant(move |_state: &mut (), msg| {
            if msg.tool_use().is_none() {
                boss_printer
                    .lock()
                    .expect("printer poisoned")
                    .line(format!("\nboss ▸ {}\n", msg.content));
            }
            [msg.into()] // seat the turn unchanged
        })
        .run((), async move |_state: &mut ()| {
            Ok(lines
                .recv()
                .await
                .map(|line| vec![(Role::User, line).into()]))
        })
        .await;

    // Fold the company: wake every worker's beat, then wait for their
    // loops to finish so `ToolBox` teardown stops the sandboxes.
    let _ = quit.send(true);
    while let Some(joined) = swarm.join_next().await {
        joined?;
    }
    outcome?;

    println!("the office is closed");
    Ok(())
}
