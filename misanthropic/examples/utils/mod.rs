//! Shared helpers for the examples.
//!
//! This is **not** an example target — it lives in a subdirectory with no
//! `main.rs`, so Cargo's example auto-discovery ignores it. Pull it into an
//! example with `mod utils;` and call e.g. [`spawn_readline_loop`].
#![allow(dead_code)]

// The chat driver needs a `Client`, so it rides the same feature.
#[cfg(feature = "client")]
mod chat;
// `BoxError` is part of the helper's API but not every example names it.
#[cfg(feature = "client")]
#[allow(unused_imports)]
pub use chat::{BoxError, Chat};

use rustyline::{DefaultEditor, ExternalPrinter, error::ReadlineError};
use tokio::sync::mpsc;

/// Spawn a dedicated thread running a `rustyline` prompt and return the entered
/// lines as a channel plus a [`Printer`] for the async side to print *through*.
///
/// # Why a thread *and* a printer
///
/// A line editor owns the terminal line while it waits, so anything else that
/// writes stdout collides with what the user is typing (the `you ▸ claude ▸ …`
/// glitch). Two moves fix it: keep the blocking `readline` on its own thread
/// (out of the `select!` future tree — its [`mpsc`] `recv()` is cancel-safe),
/// and route **all** async output through the returned [`Printer`] (rustyline's
/// [`ExternalPrinter`]), which erases the prompt, prints above it, and redraws
/// it with the user's in-progress input intact.
pub fn spawn_readline_loop(
    prompt: impl Into<String>,
) -> rustyline::Result<(mpsc::Receiver<String>, Printer)> {
    let prompt = prompt.into();
    let mut editor = DefaultEditor::new()?;
    let printer = Printer(Box::new(editor.create_external_printer()?));
    let (tx, rx) = mpsc::channel::<String>(16);

    std::thread::spawn(move || {
        loop {
            match editor.readline(&prompt) {
                Ok(line) if line.trim().is_empty() => continue,
                Ok(line) => {
                    editor.add_history_entry(&line).ok();
                    // Receiver gone (the driver exited): stop reading.
                    if tx.blocking_send(line).is_err() {
                        break;
                    }
                }
                Err(ReadlineError::Interrupted) => continue, // Ctrl-C: ignore
                Err(ReadlineError::Eof) => break,            // Ctrl-D: quit
                Err(_) => break,
            }
        }
        // Dropping `tx` signals EOF to the consumer's `recv()`.
    });

    Ok((rx, printer))
}

/// Prints *above* the live `rustyline` prompt. Use this for all async output
/// (model replies, notifications) instead of `println!`, which would collide
/// with the user's in-progress input. Returned by [`spawn_readline_loop`].
pub struct Printer(Box<dyn ExternalPrinter + Send>);

impl Printer {
    /// Print `msg` followed by a newline, above the prompt.
    pub fn line(&mut self, msg: impl std::fmt::Display) {
        // The editor thread performs the redraw; if it's gone, drop the line.
        let _ = self.0.print(format!("{msg}\n"));
    }
}
