//! API-key acquisition for the examples: environment (with a security
//! warning) → stdin prompt, then a best-effort scrub of the system clipboard.

use std::io::BufRead;

use super::BoxError;

/// Acquire the Anthropic API key.
///
/// Tries `ANTHROPIC_API_KEY` first — and if set, warns that the environment is
/// less private than stdin (it's visible to child processes and `ps`). Failing
/// that, it prompts and reads one line from stdin. After a stdin read it
/// best-effort scrubs the clipboard: an interactive user likely pasted the key,
/// so we clear it (and say so) if it's still there.
///
/// Reads stdin directly, so call this *before* [`spawn_readline_loop`] hands
/// the terminal to the line editor.
///
/// [`spawn_readline_loop`]: super::spawn_readline_loop
pub fn api_key() -> Result<String, BoxError> {
    if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
        eprintln!(
            "warning: using ANTHROPIC_API_KEY from the environment — this is \
             less private than stdin (visible to child processes and `ps`); \
             prefer pasting at the prompt."
        );
        return Ok(key);
    }

    eprintln!("Enter your API key:");
    let key = std::io::stdin()
        .lock()
        .lines()
        .next()
        .ok_or("no API key provided on stdin")??;

    scrub_clipboard();
    Ok(key)
}

/// Best-effort: clear the system clipboard if it holds something that looks
/// like an API key (`sk-ant…`), logging that we did so it isn't surprising.
/// Silently does nothing when there's no clipboard (e.g. headless CI).
fn scrub_clipboard() {
    let Ok(mut clipboard) = arboard::Clipboard::new() else {
        return;
    };
    let looks_like_key = clipboard
        .get_text()
        .map(|text| text.trim_start().starts_with("sk-ant"))
        .unwrap_or(false);
    if looks_like_key && clipboard.clear().is_ok() {
        eprintln!("note: cleared the API key from your clipboard.");
    }
}
