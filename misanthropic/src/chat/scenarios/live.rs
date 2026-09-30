//! The scenario rows marked `live`, against a local Anthropic-compatible
//! server — drama_llama's `blallama` — through a real [`Client`].
//!
//! Each test **skips** (passes, with a note on stderr) unless `BLALLAMA_URL`
//! is set, and then `BLALLAMA_MODEL` must name the model to serve, as the
//! server lists it. That is what keeps these out of CI, which builds with
//! `--all-features`. Run one model at a time:
//!
//! ```sh
//! just test-blallama cogito-32b.gguf
//! ```
//!
//! Besides the row's own expectations and the [`checks`](super::checks)
//! invariants, every response must be shaped as Anthropic's would be. A
//! deviation is a server bug unless it is a deliberate improvement.

use super::{Checked, Row, Run, StopReason, drive, expect, rows};
use crate::{Client, Prompt};

/// The server and model under test, or `None` to skip.
fn target() -> Option<(String, String)> {
    let url = std::env::var("BLALLAMA_URL")
        .ok()
        .filter(|u| !u.is_empty())?;
    let model = std::env::var("BLALLAMA_MODEL")
        .expect("BLALLAMA_MODEL names the model when BLALLAMA_URL is set");
    Some((url, model))
}

/// Run the live row `name`, unless no server is configured.
async fn run(name: &str) {
    let Some((url, model)) = target() else {
        eprintln!("skipping live row `{name}`: BLALLAMA_URL is unset");
        return;
    };
    let row: Row = rows()
        .into_iter()
        .find(|row| row.name == name)
        .expect("a row by that name");
    assert!(row.live, "`{name}` isn't marked live");

    let client = Client::new("x".repeat(108))
        .unwrap()
        .base_url(url.as_str())
        .expect("BLALLAMA_URL is a URL");
    // Room for a local model's thinking; a row may still tighten it.
    let base = Prompt::default()
        .model(model)
        .max_tokens(std::num::NonZeroU32::new(16_384).unwrap());
    let run = drive(&row, Checked::new(client), base).await;

    assert_anthropic_shaped(&run);
    expect(&row, &run, true);
}

/// Every response is shaped as the Messages API shapes one: an id, a stop
/// reason agreeing with the content, `stop_sequence` set exactly when a
/// stop sequence fired, content unless clipped, and nonzero usage.
fn assert_anthropic_shaped(run: &Run) {
    for (n, reply) in run.log.received.iter().enumerate() {
        let stop = reply
            .stop_reason
            .unwrap_or_else(|| panic!("response {n} has no stop_reason"));
        assert!(!reply.id.is_empty(), "response {n} has no id");
        let calls = reply.inner.content.tool_uses().count();
        match stop {
            StopReason::ToolUse => {
                assert!(calls > 0, "response {n}: tool_use without a call")
            }
            StopReason::EndTurn | StopReason::StopSequence => assert_eq!(
                calls, 0,
                "response {n}: a finished ({stop:?}) turn calls tools"
            ),
            _ => {}
        }
        assert_eq!(
            reply.stop_sequence.is_some(),
            stop == StopReason::StopSequence,
            "response {n}: stop_sequence {:?} with {stop:?}",
            reply.stop_sequence
        );
        assert!(
            stop == StopReason::MaxTokens || !reply.inner.content.is_empty(),
            "response {n}: an empty {stop:?} turn"
        );
        let counts = reply.usage.counts;
        let read = counts.cache_read_input_tokens.unwrap_or_default();
        let written = counts.cache_creation_input_tokens.unwrap_or_default();
        assert!(
            counts.input_tokens + read + written > 0,
            "response {n}: no input tokens counted"
        );
        assert!(counts.output_tokens > 0, "response {n}: no output tokens");
    }
}

/// One test per live row, so a model's failures are reported row by row
/// (and one row can be rerun alone).
macro_rules! live {
    ($($row:ident),* $(,)?) => {
        $(
            #[tokio::test]
            async fn $row() {
                run(stringify!($row)).await;
            }
        )*

        /// Every live row has a test here, and every test a live row.
        #[test]
        fn every_live_row_runs() {
            let mut listed = vec![$(stringify!($row)),*];
            listed.sort_unstable();
            let mut live: Vec<_> = rows()
                .into_iter()
                .filter(|row| row.live)
                .map(|row| row.name)
                .collect();
            live.sort_unstable();
            assert_eq!(listed, live);
        }
    };
}

live!(
    plain_end_turn,
    two_beats,
    stop_sequence,
    clip_first_round,
    forced_tool_final_word,
    note_rides_the_next_beat,
    user_note_drives_a_round,
);
