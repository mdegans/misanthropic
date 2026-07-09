//! Example: the **Batch API** — submit many prompts in one request at half
//! the per-token price, poll until processing ends, and match results back
//! to what you asked. Here: one haiku per topic.
//!
//! [`Client::tagged_batch`] pairs each prompt with a caller-supplied
//! [`batch::Id`], so the results can be labeled in the order the topics were
//! given (a plain [`Client::batch`] generates ids for you). Anthropic has no
//! webhook for batches, so [`Client::batch_poll`] is called in a loop until
//! the [`Batch`] flips from [`Pending`] to [`Ready`]. Batches may take up to
//! 24 hours, but a handful of haiku usually lands in under a minute.
//!
//! ```sh
//! cargo run --features batch --example batch_haiku
//!
//! # Your own topics, faster polling:
//! cargo run --features batch --example batch_haiku -- \
//!     --poll-secs 2 "monomorphization" "the orphan rule"
//! ```
//!
//! [`Client::batch`]: misanthropic::Client::batch
//! [`Client::tagged_batch`]: misanthropic::Client::tagged_batch
//! [`Client::batch_poll`]: misanthropic::Client::batch_poll
//! [`batch::Id`]: misanthropic::batch::Id
//! [`Batch`]: misanthropic::batch::Batch
//! [`Pending`]: misanthropic::batch::Pending
//! [`Ready`]: misanthropic::batch::Ready

mod utils;

use std::time::Duration;

use clap::Parser;
use misanthropic::{
    Client, Id, Prompt,
    batch::{self, Batch, BatchResult},
    prompt::message::Role,
};

#[derive(Parser, Debug)]
#[command(
    version,
    about = "Batch-generate one haiku per topic at half the per-token price \
             using the Batch API."
)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Seconds to wait between status polls.
    #[arg(long, default_value_t = 5)]
    poll_secs: u64,

    /// Topics to write haiku about, one prompt (and one haiku) each.
    #[arg(default_values_t = [
        "the borrow checker".to_string(),
        "async cancellation".to_string(),
        "feature unification".to_string(),
        "a segfault, remembered fondly".to_string(),
    ])]
    topics: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    let client = Client::new(utils::api_key()?)?;

    let system = "You are a poet. Reply with a single haiku about the \
        requested topic - three lines, nothing else.";

    // One id per topic, minted up front. `tagged_batch` sends them as each
    // request's `custom_id`, and the results come back keyed by the same ids,
    // so `ids[i]` labels `topics[i]`'s haiku no matter what order the API
    // finishes them in.
    let ids: Vec<batch::Id> =
        args.topics.iter().map(|_| batch::Id::default()).collect();

    let prompts = ids
        .iter()
        .zip(&args.topics)
        .map(|(&id, topic)| {
            Ok((
                id,
                args.common
                    .configure(
                        Prompt::default().model(Id::Haiku45).system(system),
                    )
                    .add_message((
                        Role::User,
                        format!("Write a haiku about {topic}."),
                    ))?,
            ))
        })
        .collect::<Result<Vec<_>, misanthropic::prompt::TurnOrderError>>()?;

    let mut pending = client.tagged_batch(prompts).await?;
    println!(
        "Submitted batch `{}` with {} prompts.",
        pending.meta().id,
        pending.prompts().len()
    );

    // Poll until processing ends. `batch_poll` refreshes the metadata and,
    // once the batch is done, downloads the results and returns `Ready`.
    let ready = loop {
        let stats = pending.meta().stats;
        println!(
            "processing: {} | succeeded: {} | errored: {}",
            stats.processing, stats.succeeded, stats.errored
        );

        tokio::time::sleep(Duration::from_secs(args.poll_secs)).await;

        match client.batch_poll(pending).await? {
            Batch::Pending(next) => pending = next,
            Batch::Ready(ready) => break ready,
        }
    };

    for (id, topic) in ids.iter().zip(&args.topics) {
        println!("\n## {topic}\n");
        match ready.get_result(*id) {
            Some(BatchResult::Ok(message)) => println!("{message}"),
            Some(BatchResult::Error(e)) => println!("(errored: {e})"),
            // Not every prompt in a batch is guaranteed a completion — a
            // batch can be canceled or expire with prompts unprocessed.
            // `Ready` has `remove_canceled` / `remove_expired` to pull those
            // out for resubmission; here they are only reported.
            Some(BatchResult::Canceled) => println!("(canceled)"),
            Some(BatchResult::Expired) => println!("(expired)"),
            None => println!("(no result returned)"),
        }
    }

    Ok(())
}
