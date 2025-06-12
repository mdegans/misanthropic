// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
use std::time::Duration;

use futures::{FutureExt, SinkExt, channel::mpsc, stream::StreamExt};
use futures_timer::Delay;

use super::MemorySubroutineError;
use crate::{Client, Prompt, batch::Batch, response};

use super::{ProcessingMessage, SubmissionMessage, SubroutineConfig};

/// The batch submission task that accumulates memories and submits to batch API
pub async fn batch_submission_task(
    client: Client,
    config: SubroutineConfig,
    mut rx_submission: mpsc::Receiver<SubmissionMessage>,
    mut tx_processing: mpsc::Sender<ProcessingMessage>,
) -> Result<Vec<Prompt<'static>>, MemorySubroutineError> {
    let mut pending_prompts: Vec<Prompt<'static>> = Vec::new();

    // Create timer for batch timeout
    let timeout_duration =
        Duration::from_secs(config.batch_timeout as u64 * 60);
    let mut batch_timer = Delay::new(timeout_duration).fuse();

    loop {
        futures::select! {
            msg = rx_submission.next() => {
                match msg {
                    Some(SubmissionMessage::Store { prompt }) => {
                        // The first time we are seeing this `prompt`

                        #[cfg(feature = "log")]
                        log::debug!("Received prompt for submission");

                        pending_prompts.push(prompt);

                        if pending_prompts.len() >= config.batch_size as usize {
                            #[cfg(feature = "log")]
                            log::debug!("Batch size reached, submitting {} prompts", pending_prompts.len());

                            // Submit all pending prompts
                            let batch = client.batch(pending_prompts.drain(..)).await?;

                            // Send to processing channel
                            tx_processing
                                .send(ProcessingMessage::Batch {
                                    batch,
                                    // We count down from here
                                    retry_count: config.max_retries as u32,
                                })
                                // Unwrap is safe here as we control the channel
                                .await.unwrap();
                        }
                    }
                    Some(SubmissionMessage::Retry { prompts, retry_count }) => {
                        // Something went wrong and the processing task sent us
                        // prompts to retry. We will re-submit them.
                        if retry_count == 0 {
                            // We are out of retries
                            #[cfg(feature = "log")]
                            log::warn!("Retry count is zero, not submitting prompts");
                            // TODO: failsafe dump to logs or DB
                            continue;
                        }

                        // FIXME: right now we submit retries immediately,
                        // but we might want to accumulate them too. The issue
                        // is retry count is per-batch, not per-prompt, so
                        // we can't just push them to `pending_prompts`.

                        // Re-submit the prompts
                        tx_processing
                            .send(ProcessingMessage::Batch {
                                batch: client.batch(prompts).await?,
                                // Safe because we checked != 0 above
                                retry_count: retry_count - 1,
                            })
                            .await
                            .unwrap();
                    }
                    None => {
                        // Channel closed, begin shutdown
                        #[cfg(feature = "log")]
                        log::info!("Submission channel closed, shutting down");
                        break;
                    }
                }
            }
            _ = batch_timer => {
                if !pending_prompts.is_empty() {
                    #[cfg(feature = "log")]
                    log::debug!("Batch timeout reached, submitting {} prompts", pending_prompts.len());

                    let batch = client.batch(pending_prompts.drain(..)).await?;
                    tx_processing
                        .send(ProcessingMessage::Batch {
                            batch,
                            retry_count: config.max_retries as u32,
                        })
                        .await.unwrap();
                }

                // Reset timer
                batch_timer = Delay::new(timeout_duration).fuse();
            }
        }
    }

    // Return any unsubmitted prompts
    #[cfg(feature = "log")]
    log::info!("Returning {} unsubmitted prompts", pending_prompts.len());

    Ok(pending_prompts)
}

/// The batch processing task that polls results and sends ready batches
pub async fn batch_processing_task(
    client: Client,
    config: SubroutineConfig,
    mut rx_processing: mpsc::Receiver<ProcessingMessage>,
    mut tx_submission: mpsc::Sender<SubmissionMessage>, // for retries
    mut tx_ready: mpsc::Sender<(Prompt<'static>, response::Message<'static>)>,
) -> Result<Vec<crate::batch::Pending<'static>>, MemorySubroutineError> {
    let mut pending_batches: Vec<ProcessingMessage> = Vec::new();

    // Poll timer
    let poll_duration = Duration::from_secs(10);
    let mut poll_timer = Delay::new(poll_duration).fuse();

    loop {
        futures::select! {
            msg = rx_processing.next() => {
                match msg {
                    Some(message) => {
                        #[cfg(feature = "log")]
                        log::debug!("Received new batch for processing");

                        pending_batches.push(message);
                    }
                    None => {
                        #[cfg(feature = "log")]
                        log::info!("Processing channel closed, finishing pending batches");
                        break;
                    }
                }
            }
            _ = poll_timer => {
                // It is time to check for completed batches
                let mut incomplete_batches = Vec::new();
                for ProcessingMessage::Batch { batch, retry_count } in pending_batches.drain(..) {
                    match client.batch_poll(batch).await? {
                        Batch::Pending(pending) => {
                            // Batch is still pending, keep it for next poll
                            incomplete_batches.push(ProcessingMessage::Batch {
                                batch: pending,
                                retry_count,
                            });
                        }
                        Batch::Ready(mut ready) => {
                            // Batch is ready, send it to the ready channel
                            #[cfg(feature = "log")]
                            log::debug!("Batch `{}` is ready, sending to ready channel", ready.id());

                            // Remove all Ok pairs and send them to `tx_ready`
                            let mut ok = futures::stream::iter(ready.drain_ok().map(|pair| {
                                Ok(pair)
                            }));
                            tx_ready
                                .send_all(&mut ok)
                                .await
                                .unwrap(); // Impossible because the stream always returns Ok

                            // Drain errors and send them to submission channel
                            // FIXME: We send these to the submission channel
                            // only to have them sent back here. We should
                            // remove the Retry variant from the submission
                            // message and the logic above, instead handling
                            // retries directly in this task.
                            let mut errors = futures::stream::iter(ready.drain_errors().map(|pair| {
                                Ok(pair)
                            }));
                        }
                    }
                }

                // Reset timer
                poll_timer = Delay::new(poll_duration).fuse();
            }
        }
    }

    // Return any remaining batches
    #[cfg(feature = "log")]
    log::info!("Returning {} pending batches", pending_batches.len());

    Ok(pending_batches
        .into_iter()
        .map(|msg| {
            let ProcessingMessage::Batch {
                batch,
                // FIXME: do we save this? Otherwise on load we will retry all
                // batches with equal tries regardless of how many failures.
                retry_count: _,
            } = msg;
            batch
        })
        .collect())
}
