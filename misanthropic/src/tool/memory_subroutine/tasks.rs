// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
use std::{collections::HashMap, time::Duration};

use crate::{
    Client, Prompt,
    batch::{Batch, BatchResult, Id, Pending},
    tool::{MemoryPalace, Use},
};
use futures::{FutureExt, SinkExt, channel::mpsc, stream::StreamExt};
use futures_timer::Delay;

use super::{
    MemorySubroutineError, ProcessingMessage, SubmissionMessage,
    SubroutineConfig,
};

/// The batch submission task that accumulates memories and submits to batch API
pub async fn batch_submission_task(
    client: Client,
    config: SubroutineConfig,
    mut rx_submission: mpsc::Receiver<SubmissionMessage>,
    mut tx_processing: mpsc::Sender<ProcessingMessage>,
    mut tx_dead_letter: mpsc::Sender<(Id, Prompt<'static>)>,
) -> Result<Vec<(Id, Prompt<'static>)>, MemorySubroutineError> {
    let mut identified_prompts: Vec<(Id, Prompt<'static>)> = Vec::new();
    let mut retry_counter: HashMap<Id, u8> = HashMap::new();

    // Create timer for batch timeout
    let timeout_duration =
        Duration::from_secs(config.batch_timeout as u64 * 60);
    let mut batch_timer = Delay::new(timeout_duration).fuse();

    loop {
        futures::select! {
            msg = rx_submission.next() => {
                match msg {
                    Some(SubmissionMessage::Store { id, prompt }) => {
                        // The first time we are seeing this `prompt`

                        #[cfg(feature = "log")]
                        log::debug!("Received prompt for submission");

                        identified_prompts.push((id, prompt));

                        if identified_prompts.len() >= config.batch_size as usize {
                            #[cfg(feature = "log")]
                            log::debug!("Batch size reached, submitting {} prompts", identified_prompts.len());

                            // Submit all pending prompts
                            let batch = client.tagged_batch(identified_prompts.drain(..)).await?;

                            // Send to processing channel
                            tx_processing
                                .send(ProcessingMessage::Batch {
                                    batch,
                                })
                                // Unwrap is safe here as we control the channel
                                .await.unwrap();
                        }
                    }
                    Some(SubmissionMessage::Retry { identified_prompts: retries }) => {
                        // We want to first subtract the retry count
                        for (id, prompt) in retries {
                            #[cfg(feature = "log")]
                            log::debug!("Retrying prompt with ID: {}", id);

                            // Check if we have a retry count for this ID
                            // Sub one because we already tried once
                            let retry_count = retry_counter.entry(id).or_insert(config.max_retries.saturating_sub(1));
                            if *retry_count > 0 {
                                // Decrement the retry count
                                *retry_count -= 1;

                                // Resubmit the prompt
                                identified_prompts.push((id, prompt));
                            } else {
                                #[cfg(feature = "log")]
                                log::error!("Prompt {} exhausted all {} retries, sending to dead letter queue", id, config.max_retries);
                                tx_dead_letter.send((id, prompt)).await.unwrap();
                            }
                        }
                    }
                    Some(SubmissionMessage::Complete { id }) => {
                        // Remove the prompt from the map (otherwise we will
                        // have a small memory leak)
                        retry_counter.remove(&id);
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
                if !identified_prompts.is_empty() {
                    #[cfg(feature = "log")]
                    log::debug!("Batch timeout reached, submitting {} prompts", identified_prompts.len());

                    let batch = client.tagged_batch(identified_prompts.drain(..)).await?;
                    tx_processing
                        .send(ProcessingMessage::Batch {
                            batch,
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
    log::info!("Returning {} unsubmitted prompts", identified_prompts.len());

    Ok(identified_prompts)
}

/// The batch processing task that polls results and sends ready batches
pub async fn batch_processing_task(
    client: Client,
    config: SubroutineConfig,
    mut rx_processing: mpsc::Receiver<ProcessingMessage>,
    mut tx_submission: mpsc::Sender<SubmissionMessage>, // for retries
    mut tx_ready: mpsc::Sender<(
        Id,
        Prompt<'static>,
        crate::response::Message<'static>,
    )>,
) -> Result<Vec<Pending<'static>>, MemorySubroutineError> {
    let mut pending_batches: Vec<Pending<'static>> = Vec::new();
    let mut still_pending = Vec::new();
    let mut successes = Vec::new();
    let mut completions = Vec::new();

    // Poll timer
    let poll_duration = Duration::from_secs(config.poll_frequency as u64 * 60);
    let mut poll_timer = Delay::new(poll_duration).fuse();

    loop {
        futures::select! {
            msg = rx_processing.next() => {
                match msg {
                    Some(ProcessingMessage::Batch { batch }) => {
                        #[cfg(feature = "log")]
                        log::debug!("Received new batch {} for processing", batch.id());

                        pending_batches.push(batch);
                    }
                    None => {
                        #[cfg(feature = "log")]
                        log::info!("Processing channel closed, finishing pending batches");
                        break;
                    }
                }
            }
            _ = poll_timer => {
                let mut for_retry: Vec<(Id, Prompt<'static>)> = Vec::new();

                for batch in pending_batches.drain(..) {
                    match client.batch_poll(batch).await {
                        Ok(Batch::Pending(pending)) => {
                            // Still pending, keep it for next poll
                            still_pending.push(pending);
                        }
                        Ok(Batch::Ready(ready)) => {
                            let id = ready.id().to_string();
                            let mut n_ok = 0;
                            let mut n_cancelled = 0;
                            let mut n_expired = 0;
                            let mut n_error = 0;

                            for (id, prompt, batchresult) in ready.into_iter() {
                                match batchresult {
                                    BatchResult::Ok(response) => {
                                        // Send to ready channel
                                        successes.push((id, prompt, response));
                                        completions.push(Ok(SubmissionMessage::Complete { id }));
                                        n_ok += 1;
                                    },
                                    BatchResult::Error(err) => {
                                        #[cfg(feature = "log")]
                                        log::error!("Batch {} error: {}", id, err);
                                        for_retry.push((id, prompt));
                                        n_error += 1;
                                    }
                                    BatchResult::Expired => {
                                        #[cfg(feature = "log")]
                                        log::debug!("Batch {} expired, resubmitting", id);
                                        for_retry.push((id, prompt));
                                        n_expired += 1;
                                    }
                                    BatchResult::Canceled => {
                                        // Should never happen
                                        #[cfg(feature = "log")]
                                        log::debug!("Batch {} cancelled, resubmitting", id);
                                        for_retry.push((id, prompt));
                                        n_cancelled += 1;
                                    }
                                }
                            }

                            #[cfg(feature = "log")]
                            log::info!("Batch {} processed: {} OK, {} cancelled, {} expired, {} errors",
                                id, n_ok, n_cancelled, n_expired, n_error);
                        }
                        Err(crate::client::Error::Batch {
                            submitted,
                            unsubmitted,
                            cause,
                        }) => {
                            debug_assert!(unsubmitted.is_none(), "Unsubmitted batches should not be present in this context");
                            // In this case something went wrong with checking
                            // the batch, likely a connection error or similar.
                            if let Some(pending) = submitted {
                                log::error!("Batch {} failed: {}", pending.id(), cause);
                                // These haven't necessarily failed.
                                still_pending.push(pending);
                            } else {
                                unreachable!("`batch_poll` should always return a `Batch::Pending` in it's error case.");
                            }
                        }
                        Err(_) => {
                            unreachable!("Because `batch_poll` only retuns `Error::Batch`")
                        }
                    }
                }

                // Send successes to ready channel
                let mut ready = futures::stream::iter(successes.drain(..).map(|t| Ok(t)));
                tx_ready
                    .send_all(&mut ready)
                    .await.unwrap();

                // Send completions to submission channel
                let mut completions_stream = futures::stream::iter(completions.drain(..));
                tx_submission
                    .send_all(&mut completions_stream)
                    .await.unwrap();

                // Send retries to submission channel
                if !for_retry.is_empty() {
                    let retries = SubmissionMessage::Retry {
                        identified_prompts: for_retry,
                    };
                    tx_submission.send(retries).await.unwrap();
                }

                // The old swaperoo.
                std::mem::swap(&mut pending_batches, &mut still_pending);

                // Reset timer
                poll_timer = Delay::new(poll_duration).fuse();
            }
        }
    }

    debug_assert!(
        successes.is_empty(),
        "There should be no successes left at this point",
    );

    // Return any pending batches
    #[cfg(feature = "log")]
    log::info!("Returning {} pending batches", pending_batches.len());
    Ok(pending_batches)
}

/// The archival task that archives completed batches
pub async fn batch_archival_task(
    config: SubroutineConfig,
    pool: sqlx::PgPool,
    schema: String,
    mut rx_archival: mpsc::Receiver<(
        Id,
        Prompt<'static>,
        crate::response::Message<'static>,
    )>,
    mut rx_dead_letter: mpsc::Receiver<(Id, Prompt<'static>)>,
) -> Result<(), MemorySubroutineError> {
    let mut palace =
        MemoryPalace::from_pool_with_schema(pool.clone(), schema.clone())
            .await?;

    // Accumulate tool calls for batch processing
    let mut pending_calls: Vec<Use<'static>> = Vec::new();

    // Timer for batch flushes
    let flush_duration = Duration::from_secs(30); // Flush every 30 seconds
    let mut flush_timer = Delay::new(flush_duration).fuse();

    // Batch size threshold
    const BATCH_SIZE: usize = 50;

    loop {
        futures::select! {
            msg = rx_archival.next() => {
                match msg {
                    // FIXME: We don't actually use the prompt, or need it
                    // anymore, so we should consider removing it.
                    Some((id, prompt, response)) => {
                        #[cfg(feature = "log")]
                        log::debug!("Processing response for prompt {}", id);

                        // Extract tool calls from the response
                        for block in response.inner.inner.content {
                            if let crate::prompt::message::Block::ToolUse { call } = block {
                                // Only collect MemoryPalace calls
                                if call.name.starts_with("MemoryPalace::") {
                                    pending_calls.push(call);
                                }
                            }
                        }

                        // Check if we should flush
                        if pending_calls.len() >= BATCH_SIZE {
                            flush_calls(&mut palace, &mut pending_calls).await?;
                            flush_timer = Delay::new(flush_duration).fuse();
                        }
                    }
                    None => {
                        #[cfg(feature = "log")]
                        log::info!("Archival channel closed, flushing remaining calls");

                        // Flush any remaining calls before shutdown
                        if !pending_calls.is_empty() {
                            flush_calls(&mut palace, &mut pending_calls).await?;
                        }
                        break;
                    }
                }
            }
            msg = rx_dead_letter.next() => {
                match msg {
                    Some((id, prompt)) => {
                        // TODO: Create a table for dead letters.
                        #[cfg(feature = "log")]
                        log::warn!("Dead letter received for prompt {}: {:?}", id, prompt);

                        // You could optionally create a memory about the failure
                        let failure_memory = Use {
                            id: format!("dead_letter_{}", id).into(),
                            name: "MemoryPalace::store".into(),
                            input: serde_json::json!({
                                "room": "system_failures",
                                // TODO: Add more context, like the end of the prompt?
                                "content": format!("Failed to process prompt after {} retries", config.max_retries),
                                "tags": ["dead_letter", "batch_failure", "memory_subroutine"]
                            }),
                            cache_control: None,
                        };
                        pending_calls.push(failure_memory);
                    }
                    None => {
                        #[cfg(feature = "log")]
                        log::info!("Dead letter channel closed");
                        break;
                    }
                }
            }
            _ = flush_timer => {
                if !pending_calls.is_empty() {
                    #[cfg(feature = "log")]
                    log::debug!("Flush timer expired, processing {} pending calls", pending_calls.len());

                    flush_calls(&mut palace, &mut pending_calls).await?;
                }

                flush_timer = Delay::new(flush_duration).fuse();
            }
        }
    }

    Ok(())
}

/// Helper function to flush accumulated calls to the palace
async fn flush_calls(
    palace: &mut MemoryPalace,
    pending_calls: &mut Vec<Use<'static>>,
) -> Result<(), MemorySubroutineError> {
    if pending_calls.is_empty() {
        return Ok(());
    }

    #[cfg(feature = "log")]
    log::info!(
        "Flushing {} tool calls to MemoryPalace",
        pending_calls.len()
    );

    // Process all calls in a single transaction for efficiency
    let calls = std::mem::take(pending_calls);

    // You'll need to implement batch_call on MemoryPalace
    match palace.batch_call(calls).await {
        Ok(results) =>
        {
            #[cfg(feature = "log")]
            for result in &results {
                if result.is_error {
                    log::error!("MemoryPalace error: {}", result.content);
                }
            }
        }
        Err(e) => {
            #[cfg(feature = "log")]
            log::error!("Failed to execute batch call: {}", e);
            return Err(MemorySubroutineError::MemoryPalace(e));
        }
    }

    Ok(())
}
