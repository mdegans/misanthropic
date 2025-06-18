use std::future::Future;
use std::pin::Pin;

use chrono::Utc;
use futures::{SinkExt, channel::mpsc};
use sqlx::PgPool;
use uuid::Uuid;

#[cfg(feature = "tokio")]
use crate::{Client, tool::memory_subroutine::db::SaveState};
use crate::{
    Prompt, batch,
    tool::memory_subroutine::{
        MemorySubroutineError, ProcessingMessage, SubmissionMessage,
        SubroutineConfig,
    },
};

/// Runtime-independent task handle
///
/// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
pub struct BackgroundTasks {
    /// Channel to submit [`Prompt`]s for processing
    pub(crate) to_submission: mpsc::Sender<SubmissionMessage>,
    /// Handle to the submission task. On completion it returns the [`Prompt`]s
    /// that have not yet been submitted as a [`Batch`] and their timestamps.
    submission: Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Vec<(batch::Id, Prompt<'static>)>,
                        MemorySubroutineError,
                    >,
                > + Send,
        >,
    >,
    /// Handle to the processing task. On completion it returns the pending
    /// [`Batch`]es that have been processed. Some may be complete, some may be
    /// pending. *The pending ones should be finished or you will waste money as
    /// they will eventually timeout.*
    processing: Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Vec<batch::Pending<'static>>,
                        MemorySubroutineError,
                    >,
                > + Send,
        >,
    >,
    /// Archival task that handles the tool calls to insert memories and
    /// operates the actual [`MemoryPalace`].
    archival:
        Pin<Box<dyn Future<Output = Result<(), MemorySubroutineError>> + Send>>,
}

/// A spawner that returns a handle to the spawned task
pub trait Spawn: Clone + Send + 'static {
    /// The handle type returned by the spawner
    type Handle<T: Send + 'static>: Future<Output = Result<T, MemorySubroutineError>>
        + Send;

    /// Spawn a future and return a handle
    fn spawn<F, T>(&self, future: F) -> Self::Handle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static;
}

// Tokio implementation
#[cfg(feature = "tokio")]
#[derive(Clone)]
pub struct TokioSpawn;

#[cfg(feature = "tokio")]
impl Spawn for TokioSpawn {
    type Handle<T: Send + 'static> =
        Pin<Box<dyn Future<Output = Result<T, MemorySubroutineError>> + Send>>;

    fn spawn<F, T>(&self, future: F) -> Self::Handle<T>
    where
        F: Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        let handle = tokio::spawn(future);
        Box::pin(async move { Ok(handle.await?) })
    }
}

impl BackgroundTasks {
    /// Spawn tasks with a provided spawner, load state, and return a handle.
    pub async fn spawn<S: Spawn>(
        client: Client,
        config: SubroutineConfig,
        pool: PgPool,
        schema: String,
        spawn: S,
        states: Vec<SaveState>,
    ) -> Result<Self, MemorySubroutineError> {
        let (mut tx_submission, rx_submission) = mpsc::channel(100);
        let (mut tx_processing, rx_processing) = mpsc::channel(100);
        let (tx_dead_letter, rx_dead_letter) = mpsc::channel(100);
        let (tx_ready, rx_ready) = mpsc::channel(10);

        // Create submission task
        let client1 = client.clone();
        let config1 = config.clone();
        let tx_processing1 = tx_processing.clone();
        let submission_handle = spawn.spawn(async move {
            super::tasks::batch_submission_task(
                client1,
                config1,
                rx_submission,
                tx_processing1,
                tx_dead_letter,
            )
            .await
        });

        // Create processing task
        let client2 = client;
        let config2 = config.clone();
        let tx_submission2 = tx_submission.clone();
        let processing_handle = spawn.spawn(async move {
            super::tasks::batch_processing_task(
                client2,
                config2,
                rx_processing,
                tx_submission2,
                tx_ready,
            )
            .await
        });

        // Create archival task
        let config3 = config.clone();
        let archival_handle = spawn.spawn(async move {
            super::tasks::batch_archival_task(
                config3,
                pool,
                schema,
                rx_ready,
                rx_dead_letter,
            )
            .await
        });

        // Now that the tasks are running, we can load the state if provided. We
        // will await, however the submissions should return immediately.
        // We want to push any existing state into their appropriate channels

        // FIXME: There's an assumption that the task spawns immediately which
        // may not hold true in all runtimes, in which case this could block
        // indefinitely if the tasks are not ready to receive messages.
        for state in states {
            // Create a stream from the pending submissions
            let mut submission_stream = futures::stream::iter(
                state.pending_submissions.into_iter().map(|(id, prompt)| {
                    Ok(SubmissionMessage::Store { id, prompt })
                }),
            );

            // Send all pending submissions
            tx_submission
                .send_all(&mut submission_stream)
                .await
                .unwrap(); // Impossible because the stream always returns Ok

            // Push pending batches
            let mut processing_stream =
                futures::stream::iter(state.pending_batches.into_iter().map(
                    |pending| Ok(ProcessingMessage::Batch { batch: pending }),
                ));

            tx_processing
                .send_all(&mut processing_stream)
                .await
                .unwrap(); // Impossible because the stream always returns Ok
        }

        Ok(Self {
            to_submission: tx_submission,
            submission: Box::pin(async move { submission_handle.await? }),
            processing: Box::pin(async move { processing_handle.await? }),
            archival: Box::pin(async move { archival_handle.await? }),
        })
    }

    /// Initiate shutdown and wait for tasks to return their data
    #[must_use = "Discarding SaveState may drop pending submissions/batches and cause data loss"]
    pub async fn shutdown(self) -> Result<SaveState, MemorySubroutineError> {
        // Close the submission channel to signal shutdown
        drop(self.to_submission);

        let (submission_result, processing_result, archival_result) =
            futures::join!(self.submission, self.processing, self.archival);

        // TODO: Think about whether propagating is the right thing to do since
        // we could lose data if *any* of these tasks fail.
        let pending_submissions = submission_result?;
        let pending_batches = processing_result?;
        archival_result?;

        Ok(SaveState {
            id: Uuid::new_v4(),
            shutdown_date: Utc::now(),
            pending_submissions,
            pending_batches,
        })
    }
}

// Convenience constructors
impl BackgroundTasks {
    #[cfg(feature = "tokio")]
    pub async fn spawn_tokio(
        client: Client,
        config: SubroutineConfig,
        pool: PgPool,
        schema: String,
        states: Vec<SaveState>,
    ) -> Result<Self, MemorySubroutineError> {
        Self::spawn(client, config, pool, schema, TokioSpawn, states).await
    }
}
