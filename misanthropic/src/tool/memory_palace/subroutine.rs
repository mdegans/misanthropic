use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;
use std::future::Future;
use std::pin::Pin;

use sqlx::PgPool;
use futures::channel::mpsc;
use futures::channel::oneshot;

use crate::{
    AnthropicModel, Client, Prompt, 
    batch::{Batch, Id as BatchId, Prompts},
    prompt::message::Role,
    tool::{Method, ToolBox},
};

use super::{MemoryPalaceError, models::Memory};

/// Configuration for the memory subroutine system
pub struct SubroutineConfig {
    /// API key for the subroutine agents
    pub api_key: Key,
    /// Maximum memories to accumulate before submitting batch
    pub batch_size: usize,
    /// Maximum time to wait before submitting partial batch
    pub batch_timeout: Duration,
    /// Maximum retries for failed operations
    pub max_retries: u32,
    /// Model to use for entity extraction (Haiku recommended)
    pub model: AnthropicModel,
}

impl Default for SubroutineConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            batch_size: 50,
            batch_timeout: Duration::from_secs(30),
            max_retries: 3,
            model: AnthropicModel::Haiku30,
        }
    }
}

/// Messages sent to the batch submission task
#[derive(Debug)]
pub enum SubmissionMessage {
    Store {
        memory_id: i64,
        content: String,
        room: String,
        pool: PgPool,
        schema: String,
    },
    /// Retry failed prompts from a previous batch
    Retry {
        failed_prompts: Vec<(i64, String, String, PgPool, String)>,
        retry_count: u32,
    },
    Shutdown,
}

/// Response from submission task
#[derive(Debug)]
pub enum SubmissionResponse {
    Accepted,
    Rejected(String), // Error message
}

/// Messages sent to the batch processing task  
#[derive(Debug)]
pub enum ProcessingMessage {
    NewBatch {
        batch_id: String,
        /// Maps memory_id to (content, room, pool, schema) for retry purposes
        memory_data: HashMap<i64, (String, String, PgPool, String)>,
        retry_count: u32,
    },
    Shutdown,
}

/// The search agent that handles realtime retrieval
pub struct SearchAgent {
    pool: PgPool,
    schema: String,
    /// Track memory IDs already returned in this session
    seen_memories: Arc<futures::lock::Mutex<HashSet<i64>>>,
}

impl SearchAgent {
    pub fn new(pool: PgPool, schema: String) -> Self {
        Self {
            pool,
            schema,
            seen_memories: Arc::new(futures::lock::Mutex::new(HashSet::new())),
        }
    }

    /// Search for memories, filtering out already-seen results
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryPalaceError> {
        let results = super::service::search(&self.pool, &self.schema, query).await?;
        
        let mut seen = self.seen_memories.lock().await;
        let filtered: Vec<Memory> = results
            .into_iter()
            .filter(|(_, _, memory)| {
                if seen.contains(&memory.id) {
                    false
                } else {
                    seen.insert(memory.id);
                    true
                }
            })
            .take(limit)
            .map(|(_, _, memory)| memory)
            .collect();

        Ok(filtered)
    }

    /// Reset seen memories for a new session
    pub async fn reset_session(&self) {
        self.seen_memories.lock().await.clear();
    }
}

/// Runtime-independent task handle
pub struct SubroutineHandle {
    /// Handle to the submission task
    submission_handle: Pin<Box<dyn Future<Output = Result<(), MemoryPalaceError>> + Send>>,
    /// Handle to the processing task
    processing_handle: Pin<Box<dyn Future<Output = Result<(), MemoryPalaceError>> + Send>>,
}

impl SubroutineHandle {
    /// Wait for both tasks to complete
    pub async fn join(self) -> Result<(), MemoryPalaceError> {
        // Use futures::join! to wait for both
        let (submission_result, processing_result) = futures::join!(
            self.submission_handle,
            self.processing_handle
        );
        
        submission_result?;
        processing_result?;
        Ok(())
    }
}

/// Start the memory subroutine system (runtime-independent)
pub fn start_subroutine<Spawn>(
    config: SubroutineConfig,
    pool: PgPool,
    schema: String,
    spawn: Spawn,
) -> Result<
    (
        SearchAgent,
        mpsc::Sender<SubmissionMessage>,
        SubroutineHandle,
    ),
    MemoryPalaceError,
>
where
    Spawn: Fn(Pin<Box<dyn Future<Output = ()> + Send>>) + Send + 'static + Clone,
{
    let (tx_submission, rx_submission) = mpsc::channel(100);
    let (tx_processing, rx_processing) = mpsc::channel(100);
    
    let search_agent = SearchAgent::new(pool.clone(), schema.clone());
    
    // Create submission task future
    let config1 = config.clone();
    let tx_processing1 = tx_processing.clone();
    let submission_future = Box::pin(async move {
        batch_submission_task(config1, rx_submission, tx_processing1).await
    });
    
    // Create processing task future
    let config2 = config.clone();
    let tx_submission1 = tx_submission.clone();
    let processing_future = Box::pin(async move {
        batch_processing_task(config2, rx_processing, tx_submission1).await
    });
    
    // Spawn tasks using provided spawner
    let spawn_clone = spawn.clone();
    spawn(Box::pin(async move {
        if let Err(e) = submission_future.await {
            #[cfg(feature = "log")]
            log::error!("Batch submission task error: {}", e);
        }
    }));
    
    spawn(Box::pin(async move {
        if let Err(e) = processing_future.await {
            #[cfg(feature = "log")]
            log::error!("Batch processing task error: {}", e);
        }
    }));
    
    // For the handle, we need to create futures that can be joined
    // This is a bit tricky without tokio, but we can use channels
    let (submission_done_tx, submission_done_rx) = oneshot::channel();
    let (processing_done_tx, processing_done_rx) = oneshot::channel();
    
    // Modify the spawned tasks to signal completion
    // (This would require modifying the task functions to accept completion senders)
    
    let handle = SubroutineHandle {
        submission_handle: Box::pin(async move {
            submission_done_rx.await
                .map_err(|_| MemoryPalaceError::from("Submission task panicked"))?
        }),
        processing_handle: Box::pin(async move {
            processing_done_rx.await
                .map_err(|_| MemoryPalaceError::from("Processing task panicked"))?
        }),
    };
    
    Ok((search_agent, tx_submission, handle))
}

// For convenience, provide a tokio-specific wrapper
#[cfg(feature = "tokio")]
pub fn start_subroutine_tokio(
    config: SubroutineConfig,
    pool: PgPool,
    schema: String,
) -> Result<
    (
        SearchAgent,
        mpsc::Sender<SubmissionMessage>,
        SubroutineHandle,
    ),
    MemoryPalaceError,
> {
    start_subroutine(config, pool, schema, |future| {
        tokio::spawn(future);
    })
}

/// The batch submission task that accumulates memories and submits to batch API
pub async fn batch_submission_task(
    config: SubroutineConfig,
    mut rx: mpsc::Receiver<SubmissionMessage>,
    tx_processing: mpsc::Sender<ProcessingMessage>,
) -> Result<(), MemoryPalaceError> {
    use futures::StreamExt; // for recv()
    
    let client = Client::new(config.api_key)?;
    let mut pending_memories = Vec::new();
    let mut shutting_down = false;
    
    // Use futures_timer for runtime-independent intervals
    let mut batch_timer = futures_timer::Delay::new(config.batch_timeout);

    loop {
        futures::select! {
            msg = rx.next() => {
                match msg {
                    Some(SubmissionMessage::Store { memory_id, content, room, pool, schema }) => {
                        if shutting_down {
                            #[cfg(feature = "log")]
                            log::warn!("Rejecting new submission during shutdown");
                            continue;
                        }
                        
                        pending_memories.push((memory_id, content, room, pool, schema));
                        
                        if pending_memories.len() >= config.batch_size {
                            submit_batch(
                                &client,
                                &config,
                                &mut pending_memories,
                                &tx_processing,
                            ).await?;
                        }
                    }
                    Some(SubmissionMessage::Retry { failed_prompts, retry_count }
