// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
use futures::SinkExt;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::VecDeque;

use crate::batch;
use crate::prompt::message::Block;
use crate::tool::memory_palace::{Memory, MemoryPalaceError};
use crate::tool::{self, Use};
use crate::{
    Client, Key, Prompt,
    tool::{MemoryPalace, Method, Tool},
};

mod handle;
use handle::BackgroundTasks;

mod tasks;

mod prompts;
use prompts::MEMORY_SUBROUTINE_INSTRUCTIONS;

mod db;
use db::ensure_initialized;

pub(crate) mod archivist;
pub(crate) mod navigator;
pub(crate) mod wanderer;

/// Retry count for failed batch operations.
const BATCH_RETRY_COUNT: u32 = 3;

/// Configuration for the [`MemorySubroutine`] system.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubroutineConfig {
    /// Maximum memories to accumulate before submitting batch.
    pub batch_size: u16,
    /// Maximum time to wait before submitting partial batch in minutes.
    pub batch_timeout: u16,
    /// Maximum retries for failed operations.
    pub max_retries: u8,
    /// Frequency to poll batch status in minutes.
    pub poll_frequency: u16,
}

impl Default for SubroutineConfig {
    fn default() -> Self {
        Self {
            batch_size: 50,
            batch_timeout: 5,
            max_retries: 3,
            poll_frequency: 15,
        }
    }
}

/// [`MemorySubroutine`] error type.
#[derive(Debug, thiserror::Error)]
pub enum MemorySubroutineError {
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("MemoryPalace error: {0}")]
    MemoryPalace(#[from] MemoryPalaceError),
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),
    #[error("Client error: {0}")]
    Client(#[from] crate::client::Error),
    #[cfg(feature = "tokio")]
    #[error("Tokio task join error: {0}")]
    JoinError(#[from] tokio::task::JoinError),
    #[error("Embedding error: {0}")]
    Embedding(#[from] crate::tool::embedding::EmbeddingError),
    #[error("Other error: {0}")]
    Other(String),
}

/// Messages sent to the batch submission task
#[derive(Debug)]
enum SubmissionMessage {
    Store {
        id: crate::batch::Id,
        prompt: Prompt<'static>,
    },
    /// Retry failed prompts from a previous batch
    Retry {
        identified_prompts: Vec<(crate::batch::Id, Prompt<'static>)>,
    },
    /// Signal an Id has been completed and the count should be removed from the
    /// retry map
    Complete {
        id: crate::batch::Id,
    },
}

/// Messages sent to the batch processing task  
enum ProcessingMessage {
    Batch { batch: batch::Pending<'static> },
}

/// [`State`] of the [`MemorySubroutine`].
#[atomic_enum::atomic_enum]
pub enum State {
    /// The agent has not been initialized yet.
    Uninitialized,
    /// The agent is initializing.
    Initializing,
    /// The agent is ready to process memories.
    Ready,
    /// The agent is shutting down.
    ShuttingDown,
}

/// The `MemorySubroutine` agent.
pub struct MemorySubroutine {
    /// Anthropic [`Client`].
    client: Client,
    /// The `MemoryPalace` the agent uses to store and search memories.
    palace: MemoryPalace,
    /// State of the subroutine.
    state: AtomicState,
    /// Task handles for submission and processing tasks.
    handles: Option<BackgroundTasks>,
    /// Configuration
    config: SubroutineConfig,
    /// Recent context window for memory searches
    recent_context: VecDeque<String>,
    /// Schema name for state persistence (different from palace schema)
    schema_name: String,
}

impl MemorySubroutine {
    /// Create a new `MemorySubroutine` agent from a [`SubroutineConfig`],
    /// a [`MemoryPalace`], and an Anthropic [`Client`].
    pub fn from_palace_and_client(
        client: Client,
        palace: MemoryPalace,
    ) -> Self {
        Self::from_palace_and_client_with_schemas(
            client,
            palace,
            "memory_subroutine".to_string(),
        )
    }

    /// Create from an existing palace and client with custom state schema
    pub fn from_palace_and_client_with_schemas(
        client: Client,
        palace: MemoryPalace,
        state_schema: String,
    ) -> Self {
        Self {
            client,
            palace,
            state: State::Uninitialized.into(),
            handles: None,
            config: SubroutineConfig::default(),
            recent_context: VecDeque::with_capacity(10),
            schema_name: state_schema,
        }
    }

    /// Get the current state of the subroutine.
    pub fn state(&self) -> State {
        self.state.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Returns true if the subroutine is ready to process memories.
    pub fn is_ready(&self) -> bool {
        matches!(self.state(), State::Ready)
    }

    /// Submit a prompt with id for memory storage.
    pub async fn submit_prompt_with_id(
        &mut self,
        prompt: Prompt<'static>,
        id: batch::Id,
    ) -> Result<(), MemorySubroutineError> {
        if !self.is_ready() {
            return Err(crate::tool::memory_palace::MemoryPalaceError::Other(
                "MemorySubroutine is not ready".to_string(),
            ).into());
        }

        if let Some(handles) = &mut self.handles {
            handles
                .to_submission
                .send(SubmissionMessage::Store { prompt, id })
                .await
                .map_err(|e| {
                    crate::tool::memory_palace::MemoryPalaceError::Other(
                        format!("Failed to submit prompt: {}", e),
                    )
                })?;
        }

        Ok(())
    }

    /// Submit prompt for storage and get an ID back
    pub async fn submit_prompt(
        &mut self,
        prompt: Prompt<'static>,
    ) -> Result<batch::Id, MemorySubroutineError> {
        let id = batch::Id::default();
        self.submit_prompt_with_id(prompt, id).await?;
        Ok(id)
    }


    /// Check for and process any ready batches
    pub async fn process_ready_batches(
        &mut self,
    ) -> Result<
        Vec<crate::batch::Ready<'static>>,
        crate::tool::memory_palace::MemoryPalaceError,
    > {
        // The signature needs to change too or we might not even need this
        // given the processing happens other tasks.
        todo!("Implement process_ready_batches");
    }

    /// Execute a memory search using the palace methods directly
    async fn execute_search(
        &mut self,
        query: &str,
    ) -> Result<Vec<(String, String, Memory)>, MemoryPalaceError> {
        self.palace.search(query).await
    }

    /// Execute memory storage using the palace methods directly
    async fn execute_store(
        &mut self,
        room: &str,
        content: &str,
        tags: Vec<&str>,
    ) -> Result<i64, MemoryPalaceError> {
        self.palace.store_memory(room, content, tags).await
    }
}

#[async_trait::async_trait]
impl Tool for MemorySubroutine {
    fn name(&self) -> &str {
        stringify!(MemorySubroutine)
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("MemorySubroutine::run")
                .description("Autonomously search for and summarize relevant memories based on the current conversation context. This runs automatically each turn.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build()
                .unwrap(),
        ].into_iter())
    }

    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // If we are already Ready, we should not re-initialize
        if matches!(self.state(), State::Ready) {
            return Err(Box::new(MemorySubroutineError::Other(
                "MemorySubroutine is already initialized".to_string(),
            )));
        }

        // Add memory subroutine instructions if not already present
        if let Some(system) = &mut prompt.system {
            // ## Note:
            // `Content` has only `iter_mut` because it might be a `SinglePart`
            // variant but we want to iterate over `Block`s, so on the first
            // iteration, such a part is converted to a block before iterating.
            if system.iter_mut().any(|block| {
                if let Block::Text { text, .. } = block {
                    text.contains("<memory_subroutine_instructions>")
                } else {
                    false
                }
            }) {
                // We are already initialized. We probably do not want to double
                // init. Calling this twice must be an error because it could
                // lead to an inconsistent state.
                return Err(Box::new(MemorySubroutineError::Other(
                    "MemorySubroutine is already initialized".to_string(),
                )));
            } else {
                system.push(MEMORY_SUBROUTINE_INSTRUCTIONS);
            }
        } else {
            prompt.system = Some(MEMORY_SUBROUTINE_INSTRUCTIONS.into());
        };

        // Add only our methods (not the palace's) to the main prompt
        prompt.push_methods(self.methods());

        // Ensure the state persistence schema exists
        ensure_initialized(&self.palace.pool, &self.schema_name).await?;

        // Create our retrieval agent prompt and initialize the palace
        let mut agent_prompt = prompts::create_memory_retrieval_agent_prompt();
        self.palace.on_init(&mut agent_prompt).await?;

        // Try to load any existing state(s)
        let states =
            db::load_states(&self.palace.pool, &self.schema_name).await?;

        // Start the background tasks with pending memories
        #[cfg(feature = "tokio")]
        {
            let client = self.client.clone();
            let config = self.config.clone();
            let pool = self.palace.pool.clone();
            let schema = self.palace.schema_name.clone();

            self.handles = Some(
                BackgroundTasks::spawn_tokio(
                    client,
                    config,
                    pool,
                    self.schema_name,
                    states,
                )
                .await?,
            );
        }

        self.state
            .store(State::Ready, std::sync::atomic::Ordering::Release);

        Ok(())
    }

    async fn on_turn(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Update recent context from the last few messages
        self.recent_context.clear();
        for msg in prompt.messages.iter().rev().take(5).rev() {
            self.recent_context
                .push_back(format!("{}: {}", msg.role, msg.content));
        }

        // Truncate to keep memory bounded
        while self.recent_context.len() > 10 {
            self.recent_context.pop_front();
        }

        Ok(())
    }

    async fn call<'a>(&mut self, call: Use<'a>) -> tool::Result<'a> {
        match call.name.split("::").last().unwrap_or(&call.name) {
            "run" => {
                // Build search context from recent context with security formatting
                let search_context = self
                    .recent_context
                    .iter()
                    .map(|msg| {
                        // Check for injection attempts
                        if msg.contains("<user>")
                            || msg.contains("</user>")
                            || msg.contains("<assistant>")
                            || msg.contains("</assistant>")
                        {
                            #[cfg(feature = "log")]
                            log::warn!(
                                "Potential injection attempt in context: {}",
                                msg
                            );
                            // Sanitize by escaping angle brackets
                            let sanitized =
                                msg.replace('<user>', "<fake_user>")
                                    .replace("</user>", "</fake_user>")
                                    .replace('<assistant>', "<fake_assistant>")
                                    .replace("</assistant>", "</fake_assistant>");
                            sanitized
                        } else {
                            msg.clone()
                        }
                    })
                    .enumerate()
                    .map(|(i, msg)| {
                        // Wrap messages appropriately based on role
                        if msg.starts_with("User:") {
                            format!(
                                "<user>{}</user>",
                                msg.strip_prefix("User:").unwrap().trim()
                            )
                        } else if msg.starts_with("Assistant:") {
                            format!(
                                "<assistant>{}</assistant>",
                                msg.strip_prefix("Assistant:").unwrap().trim()
                            )
                        } else {
                            // Shouldn't happen but handle gracefully
                            format!("<unknown>{}</unknown>", msg)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                if search_context.is_empty() {
                    return tool::Result {
                        tool_use_id: call.id,
                        content: "No conversation context available for memory search.".into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    };
                }

                // Search for relevant memories
                match self.execute_search(&search_context).await {
                    Ok(memories) => {
                        if memories.is_empty() {
                            tool::Result {
                                tool_use_id: call.id,
                                content: "None".into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            // Format memories with citations
                            let mut response = String::new();
                            let mut citations = Vec::new();

                            for (idx, (room, id, memory)) in
                                memories.iter().take(5).enumerate()
                            {
                                if idx > 0 {
                                    response.push_str("\n\n");
                                }
                                response.push_str(&format!(
                                    "[{}] {}",
                                    room, memory.content
                                ));
                                citations.push(format!("memory:{}", id));
                            }

                            if !citations.is_empty() {
                                response.push_str(&format!(
                                    "\n\n[Sources: {}]",
                                    citations.join(", ")
                                ));
                            }

                            tool::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(e) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Error searching memories: {}", e)
                            .into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            _ => tool::Result {
                tool_use_id: call.id,
                content: format!("Unknown method: {}", call.name).into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    async fn on_shutdown(
        &mut self,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.state
            .store(State::ShuttingDown, std::sync::atomic::Ordering::Release);

        // Shutdown and retrieve any pending state
        if let Some(mut handles) = self.handles.take() {
            match handles.shutdown().await {
                Ok(state) => {
                    match db::save_state(
                        &self.palace.pool,
                        &self.palace.schema_name,
                        state,
                    )
                    .await
                    {
                        Ok(shutdown_id) => {
                            #[cfg(feature = "log")]
                            log::info!(
                                "Saved shutdown state with ID: {}",
                                shutdown_id
                            );
                        }
                        Err(e) => {
                            #[cfg(feature = "log")]
                            log::error!("Failed to save shutdown state: {}", e);
                            return Err(Box::new(e));
                        }
                    }
                }
                Err(e) => {
                    #[cfg(feature = "log")]
                    log::error!("Error during shutdown: {}", e);
                    return Err(Box::new(e));
                }
            }
        }

        Ok(())
    }

    async fn save_json(&mut self) -> serde_json::Value {
        // Like with the palace itself, we just store the metadata and config
        // necessary to connect to the database and restore the palace.
        json!({
            "schema": self.schema_name,
            "config": self.config,
            "palace": self.palace.save_json().await,
            // We don't store the most recent context as it is transient
        })
    }

    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> Result<(), String> {
        let obj = if let serde_json::Value::Object(obj) = json {
            obj
        } else {
            return Err("Invalid JSON format for MemorySubroutine".to_string());
        };

        // Load schema
        self.schema_name = obj
            .remove("schema")
            .and_then(|v| {
                if let serde_json::Value::String(s) = v {
                    Some(s)
                } else {
                    None
                }
            })
            .ok_or("Missing or invalid 'schema' field")?;

        // Load config
        self.config = serde_json::from_value(
            obj.remove("config").ok_or("Missing 'config' field")?,
        )
        .map_err(|e| format!("Failed to parse 'config': {}", e))?;

        // Load palace
        let palace_json =
            obj.remove("palace").ok_or("Missing 'palace' field")?;
        self.palace
            .load_json(palace_json) // ensures the palace is initialized
            .await
            .map_err(|e| format!("Failed to load palace: {}", e))?;

        // Ensure the palace is initialized
        ensure_initialized(&self.palace.pool, &self.schema_name)
            .await
            .map_err(|e| {
                format!("Failed to ensure palace initialization: {}", e)
            })?;

        Ok(())
    }
}
