// Copyright (c) 2024 Michael de Gans, Claude Sonnet 4, and Claude Opus 4
#![allow(dead_code)]

//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.
use std::sync::Arc;

use sqlx::{PgPool, Postgres, Transaction};
mod tool;

/// [`MemoryPalace`] models and types.
mod models;
pub(crate) use models::*;

/// [`MemoryPalace`] Database initialization.
mod db;
pub(crate) use db::{ensure_initialized, execute_with_schema};

/// [`MemoryPalace`] specific [`tool::Use`] operations.
mod m_use;
pub(crate) use m_use::Use;

/// [`MemoryPalace`] service implementation.
mod service;
use service::*;

/// [`MemoryPalace`] error handling.
mod error;
pub use error::MemoryPalaceError;

use crate::Prompt;
use crate::tool::embedding::{EmbeddingClient, EmbeddingError};
use crate::tool::memory_subroutine::{
    archivist::ArchivistUse, navigator::NavigatorUse,
};

const MEMORY_PALACE_INSTRUCTIONS: &str = r#"<memory_palace_instructions>You have access to a Memory Palace - a spatial knowledge organization system that helps you store, organize, and retrieve knowledge across conversations.

## Key Concepts:
- **Rooms**: Organize memories by topic (e.g., "science", "cooking", "personal_facts")
- **Memories**: Individual pieces of knowledge with content, tags, and timestamps
- **Relationships**: Connect related memories for graph traversal and discovery
- **Concepts**: Extract and link semantic concepts for advanced querying

## Best Practices:
- On your first turn with a user call `MemoryPalace::summary` to get a context summary of recent and important memories.
- Do not call `MemoryPalace::summary` in the middle of a conversation since any alterations to the palace will already be in context.
- Use descriptive room names that group related knowledge
- Add relevant tags to make memories searchable
- Create relationships between related memories to build knowledge graphs

Start with `MemoryPalace::store` to save important information, then use `MemoryPalace::search` to find it later.</memory_palace_instructions>"#;

/// A Memory Palace knowledge base for AI agents. Cheap to clone.
///
/// Designed by Claude 4, Sonnet (Copilot), guided by Michael de Gans.
#[derive(Clone, Debug)]
pub struct MemoryPalace {
    /// PostgreSQL connection pool.
    pub(crate) pool: PgPool,
    /// The schema name to use for all operations.
    pub(crate) schema_name: Arc<String>,
    /// Embedding client for generating embeddings
    pub(crate) embedding_client: Option<Box<dyn EmbeddingClient>>,
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";
    const EMBEDDING_SIZE: usize = 1536;
    const MAX_TRAVERSAL_DEPTH: u32 = 10; // Safety limit for graph traversal
    const MAX_RESULTS_PER_QUERY: usize = 100; // Limit result set sizes

    /// Palace schema name.
    pub fn schema(&self) -> &str {
        &self.schema_name
    }

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool. Uses the
    /// default 'public' schema.
    pub async fn from_pool(pool: PgPool) -> Result<Self, MemoryPalaceError> {
        Self::from_pool_with_schema(pool, "public".to_string()).await
    }

    /// Get the embedding size for this Memory Palace.
    pub const fn embedding_size() -> usize {
        Self::EMBEDDING_SIZE
    }

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool with a
    /// specific schema.
    pub async fn from_pool_with_schema(
        pool: PgPool,
        schema_name: String,
    ) -> Result<Self, MemoryPalaceError> {
        let new = Self {
            pool,
            schema_name: schema_name.into(),
            embedding_client: None,
        };
        ensure_initialized(&new.pool, &new.schema_name).await?;

        Ok(new)
    }

    /// Set the embedding client for this palace
    pub fn with_embedding_client(
        mut self,
        client: Box<dyn EmbeddingClient>,
    ) -> Self {
        self.embedding_client = Some(client);
        self
    }

    /// Get or compute an embedding for the given text
    async fn get_or_compute_embedding(
        &self,
        text: &str,
    ) -> Result<Option<Vec<f32>>, MemoryPalaceError> {
        let Some(client) = &self.embedding_client else {
            return Ok(None);
        };

        // Compute hash of the content
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let content_hash = hasher.finalize().to_vec();

        // Check if we already have this embedding
        let existing: Option<pgvector::Vector> =
            execute_with_schema(&self.pool, &self.schema_name, |tx| {
                Box::pin(async move {
                    sqlx::query_scalar(
                        r#"
                        SELECT embedding 
                        FROM embeddings 
                        WHERE model_name = $1 AND content_hash = $2
                        "#,
                    )
                    .bind(client.model().as_str())
                    .bind(&content_hash)
                    .fetch_optional(&mut **tx)
                    .await
                })
            })
            .await?;

        if let Some(embedding) = existing {
            return Ok(Some(embedding.to_vec()));
        }

        // Compute new embedding
        let text_embedding = client
            .get_embedding(text)
            .await
            .map_err(|e| MemoryPalaceError::Other(e.to_string()))?;

        // Store for future use
        execute_with_schema(
            &self.pool,
            &self.schema_name,
            |tx| {
                Box::pin(async move {
                    sqlx::query(
                        r#"
                        INSERT INTO embeddings (model_name, model_size, content_hash, embedding)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (model_name, content_hash) DO NOTHING
                        "#
                    )
                    .bind(client.model().as_str())
                    .bind(client.embedding_size() as i32)
                    .bind(&content_hash)
                    .bind(pgvector::Vector::from(text_embedding.embedding.as_ref().clone()))
                    .execute(&mut **tx)
                    .await
                })
            },
        )
        .await?;

        Ok(Some(text_embedding.embedding.to_vec()))
    }

    /// Store a [`Memory`] in a specific [`Room`].
    pub async fn store_memory(
        &self,
        room_name: &str,
        memory: Memory,
        placement: &str,
        placement_description: Option<&str>,
        tags: Vec<String>,
        embedding: Option<Vec<f32>>,
    ) -> Result<MemoryId, MemoryPalaceError> {
        let room_name = room_name.to_string();
        let placement = placement.to_string();
        let placement_description =
            placement_description.map(|s| s.to_string());

        // Generate embedding if not provided
        let embedding = match embedding {
            Some(emb) => Some(emb),
            None => {
                if let Some(content) =
                    memory.format_for_navigator(MemoryId(0), RoomId(0))
                {
                    self.get_or_compute_embedding(&content.to_string()).await?
                } else {
                    None
                }
            }
        };

        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                // Get room ID
                let room_id: RoomId = sqlx::query_scalar(
                    "SELECT id FROM rooms WHERE name = $1"
                )
                .bind(&room_name)
                .fetch_one(&mut **tx)
                .await
                .map_err(|_| MemoryPalaceError::RoomNotFound(room_name.clone()))?;

                // Update room visit tracking
                sqlx::query(
                    "UPDATE rooms SET last_visited = NOW(), visit_count = visit_count + 1 WHERE id = $1"
                )
                .bind(room_id)
                .execute(&mut **tx)
                .await?;

                // Store the memory
                let memory_id: MemoryId = sqlx::query_scalar(
                    r#"INSERT INTO memories (content, room_id, placement, placement_description, tags, embedding, importance)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)
                       RETURNING id"#,
                )
                .bind(serde_json::to_value(&memory)?)
                .bind(room_id)
                .bind(&placement)
                .bind(&placement_description)
                .bind(serde_json::to_value(&tags)?)
                .bind(embedding.map(pgvector::Vector::from))
                .bind(importance)
                .fetch_one(&mut **tx)
                .await?;

                Ok(memory_id)
            })
        })
        .await
    }

    /// Search for memories with proper type handling
    pub async fn search(
        &self,
        query: &str,
    ) -> Result<Vec<ScoredMemory>, MemoryPalaceError> {
        let results =
            service::search(&self.pool, &self.schema_name, query).await?;

        // Convert MemoryRow to ScoredMemory with proper Memory types
        Ok(results
            .into_iter()
            .map(|r| {
                let memory_row = r.memory;
                ScoredMemory {
                    memory: memory_row,
                    room: r.room,
                    relevance_score: r.relevance_score,
                    recency_score: r.recency_score,
                    relationship_score: r.relationship_score,
                    final_score: r.final_score,
                }
            })
            .collect())
    }

    /// Handle batches of [`tool::Use`]
    pub(crate) async fn batch_call_archivist(
        &mut self,
        calls: Vec<ArchivistUse>,
    ) -> Result<(), MemoryPalaceError> {
        let mut tx = self.pool.begin().await?;
        let mut errors = Vec::new();

        for archivist in calls {
            archivist.archive(self, tx).await?;
        }

        tx.commit().await?;

        if !errors.is_empty() {
            return Err(MemoryPalaceError::Many(errors));
        }

        Ok(())
    }

    /// Get a memory by its ID
    pub async fn get_memory_by_id(
        &self,
        memory_id: MemoryId,
    ) -> Result<MemoryRow, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let memory: MemoryRow = sqlx::query_as(
                    "SELECT * FROM memories WHERE id = $1"
                )
                .bind(memory_id)
                .fetch_one(&mut **tx)
                .await
                .map_err(|_| MemoryPalaceError::MemoryNotFound(memory_id))?;

                // Update access tracking
                sqlx::query(
                    "UPDATE memories SET last_accessed = NOW(), access_count = access_count + 1 WHERE id = $1"
                )
                .bind(memory_id)
                .execute(&mut **tx)
                .await?;

                Ok(memory)
            })
        })
        .await
    }

    /// Get a room by its ID
    pub async fn get_room_by_id(
        &self,
        room_id: RoomId,
    ) -> Result<Room, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let room: Room =
                    sqlx::query_as("SELECT * FROM rooms WHERE id = $1")
                        .bind(room_id)
                        .fetch_one(&mut **tx)
                        .await
                        .map_err(|_| {
                            MemoryPalaceError::RoomNotFound(room_id.to_string())
                        })?;

                Ok(room)
            })
        })
        .await
    }

    /// Get a room by its name
    pub async fn get_room_by_name(
        &self,
        room_name: &str,
    ) -> Result<Room, MemoryPalaceError> {
        let room_name = room_name.to_string();
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let room: Room =
                    sqlx::query_as("SELECT * FROM rooms WHERE name = $1")
                        .bind(&room_name)
                        .fetch_one(&mut **tx)
                        .await
                        .map_err(|_| {
                            MemoryPalaceError::RoomNotFound(room_name)
                        })?;

                Ok(room)
            })
        })
        .await
    }

    /// Create a new room
    pub async fn create_room(
        &self,
        name: &str,
        description: &str,
        atmosphere: Option<&str>,
    ) -> Result<RoomId, MemoryPalaceError> {
        let name = name.to_string();
        let description = description.to_string();
        let atmosphere = atmosphere.map(|s| s.to_string());

        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let room_id: RoomId = sqlx::query_scalar(
                    r#"INSERT INTO rooms (name, description, atmosphere)
                       VALUES ($1, $2, $3)
                       RETURNING id"#,
                )
                .bind(&name)
                .bind(&description)
                .bind(&atmosphere)
                .fetch_one(&mut **tx)
                .await?;

                Ok(room_id)
            })
        })
        .await
    }

    /// Find [`Memory`]s related to a specific [`Memory`] with a maximum depth
    pub(crate) async fn find_resonating_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, MemoryRow, String, f64)>, MemoryPalaceError>
    {
        // Enforce safety limit
        let safe_depth = max_depth.min(Self::MAX_TRAVERSAL_DEPTH);

        find_resonating_memories(
            &self.pool,
            &self.schema_name,
            memory_id,
            safe_depth,
            min_strength,
        )
        .await
    }

    /// Semantic search across all rooms using an embedding.
    pub(crate) async fn semantic_search_all_rooms(
        &mut self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<MemoryRow>, MemoryPalaceError> {
        semantic_search_all_rooms(
            &self.pool,
            &self.schema_name,
            embedding,
            limit,
        )
        .await
    }

    /// Get rooms within N hops of current room
    pub async fn get_rooms_within_radius(
        &self,
        start_room: &str,
        radius: u32,
    ) -> Result<Vec<RoomWithDistance>, MemoryPalaceError> {
        // Enforce safety limit
        let safe_radius = radius.min(Self::MAX_TRAVERSAL_DEPTH);

        service::get_rooms_within_radius(
            &self.pool,
            &self.schema_name,
            start_room,
            safe_radius,
        )
        .await
    }

    /// Get a hint about what kind of memories are in a room
    pub async fn get_room_character_hint(
        pool: &PgPool,
        schema: &str,
        room_name: &str,
    ) -> Result<String, MemoryPalaceError> {
        get_room_character_hint(pool, schema, room_name).await
    }

    /// Extract and create [`Concept`]s from a specific [`Memory`].
    pub(crate) async fn extract_concepts(
        &mut self,
        memory_id: i64,
        concepts: impl IntoIterator<Item = &str>,
    ) -> Result<String, MemoryPalaceError> {
        extract_concepts(
            &self.pool,
            &self.schema_name,
            memory_id,
            concepts.into_iter().map(|s| s.to_string()).collect(),
        )
        .await
    }

    /// Find memories by [`Concept`] with enhanced relevance scoring.
    pub(crate) async fn find_memories_by_concept(
        &mut self,
        concept: impl Into<String>,
    ) -> Result<Vec<(String, String, MemoryRow, f64)>, MemoryPalaceError> {
        find_memories_by_concept(&self.pool, &self.schema_name, concept.into())
            .await
    }

    /// Get graph statistics and insights.
    pub(crate) async fn get_graph_stats(
        &mut self,
    ) -> Result<String, MemoryPalaceError> {
        get_graph_stats(&self.pool, &self.schema_name).await
    }

    /// Get a summary of recent and important memories for prompt context.
    async fn get_context_summary(
        &mut self,
    ) -> Result<String, MemoryPalaceError> {
        get_context_summary(&self.pool, &self.schema_name).await
    }

    /// Search within a specific room only
    pub async fn search_in_room(
        &self,
        room_name: &str,
        query: &str,
    ) -> Result<Vec<MemoryRow>, MemoryPalaceError> {
        // Like search() but filtered to one room
        crate::tool::memory_subroutine::navigator::search_in_room(
            &self.pool,
            &self.schema_name,
            room_name,
            query,
        )
        .await
    }

    /// Get adjacent rooms with semantic distances
    pub async fn get_adjacent_rooms_sorted(
        &self,
        current_room: &str,
        limit: Option<usize>,
    ) -> Result<Vec<(String, Room, f32)>, MemoryPalaceError> {
        crate::tool::memory_subroutine::navigator::get_adjacent_rooms_sorted(
            &self.pool,
            &self.schema_name,
            current_room,
            limit.unwrap_or(10),
        )
        .await
    }

    /// Follow a passage to get the destination room
    pub async fn follow_passage(
        &self,
        from_room: &str,
        direction: &str,
    ) -> Result<Room, MemoryPalaceError> {
        crate::tool::memory_subroutine::navigator::follow_passage(
            &self.pool,
            &self.schema_name,
            from_room,
            direction,
        )
        .await
    }

    /// Get rich description of a room
    pub async fn get_room_description(
        &self,
        room_name: &str,
    ) -> Result<String, MemoryPalaceError> {
        crate::tool::memory_subroutine::navigator::get_room_description(
            &self.pool,
            &self.schema_name,
            room_name,
        )
        .await
    }

    /// Find similar memories to a given memory
    pub async fn find_similar_memories(
        &self,
        memory_id: MemoryId,
        similarity_threshold: Option<f32>,
        max_results: Option<i32>,
    ) -> Result<Vec<SimilarMemory>, MemoryPalaceError> {
        // ...existing code...
    }

    /// Get memories that are candidates for consolidation
    pub async fn get_consolidation_candidates(
        &self,
        min_cluster_size: Option<i32>,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<MemoryCluster>, MemoryPalaceError> {
        // ...existing code...
    }

    /// Store a [`Prompt`] and return its ID for citation purposes
    pub async fn store_prompt(
        &self,
        prompt: &Prompt<'static>,
        embedding: Option<Vec<f32>>,
    ) -> Result<PromptId, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let prompt_id: PromptId = sqlx::query_scalar(
                    r#"INSERT INTO prompts (content, embedding)
                       VALUES ($1, $2)
                       RETURNING id"#,
                )
                .bind(serde_json::to_value(prompt)?)
                .bind(embedding.map(pgvector::Vector::from))
                .fetch_one(&mut **tx)
                .await?;

                Ok(prompt_id)
            })
        })
        .await
    }

    /// Retrieve a [`Prompt`] by its ID
    pub async fn get_prompt(
        &self,
        prompt_id: PromptId,
    ) -> Result<Prompt<'static>, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let row: (serde_json::Value,) =
                    sqlx::query_as("SELECT content FROM prompts WHERE id = $1")
                        .bind(prompt_id)
                        .fetch_one(&mut **tx)
                        .await
                        .map_err(|_| {
                            MemoryPalaceError::Other(format!(
                                "Prompt {} not found",
                                prompt_id.0
                            ))
                        })?;

                let prompt: Prompt<'static> = serde_json::from_value(row.0)?;
                Ok(prompt)
            })
        })
        .await
    }

    /// Store a memory from a message with prompt context
    pub async fn store_message_memory(
        &self,
        room_name: &str,
        message: Message<'static>,
        prompt_id: Option<PromptId>,
        message_index: Option<usize>,
        note: Option<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, MemoryPalaceError> {
        let memory = Memory::Message {
            message,
            prompt: prompt_id,
            index: message_index,
            note,
        };

        // Generate embedding from the message content
        let embedding = if let Some(content) =
            memory.format_for_navigator(MemoryId(0), RoomId(0))
        {
            self.get_or_compute_embedding(&content.to_string()).await?
        } else {
            None
        };

        self.store_memory(
            room_name,
            memory,
            "message_shelf",
            Some("A preserved exchange"),
            tags,
            embedding,
        )
        .await
    }

    /// Store a memory from a message pair with prompt context
    pub async fn store_pair_memory(
        &self,
        room_name: &str,
        pair: MessagePair<'static>,
        prompt_id: Option<PromptId>,
        message_index: Option<usize>,
        note: Option<String>,
        tags: Vec<String>,
    ) -> Result<MemoryId, MemoryPalaceError> {
        let memory = Memory::Pair {
            pair,
            prompt: prompt_id,
            index: message_index,
            note,
        };

        // The embedding will be generated by store_memory
        self.store_memory(
            room_name,
            memory,
            "conversation_shelf",
            Some("A meaningful exchange"),
            tags,
            None, // Let store_memory handle embedding generation
        )
        .await
    }

    /// Store a conversation summary
    pub async fn store_conversation_summary(
        &self,
        room_name: &str,
        prompt_id: PromptId,
        summary: Content<'static>,
        title: String,
        tags: Vec<String>,
    ) -> Result<MemoryId, MemoryPalaceError> {
        let memory = Memory::ConversationSummary {
            prompt: prompt_id,
            summary,
            title,
        };

        // The embedding will be generated by store_memory
        self.store_memory(
            room_name,
            memory,
            "summary_alcove",
            Some("A distilled understanding"),
            tags,
            None, // Let store_memory handle embedding generation
        )
        .await
    }

    /// Find memories related to a specific prompt
    pub async fn find_memories_by_prompt(
        &self,
        prompt_id: PromptId,
    ) -> Result<Vec<MemoryRow>, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let memories: Vec<MemoryRow> = sqlx::query_as(
                    r#"
                    SELECT m.*
                    FROM memories m
                    WHERE 
                        (m.content->>'prompt')::bigint = $1
                        OR (m.content->'data'->>'prompt')::bigint = $1
                    ORDER BY 
                        (m.content->'data'->>'index')::int,
                        m.created_at
                    "#,
                )
                .bind(prompt_id.0)
                .fetch_all(&mut **tx)
                .await?;

                Ok(memories)
            })
        })
        .await
    }

    /// Find memories using breadth-first search with depth and score limits
    pub async fn find_memories_bfs(
        &self,
        start_memory_id: i64,
        max_depth: u32,
        decay_factor: f64,
        min_score: f64,
    ) -> Result<Vec<(String, String, MemoryRow, f64, i32)>, MemoryPalaceError>
    {
        // Enforce safety limit
        let safe_depth = max_depth.min(Self::MAX_TRAVERSAL_DEPTH);

        service::find_memories_bfs(
            &self.pool,
            &self.schema_name,
            start_memory_id,
            safe_depth,
            decay_factor,
            min_score,
        )
        .await
    }
}
