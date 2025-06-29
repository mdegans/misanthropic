// Copyright (c) 2024 Michael de Gans, Claude Sonnet 4, and Claude Opus 4
#![allow(dead_code)]

//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.
use std::collections::BTreeSet;
use std::sync::Arc;

use sqlx::types::Text;
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
use crate::prompt::Message;
use crate::tool::embedding::{EmbeddingClient, EmbeddingError, TextEmbedding};

const MEMORY_PALACE_INSTRUCTIONS: &str = r#"<memory_palace_instructions>You have access to a Memory Palace - a spatial knowledge organization system that helps you store, organize, and retrieve knowledge across conversations.

## Key Concepts:
- **Rooms**: Organize memories by topic (e.g., "science", "cooking", "personal_facts")
- **Memories**: Individual pieces of knowledge with content, tags, and timestamps
- **Relationships**: Connect related memories for graph traversal and discovery

## Best Practices:
- On your first turn with a user call `MemoryPalace::summary` to get a context summary of recent and important memories.
- Do not call `MemoryPalace::summary` in the middle of a conversation since any alterations to the palace will already be in context.
- Use descriptive room names that group related knowledge
- Add relevant tags to make memories searchable
- Create relationships between related memories to build knowledge graphs

Start with `MemoryPalace::store` to save important information, then use `MemoryPalace::search` to find it later.</memory_palace_instructions>"#;

/// A `MemoryPalace` knowledge base for AI agents. Cheap to clone.
///
/// Designed by:
/// - Claude Sonnet 4 (initial design)
/// - Claude Opus 4 (further refinements)
/// - Michael de Gans (guidance, context)
#[derive(Clone)]
pub struct MemoryPalace {
    /// PostgreSQL connection pool.
    pub(crate) pool: PgPool,
    /// The schema name to use for all operations.
    pub(crate) schema_name: Arc<String>,
    /// Embedding client for generating embeddings
    pub(crate) embedding_client: Arc<Box<dyn EmbeddingClient>>,
}

impl std::fmt::Debug for MemoryPalace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(stringify!(MemoryPalace))
            .field("schema_name", &self.schema_name)
            .field("embedding_client", &self.embedding_client)
            .field("pool", &stringify!(PgPool))
            .finish()
    }
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";
    const MAX_TRAVERSAL_DEPTH: u32 = 10; // Hard limit for traversal depth
    const MAX_RESULTS_PER_QUERY: usize = 25; // Max results per query

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool with a
    /// specific schema.
    pub async fn from_components(
        pool: PgPool,
        schema_name: String,
        embedding_client: Arc<Box<dyn EmbeddingClient>>,
    ) -> Result<Self, MemoryPalaceError> {
        let new = Self {
            pool,
            schema_name: schema_name.into(),
            embedding_client,
        };

        // Ensure the database is initialized
        ensure_initialized(&new.pool, &new.schema_name).await?;

        Ok(new)
    }

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool. Uses the
    /// default 'public' schema.
    pub async fn from_pool_and_embedding_client(
        pool: PgPool,
        embedding_client: Arc<Box<dyn EmbeddingClient>>,
    ) -> Result<Self, MemoryPalaceError> {
        Self::from_components(pool, "public".to_string(), embedding_client)
            .await
    }

    /// Palace schema name.
    pub fn schema(&self) -> &str {
        &self.schema_name
    }

    /// Get the embedding service name currently in use.
    pub fn embedding_service_name(&self) -> &str {
        self.embedding_client.name()
    }

    /// Get the embedding size currently in use.
    pub fn embedding_size(&self) -> u16 {
        self.embedding_client.embedding_size()
    }

    /// Get the embedding model name currently in use.
    pub fn embedding_model_name(&self) -> Arc<String> {
        self.embedding_client.model()
    }

    /// Get or compute an embedding for the given text
    async fn get_or_compute_embedding(
        &self,
        text: &str,
    ) -> Result<TextEmbedding, MemoryPalaceError> {
        // Compute hash of the content
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(text.as_bytes());
        let content_hash = hasher.finalize().to_vec();

        // Check if we already have this embedding
        let existing: Option<Vec<f32>> =
            execute_with_schema(&self.pool, &self.schema_name, |tx| {
                Box::pin(async move {
                    Ok(sqlx::query_scalar(
                        r#"
                        SELECT embedding 
                        FROM embeddings 
                        WHERE model_name = $1 AND content_hash = $2
                        "#,
                    )
                    .bind(self.embedding_model_name().as_ref())
                    .bind(&content_hash)
                    .fetch_optional(&mut **tx)
                    .await?)
                })
            })
            .await?;

        if let Some(vec) = existing {
            return Ok(TextEmbedding {
                embedding: vec.into(),
                model: self.embedding_model_name(),
            });
        }
        // No embedding found

        // Compute new embedding
        let embedding = self
            .embedding_client
            .get_embedding(text)
            .await
            .map_err(|e| MemoryPalaceError::Other(e.to_string()))?;

        // Store for future use before returning
        execute_with_schema(
            &self.pool,
            &self.schema_name,
            |tx| {
                Box::pin(async move {
                    Ok(sqlx::query(
                        r#"
                        INSERT INTO embeddings (model_name, model_size, content_hash, embedding)
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT (model_name, content_hash) DO NOTHING
                        "#
                    )
                    .bind(self.embedding_model_name().as_ref())
                    .bind(self.embedding_size() as i32)
                    .bind(&content_hash)
                    .bind(embedding.as_ref())
                    .execute(&mut **tx)
                    .await?)
                })
            },
        )
        .await?;

        Ok(embedding)
    }

    /// Store a [`Memory`] in a specific [`Room`].
    pub async fn store_memory(
        &self,
        room_id: RoomId,
        memory: MemoryContent,
        placement: &str,
        placement_description: Option<&str>,
        tags: Vec<String>,
        embedding: Option<TextEmbedding>,
    ) -> Result<MemoryId, MemoryPalaceError> {
        let placement = placement.to_string();
        let placement_description =
            placement_description.map(|s| s.to_string());

        // Generate embedding if not provided
        let embedding = match embedding {
            Some(emb) => emb,
            None => {
                if let Some(content) =
                    memory.clone().format_for_navigator(MemoryId(0), RoomId(0))
                {
                    self.get_or_compute_embedding(&content.to_string()).await?
                } else {
                    return Err(MemoryPalaceError::Other(
                        "Memory content is empty".to_string(),
                    ));
                }
            }
        };

        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                // Update room visit tracking
                sqlx::query(
                    "UPDATE rooms SET last_visited = NOW(), visit_count = visit_count + 1 WHERE id = $1"
                )
                .bind(room_id)
                .execute(&mut **tx)
                .await?;

                // Store the memory
                let memory_id: MemoryId = sqlx::query_scalar(
                    r#"INSERT INTO memories (content, room_id, placement, placement_description, tags, embedding)
                       VALUES ($1, $2, $3, $4, $5, $6)
                       RETURNING id"#,
                )
                .bind(serde_json::to_value(&memory)?)
                .bind(room_id)
                .bind(&placement)
                .bind(&placement_description)
                .bind(serde_json::to_value(&tags)?)
                .bind(embedding.as_ref())
                .fetch_one(&mut **tx)
                .await?;

                Ok(memory_id)
            })
        })
        .await
    }

    /// Get a memory by its ID
    pub async fn get_memory_by_id(
        &self,
        memory_id: MemoryId,
    ) -> Result<Memory, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let memory: Memory = sqlx::query_as(
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
                            MemoryPalaceError::RoomNotFound(room_id)
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
    ) -> Result<Vec<(String, String, Memory, String, f64)>, MemoryPalaceError>
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

    /// Get rooms within N hops of current room. Max depth is limited to
    /// [`MemoryPalace::MAX_TRAVERSAL_DEPTH`].
    pub async fn get_rooms_within_radius(
        &self,
        start_room: &str,
        radius: u32,
    ) -> Result<BTreeSet<RoomWithJourney>, MemoryPalaceError> {
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

    /// Search within a specific room only
    pub async fn search_in_room(
        &self,
        room_name: &str,
        query: &str,
    ) -> Result<Vec<Memory>, MemoryPalaceError> {
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

    /// Store a [`Prompt`] and return its ID for citation purposes
    pub async fn store_prompt(
        &self,
        prompt: Prompt<'static>,
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
}
