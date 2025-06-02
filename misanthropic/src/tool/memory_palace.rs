//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.
use sqlx::{PgPool, Postgres, Transaction}; // add Transaction, Postgres

mod tool;

/// [`MemoryPalace`] models and types.
mod models;
use models::*;

/// [`MemoryPalace`] Database initialization.
mod db;
use db::ensure_initialized;

/// [`MemoryPalace`] service implementation.
mod service;
use service::*;

/// [`MemoryPalace`] error handling.
mod error;
pub use error::MemoryPalaceError;

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

/// A Memory Palace knowledge base for AI agents.
///
/// Designed by Claude 4, Sonnet (Copilot), guided by Michael de Gans.
#[derive(Debug)]
pub struct MemoryPalace {
    /// PostgreSQL connection pool.
    pub(crate) pool: PgPool,
    /// The schema name to use for all operations.
    pub(crate) schema_name: String,
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool. Uses the
    /// default 'public' schema.
    pub async fn from_pool(pool: PgPool) -> Result<Self, MemoryPalaceError> {
        Self::from_pool_with_schema(pool, "public".to_string()).await
    }

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool with a
    /// specific schema.
    pub async fn from_pool_with_schema(
        pool: PgPool,
        schema_name: String,
    ) -> Result<Self, MemoryPalaceError> {
        let new = Self { pool, schema_name };
        ensure_initialized(&new.pool, &new.schema_name).await?;

        Ok(new)
    }

    /// Store a [`Memory`] in a specific [`Room`].
    pub(crate) async fn store_memory(
        &mut self,
        room: impl Into<String>,
        content: impl Into<String>,
        tags: impl IntoIterator<Item = &str>,
    ) -> Result<i64, MemoryPalaceError> {
        store(
            &self.pool,
            &self.schema_name,
            room.into(),
            content.into(),
            tags.into_iter().map(|s| s.to_string()).collect(),
        )
        .await
    }

    /// Search for [`Memory`]s using blended scoring that combines relevance,
    /// recency, and relationships.
    pub(crate) async fn search(
        &mut self,
        query: &str,
    ) -> Result<Vec<(String, String, Memory)>, MemoryPalaceError> {
        search(&self.pool, &self.schema_name, query).await
    }

    /// Find [`Memory`]s using BFS with decay factor for distance.
    pub(crate) async fn find_memories_bfs(
        &mut self,
        start_memory_id: i64,
        max_distance: u32,
        decay_factor: f64,
        min_score: f64,
    ) -> Result<Vec<(String, String, Memory, f64, i32)>, MemoryPalaceError>
    {
        find_memories_bfs(
            &self.pool,
            &self.schema_name,
            start_memory_id,
            max_distance,
            decay_factor,
            min_score,
        )
        .await
    }

    /// Connect two [`Room`] in the palace.
    pub(crate) async fn connect_rooms(
        &mut self,
        room1: impl Into<String>,
        room2: impl Into<String>,
    ) -> Result<(), MemoryPalaceError> {
        let room1 = room1.into();
        let room2 = room2.into();
        connect_rooms(&self.pool, &self.schema_name, room1, room2).await
    }

    /// List all [`Room`]s with their [`Memory`] counts and [`Connection`]s.
    pub(crate) async fn list_rooms(
        &mut self,
    ) -> Result<Vec<(String, String, usize, Vec<String>)>, MemoryPalaceError>
    {
        list_rooms(&self.pool, &self.schema_name).await
    }

    /// Create a [`RelatedMemory`] between two [`Memory`]s with a specified
    /// relationship type and strength.
    pub(crate) async fn relate_memories(
        &mut self,
        memory_id1: i64,
        memory_id2: i64,
        relationship_type: impl Into<String>,
        strength: f64,
    ) -> Result<String, MemoryPalaceError> {
        let relationship_type = relationship_type.into();
        relate_memories(
            &self.pool,
            &self.schema_name,
            memory_id1,
            memory_id2,
            relationship_type,
            strength,
        )
        .await
    }

    /// Find [`Memory`]s related to a specific [`Memory`] with a maximum depth
    pub(crate) async fn find_related_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, Memory, String, f64)>, MemoryPalaceError>
    {
        find_related_memories(
            &self.pool,
            &self.schema_name,
            memory_id,
            max_depth,
            min_strength,
        )
        .await
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
    ) -> Result<Vec<(String, String, Memory, f64)>, MemoryPalaceError> {
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
}
