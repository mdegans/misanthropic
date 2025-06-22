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
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";
    const EMBEDDING_SIZE: usize = 1536;

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
        };
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
    pub(crate) async fn find_resonating_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, Memory, String, f64)>, MemoryPalaceError>
    {
        find_resonating_memories(
            &self.pool,
            &self.schema_name,
            memory_id,
            max_depth,
            min_strength,
        )
        .await
    }

    /// Semantic search across all rooms using an embedding.
    pub(crate) async fn semantic_search_all_rooms(
        &mut self,
        embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Memory>, MemoryPalaceError> {
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
    ) -> Result<Vec<(String, String, u32)>, MemoryPalaceError> {
        get_rooms_within_radius(
            &self.pool,
            &self.schema_name,
            start_room,
            radius,
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

    /// Get all memories in a specific room
    pub async fn get_room_memories(
        &self,
        room_name: &str,
    ) -> Result<Vec<Memory>, MemoryPalaceError> {
        // Query memories WHERE room = room_name
        // Format with placement info from tags
        crate::tool::memory_subroutine::navigator::get_room_memories(
            &self.pool,
            &self.schema_name,
            room_name,
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
        radius: u32,
        mission: Option<&String>,
    ) -> Result<Vec<(String, String, f32)>, MemoryPalaceError> {
        // Use room_connections + centroid embeddings
        // Return (direction, room_name, distance_meters)
        crate::tool::memory_subroutine::navigator::get_adjacent_rooms_sorted(
            &self.pool,
            &self.schema_name,
            current_room,
            radius,
            mission,
        )
        .await
    }

    /// Follow a passage to get the destination room
    pub async fn follow_passage(
        &self,
        from_room: &str,
        direction: &str,
    ) -> Result<Room, MemoryPalaceError> {
        // Parse direction, find matching connection
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
        room_name: String,
    ) -> Result<String, MemoryPalaceError> {
        // Format room with memory count, connections, atmosphere
        crate::tool::memory_subroutine::navigator::get_room_description(
            &self.pool,
            &self.schema_name,
            room_name,
        )
        .await
    }
}
