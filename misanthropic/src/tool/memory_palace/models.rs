use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::FromRow;

/// A memory item stored in the palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub(crate) struct Memory {
    pub(crate) id: i64,
    pub(crate) content: String,
    pub(crate) room: String,
    #[sqlx(json)]
    pub(crate) tags: Vec<String>,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_updated: DateTime<Utc>,
}

/// A room in the memory palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub(crate) struct Room {
    pub(crate) id: i64,
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) created_at: DateTime<Utc>,
}

/// A connection between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub(crate) struct Connection {
    pub(crate) id: i64,
    pub(crate) from_room: String,
    pub(crate) to_room: String,
    pub(crate) description: Option<String>,
    pub(crate) strength: i32,
    pub(crate) created_at: DateTime<Utc>,
}

/// Helper struct for room listing with memory count
#[derive(Debug, Clone, FromRow)]
pub(crate) struct RoomWithCount {
    pub(crate) name: String,
    pub(crate) description: String,
    pub(crate) memory_count: i64,
}

/// Helper struct for connection listing
#[derive(Debug, Clone, FromRow)]
pub(crate) struct RoomConnection {
    pub(crate) to_room: String,
}

/// Helper struct for memory relationships
#[derive(Debug, Clone, FromRow)]
pub(crate) struct RelatedMemory {
    pub(crate) id: i64,
    pub(crate) content: String,
    pub(crate) room: String,
    pub(crate) tags: Value,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_updated: DateTime<Utc>,
    pub(crate) relationship_type: String,
    pub(crate) strength: f64,
}

/// Helper struct for concept-based memory search
#[derive(Debug, Clone, FromRow)]
pub(crate) struct ConceptMemory {
    pub(crate) id: i64,
    pub(crate) content: String,
    pub(crate) room: String,
    pub(crate) tags: Value,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_updated: DateTime<Utc>,
    pub(crate) confidence: f64,
}

/// Helper struct for graph statistics
#[derive(Debug, Clone, FromRow)]
pub(crate) struct GraphStats {
    pub(crate) total_memories: i64,
    pub(crate) total_rooms: i64,
    pub(crate) total_relationships: i64,
    pub(crate) total_concepts: i64,
    pub(crate) total_mentions: i64,
}

/// Helper struct for recent memories summary
#[derive(Debug, Clone, FromRow)]
pub(crate) struct RecentMemory {
    pub(crate) content: String,
    pub(crate) room: String,
    pub(crate) tags: Value,
    pub(crate) created_at: DateTime<Utc>,
}

/// Helper struct for top relationships summary
#[derive(Debug, Clone, FromRow)]
pub(crate) struct TopRelationship {
    pub(crate) from_content: String,
    pub(crate) to_content: String,
    pub(crate) relationship_type: String,
    pub(crate) strength: f64,
}

/// Helper struct for BFS memory discovery
#[derive(Debug, Clone, FromRow)]
pub(crate) struct BfsMemory {
    pub(crate) id: i64,
    pub(crate) content: String,
    pub(crate) room: String,
    pub(crate) tags: Value,
    pub(crate) created_at: DateTime<Utc>,
    pub(crate) last_updated: DateTime<Utc>,
    pub(crate) distance: i32,
    pub(crate) path_strength: f64,
}

/// Helper struct for blended search results
#[derive(Debug, Clone)]
pub(crate) struct ScoredMemory {
    pub(crate) memory: Memory,
    pub(crate) room: String,
    pub(crate) relevance_score: f64,
    pub(crate) recency_score: f64,
    pub(crate) relationship_score: f64,
    pub(crate) final_score: f64,
}
