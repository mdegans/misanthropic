use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sqlx::FromRow;

/// A room in the memory palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Room {
    pub id: i64,
    pub name: String,
    pub description: String,
    pub atmosphere: Option<String>,
    pub centroid_embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
}
/// A memory item stored in the palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Memory {
    pub id: i64,
    pub content: String,
    pub room: String,
    pub placement: String,
    pub placement_description: Option<String>,
    #[sqlx(json)]
    pub tags: Vec<String>,
    pub embedding: Option<Vec<f32>>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
}

/// A connection between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Connection {
    pub id: i64,
    pub from_room: String,
    pub to_room: String,
    pub passage_type: String,
    pub description: Option<String>,
    pub strength: i32,
    pub created_at: DateTime<Utc>,
}

/// Helper struct for room listing with memory count
#[derive(Debug, Clone, FromRow)]
pub struct RoomWithCount {
    pub name: String,
    pub description: String,
    pub memory_count: i64,
}

/// Helper struct for connection listing
#[derive(Debug, Clone, FromRow)]
pub struct RoomConnection {
    pub to_room: String,
}

/// Helper struct for recent memories summary
#[derive(Debug, Clone, FromRow)]
pub struct RecentMemory {
    pub content: String,
    pub room: String,
    pub tags: Value,
    pub created_at: DateTime<Utc>,
}

/// Helper struct for top relationships summary
#[derive(Debug, Clone, FromRow)]
pub struct TopRelationship {
    pub from_content: String,
    pub to_content: String,
    pub relationship_type: String,
    pub strength: f64,
}

/// Helper struct for blended search results
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub room: String,
    pub relevance_score: f64,
    pub recency_score: f64,
    pub relationship_score: f64,
    pub final_score: f64,
}
