use crate::tool::embedding::EmbeddingError;
use crate::tool::memory_palace::{MemoryId, RoomId};
use serde_json;
use sqlx;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryPalaceError {
    #[error("Database error: {0}")]
    Db(#[from] sqlx::Error),

    #[error("JSON serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),

    #[error("Many errors occurred.")]
    Many(Vec<MemoryPalaceError>),

    #[error("Not found: {0:?}")]
    RoomNotFound(RoomId),

    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),

    #[error("Room ID not found: {0}")]
    RoomIdNotFound(RoomId),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Embedding error: {0}")]
    Embedding(#[from] EmbeddingError),
}
