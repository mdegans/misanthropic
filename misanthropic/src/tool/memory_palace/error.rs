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

    #[error("Room not found: {0}")]
    RoomNotFound(String),
}
