use crate::{request, Model};
use serde::{Deserialize, Serialize};

/// A [`request::Message`] with additional response metadata.
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[display("{}", message)]
pub struct Message {
    /// Unique `id` for the message.
    pub id: String,
    /// Inner [`request::Message`].
    #[serde(flatten)]
    pub message: request::Message,
    /// [`Model`] that generated the message.
    pub model: Model,
    /// The reason the model stopped generating tokens.
    pub stop_reason: Option<StopReason>,
    /// If the [`StopReason`] was [`StopSequence`], this is the sequence that
    /// triggered it.
    ///
    /// [`StopSequence`]: StopReason::StopSequence
    pub stop_sequence: Option<String>,
    /// Usage statistics for the message.
    pub usage: Usage,
}

/// Reason the model stopped generating tokens.
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// The model reached a natural stopping point.
    EndTurn,
    /// Maximum tokens reached.
    MaxTokens,
    /// A stop sequence was generated.
    StopSequence,
    /// A tool was used.
    ToolUse,
}

/// Usage statistics from the API. This is used in multiple contexts, not just
/// for messages.
#[derive(Debug, Serialize, Deserialize)]
pub struct Usage {
    /// Number of input tokens used.
    pub input_tokens: u64,
    /// Number of input tokens used to create the cache entry.
    #[cfg(feature = "prompt-caching")]
    pub cache_creation_input_tokens: Option<u64>,
    /// Number of input tokens read from the cache.
    #[cfg(feature = "prompt-caching")]
    pub cache_read_input_tokens: Option<u64>,
    /// Number of output tokens generated.
    pub output_tokens: u64,
}
