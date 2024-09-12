use crate::{request, stream::MessageDelta, Model};
use serde::{Deserialize, Serialize};

/// A [`request::Message`] with additional response metadata.
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
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

impl Message {
    /// Apply a [`MessageDelta`] with metadata to the message.
    pub fn apply_delta(&mut self, delta: MessageDelta) {
        self.stop_reason = delta.stop_reason;
        self.stop_sequence = delta.stop_sequence;
        if let Some(usage) = delta.usage {
            self.usage = usage;
        }
    }

    /// Get the [`tool::Use`] from the message if the [`StopReason`] was
    /// [`StopReason::ToolUse`] and the final message [`Content`] [`Block`] is
    /// [`ToolUse`].
    ///
    /// [`Content`]: crate::request::message::Content
    /// [`Block`]: crate::request::message::Block
    /// [`tool::Use`]: crate::tool::Use
    /// [`ToolUse`]: crate::request::message::Block::ToolUse
    pub fn tool_use(&self) -> Option<&crate::tool::Use> {
        if !matches!(self.stop_reason, Some(StopReason::ToolUse)) {
            return None;
        }

        self.message.content.last()?.tool_use()
    }
}

/// Reason the model stopped generating tokens.
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
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
#[derive(Debug, Serialize, Deserialize, Default)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
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
