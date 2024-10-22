use std::borrow::Cow;

use crate::{prompt, stream::MessageDelta, Model};
use serde::{Deserialize, Serialize};

/// A [`prompt::message`] with additional response metadata.
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[display("{}", message)]
pub struct Message<'a> {
    /// Unique `id` for the message.
    pub id: Cow<'a, str>,
    /// Inner [`prompt::message`].
    #[serde(flatten)]
    pub message: prompt::Message<'a>,
    /// [`Model`] that generated the message.
    pub model: Model,
    /// The reason the model stopped generating tokens.
    pub stop_reason: Option<StopReason>,
    /// If the [`StopReason`] was [`StopSequence`], this is the sequence that
    /// triggered it.
    ///
    /// [`StopSequence`]: StopReason::StopSequence
    pub stop_sequence: Option<Cow<'a, str>>,
    /// Usage statistics for the message.
    pub usage: Usage,
}

impl Message<'_> {
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
    /// [`Content`]: crate::prompt::message::Content
    /// [`Block`]: crate::prompt::message::Block
    /// [`tool::Use`]: crate::tool::Use
    /// [`ToolUse`]: crate::prompt::message::Block::ToolUse
    pub fn tool_use(&self) -> Option<&crate::tool::Use> {
        if !matches!(self.stop_reason, Some(StopReason::ToolUse)) {
            return None;
        }

        self.message.content.last()?.tool_use()
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> Message<'static> {
        Message {
            id: Cow::Owned(self.id.into_owned()),
            message: self.message.into_static(),
            model: self.model,
            stop_reason: self.stop_reason,
            stop_sequence: self
                .stop_sequence
                .map(|s| Cow::Owned(s.into_owned())),
            usage: self.usage,
        }
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

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Message<'_> {
    fn markdown_events_custom<'a>(
        &'a self,
        options: &'a crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        self.message.markdown_events_custom(options)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // FIXME: This is Copilot generated JSON. It should be replaced with actual
    // response JSON, however this is pretty close to what the actual JSON looks
    // like.
    pub const RESPONSE_JSON: &str = r#"{
    "content": [
        {
        "text": "Hi! My name is Claude.",
        "type": "text"
        }
    ],
    "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
    "model": "claude-3-5-sonnet-20240620",
    "role": "assistant",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "type": "message",
    "usage": {
        "input_tokens": 2095,
        "output_tokens": 503
    }
}"#;

    #[test]
    fn deserialize_response_message() {
        let message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        assert_eq!(message.message.content.len(), 1);
        assert_eq!(message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(message.model, crate::Model::Sonnet35_20240620);
        assert!(matches!(message.stop_reason, Some(StopReason::EndTurn)));
        assert_eq!(message.stop_sequence, None);
        assert_eq!(message.usage.input_tokens, 2095);
        assert_eq!(message.usage.output_tokens, 503);
    }

    #[test]
    fn test_apply_delta() {
        let mut message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        let delta = MessageDelta {
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: Some("sequence".into()),
            usage: Some(Usage {
                input_tokens: 100,
                output_tokens: 200,
                ..Default::default()
            }),
        };

        message.apply_delta(delta);

        assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(message.stop_sequence, Some("sequence".into()));
        assert_eq!(message.usage.input_tokens, 100);
        assert_eq!(message.usage.output_tokens, 200);
    }

    #[test]
    fn test_tool_use() {
        let mut message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        assert!(message.tool_use().is_none());

        message.stop_reason = Some(StopReason::ToolUse);
        assert!(message.tool_use().is_none());

        message.message.content.push(crate::tool::Use {
            id: "id".into(),
            name: "name".into(),
            input: serde_json::json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        });
        assert!(message.tool_use().is_some());
    }
}
