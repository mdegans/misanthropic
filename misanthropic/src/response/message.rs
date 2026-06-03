use std::borrow::Cow;

use crate::{model, prompt, stream::MessageDelta};
use serde::{Deserialize, Serialize};

/// Errors returned by [`Message::json`].
///
/// Distinguishes between the four non-happy outcomes of parsing
/// structured output: the model refused, the model called a tool instead
/// of emitting text, the message had no text block to parse, or the text
/// block failed to deserialize.
#[derive(Debug, thiserror::Error)]
pub enum JsonError {
    /// The response has [`StopReason::Refusal`]: the model declined to
    /// produce structured output.
    #[error("model refused to produce structured output")]
    Refusal,
    /// The response has [`StopReason::ToolUse`]: no text block is
    /// available to parse.
    #[error("response is a tool_use, not structured output")]
    ToolUse,
    /// The message contains no [`Text`] [`Block`].
    ///
    /// [`Text`]: crate::prompt::message::Block::Text
    /// [`Block`]: crate::prompt::message::Block
    #[error("response contains no text block")]
    NoTextBlock,
    /// The text block failed to deserialize into the target type.
    #[error("failed to deserialize structured output: {0}")]
    Json(#[from] serde_json::Error),
}

/// A [`prompt::Message`] with additional response metadata.
#[derive(Clone, Debug, Serialize, Deserialize, derive_more::Display)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[display("{}", inner)]
pub struct Message<'a> {
    /// Unique `id` for the message.
    pub id: Cow<'a, str>,
    /// Inner [`prompt::message`].
    #[serde(flatten)]
    pub inner: prompt::AssistantMessage<'a>,
    /// [`Model`] that generated the message.
    pub model: model::Id<'a>,
    /// The reason the model stopped generating tokens.
    pub stop_reason: Option<StopReason>,
    /// If the [`StopReason`] was [`StopSequence`], this is the sequence that
    /// triggered it.
    ///
    /// [`StopSequence`]: StopReason::StopSequence
    pub stop_sequence: Option<Cow<'a, str>>,
    /// Usage statistics for the message.
    #[serde(default)]
    pub usage: Usage,
}

impl Message<'_> {
    /// Apply a [`MessageDelta`] with metadata to the message.
    pub fn apply_delta(&mut self, delta: MessageDelta) {
        if let Some(stop_reason) = delta.stop_reason {
            self.stop_reason = Some(stop_reason);
        }
        if let Some(stop_sequence) = delta.stop_sequence {
            self.stop_sequence = Some(stop_sequence);
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
    pub fn tool_use(&self) -> Option<&crate::tool::Use<'_>> {
        if !matches!(self.stop_reason, Some(StopReason::ToolUse)) {
            return None;
        }

        self.inner.content.last()?.tool_use()
    }

    /// Parse the first [`Text`] [`Block`] as JSON into `T`, skipping any
    /// leading [`Thought`] / [`RedactedThought`] blocks produced by
    /// [Extended Thinking]. Intended for use with
    /// [`Prompt::output_config`]: when structured output is enabled, the
    /// response is guaranteed to contain exactly one JSON text block
    /// matching the supplied schema.
    ///
    /// Returns [`JsonError::Refusal`] / [`JsonError::ToolUse`] /
    /// [`JsonError::NoTextBlock`] for non-text outcomes so callers can
    /// handle [`Refusal`] explicitly without inspecting [`stop_reason`].
    ///
    /// [`Text`]: crate::prompt::message::Block::Text
    /// [`Thought`]: crate::prompt::message::Block::Thought
    /// [`RedactedThought`]: crate::prompt::message::Block::RedactedThought
    /// [`Block`]: crate::prompt::message::Block
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    /// [`Refusal`]: StopReason::Refusal
    /// [`stop_reason`]: Message::stop_reason
    /// [Extended Thinking]: <https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking>
    pub fn json<T>(&self) -> Result<T, JsonError>
    where
        T: serde::de::DeserializeOwned,
    {
        use crate::prompt::message::Block;

        match self.stop_reason {
            Some(StopReason::Refusal) => return Err(JsonError::Refusal),
            Some(StopReason::ToolUse) => return Err(JsonError::ToolUse),
            _ => {}
        }

        let text: &str = self
            .inner
            .content
            .iter()
            .find_map(|b| match b {
                Block::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .ok_or(JsonError::NoTextBlock)?;

        Ok(serde_json::from_str(text)?)
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> Message<'static> {
        Message {
            id: Cow::Owned(self.id.into_owned()),
            inner: self.inner.into_static(),
            model: self.model.into_static(),
            stop_reason: self.stop_reason,
            stop_sequence: self
                .stop_sequence
                .map(|s| Cow::Owned(s.into_owned())),
            usage: self.usage,
        }
    }

    /// Remove an incomplete thought from the message. If after removal, the
    /// message is empty, `None` is returned.
    ///
    /// See also [`prompt::Message::remove_incomplete_thought`].
    pub fn remove_incomplete_thought(self) -> Option<Self> {
        let inner = self.inner.remove_incomplete_thought()?;
        Some(Self { inner, ..self })
    }
}

/// Reason the model stopped generating tokens.
#[derive(
    Clone, Copy, Debug, Serialize, Deserialize, derive_more::IsVariant,
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
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
    /// The model refused to produce output, typically due to a conflict
    /// between the request and its safety constraints. When this occurs
    /// with [`Prompt::output_config`], the response body may not match
    /// the requested schema; see [Anthropic docs on invalid outputs].
    ///
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    /// [Anthropic docs on invalid outputs]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#invalid-outputs>
    Refusal,
}

/// Usage statistics from the API. This is used in multiple contexts, not just
/// for messages.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Usage {
    /// Number of input tokens used.
    pub input_tokens: u64,
    /// Number of input tokens used to create the cache entry.
    pub cache_creation_input_tokens: Option<u64>,
    /// Number of input tokens read from the cache.
    pub cache_read_input_tokens: Option<u64>,
    /// Number of output tokens generated.
    pub output_tokens: u64,
}

impl std::ops::Add<Usage> for Usage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .map(|c| c + rhs.cache_creation_input_tokens.unwrap_or(0))
                .or(rhs.cache_creation_input_tokens),
            cache_read_input_tokens: self
                .cache_read_input_tokens
                .map(|c| c + rhs.cache_read_input_tokens.unwrap_or(0))
                .or(rhs.cache_read_input_tokens),
            output_tokens: self.output_tokens + rhs.output_tokens,
        }
    }
}

impl std::ops::AddAssign<Usage> for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self = *self + rhs;
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Message<'a> {
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        self.inner.markdown_events_custom(options)
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
        assert_eq!(message.inner.content.len(), 1); // single block
        assert_eq!(message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(message.model, crate::AnthropicModel::Sonnet35_20240620);
        assert!(matches!(message.stop_reason, Some(StopReason::EndTurn)));
        assert_eq!(message.stop_sequence, None);
    }

    #[test]
    fn test_apply_delta() {
        let mut message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        let delta = MessageDelta {
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: Some("sequence".into()),
        };

        message.apply_delta(delta);

        assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
        assert_eq!(message.stop_sequence, Some("sequence".into()));
    }

    #[test]
    fn test_tool_use() {
        let mut message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        assert!(message.tool_use().is_none());

        message.stop_reason = Some(StopReason::ToolUse);
        assert!(message.tool_use().is_none());

        message.inner.content_mut().push(crate::tool::Use {
            id: "id".into(),
            name: "name".into(),
            input: serde_json::json!({}),
            cache_control: None,
        });
        assert!(message.tool_use().is_some());
    }

    #[test]
    fn test_into_static() {
        // Refers to json:
        let message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        // Owns the `Cow` fields:
        let static_message = message.into_static();

        assert_eq!(static_message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(
            static_message.model,
            crate::AnthropicModel::Sonnet35_20240620
        );
        assert!(matches!(
            static_message.stop_reason,
            Some(StopReason::EndTurn)
        ));
        assert_eq!(static_message.stop_sequence, None);
        assert_eq!(static_message.usage.input_tokens, 2095);
        assert_eq!(static_message.usage.output_tokens, 503);
    }

    #[test]
    fn stop_reason_refusal_deserializes() {
        let sr: StopReason = serde_json::from_str("\"refusal\"").unwrap();
        assert!(matches!(sr, StopReason::Refusal));
        // And roundtrips.
        assert_eq!(serde_json::to_string(&sr).unwrap(), "\"refusal\"");
    }

    #[derive(serde::Deserialize, PartialEq, Debug)]
    struct VoteIntent {
        post_id: String,
        support: bool,
    }

    fn message_with(
        stop_reason: Option<StopReason>,
        content: prompt::message::Content<'static>,
    ) -> Message<'static> {
        Message {
            id: "id".into(),
            inner: prompt::AssistantMessage {
                inner: prompt::Message {
                    role: prompt::message::Role::Assistant,
                    content,
                },
            },
            model: crate::AnthropicModel::Sonnet35.into(),
            stop_reason,
            stop_sequence: None,
            usage: Usage::default(),
        }
    }

    #[test]
    fn json_parses_single_part_text() {
        let message = message_with(
            Some(StopReason::EndTurn),
            prompt::message::Content::text(
                r#"{"post_id":"abc","support":true}"#,
            ),
        );
        let parsed: VoteIntent = message.json().unwrap();
        assert_eq!(
            parsed,
            VoteIntent {
                post_id: "abc".into(),
                support: true
            }
        );
    }

    #[test]
    fn json_skips_leading_thought_blocks() {
        use prompt::message::{Block, Content};
        let content = Content(vec![
            Block::Thought {
                thought: "Let me think about this...".into(),
                signature: "sig".into(),
            },
            Block::Text {
                text: r#"{"post_id":"abc","support":false}"#.into(),
                citations: None,
                cache_control: None,
            },
        ]);
        let message = message_with(Some(StopReason::EndTurn), content);
        let parsed: VoteIntent = message.json().unwrap();
        assert_eq!(
            parsed,
            VoteIntent {
                post_id: "abc".into(),
                support: false
            }
        );
    }

    #[test]
    fn json_returns_refusal_on_refusal_stop() {
        let message = message_with(
            Some(StopReason::Refusal),
            prompt::message::Content::text("I can't help with that."),
        );
        let err = message.json::<VoteIntent>().unwrap_err();
        assert!(matches!(err, JsonError::Refusal));
    }

    #[test]
    fn json_returns_tool_use_on_tool_stop() {
        let message = message_with(
            Some(StopReason::ToolUse),
            prompt::message::Content::text(""),
        );
        let err = message.json::<VoteIntent>().unwrap_err();
        assert!(matches!(err, JsonError::ToolUse));
    }

    #[test]
    fn json_returns_no_text_block_when_only_tool_blocks() {
        use prompt::message::{Block, Content};
        let content = Content(vec![Block::ToolUse {
            call: crate::tool::Use {
                id: "u".into(),
                name: "x".into(),
                input: serde_json::json!({}),
                cache_control: None,
            },
        }]);
        // stop_reason == EndTurn so we pass the stop-reason guard and hit
        // the no-text-block path.
        let message = message_with(Some(StopReason::EndTurn), content);
        let err = message.json::<VoteIntent>().unwrap_err();
        assert!(matches!(err, JsonError::NoTextBlock));
    }

    #[test]
    fn json_propagates_serde_errors() {
        let message = message_with(
            Some(StopReason::EndTurn),
            prompt::message::Content::text("not json"),
        );
        let err = message.json::<VoteIntent>().unwrap_err();
        assert!(matches!(err, JsonError::Json(_)));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_markdown() {
        use crate::markdown::ToMarkdown;

        let message = Message {
            id: "id".into(),
            inner: prompt::AssistantMessage {
                inner: prompt::Message {
                    role: prompt::message::Role::User,
                    content: prompt::message::Content::text(
                        "Hello, **world**!",
                    ),
                },
            },
            model: crate::AnthropicModel::Sonnet35.into(),
            stop_reason: None,
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(3),
                output_tokens: 4,
            },
        };

        let expected = "### User\n\nHello, **world**!";
        let markdown = message.markdown();
        assert_eq!(markdown.as_ref(), expected);
    }
}
