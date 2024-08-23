//! [`Event`] [`Stream`] for streaming responses from the API as well as
//! associated types and errors only used when streaming.
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::pin::Pin;

use crate::{
    client::AnthropicError,
    response::{self, StopReason, Usage},
};

/// Sucessful Event from the API. See [`stream::Error`] for errors.
///
/// [`stream::Error`]: Error
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Event {
    /// Periodic ping.
    Ping,
    /// [`response::Message`] with empty content. The deltas arrive in
    /// [`ContentBlock`]s.
    MessageStart {
        /// The message.
        message: response::Message,
    },
    /// Content block with empty content.
    ContentBlockStart {
        /// Index of the content block.
        index: usize,
        /// Empty content block.
        content_block: ContentBlock,
    },
    /// Content block delta.
    ContentBlockDelta {
        /// Index of the content block.
        index: usize,
        /// Delta to apply to the content block.
        delta: ContentBlock,
    },
    /// Content block end.
    ContentBlockStop {
        /// Index of the content block.
        index: usize,
    },
    /// Message delta. Confusingly this does not contain message content rather
    /// metadata about the message in progress.
    MessageDelta {
        /// Delta to apply to the [`response::Message`].
        delta: MessageDelta,
    },
    /// Message end.
    MessageStop,
}

/// Internal enum for the API result so we don't have to add an error variant to
/// the `Event` enum.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ApiResult {
    /// Successful Event.
    Event {
        #[serde(flatten)]
        event: Event,
    },
    /// Error Event.
    Error { error: AnthropicError },
}

/// A content block or delta. This can be [`Text`], [`Json`], or [`Tool`] use.
///
/// [`Text`]: ContentBlock::Text
/// [`Json`]: ContentBlock::Json
/// [`Tool`]: ContentBlock::Tool
#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ContentBlock {
    /// Text content.
    #[serde(alias = "text_delta")]
    Text {
        /// The text content.
        text: String,
    },
    /// JSON delta.
    #[serde(rename = "input_json_delta")]
    Json {
        /// The JSON delta.
        partial_json: String,
    },
    /// Tool use.
    #[serde(rename = "tool_use")]
    Tool {
        /// ID of the request.
        id: String,
        /// Name of the tool.
        name: String,
        /// Input to the tool.
        input: serde_json::Value,
    },
}

/// Error when applying a [`ContentBlock`] delta to a target [`ContentBlock`].
#[derive(Serialize, thiserror::Error, Debug)]
#[error("Cannot apply delta {from:?} to {to:?}.")]
pub struct ContentMismatch<'a> {
    /// The content block that failed to apply.
    from: ContentBlock,
    /// The target content block.
    to: &'a ContentBlock,
}

impl ContentBlock {
    /// Apply a [`ContentBlock`] delta to self.
    pub fn append(
        &mut self,
        delta: ContentBlock,
    ) -> Result<(), ContentMismatch> {
        match (self, delta) {
            (
                ContentBlock::Text { text },
                ContentBlock::Text { text: delta },
            ) => {
                text.push_str(&delta);
            }
            (
                ContentBlock::Json { partial_json },
                ContentBlock::Json {
                    partial_json: delta,
                },
            ) => {
                partial_json.push_str(&delta);
            }
            (to, from) => {
                return Err(ContentMismatch { from, to });
            }
        }

        Ok(())
    }
}

/// Metadata about a message in progress. This does not contain actual text
/// deltas. That's the [`ContentBlock`] in [`Event::ContentBlockDelta`].
#[derive(Debug, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Stop reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<String>,
    /// Token usage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Stream error. This can be JSON parsing errors or errors from the API.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// [`eventsource_stream::EventStreamError`] wrapping a [`reqwest::Error`].
    #[error("HTTP error: {error}")]
    Stream {
        #[from]
        /// Error from the `eventsource_stream` crate.
        error: eventsource_stream::EventStreamError<reqwest::Error>,
    },
    /// JSON parsing error.
    #[error("JSON error: {error}")]
    Parse {
        /// Error from [`serde_json`].
        error: serde_json::Error,
        /// [`eventsource_stream::Event`] that did not parse.
        event: eventsource_stream::Event,
    },
    /// Error from the API.
    #[error("API error: {error}")]
    Anthropic {
        /// Error from the API.
        error: AnthropicError,
        /// [`eventsource_stream::Event`] containing the error.
        event: eventsource_stream::Event,
    },
}

/// Stream of [`Event`]s or [`Error`]s.
pub struct Stream {
    inner: Pin<Box<dyn futures::Stream<Item = Result<Event, Error>>>>,
}

impl Stream {
    /// Create a new stream from an [`eventsource_stream::EventStream`] or
    /// similar stream of [`eventsource_stream::Event`]s.
    pub fn new<S>(stream: S) -> Self
    where
        S: futures::Stream<
                Item = Result<
                    eventsource_stream::Event,
                    eventsource_stream::EventStreamError<reqwest::Error>,
                >,
            > + 'static,
    {
        Self {
            inner: Box::pin(stream.map(|event| match event {
                Ok(event) => {
                    #[cfg(feature = "log")]
                    log::trace!("Event: {:?}", event);

                    match serde_json::from_str::<ApiResult>(&event.data) {
                        Ok(ApiResult::Event { event }) => Ok(event),
                        Ok(ApiResult::Error { error }) => {
                            Err(Error::Anthropic { error, event })
                        }
                        Err(error) => Err(Error::Parse { error, event }),
                    }
                }
                Err(error) => {
                    #[cfg(feature = "log")]
                    log::error!("Stream error: {:?}", error);
                    Err(Error::Stream { error })
                }
            })),
        }
    }

    /// Filter out rate limit and overload errors. Because the server sends
    /// these events there isn't a need to retry or backoff. The stream will
    /// continue when ready.
    ///
    /// This is recommended for most use cases.
    pub fn filter_rate_limit(self) -> Self {
        Self {
            inner: Box::pin(self.inner.filter_map(|result| async move {
                match result {
                    Ok(event) => Some(Ok(event)),
                    Err(Error::Anthropic {
                        error:
                            AnthropicError::Overloaded { .. }
                            | AnthropicError::RateLimit { .. },
                        ..
                    }) => None,
                    Err(error) => Some(Err(error)),
                }
            })),
        }
    }

    /// Filter out everything but [`Event::ContentBlockDelta`]. This can include
    /// text, JSON, and tool use.
    pub fn deltas(
        self,
    ) -> impl futures::Stream<Item = Result<ContentBlock, Error>> {
        self.inner.filter_map(|result| async move {
            match result {
                Ok(Event::ContentBlockDelta { delta, .. }) => Some(Ok(delta)),
                _ => None,
            }
        })
    }

    /// Filter out everything but text pieces.
    pub fn text(self) -> impl futures::Stream<Item = Result<String, Error>> {
        self.deltas().filter_map(|result| async move {
            match result {
                Ok(ContentBlock::Text { text }) => Some(Ok(text)),
                _ => None,
            }
        })
    }
}

impl futures::Stream for Stream {
    type Item = Result<Event, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    // Actual JSON from the API.

    pub const CONTENT_BLOCK_START: &str = "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"} }";
    pub const CONTENT_BLOCK_DELTA: &str = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Certainly! I\"}     }";

    #[test]
    fn test_content_block_start() {
        let event: Event = serde_json::from_str(CONTENT_BLOCK_START).unwrap();
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                assert_eq!(
                    content_block,
                    ContentBlock::Text { text: "".into() }
                );
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    fn test_content_block_delta() {
        let event: Event = serde_json::from_str(CONTENT_BLOCK_DELTA).unwrap();
        match event {
            Event::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert_eq!(
                    delta,
                    ContentBlock::Text {
                        text: "Certainly! I".into()
                    }
                );
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }
}
