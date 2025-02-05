//! [`Event`] [`Stream`] for streaming responses from the API as well as
//! associated types and errors only used when streaming.
use crate::tool;
#[allow(unused_imports)] // `Content`, `request` Used in docs.
use crate::{
    client::AnthropicError,
    prompt::{
        self,
        message::{Block, Content},
    },
    response::{self, StopReason, Usage},
};
use futures::{pin_mut, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{borrow::Cow, pin::Pin, task::Poll};

/// Sucessful Event from the API. See [`stream::Error`] for errors.
///
/// [`stream::Error`]: Error
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Event {
    /// Periodic ping.
    Ping,
    /// [`response::Message`] with empty content. [`MessageDelta`] and
    /// [`Content`] [`Delta`]s must be applied to this message.
    MessageStart {
        /// The message.
        message: response::Message<'static>,
    },
    /// [`Content`] [`Block`] with empty content.
    ContentBlockStart {
        /// Index of the [`Content`] [`Block`] in [`prompt::message::Content`].
        // TODO: Indexing. Issue is the Content::SinglePart is a String and
        // Content::MultiPart is a Vec of Block. This is for serialization
        // purposes. We should probably just use a Vec for both and write a
        // custom serializer for that field.
        index: usize,
        /// Empty content block.
        content_block: Block<'static>,
    },
    /// Content block delta.
    ContentBlockDelta {
        /// Index of the [`Content`] [`Block`] in [`prompt::message::Content`].
        index: usize,
        /// Delta to apply to the content block.
        delta: Delta<'static>,
    },
    /// Content block end.
    ContentBlockStop {
        /// Index of the [`Content`] [`Block`] in [`prompt::message::Content`].
        index: usize,
    },
    /// [`MessageDelta`]. Contains metadata, not [`Content`] [`Delta`]s. Apply
    /// to the [`response::Message`].
    MessageDelta {
        /// Delta to apply to the [`response::Message`].
        delta: MessageDelta,
    },
    /// Message end.
    MessageStop,
    /// Complete message. Provided by [`StreamExt::with_message`], not the API.
    Message {
        /// The message.
        message: response::Message<'static>,
    },
    /// Tool use. Provided by [`StreamExt::with_tool_use`], not the API.
    ToolUse {
        /// The tool use.
        tool_use: tool::Use<'static>,
    },
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

/// [`Text`] or [`Json`] to be applied to a [`Block::Text`] or
/// [`Block::ToolUse`] [`Content`] [`Block`].
///
/// [`Text`]: Delta::Text
/// [`Json`]: Delta::Json
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Delta<'a> {
    /// Text delta for a [`Text`] [`Content`] [`Block`].
    ///
    /// [`Text`]: Block::Text
    #[serde(alias = "text_delta")]
    Text {
        /// The text content.
        text: Cow<'a, str>,
    },
    /// JSON delta for the input field of a [`ToolUse`] [`Content`] [`Block`].
    ///
    /// [`ToolUse`]: Block::ToolUse
    #[serde(rename = "input_json_delta")]
    Json {
        /// The JSON delta.
        partial_json: Cow<'a, str>,
    },
}

impl Delta<'_> {
    /// Convert to a static lifetime. This is useful for when the delta is
    /// stored in a `Pin<Box<dyn Stream<Item = Result<Event, Error>>>`.
    pub fn into_static(self) -> Delta<'static> {
        match self {
            Delta::Text { text } => Delta::Text {
                text: text.into_owned().into(),
            },
            Delta::Json { partial_json } => Delta::Json {
                partial_json: partial_json.into_owned().into(),
            },
        }
    }
}

/// Error when applying a [`Delta`] to a [`Content`] [`Block`] and the types do
/// not match.
#[derive(Serialize, thiserror::Error, Debug)]
#[error("`Delta::{from:?}` canot be applied to `{to}`.")]
pub struct ContentMismatch<'a> {
    /// The content block that failed to apply.
    pub from: Delta<'a>,
    /// The target [`Content`].
    pub to: &'static str,
}

/// Error when applying a [`Delta`] to a [`Content`] [`Block`] and the index is
/// out of bounds.
#[derive(Serialize, thiserror::Error, Debug)]
#[error("Index {index} out of bounds. Max index is {max}.")]
pub struct OutOfBounds {
    /// The index that was out of bounds.
    pub index: usize,
    /// The maximum index.
    pub max: usize,
}

/// Error when applying a [`Delta`].
#[derive(Serialize, thiserror::Error, Debug, derive_more::From)]
#[allow(missing_docs)]
pub enum DeltaError<'a> {
    #[error("Cannot apply delta because: {error}")]
    ContentMismatch { error: ContentMismatch<'a> },
    #[error("Cannot apply delta because: {error}")]
    OutOfBounds { error: OutOfBounds },
    #[error(
        "Cannot apply delta because deserialization failed because: {error}"
    )]
    Parse { error: String },
}

impl Delta<'_> {
    /// Merge another [`Delta`] onto the end of `self`.
    pub fn merge(mut self, delta: Delta) -> Result<Self, ContentMismatch> {
        match (&mut self, delta) {
            (Delta::Text { text }, Delta::Text { text: delta }) => {
                text.to_mut().push_str(&delta);
            }
            (
                Delta::Json { partial_json },
                Delta::Json {
                    partial_json: delta,
                },
            ) => {
                partial_json.to_mut().push_str(&delta);
            }
            (to, from) => {
                return Err(ContentMismatch {
                    from,
                    to: match to {
                        Delta::Text { .. } => stringify!(Delta::Text),
                        Delta::Json { .. } => stringify!(Delta::Json),
                    },
                });
            }
        }

        Ok(self)
    }
}

/// Metadata about a message in progress. This does not contain actual text
/// deltas. That's the [`Delta`] in [`Event::ContentBlockDelta`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MessageDelta {
    /// Stop reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<Cow<'static, str>>,
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
    /// Message assembly error (delta application, etc).
    #[error("Message assembly error: {message}")]
    MessageAssembly {
        /// Error message.
        message: Cow<'static, str>,
        /// Any delta that failed to apply.
        delta: Option<Delta<'static>>,
    },
}

// Some of the error types do not implement `Serialize` so we do it manually.
impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let message = self.to_string();
        match self {
            Error::Stream { .. } => json!({
                "type": "stream",
                "message": message,
            })
            .serialize(serializer),
            Error::Parse { event, .. } => json!({
                "type": "parse",
                "message": message,
                "event": {
                    "event": event.event,
                    "data": event.data,
                    "id": event.id,
                    "retry": event.retry,
                },
            })
            .serialize(serializer),
            Error::Anthropic { error, event } => json!({
                "type": "anthropic",
                "message": message,
                "error": error,
                "event": {
                    "event": event.event,
                    "data": event.data,
                    "id": event.id,
                    "retry": event.retry,
                },
            })
            .serialize(serializer),
            Error::MessageAssembly { delta, .. } => json!({
                "type": "message_assembly",
                "message": message,
                "delta": delta,
            })
            .serialize(serializer),
        }
    }
}

/// Stream of [`Event`]s or [`Error`]s.
pub struct Stream {
    inner: Pin<
        Box<dyn futures::Stream<Item = Result<Event, Error>> + Send + 'static>,
    >,
}

static_assertions::assert_impl_all!(Stream: futures::Stream, Send);

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
            > + Send
            + 'static,
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

    // TODO: Figure out an ergonomic way to handle tool use when streaming. We
    // may need another wrapper stream to store json deltas until a full block
    // is received. This would allow us to merge json deltas and then emit a
    // tool use event. Emitting `Block`s might not be a bad idea, but it would
    // delay the text output, which is the primary use case for streaming. Even
    // though events can be made up of multiple text blocks, generally the model
    // only generates a single block per message per type. Waiting for an entire
    // text block would mean waiting for the entire message. Waiting on JSON, is
    // however necessary since we can't do anything useful with partial JSON.
}

impl futures::Stream for Stream {
    type Item = Result<Event, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

/// Extension trait for our crate [`Event`] [`Stream`]s to filter out
/// [`RateLimit`] and [`Overloaded`] [`AnthropicError`]s, as well as several
/// other common use cases.
///
/// This is recommended for most use cases.
///
/// [`RateLimit`]: AnthropicError::RateLimit
/// [`Overloaded`]: AnthropicError::Overloaded
pub trait FilterExt:
    futures::stream::Stream<Item = Result<Event, Error>> + Sized + Send
{
    /// Filter out rate limit and overload errors. Because the server sends
    /// these events there isn't a need to retry or backoff. The stream will
    /// continue when ready.
    ///
    /// This is recommended for most use cases.
    fn filter_rate_limit(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        self.filter_map(|result| async move {
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
        })
    }

    /// Filter out everything but [`Event::ContentBlockDelta`]. This can include
    /// text, JSON, and tool use.
    fn deltas(
        self,
    ) -> impl futures::Stream<Item = Result<Delta<'static>, Error>> + Send {
        self.filter_map(|result| async move {
            match result {
                Ok(Event::ContentBlockDelta { delta, .. }) => Some(Ok(delta)),
                _ => None,
            }
        })
    }

    /// Filter out everything but text pieces.
    fn text(self) -> impl futures::Stream<Item = Result<String, Error>> + Send {
        self.deltas().filter_map(|result| async move {
            match result {
                Ok(Delta::Text { text }) => Some(Ok(text.into_owned())),
                _ => None,
            }
        })
    }

    /// Adds [`Event::Message`] to the stream by assembling a message from the
    /// stream in place. This is useful for when you want to assemble a message
    /// as well as use the deltas and interrupt the stream, taking any
    /// partiallly assembled message with you. If the stream is allowed to
    /// complete, the `message` supplied will be `None` and the complete message
    /// yielded as with [`with_message`].
    fn with_message_ip(
        self,
        message: &mut Option<response::Message<'static>>,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = self;

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(Event::MessageStart { message: msg }) => {
                        // Beginning of stream, Anthropic sends an empty message
                        *message = Some(msg.clone());

                        yield Ok(Event::MessageStart { message: message.clone().unwrap() });
                    }
                    Ok(Event::ContentBlockStart { index, content_block }) => {
                        if let Some(message) = message {
                            if index == message.message.content.len() {
                                // Most common case, append to the end.
                                message.message.content.push(content_block.clone());
                            } else if index == 0 {
                                // Insert at the beginning.
                                message.message.content = Content::MultiPart(
                                    vec![content_block.clone()]
                                );
                            } else {
                                yield Err(Error::MessageAssembly {
                                    message: format!("Index {} out of bounds. Max index is {}.", index, message.message.content.len()).into(),
                                    delta: None,
                                });
                            }
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Content block start without message start.".into(),
                                delta: None,
                            });
                        }

                        yield Ok(Event::ContentBlockStart { index, content_block });
                    }
                    Ok(Event::ContentBlockDelta { index, delta }) => {
                        if let Some(message) = message {
                            if index != message.message.content.len() - 1 {
                                // A message delta appends to an existing index,
                                // so the index should not be the len.
                                yield Err(Error::MessageAssembly {
                                    message: format!("Unexpected index for delta. Got `{}`, expected `{}`.", index, message.message.content.len() - 1).into(),
                                    delta: Some(delta.clone()),
                                });
                            }

                            if let Err(err) = message.message.content.push_delta(delta.clone()) {
                                yield Err(Error::MessageAssembly {
                                    message: err.to_string().into(),
                                    delta: Some(delta.clone()),
                                });
                            }
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Content block delta without message start.".into(),
                                delta: Some(delta.clone()),
                            });
                        }

                        yield Ok(Event::ContentBlockDelta { index, delta });
                    }
                    Ok(Event::ContentBlockStop { index }) => {
                        if let Some(message) = message {
                            if index != message.message.content.len() - 1 {
                                yield Err(Error::MessageAssembly {
                                    message: format!("Unexpected index for stop. Got `{}`, expected `{}`.", index, message.message.content.len() - 1).into(),
                                    delta: None,
                                });
                            }
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Content block stop without message start.".into(),
                                delta: None,
                            });
                        }

                        yield Ok(Event::ContentBlockStop { index });
                    }
                    Ok(Event::MessageDelta { delta }) => {
                        if let Some(message) = message {
                            message.apply_delta(delta.clone())
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: format!("Message metadata delta without start: {:?}", delta).into(),
                                delta: None,
                            });
                        }

                        yield Ok(Event::MessageDelta { delta });
                    }
                    Ok(Event::MessageStop) => {
                        if let Some(message) = message.take() {
                            yield Ok(Event::Message { message });
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Message stop without start.".into(),
                                delta: None,
                            });
                        }

                        yield Ok(Event::MessageStop);
                    }
                    event => yield event,
                }
            }
        }
    }

    /// Adds [`Event::Message`] to the stream by assembling a message from
    /// the stream. If you need to interrupt the stream and take the partially
    /// assembled message with you, use [`with_message_ip`].
    fn with_message(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let mut message = None;

            let stream = self.with_message_ip(&mut message);

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                yield result;
            }
        }
    }

    /// Adds [`Event::ToolUse`] to the stream. This is useful for when you don't
    /// want to bother with assembling tool use from pieces of JSON deltas. In
    /// the case a tool::Use fails to deserialize, the JSON will be included in
    /// the [`Error::MessageAssembly`] error.
    fn with_tool_use(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = self;
            let mut json_buf = String::new();

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(Event::ContentBlockStart { index, content_block }) => {
                        // New block, clear the buffer.
                        json_buf.clear();
                        yield Ok(Event::ContentBlockStart { index, content_block });
                    }
                    Ok(Event::ContentBlockDelta { index, delta }) => {
                        // Content delta, if it's JSON, append to the buffer.
                        if let Delta::Json { partial_json } = &delta {
                            json_buf.push_str(partial_json);
                        }

                        yield Ok(Event::ContentBlockDelta { index, delta });
                    }
                    Ok(Event::ContentBlockStop { index }) => {

                        if !json_buf.is_empty() {
                            let tool_use = match serde_json::from_str(&json_buf) {
                                Ok(tool_use) => tool_use,
                                Err(error) => {
                                    yield Err(Error::MessageAssembly {
                                        message: error.to_string().into(),
                                        delta: Some(Delta::Json { partial_json: json_buf.clone().into() }),
                                    });
                                    continue;
                                }
                            };

                            yield Ok(Event::ToolUse { tool_use });
                        }

                        yield Ok(Event::ContentBlockStop { index });
                    }
                    event => yield event,
                }
            }
        }
    }
}

impl<S> FilterExt for S where
    S: futures::Stream<Item = Result<Event, Error>> + Send
{
}

#[cfg(test)]
pub(crate) mod tests {
    use futures::TryStreamExt;

    use super::*;

    // Actual JSON from the API.

    pub const CONTENT_BLOCK_START: &str = "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"} }";
    pub const CONTENT_BLOCK_DELTA: &str = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Certainly! I\"}     }";

    /// Creates a mock stream from a string (likely `include_str!`). The string
    /// should be a series of `event`, `data`, and empty lines (a SSE stream).
    /// Anthropic provides such example data in the API documentation.
    pub fn mock_stream(text: &'static str) -> Stream {
        use itertools::Itertools;

        // TODO: one of every possible variants, even if it doesn't make sense.
        let inner = futures::stream::iter(
            // first line should be `event`, second line should be `data`, third
            // line should be empty.
            text.lines().tuples().map(|(event, data, _empty)| {
                assert!(_empty.is_empty());

                Ok(eventsource_stream::Event {
                    event: event.strip_prefix("event: ").unwrap().into(),
                    data: data.strip_prefix("data: ").unwrap().into(),
                    id: "".into(),
                    retry: None,
                })
            }),
        );

        Stream::new(inner)
    }

    #[test]
    fn test_content_block_start() {
        let event: Event = serde_json::from_str(CONTENT_BLOCK_START).unwrap();
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 0);
                #[cfg(feature = "prompt-caching")]
                if let Block::Text {
                    text,
                    cache_control,
                } = content_block
                {
                    assert_eq!(text.as_ref(), "");
                    assert!(cache_control.is_none());
                } else {
                    panic!("Unexpected content block: {:?}", content_block);
                }
                #[cfg(not(feature = "prompt-caching"))]
                if let Block::Text { text } = content_block {
                    assert_eq!(text.as_ref(), "");
                } else {
                    panic!("Unexpected content block: {:?}", content_block);
                }
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
                    Delta::Text {
                        text: "Certainly! I".into()
                    }
                );
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    fn test_content_block_delta_merge() {
        // Merge text deltas.
        let text_delta = Delta::Text {
            text: "Certainly! I".into(),
        }
        .merge(Delta::Text {
            text: " can".into(),
        })
        .unwrap()
        .merge(Delta::Text { text: " do".into() })
        .unwrap();

        assert_eq!(
            text_delta,
            Delta::Text {
                text: "Certainly! I can do".into()
            }
        );

        // Merge JSON deltas.
        let json_delta = Delta::Json {
            partial_json: r#"{"key":"#.into(),
        }
        .merge(Delta::Json {
            partial_json: r#""value"}"#.into(),
        })
        .unwrap();

        assert_eq!(
            json_delta,
            Delta::Json {
                partial_json: r#"{"key":"value"}"#.into()
            }
        );

        // Content mismatch.
        let mismatch = json_delta.merge(text_delta).unwrap_err();

        assert_eq!(
            mismatch.to_string(),
            ContentMismatch {
                from: Delta::Text {
                    text: "Certainly! I can do".into()
                },
                to: "Delta::Json"
            }
            .to_string()
        );

        // Other way around, for coverage.
        let text_delta = Delta::Text {
            text: "Certainly!".into(),
        };
        let json_delta = Delta::Json {
            partial_json: r#"{"key":"value"}"#.into(),
        };

        let mismatch = text_delta.merge(json_delta).unwrap_err();

        assert_eq!(
            mismatch.to_string(),
            ContentMismatch {
                from: Delta::Json {
                    partial_json: r#"{"key":"value"}"#.into()
                },
                to: "Delta::Text"
            }
            .to_string()
        );
    }

    #[tokio::test]
    async fn test_stream() {
        // sse.stream.txt is from the API docs and includes one of every event
        // type, with the exception of fatal errors, but they all have the same
        // structure, so if one works, they all should. It covers every code
        // path in the `Stream` struct and every event type.
        let stream = mock_stream(include_str!("../test/data/sse.stream.txt"));

        let text: String = stream
            .filter_rate_limit()
            .text()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(
            text,
            "Okay, let's check the weather for San Francisco, CA:"
        );
    }
}
