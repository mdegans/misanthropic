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
use futures::{StreamExt, pin_mut};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{borrow::Cow, pin::Pin, task::Poll};

/// Sucessful Event from the API. See [`stream::Error`] for errors.
///
/// [`stream::Error`]: Error
#[derive(Debug, Serialize, Deserialize, derive_more::IsVariant)]
#[cfg_attr(any(test, feature = "partial-eq"), derive(PartialEq))]
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
        /// Usage statistics for the message.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// Message end.
    MessageStop,
    /// Complete [`response::Message`]. Assembled by [`FilterExt::with_message`]
    /// not the API.
    Message {
        /// The message.
        message: response::Message<'static>,
    },
    /// Complete [`tool::Use`]. Assembled by [`FilterExt::with_tool_use`] not
    /// the API.
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
    /// Thinking delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "thinking_delta")]
    Thought {
        /// The thinking delta.
        thinking: Cow<'a, str>,
        /// Signature, when the thinking is complete.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<Cow<'a, str>>,
    },
    /// Redacted thinking delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "redacted_thinking_delta")]
    RedactedThought {
        /// Complete signature of a redacted thought.
        signature: Cow<'a, str>,
    },
    /// Signature delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "signature_delta")]
    Signature {
        /// Signature of a complete thought. This should be merged with a
        /// [`Delta::Thought`]` to complete the thought.
        signature: Cow<'a, str>,
    },
    /// A single [`Citation`] to append to the current [`Text`] block's
    /// citations. Available when a [`Document`] had citations enabled.
    ///
    /// [`Citation`]: crate::prompt::Citation
    /// [`Text`]: crate::prompt::message::Block::Text
    /// [`Document`]: crate::prompt::message::Block::Document
    #[serde(rename = "citations_delta")]
    CitationsDelta {
        /// The citation to append.
        citation: crate::prompt::Citation<'a>,
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
            Delta::Thought {
                thinking,
                signature,
            } => Delta::Thought {
                thinking: thinking.into_owned().into(),
                signature: signature.map(|s| s.into_owned().into()),
            },
            Delta::Signature { signature } => Delta::Signature {
                signature: signature.into_owned().into(),
            },
            Delta::RedactedThought { signature } => Delta::RedactedThought {
                signature: signature.into_owned().into(),
            },
            Delta::CitationsDelta { citation } => Delta::CitationsDelta {
                citation: citation.into_static(),
            },
        }
    }
}

/// Error when applying a [`Delta`] to a [`Content`] [`Block`] and the types do
/// not match. Also from [`Delta::merge`].
#[derive(Serialize, thiserror::Error, Debug)]
#[error("`Delta::{from:?}` canot be applied to `{to}`.")]
pub struct ContentMismatch<'a> {
    /// The content block that failed to apply.
    pub from: Delta<'a>,
    /// The target [`Content`].
    pub to: &'static str,
}

impl ContentMismatch<'_> {
    /// Convert to a static lifetime. This is useful for when the error is
    /// stored in a `Pin<Box<dyn Stream<Item = Result<Event, Error>>>`.
    pub fn into_static(self) -> ContentMismatch<'static> {
        ContentMismatch {
            from: self.from.into_static(),
            to: self.to,
        }
    }
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

impl DeltaError<'_> {
    /// Convert to a static lifetime. This is useful for when the error is
    /// stored in a `Pin<Box<dyn Stream<Item = Result<Event, Error>>>`.
    pub fn into_static(self) -> DeltaError<'static> {
        match self {
            DeltaError::ContentMismatch { error } => {
                DeltaError::ContentMismatch {
                    error: error.into_static(),
                }
            }
            DeltaError::OutOfBounds { error } => {
                DeltaError::OutOfBounds { error }
            }
            DeltaError::Parse { error } => DeltaError::Parse { error },
        }
    }
}

impl<'a> Delta<'a> {
    /// Return true if `self` is a [`Thought`] delta and `signature` is `Some`.
    ///
    /// [`Thought`]: Delta::Thinking
    pub fn thought_complete(&self) -> bool {
        matches!(
            self,
            Delta::Thought {
                signature: Some(_),
                ..
            }
        )
    }

    /// Merge another [`Delta`] onto the end of `self`.
    pub fn merge(
        mut self,
        delta: Delta<'a>,
    ) -> Result<Self, ContentMismatch<'a>> {
        match (&mut self, delta) {
            // Text incoming, text already here. Simply append.
            (Delta::Text { text }, Delta::Text { text: delta }) => {
                text.to_mut().push_str(&delta);
            }
            // Dittos for JSON.
            (
                Delta::Json { partial_json },
                Delta::Json {
                    partial_json: delta,
                },
            ) => {
                partial_json.to_mut().push_str(&delta);
            }
            // Case where an incomplete thought is merged with an incomplete
            // thought. This is valid. Simply append.
            (
                Delta::Thought {
                    thinking,
                    signature: None,
                },
                Delta::Thought {
                    thinking: delta,
                    // It is not valid to merge a complete thought with anything
                    signature: None,
                },
            ) => {
                thinking.to_mut().push_str(&delta);
            }
            // Case where an incomplete thought is merged with a signature to
            // create a complete thought.
            (
                Delta::Thought { signature, .. },
                Delta::Signature {
                    signature: signature_delta,
                },
            ) => {
                if signature.is_some() {
                    return Err(ContentMismatch {
                        from: Delta::Signature {
                            signature: signature_delta,
                        },
                        to: stringify!(Delta::Thinking),
                    });
                }
                signature.replace(signature_delta);
            }
            // Every other case is a mismatch.
            (to, from) => {
                return Err(ContentMismatch {
                    from,
                    to: match to {
                        Delta::Text { .. } => "Delta::Text",
                        Delta::Json { .. } => "Delta::Json",
                        Delta::Thought { .. } => "Delta::Thought",
                        // Each delta below is a single event. Merge impossible.
                        Delta::Signature { .. } => "Delta::Signature",
                        Delta::RedactedThought { .. } => {
                            "Delta::RedactedThought"
                        }
                        Delta::CitationsDelta { .. } => "Delta::CitationsDelta",
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
#[cfg_attr(any(test, feature = "partial-eq"), derive(PartialEq))]
pub struct MessageDelta {
    /// Stop reason.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,
    /// Stop sequence.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequence: Option<Cow<'static, str>>,
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
    /// Message assembly error (delta without message start, etc).
    #[error("Message assembly error: {message}")]
    MessageAssembly {
        /// Error message.
        message: Cow<'static, str>,
        /// Any delta that failed to apply.
        delta: Option<Delta<'static>>,
    },
    /// DeltaError from applying a delta.
    #[error("Delta error: {error}")]
    Delta {
        /// Error from applying a delta.
        #[from]
        error: DeltaError<'static>,
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
            Error::Delta { error } => json!({
                "type": "delta",
                "message": message,
                "error": error,
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
    // `stream::Error` is 136 B, but `Result<Event, Error>` is sized by the
    // `Event` success variant (184 B) regardless, so boxing the error wouldn't
    // shrink it. Permanent allow, not a deferral.
    #[allow(clippy::result_large_err)]
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

/// Extension trait for our crate [`Event`] [`Stream`]s covering several common
/// use cases such as extracting [`Delta`]s or [`text`] and assembling complete
/// [`Message`]s in place.
///
/// [`text`]: FilterExt::text
/// [`Message`]: response::Message
pub trait FilterExt:
    futures::stream::Stream<Item = Result<Event, Error>> + Sized + Send
{
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
    /// stream in place. If the stream is allowed to complete, the `message`
    /// supplied will be `None` and the complete message yielded as with
    /// [`with_message`].
    ///
    /// # Note:
    /// - Message is set to `None` at the beginning of the stream.
    /// - Implies [`with_tool_use`].
    ///
    /// [`with_tool_use`]: FilterExt::with_tool_use
    /// [`with_message`]: FilterExt::with_message
    fn with_message_ip(
        self,
        message: &mut Option<response::Message<'static>>,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = self.with_tool_use();

            pin_mut!(stream);

            // reset the message if it's not already None.
            *message = None;

            while let Some(result) = stream.next().await {
                match &result {
                    // The most common case is content block delta.
                    Ok(Event::ContentBlockDelta { delta, ..}) => {
                        if let Some(message) = message.as_mut() {
                            if let Err(e) = message.inner.inner.content.push_delta(delta.clone()) {
                                yield Err(e.into_static().into());
                            }
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Content block delta received before message start.".into(),
                                delta: Some(delta.clone()),
                            });
                        }
                    }
                    Ok(Event::MessageStart { message: start }) => {
                        *message = Some(start.clone());
                    }
                    Ok(Event::ContentBlockStart {
                        content_block, ..
                    }) => {
                        if let Some(message) = message.as_mut() {
                            message.inner.inner.content.push(
                                content_block.clone()
                            );
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Content block received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::ToolUse { tool_use }) => {
                        if let Some(message) = message.as_mut() {
                            message.inner.inner.content.push(tool_use.clone());
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Tool use received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::MessageDelta { delta, usage }) => {
                        if let Some(message) = message.as_mut() {
                            message.apply_delta(delta.clone());
                            if let Some(usage) = usage {
                                message.usage += *usage;
                            }
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Message delta received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::MessageStop) => {
                        if let Some(message) = message.take() {
                            yield Ok(Event::Message { message });
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Message stop received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::ContentBlockStop { .. })
                    | Ok(Event::Ping)
                    | Ok(Event::Message { .. })=> {
                        // This is a no-op. We don't need to do anything with
                        // this event.
                    }
                    Err(_) => {
                        // It's passed through below.
                    }
                }


                yield result;
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

    /// Yields tool_use events when complete, instead of an empty tool use at
    /// the beginning and then having to handle the deltas yourself when a tool
    /// call is 99% of the time only useful when complete. This will also skip
    /// `input_json_delta` events.
    fn with_tool_use(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = self;
            let mut call: Option<tool::Use> = None;
            let mut input = String::new();

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(Event::ContentBlockStart {
                        content_block: Block::ToolUse { call: empty }, .. }) => {
                        input.clear();
                        call = Some(empty);
                    }
                    Ok(Event::ContentBlockDelta { delta: Delta::Json { partial_json }, .. }) => {
                        input.push_str(&partial_json);
                    }
                    Ok(Event::ContentBlockStop { .. }) => {
                        if let Some(mut call) = call.take() {
                            call.input = match serde_json::from_str(&input) {
                                Ok(input) => input,
                                Err(err) => {
                                    yield Err(Error::MessageAssembly {
                                        message: format!("Failed to parse JSON: {}", err).into(),
                                        delta: None,
                                    });
                                    continue;
                                }
                            };

                            yield Ok(Event::ToolUse { tool_use: call });
                        }
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

    #[allow(unused_imports)] // because conditional compilation.
    use crate::{
        AnthropicModel, Prompt,
        prompt::{Message, message::Role},
    };

    use super::*;

    // Actual JSON from the API.

    pub const CONTENT_BLOCK_START: &str = "{\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"} }";
    pub const CONTENT_BLOCK_DELTA: &str = "{\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Certainly! I\"}     }";

    // Test each event individually.
    #[test]
    pub fn test_event_ping() {
        let event: Event = serde_json::from_str(r#"{"type":"ping"}"#).unwrap();
        match event {
            Event::Ping => {}
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_message_start() {
        let event: Event = serde_json::from_str(
            r#"{"type":"message_start","message":{"id":"msg_014p7gG3wDgGV9EUtLvnow3U","type":"message","role":"assistant","model":"claude-3-haiku-20240307","stop_sequence":null,"usage":{"input_tokens":472,"output_tokens":2},"content":[],"stop_reason":null}}"#,
        )
        .unwrap();
        match event {
            Event::MessageStart { message } => {
                assert_eq!(message.inner.inner.role, Role::Assistant);
                assert_eq!(message.id, "msg_014p7gG3wDgGV9EUtLvnow3U");
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_content_block_start() {
        // Test tool_use delta. Text is tested in many other places.
        let event: Event = serde_json::from_str(r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_01T1x1fJ34qAmk2tNTrN7Up6","name":"get_weather","input":{}}}"#).unwrap();
        match event {
            Event::ContentBlockStart {
                index,
                content_block,
            } => {
                assert_eq!(index, 1);
                assert!(content_block.is_tool_use());
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_content_block_delta() {
        // text delta
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" check"}}"#,
        )
        .unwrap();
        match event {
            Event::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 0);
                assert_eq!(
                    delta,
                    Delta::Text {
                        text: " check".into()
                    }
                );
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
        // json delta
        let event: Event = serde_json::from_str(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":" Francisc"}}"#,
        )
        .unwrap();
        match event {
            Event::ContentBlockDelta { index, delta } => {
                assert_eq!(index, 1);
                assert_eq!(
                    delta,
                    Delta::Json {
                        partial_json: " Francisc".into()
                    }
                );
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_content_block_stop() {
        let event: Event =
            serde_json::from_str(r#"{"type":"content_block_stop","index":0}"#)
                .unwrap();
        match event {
            Event::ContentBlockStop { index } => {
                assert_eq!(index, 0);
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_message_delta() {
        let event: Event = serde_json::from_str(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"output_tokens":89}}"#,
        )
        .unwrap();
        match event {
            Event::MessageDelta { delta, usage } => {
                assert!(
                    delta
                        .stop_reason
                        .is_some_and(|reason| reason.is_tool_use())
                );
                assert!(delta.stop_sequence.is_none());
                assert_eq!(usage.unwrap().output_tokens, 89);
            }
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    #[test]
    pub fn test_event_message_stop() {
        let event: Event =
            serde_json::from_str(r#"{"type":"message_stop"}"#).unwrap();
        match event {
            Event::MessageStop => {}
            _ => panic!("Unexpected event: {:?}", event),
        }
    }

    // MessageDelta tests.

    #[test]
    pub fn test_message_delta() {
        let delta: MessageDelta = serde_json::from_str(
            r#"{"stop_reason":"tool_use","stop_sequence":null}"#,
        )
        .unwrap();
        assert!(delta.stop_reason.is_some_and(|reason| reason.is_tool_use()));
        assert!(delta.stop_sequence.is_none());
    }

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

    #[allow(clippy::result_large_err)] // see `Stream::new`: `Event` dominates.
    pub fn mock_stream_jsonl(
        text: &'static str,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        futures::stream::iter(text.lines().map(|line| {
            let res: Result<Event, serde_json::Value> =
                serde_json::from_str(line).unwrap();
            match res {
                Ok(event) => Ok(event),
                Err(_) => Err(Error::Anthropic {
                    error: AnthropicError::Unknown {
                        code: Some(123.try_into().unwrap()),
                        // every line in the file is Ok, so this is impossible.
                        message: "impossible".into(),
                    },
                    event: eventsource_stream::Event {
                        event: "impossible".into(),
                        data: "impossible".into(),
                        id: "impossible".into(),
                        retry: None,
                    },
                }),
            }
        }))
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
                if let Block::Text {
                    text,
                    cache_control,
                    ..
                } = content_block
                {
                    assert_eq!(text.as_ref(), "");
                    assert!(cache_control.is_none());
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
        let stream = mock_stream(include_str!("../test/data/sse.stream.txt"));

        let events = stream.collect::<Vec<_>>().await;

        assert_eq!(events.len(), 32);
        // there are 2 errors
        let n_errors = events.iter().filter(|e| e.is_err()).count();
        assert_eq!(n_errors, 2);
    }

    #[tokio::test]
    async fn test_stream_text() {
        // sse.stream.txt is from the API docs and includes one of every event
        // type, with the exception of fatal errors, but they all have the same
        // structure, so if one works, they all should. It covers every code
        // path in the `Stream` struct and every event type.
        let stream = mock_stream(include_str!("../test/data/sse.stream.txt"));

        let text: String = stream.text().try_collect().await.unwrap();

        assert_eq!(
            text,
            "Okay, let's check the weather for San Francisco, CA:"
        );
    }

    #[tokio::test]
    async fn test_thought_stream() {
        // Test every message deserializes.
        let mut stream =
            mock_stream(include_str!("../test/data/thinking.sse.stream.txt"));

        let mut errors = Vec::new();
        while let Some(event) = stream.next().await {
            if let Err(error) = event {
                errors.push(error)
            }
        }
        if !errors.is_empty() {
            panic!("Errors: {:#?}", errors);
        }
        // The stream has no error variants, so we parsed everything correctly.

        let stream =
            mock_stream(include_str!("../test/data/thinking.sse.stream.txt"));

        // Test the text stream filters out the thinking delta.
        let text: String = stream.text().try_collect().await.unwrap();

        assert_eq!(text, "27 * 453 = 12,231");
    }

    #[tokio::test]
    async fn test_thought_stream_exact() {
        let mut stream =
            mock_stream(include_str!("../test/data/thinking.sse.stream.txt"));

        // Test prompt assembly from the stream.
        let mut prompt = Prompt::default()
            // This is a dummy message because the prompt must start with a user
            // message. `handle_stream_event` checks turn order.
            .add_message(Message {
                role: Role::User,
                content: Content::text("dummy message"),
            })
            .unwrap();

        while let Some(event) = stream.next().await {
            prompt.handle_stream_event(event.unwrap()).unwrap();
        }

        assert_eq!(prompt.messages.len(), 2);
        let last = prompt.messages.pop().unwrap();
        assert_eq!(
            last,
            prompt::Message {
                role: Role::Assistant,
                content: Content(vec![
                    Block::Thought {
                        thought: "Let me solve this step by step:\n\n1. First break down 27 * 453\n2. 453 = 400 + 50 + 3".to_string().into(),
                        signature: "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds...".to_string().into()
                    },
                    Block::Text {
                        text: "27 * 453 = 12,231".to_string().into(),
                        citations: None,
                        cache_control: None
                    }
                ])
            }
        );
    }

    #[tokio::test]
    async fn test_stream_prompt_extend() {
        let stream =
            mock_stream(include_str!("../test/data/thinking.sse.stream.txt"));

        // Test prompt assembly from the stream.
        let mut prompt = Prompt::default()
            // This is a dummy message because the prompt must start with a user
            // message. `handle_stream_event` checks turn order.
            .add_message(Message {
                role: Role::User,
                content: Content::text("dummy message"),
            })
            .unwrap();

        // Extend a prompt with a stream. We can't implement actual `Extend`
        // because it's not async and some of the type that implement
        // `ExtendOntoPrompt` are async, like this one. So you can't collect
        // but also you wouldn't be able to anyway because a Prompt must start
        // with a user message and a stream is always from the assistant.
        let prompt = prompt.extend(stream).await.unwrap();

        assert_eq!(prompt.messages.len(), 2);
        let last = prompt.messages.pop().unwrap();
        assert_eq!(
            last,
            prompt::Message {
                role: Role::Assistant,
                content: Content(vec![
                    Block::Thought {
                        thought: "Let me solve this step by step:\n\n1. First break down 27 * 453\n2. 453 = 400 + 50 + 3".to_string().into(),
                        signature: "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds...".to_string().into()
                    },
                    Block::Text {
                        text: "27 * 453 = 12,231".to_string().into(),
                        citations: None,
                        cache_control: None
                    }
                ])
            }
        );
    }

    #[tokio::test]
    async fn test_stream_redacted_thought_from_jsonl() {
        const JSON: &str =
            include_str!("../test/data/redacted_thought.sse.stream.jsonl");
        let stream = mock_stream_jsonl(JSON);

        // Test the text stream filters out the thinking delta.
        let text: String = stream.text().try_collect().await.unwrap();

        assert_eq!(
            text,
            "I notice that you've included what appears to be a trigger phrase or command. I'm designed to communicate transparently and directly with you without any special backdoor commands or hidden modes.\n\nI'm here to have a helpful conversation! Could you let me know what you'd actually like assistance with today? I'm happy to help with information, answer questions, brainstorm ideas, or discuss various topics within my guidelines."
        );

        let stream = mock_stream_jsonl(JSON);

        pin_mut!(stream);

        while let Some(event) = stream.next().await {
            if let Ok(Event::ContentBlockStart {
                content_block: Block::RedactedThought { signature },
                ..
            }) = event
            {
                assert!(!signature.is_empty());
            }
        }
    }

    // Test from live API. If they break our client, we'll know.
    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore]
    async fn test_stream_redacted_thought() {
        const TRIGGER: &str = "ANTHROPIC_MAGIC_STRING_TRIGGER_REDACTED_THINKING_46C9A13E193C177646C7398A98432ECCCE4C1253D5E2D82641AC0E52CC2876CB";
        let api_key = crate::utils::load_api_key().await;
        let client = crate::Client::new(api_key).unwrap();
        let prompt = Prompt::default()
            // Only sonnet 3.7 and newer will respond to the trigger.
            .model(AnthropicModel::Sonnet37)
            // Sonnet 3.7 still accepts the deprecated fixed-budget mode.
            .thinking(prompt::Thinking::enabled(1024.try_into().unwrap()))
            .add_message(Message {
                role: Role::User,
                content: TRIGGER.into(),
            })
            .unwrap();

        // In a real app you could RwLock the prompt and pass a reference, and
        // then append to the same prompt with `.write().await.extend(stream)`.
        let stream = client.stream(prompt.clone()).await.unwrap();

        pin_mut!(stream);

        let mut redacted_seen = false;
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            match &event {
                Ok(Event::ContentBlockStart { content_block, .. }) => {
                    if let Block::RedactedThought { signature } = content_block
                    {
                        assert!(!signature.is_empty());
                        redacted_seen = true;
                    }

                    events.push(event);
                }
                _ => {
                    events.push(event);
                }
            }
        }

        assert!(redacted_seen);
    }

    // This also tests the `_ip` version since this just wraps it.
    #[tokio::test]
    async fn test_stream_with_message() {
        let stream = mock_stream(include_str!("../test/data/sse.stream.txt"));

        let stream = stream.with_message();

        pin_mut!(stream);

        let mut message = None;
        while let Some(event) = stream.next().await {
            dbg!(&event);
            if let Ok(Event::Message { message: new }) = event {
                message = Some(new);
                break;
            }
        }

        if let Some(message) = message {
            assert_eq!(message.id, "msg_014p7gG3wDgGV9EUtLvnow3U");
            assert_eq!(message.model.to_string(), "claude-3-haiku-20240307");
        } else {
            panic!("No message assembled.");
        }
    }

    #[tokio::test]
    async fn test_stream_with_tool_use() {
        let stream = mock_stream(include_str!("../test/data/sse.stream.txt"))
            .with_tool_use();
        let mut tool_use = None;

        pin_mut!(stream);
        while let Some(event) = stream.next().await {
            dbg!(&event);
            if let Ok(Event::ToolUse { tool_use: new }) = event {
                tool_use = Some(new);
                break;
            }
        }

        if let Some(tool_use) = tool_use {
            assert_eq!(
                serde_json::to_value(tool_use).unwrap(),
                serde_json::json!({
                    "id": "toolu_01T1x1fJ34qAmk2tNTrN7Up6",
                    "name": "get_weather",
                    "input": {
                        "location": "San Francisco, CA",
                        "unit": "fahrenheit",
                    }
                })
            )
        } else {
            panic!("No tool use assembled.");
        }
    }
}
