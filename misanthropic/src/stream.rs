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
        message: response::Message,
    },
    /// [`Content`] [`Block`] with empty content.
    ContentBlockStart {
        /// Index of the [`Content`] [`Block`] in [`prompt::message::Content`].
        index: usize,
        /// Empty content block.
        content_block: Block,
    },
    /// Content block delta.
    ContentBlockDelta {
        /// Index of the [`Content`] [`Block`] in [`prompt::message::Content`].
        index: usize,
        /// Delta to apply to the content block.
        delta: Delta,
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
        message: response::Message,
    },
    /// Complete [`tool::Use`]. Assembled by [`FilterExt::with_tool_use`] not
    /// the API.
    ToolUse {
        /// The tool use.
        tool_use: tool::Use,
    },
    /// Complete *server* [`tool::Use`] — a [`ServerToolUse`] block (e.g.
    /// [`web_search`]) the API ran itself. Assembled by
    /// [`FilterExt::with_tool_use`], not the API. Distinct from
    /// [`ToolUse`](Event::ToolUse) so callers can tell that this call was
    /// executed server-side and needs no [`tool::Result`].
    ///
    /// [`ServerToolUse`]: crate::prompt::message::Block::ServerToolUse
    /// [`web_search`]: crate::tool::ServerMethodDef::web_search
    /// [`tool::Result`]: crate::tool::Result
    ServerToolUse {
        /// The server tool use.
        tool_use: tool::Use,
    },
    /// A completed element of the outermost JSON array in a [`Text`] or
    /// [`ToolUse`] block — see [`Items`] for the conventional shape.
    /// Assembled by [`FilterExt::with_json`], not the API.
    ///
    /// [`Text`]: Block::Text
    /// [`ToolUse`]: Block::ToolUse
    /// [`Items`]: crate::prompt::Items
    JsonObject {
        /// Index of the [`Content`] [`Block`] the element belongs to.
        index: usize,
        /// The parsed element. Conforms to the schema when
        /// [`output_config`] is set.
        ///
        /// [`output_config`]: crate::Prompt::output_config
        value: serde_json::Value,
    },
}

/// Internal enum for the API result so we don't have to add an error variant to
/// the `Event` enum.
// Transient: parsed and immediately destructured into `Result<Event, Error>`,
// which is sized by `Event` regardless (see `Stream::new`). Boxing here would
// shrink nothing downstream. Permanent allow, not a deferral.
#[allow(clippy::large_enum_variant)]
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum ApiResult {
    /// Successful Event.
    Event {
        #[serde(flatten)]
        event: Event,
    },
    /// Error Event.
    Error(ErrorEvent),
}

/// A wire `error` event — the `data:` payload `{"type":"error","error":{…}}`.
/// Surfaced as [`Error::Anthropic`]; also the typed `Err` arm of the wrapped
/// `*.sse.stream.jsonl` fixtures (see `test/data/README.md`), so captured
/// error frames round-trip through the real error types, not a `Value`.
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "partial-eq"), derive(PartialEq))]
pub(crate) struct ErrorEvent {
    /// The literal `"type": "error"` tag.
    #[serde(rename = "type")]
    tag: ErrorTag,
    /// The API error.
    pub(crate) error: AnthropicError,
}

impl From<AnthropicError> for ErrorEvent {
    fn from(error: AnthropicError) -> Self {
        Self {
            tag: ErrorTag::Error,
            error,
        }
    }
}

/// The literal `"error"` tag on an [`ErrorEvent`]. Requiring it on
/// deserialization keeps [`ApiResult`] strict — a payload is only an API error
/// if it says so, not merely because an `error` key appears somewhere.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[cfg_attr(any(test, feature = "partial-eq"), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
enum ErrorTag {
    /// `"error"`.
    #[default]
    Error,
}

/// [`Text`] or [`Json`] to be applied to a [`Block::Text`] or
/// [`Block::ToolUse`] [`Content`] [`Block`].
///
/// [`Text`]: Delta::Text
/// [`Json`]: Delta::Json
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Delta {
    /// Text delta for a [`Text`] [`Content`] [`Block`].
    ///
    /// Serializes as the wire's `text_delta` (the `text` alias is accepted
    /// for backward compatibility), so captured `content_block_delta` frames
    /// round-trip exactly — see `test/data/README.md`.
    ///
    /// [`Text`]: Block::Text
    #[serde(rename = "text_delta", alias = "text")]
    Text {
        /// The text content.
        text: Cow<'static, str>,
    },
    /// JSON delta for the input field of a [`ToolUse`] [`Content`] [`Block`].
    ///
    /// [`ToolUse`]: Block::ToolUse
    #[serde(rename = "input_json_delta")]
    Json {
        /// The JSON delta.
        partial_json: Cow<'static, str>,
    },
    /// Thinking delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "thinking_delta")]
    Thought {
        /// The thinking delta.
        thinking: Cow<'static, str>,
        /// Signature, when the thinking is complete.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<Cow<'static, str>>,
    },
    /// Redacted thinking delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "redacted_thinking_delta")]
    RedactedThought {
        /// Complete signature of a redacted thought.
        signature: Cow<'static, str>,
    },
    /// Signature delta. Availalble with Sonnet 3.7 and newer when
    /// [`Prompt::thinking`] is set.
    ///
    /// [`Prompt::thinking`]: crate::prompt::Prompt::thinking
    #[serde(rename = "signature_delta")]
    Signature {
        /// Signature of a complete thought. This should be merged with a
        /// [`Delta::Thought`]` to complete the thought.
        signature: Cow<'static, str>,
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
        citation: crate::prompt::Citation,
    },
}

impl Delta {}

/// Error when applying a [`Delta`] to a [`Content`] [`Block`] and the types do
/// not match. Also from [`Delta::merge`].
#[derive(Serialize, thiserror::Error, Debug)]
#[error("`Delta::{from:?}` canot be applied to `{to}`.")]
pub struct ContentMismatch {
    /// The content block that failed to apply.
    pub from: Delta,
    /// The target [`Content`].
    pub to: &'static str,
}

impl ContentMismatch {}

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
pub enum DeltaError {
    #[error("Cannot apply delta because: {error}")]
    ContentMismatch { error: ContentMismatch },
    #[error("Cannot apply delta because: {error}")]
    OutOfBounds { error: OutOfBounds },
    #[error(
        "Cannot apply delta because deserialization failed because: {error}"
    )]
    Parse { error: String },
}

impl DeltaError {}

impl Delta {
    /// Return true if `self` is a [`Thought`] delta and `signature` is `Some`.
    ///
    /// [`Thought`]: Delta::Thought
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
    pub fn merge(mut self, delta: Delta) -> Result<Self, ContentMismatch> {
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
    /// Structured stop detail — populated on
    /// [`Refusal`](StopReason::Refusal), explicitly `null` otherwise. Boxed
    /// for the same reason as
    /// [`Message::stop_details`](crate::response::Message::stop_details).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Box<crate::response::StopDetails>>,
    /// The [code execution] container backing this turn — streamed in the
    /// final `message_delta`, *not* `message_start`. Dropping it would make a
    /// streamed [programmatic tool call] impossible to resume (the container
    /// id must be passed back via
    /// [`Prompt::container`](crate::Prompt::container)).
    ///
    /// [code execution]: crate::tool::ServerMethodDef::code_execution
    /// [programmatic tool call]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<Box<crate::response::Container>>,
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
        delta: Option<Delta>,
    },
    /// DeltaError from applying a delta.
    #[error("Delta error: {error}")]
    Delta {
        /// Error from applying a delta.
        #[from]
        error: DeltaError,
    },
    /// JSON assembly error from [`FilterExt::with_json`] — an array element
    /// failed to parse, or the block ended mid-value (e.g. on
    /// [`MaxTokens`]).
    ///
    /// [`MaxTokens`]: crate::response::StopReason::MaxTokens
    #[error("JSON assembly error: {message}")]
    JsonAssembly {
        /// Error message.
        message: Cow<'static, str>,
        /// Index of the [`Content`] [`Block`] that failed.
        index: usize,
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
            Error::JsonAssembly { index, .. } => json!({
                "type": "json_assembly",
                "message": message,
                "index": index,
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
                        Ok(ApiResult::Error(ErrorEvent { error, .. })) => {
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

/// Incremental scanner yielding each completed element of the outermost JSON
/// array as its bytes arrive. The target array is the first one opened at the
/// root or as a direct value of the root object — the shape [`Items`]
/// serializes to. Drives [`FilterExt::with_json`].
///
/// [`Items`]: crate::prompt::Items
#[derive(Debug, Default)]
struct ArrayScanner {
    /// Bytes seen so far for the block.
    buf: String,
    /// Scan cursor into [`buf`](Self::buf). Structural bytes are ASCII, so
    /// it always lands on a char boundary.
    pos: usize,
    /// Open `{` / `[` containers at the cursor.
    depth: usize,
    in_string: bool,
    escaped: bool,
    /// [`depth`](Self::depth) of elements inside the target array, once
    /// found. Cleared when the array closes so a sibling array can't
    /// re-target.
    element_depth: Option<usize>,
    /// Offset of the first byte of the element being scanned.
    element_start: Option<usize>,
    /// The target array closed cleanly.
    done: bool,
    /// An element failed to parse; scanning is abandoned.
    failed: bool,
}

impl ArrayScanner {
    /// Feed a chunk, returning the elements it completed.
    fn feed(
        &mut self,
        chunk: &str,
    ) -> Result<Vec<serde_json::Value>, serde_json::Error> {
        self.buf.push_str(chunk);
        let mut out = Vec::new();

        if self.failed {
            self.pos = self.buf.len();
            return Ok(out);
        }

        while self.pos < self.buf.len() {
            let i = self.pos;
            let b = self.buf.as_bytes()[i];
            self.pos += 1;

            if self.in_string {
                if self.escaped {
                    self.escaped = false;
                } else if b == b'\\' {
                    self.escaped = true;
                } else if b == b'"' {
                    self.in_string = false;
                }
                continue;
            }

            match b {
                b'"' => {
                    self.start_element(i);
                    self.in_string = true;
                }
                b'{' => {
                    self.start_element(i);
                    self.depth += 1;
                }
                b'[' => {
                    if !self.done
                        && self.element_depth.is_none()
                        && self.depth <= 1
                    {
                        // The first root-or-root-field array: the target.
                        self.depth += 1;
                        self.element_depth = Some(self.depth);
                    } else {
                        self.start_element(i);
                        self.depth += 1;
                    }
                }
                b'}' => {
                    self.depth = self.depth.saturating_sub(1);
                }
                b']' => {
                    if self.element_depth == Some(self.depth) {
                        // Closes the target array.
                        if let Some(start) = self.element_start.take() {
                            out.push(self.parse(start, i)?);
                        }
                        self.element_depth = None;
                        self.done = true;
                    }
                    self.depth = self.depth.saturating_sub(1);
                }
                b',' => {
                    if self.element_depth == Some(self.depth)
                        && let Some(start) = self.element_start.take()
                    {
                        out.push(self.parse(start, i)?);
                    }
                }
                b' ' | b'\t' | b'\n' | b'\r' => {}
                _ => self.start_element(i),
            }
        }

        Ok(out)
    }

    /// Mark `i` as the start of an element when the cursor sits directly
    /// inside the target array and no element is in progress.
    fn start_element(&mut self, i: usize) {
        if self.element_depth == Some(self.depth)
            && self.element_start.is_none()
        {
            self.element_start = Some(i);
        }
    }

    /// Parse one element's bytes, abandoning the scan on failure.
    fn parse(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<serde_json::Value, serde_json::Error> {
        serde_json::from_str(&self.buf[start..end])
            .inspect_err(|_| self.failed = true)
    }

    /// Whether the block ended mid-value — e.g. cut off by
    /// [`MaxTokens`](StopReason::MaxTokens). Parse failures already
    /// surfaced, so they don't count.
    fn is_truncated(&self) -> bool {
        !self.failed && (self.depth > 0 || self.in_string)
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
    ) -> impl futures::Stream<Item = Result<Delta, Error>> + Send {
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
        message: &mut Option<response::Message>,
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
                            if let Err(e) = message.inner.content.push_delta(delta.clone()) {
                                yield Err(e.into());
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
                            message.inner.content.push(
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
                            message.inner.content.push(tool_use.clone());
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Tool use received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::ServerToolUse { tool_use }) => {
                        if let Some(message) = message.as_mut() {
                            // No `From<tool::Use>` shortcut here: that builds a
                            // `Block::ToolUse`. A server tool use is its own
                            // block.
                            message.inner.content.push(
                                crate::prompt::message::Block::ServerToolUse {
                                    call: tool_use.clone(),
                                },
                            );
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Server tool use received before message start.".into(),
                                delta: None,
                            });
                        }
                    }
                    Ok(Event::MessageDelta { delta, usage }) => {
                        if let Some(message) = message.as_mut() {
                            message.apply_delta(delta.clone());
                            if let Some(usage) = usage {
                                message.usage += usage.clone();
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
                    | Ok(Event::JsonObject { .. })
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
    /// assembled message with you, use [`Self::with_message_ip`].
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
            // Whether the block being assembled is a server tool use, so we
            // emit the matching `Event` variant at the block's end.
            let mut is_server = false;
            let mut input = String::new();

            pin_mut!(stream);

            while let Some(result) = stream.next().await {
                match result {
                    Ok(Event::ContentBlockStart {
                        content_block: Block::ToolUse { call: empty }, .. }) => {
                        input.clear();
                        call = Some(empty);
                        is_server = false;
                    }
                    Ok(Event::ContentBlockStart {
                        content_block: Block::ServerToolUse { call: empty }, .. }) => {
                        input.clear();
                        call = Some(empty);
                        is_server = true;
                    }
                    Ok(Event::ContentBlockDelta { delta: Delta::Json { partial_json }, .. }) => {
                        input.push_str(&partial_json);
                    }
                    Ok(Event::ContentBlockStop { .. }) => {
                        if let Some(mut call) = call.take() {
                            // No deltas means the call arrived complete in
                            // `content_block_start` — a PTC / resumed-turn
                            // `tool_use`, or a zero-argument call. Keep its
                            // input as-is (captured in
                            // `ptc.sse.stream.jsonl`).
                            if !input.is_empty() {
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
                            }

                            if is_server {
                                yield Ok(Event::ServerToolUse { tool_use: call });
                            } else {
                                yield Ok(Event::ToolUse { tool_use: call });
                            }
                        }
                    }
                    event => yield event,
                }
            }
        }
    }

    /// Adds [`Event::JsonObject`] to the stream by incrementally scanning
    /// [`Text`] and tool-input JSON for completed elements of the outermost
    /// array (the [`Items`] shape). Elements are yielded the moment their
    /// closing byte arrives — before the block, let alone the message,
    /// completes. All original events still pass through.
    ///
    /// # Note:
    /// - Text blocks are scanned unconditionally — call this when
    ///   [`output_config`] is set (the block is then guaranteed to be JSON).
    /// - Apply *upstream* of [`with_tool_use`] / [`with_message`], which
    ///   consume the input JSON deltas this scans.
    /// - A block that ends mid-value (e.g. on [`MaxTokens`]) yields
    ///   [`Error::JsonAssembly`].
    ///
    /// [`Text`]: Block::Text
    /// [`Items`]: crate::prompt::Items
    /// [`output_config`]: crate::Prompt::output_config
    /// [`with_tool_use`]: FilterExt::with_tool_use
    /// [`with_message`]: FilterExt::with_message
    /// [`MaxTokens`]: StopReason::MaxTokens
    fn with_json(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = self;
            pin_mut!(stream);

            // Scanner for the in-progress block, keyed by its index.
            let mut scan: Option<(usize, ArrayScanner)> = None;

            while let Some(result) = stream.next().await {
                match &result {
                    Ok(Event::ContentBlockStart { index, content_block }) => {
                        scan = match content_block {
                            Block::Text { .. }
                            | Block::ToolUse { .. }
                            | Block::ServerToolUse { .. } => {
                                Some((*index, ArrayScanner::default()))
                            }
                            _ => None,
                        };
                    }
                    Ok(Event::ContentBlockDelta { index, delta }) => {
                        let chunk = match delta {
                            Delta::Text { text } => Some(text.as_ref()),
                            Delta::Json { partial_json } => {
                                Some(partial_json.as_ref())
                            }
                            _ => None,
                        };
                        if let Some(chunk) = chunk
                            && let Some((block, scanner)) = scan
                                .as_mut()
                                .filter(|(block, _)| block == index)
                        {
                            match scanner.feed(chunk) {
                                Ok(values) => for value in values {
                                    yield Ok(Event::JsonObject {
                                        index: *block,
                                        value,
                                    });
                                },
                                Err(error) => {
                                    yield Err(Error::JsonAssembly {
                                        message: format!(
                                            "Array element does not parse: {error}"
                                        ).into(),
                                        index: *block,
                                    });
                                }
                            }
                        }
                    }
                    Ok(Event::ContentBlockStop { index }) => {
                        if let Some((block, scanner)) =
                            scan.take_if(|(block, _)| block == index)
                            && scanner.is_truncated()
                        {
                            yield Err(Error::JsonAssembly {
                                message: "Block ended mid-JSON \
                                    (truncated output?)".into(),
                                index: block,
                            });
                        }
                    }
                    _ => {}
                }

                yield result;
            }
        }
    }

    /// [`with_json`], typed: yields each completed element of the outermost
    /// array deserialized as a `T`, dropping all other events (errors still
    /// pass through). Pair with a [`Prompt::structured_output`] of
    /// [`Items<T>`] so every element is guaranteed by the schema to be a
    /// `T`.
    ///
    /// [`with_json`]: FilterExt::with_json
    /// [`Prompt::structured_output`]: crate::Prompt::structured_output
    /// [`Items<T>`]: crate::prompt::Items
    fn json_items<T>(
        self,
    ) -> impl futures::Stream<Item = Result<T, Error>> + Send
    where
        T: serde::de::DeserializeOwned + Send,
    {
        self.with_json().filter_map(|result| async move {
            match result {
                Ok(Event::JsonObject { index, value }) => {
                    Some(serde_json::from_value(value).map_err(|error| {
                        Error::JsonAssembly {
                            message: format!(
                                "Element does not deserialize: {error}"
                            )
                            .into(),
                            index,
                        }
                    }))
                }
                Ok(_) => None,
                Err(error) => Some(Err(error)),
            }
        })
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
        Id, Prompt,
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
                assert_eq!(Role::from(message.inner.role), Role::Assistant);
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

    /// Replay a wrapped `*.sse.stream.jsonl` fixture — one
    /// `{"Ok": <event>}` / `{"Err": <error event>}` per line (see
    /// `test/data/README.md`) — as a stream. `Err` lines surface as the real
    /// typed [`Error::Anthropic`], exactly as the live stream would, so error
    /// frames get parse coverage rather than a placeholder.
    #[allow(clippy::result_large_err)] // see `Stream::new`: `Event` dominates.
    pub fn mock_stream_jsonl(
        text: &'static str,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        futures::stream::iter(text.lines().map(|line| {
            let res: Result<Event, ErrorEvent> =
                serde_json::from_str(line).unwrap();
            match res {
                Ok(event) => Ok(event),
                Err(error_event) => {
                    let data = serde_json::to_string(&error_event).unwrap();
                    Err(Error::Anthropic {
                        error: error_event.error,
                        event: eventsource_stream::Event {
                            event: "error".into(),
                            data,
                            id: "".into(),
                            retry: None,
                        },
                    })
                }
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

    // ArrayScanner unit tests. Chunk splits mirror the wire: the captures in
    // `test/data/incremental/` split mid-token, so feeds here split at the
    // nastiest spots (mid-escape, mid-number) on purpose.

    #[test]
    fn test_array_scanner_split_mid_escape() {
        let mut scanner = ArrayScanner::default();
        // First chunk ends on a pending escape inside a string element.
        let out = scanner.feed(r#"["a\"#).unwrap();
        assert!(out.is_empty());
        let out = scanner.feed(r#""x", 42"#).unwrap();
        assert_eq!(out, vec![serde_json::json!("a\"x")]);
        let out = scanner.feed(r#"]"#).unwrap();
        assert_eq!(out, vec![serde_json::json!(42)]);
        assert!(!scanner.is_truncated());
    }

    #[test]
    fn test_array_scanner_nested_containers() {
        let mut scanner = ArrayScanner::default();
        // Elements that are themselves arrays and objects; trailing root
        // fields after the target array are ignored.
        let out = scanner
            .feed(r#"{"items":[[1,2],{"a":[3]}],"x":1}"#)
            .unwrap();
        assert_eq!(
            out,
            vec![serde_json::json!([1, 2]), serde_json::json!({"a": [3]})]
        );
        assert!(!scanner.is_truncated());
    }

    #[test]
    fn test_array_scanner_targets_first_array_only() {
        let mut scanner = ArrayScanner::default();
        let out = scanner.feed(r#"{"a":[1],"b":[2]}"#).unwrap();
        assert_eq!(out, vec![serde_json::json!(1)]);
        assert!(!scanner.is_truncated());
    }

    #[test]
    fn test_array_scanner_no_array_no_emission() {
        let mut scanner = ArrayScanner::default();
        // A plain object — the common non-list tool input. Nothing to emit,
        // nothing truncated. The nested array is too deep to target.
        let out = scanner
            .feed(r#"{"location": {"coords": [1, 2]}, "unit": "C"}"#)
            .unwrap();
        assert!(out.is_empty());
        assert!(!scanner.is_truncated());
    }

    #[test]
    fn test_array_scanner_truncated() {
        let mut scanner = ArrayScanner::default();
        let out = scanner.feed(r#"{"items":[{"a":1},{"b""#).unwrap();
        assert_eq!(out, vec![serde_json::json!({"a": 1})]);
        assert!(scanner.is_truncated());
    }

    #[test]
    fn test_array_scanner_unicode() {
        let mut scanner = ArrayScanner::default();
        let out = scanner.feed(r#"{ "items" : [ "héllo 🌍" ,"#).unwrap();
        assert_eq!(out, vec![serde_json::json!("héllo 🌍")]);
        let out = scanner.feed(r#" {"emoji": "🦀"} ] }"#).unwrap();
        assert_eq!(out, vec![serde_json::json!({"emoji": "🦀"})]);
        assert!(!scanner.is_truncated());
    }

    /// The shared `Item` schema both incremental fixtures were captured
    /// against (see `test/data/README.md`).
    #[derive(Debug, Deserialize, PartialEq)]
    struct GroceryItem {
        name: String,
        quantity: u32,
        #[serde(default)]
        note: Option<String>,
    }

    #[tokio::test]
    async fn test_json_items_structured_output() {
        // Structured output (`output_config`): JSON arrives in `text_delta`s.
        let items: Vec<GroceryItem> = mock_stream_jsonl(include_str!(
            "../test/data/incremental/structured_items.sse.stream.jsonl"
        ))
        .json_items()
        .try_collect()
        .await
        .unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].name, "granny smith apples");
        assert_eq!(items[0].quantity, 3);
        assert_eq!(items[2].note.as_deref(), Some("carton"));
    }

    #[tokio::test]
    async fn test_json_items_tool_use() {
        // The same list arriving as a tool call's `input_json_delta`s, split
        // mid-token across 21 frames.
        let items: Vec<GroceryItem> = mock_stream_jsonl(include_str!(
            "../test/data/incremental/tool_items.sse.stream.jsonl"
        ))
        .json_items()
        .try_collect()
        .await
        .unwrap();

        assert_eq!(items.len(), 3);
        assert_eq!(items[0].name, "granny smith apples");
        assert_eq!(items[2].name, "oat milk");
    }

    #[tokio::test]
    async fn test_with_json_composes_with_message() {
        // `with_json` upstream of `with_message`: elements stream out early
        // and the complete message still assembles.
        let events: Vec<Event> = mock_stream_jsonl(include_str!(
            "../test/data/incremental/structured_items.sse.stream.jsonl"
        ))
        .with_json()
        .with_message()
        .try_collect()
        .await
        .unwrap();

        let n_json = events.iter().filter(|e| e.is_json_object()).count();
        assert_eq!(n_json, 3);
        let message = events
            .iter()
            .find_map(|e| match e {
                Event::Message { message } => Some(message),
                _ => None,
            })
            .expect("with_message should assemble a complete message");
        // The assembled message carries the full JSON text block.
        assert!(message.inner.content.to_string().contains("oat milk"));
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

    // A real `web_fetch` server tool use, streamed (captured from the live API
    // via `curl`): an empty `server_tool_use` start, then `input_json_delta`s
    // spelling out `{"url": "https://www.rust-lang.org"}`, then a stop.
    const SERVER_TOOL_USE_STREAM: &str = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srvtoolu_012jyo3ThP6CEiKLRKJUrBXA\",\"name\":\"web_fetch\",\"input\":{}}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"url\"}}\n",
        "\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\": \\\"https://www.rust-lang.org\\\"}\"}}\n",
        "\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n",
        "\n",
    );

    #[tokio::test]
    async fn test_stream_with_server_tool_use() {
        let stream = mock_stream(SERVER_TOOL_USE_STREAM).with_tool_use();
        let mut server_tool_use = None;

        pin_mut!(stream);
        while let Some(event) = stream.next().await {
            dbg!(&event);
            match event {
                // A server tool use must NOT come back as a plain `ToolUse`.
                Ok(Event::ToolUse { .. }) => {
                    panic!("server tool use mis-assembled as a client ToolUse")
                }
                Ok(Event::ServerToolUse { tool_use: new }) => {
                    server_tool_use = Some(new);
                    break;
                }
                _ => {}
            }
        }

        let server_tool_use =
            server_tool_use.expect("no server tool use assembled");
        assert_eq!(
            serde_json::to_value(server_tool_use).unwrap(),
            serde_json::json!({
                "id": "srvtoolu_012jyo3ThP6CEiKLRKJUrBXA",
                "name": "web_fetch",
                "input": { "url": "https://www.rust-lang.org" }
            })
        );
    }

    // The server-tool *result* arrives mid-stream as a `content_block_start`
    // carrying the whole block inline (no deltas) — shape captured verbatim
    // from the live API via `curl`. `with_tool_use` must pass it through
    // untouched (it only intercepts tool-use *calls*), so it reaches
    // `with_message` assembly as a `Block::WebFetchToolResult`.
    const WEB_FETCH_RESULT_STREAM: &str = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":2,\"content_block\":{\"type\":\"web_fetch_tool_result\",\"tool_use_id\":\"srvtoolu_012jyo3ThP6CEiKLRKJUrBXA\",\"content\":{\"type\":\"web_fetch_result\",\"url\":\"https://www.rust-lang.org\",\"retrieved_at\":\"2026-06-04T11:50:09.370326\",\"content\":{\"type\":\"document\",\"source\":{\"type\":\"text\",\"media_type\":\"text/plain\",\"data\":\"Rust is a language...\"}}}}}\n",
        "\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":2}\n",
        "\n",
    );

    #[tokio::test]
    async fn test_stream_web_fetch_result_passes_through() {
        use crate::prompt::message::{Block, WebFetchToolResultContent};

        let stream = mock_stream(WEB_FETCH_RESULT_STREAM).with_tool_use();
        let mut seen = false;

        pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let Ok(Event::ContentBlockStart {
                content_block:
                    Block::WebFetchToolResult {
                        tool_use_id,
                        content,
                        ..
                    },
                ..
            }) = event
            {
                assert_eq!(tool_use_id, "srvtoolu_012jyo3ThP6CEiKLRKJUrBXA");
                let WebFetchToolResultContent::Result { url, .. } = content
                else {
                    panic!("expected a successful fetch result");
                };
                assert_eq!(url, "https://www.rust-lang.org");
                seen = true;
            }
        }

        assert!(seen, "web_fetch_tool_result did not survive with_tool_use");
    }

    // A real `bash_code_execution` result, streamed (captured from the live API
    // via `curl`): the whole result block arrives inline in a single
    // `content_block_start` with no deltas, exactly like the other server-tool
    // result blocks. `with_tool_use` must pass it through untouched.
    const BASH_CODE_EXECUTION_RESULT_STREAM: &str = concat!(
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"bash_code_execution_tool_result\",\"tool_use_id\":\"srvtoolu_01V2pLZmnVF7hwGxJQQb1uD1\",\"content\":{\"type\":\"bash_code_execution_result\",\"stdout\":\"streaming-test\\n\",\"stderr\":\"\",\"return_code\":0,\"content\":[]}}}\n",
        "\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n",
        "\n",
    );

    #[tokio::test]
    async fn test_stream_bash_code_execution_result_passes_through() {
        use crate::prompt::message::{BashCodeExecutionResultContent, Block};

        let stream =
            mock_stream(BASH_CODE_EXECUTION_RESULT_STREAM).with_tool_use();
        let mut seen = false;

        pin_mut!(stream);
        while let Some(event) = stream.next().await {
            if let Ok(Event::ContentBlockStart {
                content_block:
                    Block::BashCodeExecutionToolResult {
                        tool_use_id,
                        content,
                        ..
                    },
                ..
            }) = event
            {
                assert_eq!(tool_use_id, "srvtoolu_01V2pLZmnVF7hwGxJQQb1uD1");
                let BashCodeExecutionResultContent::Result { stdout, .. } =
                    content
                else {
                    panic!("expected a ran command");
                };
                assert_eq!(stdout, "streaming-test\n");
                seen = true;
            }
        }

        assert!(
            seen,
            "bash_code_execution_tool_result did not survive with_tool_use"
        );
    }

    /// The full PTC turn-1 stream, captured live
    /// (`test/data/server_tools/ptc.sse.stream.jsonl`): a `server_tool_use`
    /// assembled from `input_json_delta`s, a complete `tool_use` with a
    /// `code_execution` caller, and — crucially — the `container` arriving in
    /// the final `message_delta`, *not* `message_start`. Dropping it would
    /// make the paused turn impossible to resume; this pins the
    /// [`MessageDelta::container`] fix.
    #[tokio::test]
    async fn test_stream_ptc_container_survives_assembly() {
        const JSONL: &str =
            include_str!("../test/data/server_tools/ptc.sse.stream.jsonl");

        let stream = mock_stream_jsonl(JSONL).with_message();
        pin_mut!(stream);

        let mut assembled = None;
        while let Some(event) = stream.next().await {
            if let Ok(Event::Message { message }) = event {
                assembled = Some(message);
            }
        }

        let message = assembled.expect("stream assembles a message");
        let container = message
            .container
            .as_ref()
            .expect("container survives assembly");
        assert!(container.id.starts_with("container_"));
        assert!(matches!(
            message.stop_reason,
            Some(response::StopReason::ToolUse)
        ));
        let call = message.tool_use().expect("PTC tool_use assembled");
        assert_eq!(call.name, "query_sales");
    }

    /// A *resumed* PTC turn, captured live
    /// (`test/data/server_tools/ptc_resume.sse.stream.jsonl`): the API
    /// replays the paused message as a `message_start` with **pre-populated
    /// content** (a complete `tool_use` with caller), `container`, and
    /// `stop_reason` already set — followed immediately by `message_stop`,
    /// with no content_block or message_delta events at all. Assembly must
    /// surface that message as-is.
    #[tokio::test]
    async fn test_stream_ptc_resume_prepopulated_message_start() {
        const JSONL: &str = include_str!(
            "../test/data/server_tools/ptc_resume.sse.stream.jsonl"
        );

        let stream = mock_stream_jsonl(JSONL).with_message();
        pin_mut!(stream);

        let mut assembled = None;
        while let Some(event) = stream.next().await {
            if let Ok(Event::Message { message }) = event {
                assembled = Some(message);
            }
        }

        let message = assembled.expect("resumed stream assembles a message");
        assert!(message.container.is_some(), "container from message_start");
        assert!(matches!(
            message.stop_reason,
            Some(response::StopReason::ToolUse)
        ));
        let call = message.tool_use().expect("pre-populated tool_use");
        assert_eq!(call.name, "query_sales");
    }

    /// A paused server-tool turn and its continuation, captured live
    /// (`test/data/server_tools/pause_turn{,_resume}.sse.stream.jsonl`): 11
    /// sequential `web_fetch` rounds hit the server-side iteration cap and
    /// the turn pauses with [`StopReason::PauseTurn`] in the `message_delta`.
    /// Echoing the assembled assistant turn back (same tools, no new user
    /// message) resumes it — and unlike a PTC resume, the continuation's
    /// `message_start` is **empty** (a fresh adjacent assistant turn the API
    /// merges server-side, not a pre-populated replay): it delivers the
    /// in-flight result, runs the last fetch, and ends normally.
    #[tokio::test]
    async fn test_stream_pause_turn_and_resume() {
        use crate::prompt::message::Block;

        async fn assemble(jsonl: &'static str) -> response::Message {
            let stream = mock_stream_jsonl(jsonl).with_message();
            pin_mut!(stream);
            let mut assembled = None;
            while let Some(event) = stream.next().await {
                if let Ok(Event::Message { message }) = event {
                    assembled = Some(message);
                }
            }
            assembled.expect("stream assembles a message")
        }

        let fetches = |m: &response::Message| {
            m.inner
                .iter()
                .filter(|b| matches!(b, Block::WebFetchToolResult { .. }))
                .count()
        };

        let paused = assemble(include_str!(
            "../test/data/server_tools/pause_turn.sse.stream.jsonl"
        ))
        .await;
        assert!(matches!(
            paused.stop_reason,
            Some(response::StopReason::PauseTurn)
        ));
        // The turn pauses *mid-call*: the 11th `server_tool_use` is issued
        // but its result never arrives in this turn.
        let calls = paused
            .inner
            .iter()
            .filter(|b| matches!(b, Block::ServerToolUse { .. }))
            .count();
        assert_eq!(calls, 11, "11 fetch calls issued");
        assert_eq!(fetches(&paused), 10, "only 10 results before the pause");

        let resumed = assemble(include_str!(
            "../test/data/server_tools/pause_turn_resume.sse.stream.jsonl"
        ))
        .await;
        assert!(matches!(
            resumed.stop_reason,
            Some(response::StopReason::EndTurn)
        ));
        // The continuation first delivers the in-flight 11th result, then
        // issues and completes the 12th fetch.
        assert_eq!(fetches(&resumed), 2, "11th (pending) + 12th results");
    }

    /// A tool call whose input is a *list of objects*, captured live
    /// (`test/data/incremental/tool_items.sse.stream.jsonl`): the array
    /// arrives as 21 `input_json_delta` frames split mid-token. The #58
    /// incremental-parsing substrate; here, end-to-end through
    /// [`FilterExt::with_tool_use`] assembly.
    #[tokio::test]
    async fn test_stream_tool_items_list_assembles() {
        const JSONL: &str = include_str!(
            "../test/data/incremental/tool_items.sse.stream.jsonl"
        );

        let stream = mock_stream_jsonl(JSONL).with_tool_use();
        pin_mut!(stream);

        let mut call = None;
        while let Some(event) = stream.next().await {
            if let Ok(Event::ToolUse { tool_use }) = event {
                call = Some(tool_use);
            }
        }

        let call = call.expect("tool_use assembles from json deltas");
        assert_eq!(call.name, "add_items");
        let items = call.input["items"]
            .as_array()
            .expect("input.items is an array");
        assert_eq!(items.len(), 3, "three shopping-list items");
        assert!(
            items
                .iter()
                .all(|i| i["name"].is_string() && i["quantity"].is_u64())
        );
    }

    /// A structured-output generation with the same list-of-items schema,
    /// captured live
    /// (`test/data/incremental/structured_items.sse.stream.jsonl`): with
    /// [`Prompt::output_config`] the JSON arrives as plain `text_delta`s in a
    /// [`Block::Text`]. End-to-end: assemble with
    /// [`FilterExt::with_message`], parse with [`response::Message::json`].
    ///
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    #[tokio::test]
    async fn test_stream_structured_items_list_assembles() {
        const JSONL: &str = include_str!(
            "../test/data/incremental/structured_items.sse.stream.jsonl"
        );

        #[derive(serde::Deserialize)]
        struct Item {
            name: String,
            quantity: u64,
        }
        #[derive(serde::Deserialize)]
        struct ShoppingList {
            items: Vec<Item>,
        }

        let stream = mock_stream_jsonl(JSONL).with_message();
        pin_mut!(stream);

        let mut assembled = None;
        while let Some(event) = stream.next().await {
            if let Ok(Event::Message { message }) = event {
                assembled = Some(message);
            }
        }

        let message = assembled.expect("stream assembles a message");
        let list: ShoppingList =
            message.json().expect("text block parses as the schema");
        assert_eq!(list.items.len(), 3);
        assert!(
            list.items
                .iter()
                .any(|i| i.name.contains("apple") && i.quantity == 3)
        );
    }
}
