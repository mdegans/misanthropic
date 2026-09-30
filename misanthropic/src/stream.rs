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
        /// The turn's usage so far — cumulative, so it supersedes
        /// [`MessageStart`](Self::MessageStart)'s rather than adding to it.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        usage: Option<Usage>,
    },
    /// Message end.
    MessageStop,
    /// Complete [`response::Message`]. Assembled by [`FilterExt::with_message`]
    /// not the API. A call the turn left open (a [`MaxTokens`] clip) is
    /// closed as the non-streaming response closes it — its completed
    /// arguments only — so the message matches its non-streaming twin.
    ///
    /// [`MaxTokens`]: StopReason::MaxTokens
    Message {
        /// The message.
        message: response::Message,
    },
    /// Complete [`tool::Use`]. Assembled by [`FilterExt::with_tool_use`] not
    /// the API. Arrives when the block closes, *before* the turn's
    /// [`StopReason`] — don't run it here; dispatch from [`Event::Message`]
    /// via [`response::Message::tool_uses`].
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

/// The serialized shape of an [`Error`], tagged by `type`. Some of the wrapped
/// errors don't implement `Serialize`, so they ride as their message.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ErrorRepr<'a> {
    Stream {
        message: String,
    },
    Parse {
        message: String,
        event: EventRepr<'a>,
    },
    Anthropic {
        message: String,
        error: &'a AnthropicError,
        event: EventRepr<'a>,
    },
    MessageAssembly {
        message: String,
        delta: &'a Option<Delta>,
    },
    Delta {
        message: String,
        error: &'a DeltaError,
    },
    JsonAssembly {
        message: String,
        index: usize,
    },
}

/// An [`eventsource_stream::Event`], which isn't `Serialize` itself.
#[derive(Serialize)]
struct EventRepr<'a> {
    event: &'a str,
    data: &'a str,
    id: &'a str,
    retry: Option<std::time::Duration>,
}

impl<'a> From<&'a eventsource_stream::Event> for EventRepr<'a> {
    fn from(event: &'a eventsource_stream::Event) -> Self {
        Self {
            event: &event.event,
            data: &event.data,
            id: &event.id,
            retry: event.retry,
        }
    }
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let message = self.to_string();
        match self {
            Error::Stream { .. } => ErrorRepr::Stream { message },
            Error::Parse { event, .. } => ErrorRepr::Parse {
                message,
                event: event.into(),
            },
            Error::Anthropic { error, event } => ErrorRepr::Anthropic {
                message,
                error,
                event: event.into(),
            },
            Error::MessageAssembly { delta, .. } => {
                ErrorRepr::MessageAssembly { message, delta }
            }
            Error::Delta { error } => ErrorRepr::Delta { message, error },
            Error::JsonAssembly { index, .. } => ErrorRepr::JsonAssembly {
                message,
                index: *index,
            },
        }
        .serialize(serializer)
    }
}

/// A raw `data:` payload held for the [recorder](Stream::record): `Ok` for an
/// event line, `Err` for a wire `{"type":"error"}` line — the two arms of the
/// wrapped fixture format (see `test/data/README.md`). Pure bytes, never
/// re-serialized through our types (that would hide dropped fields, the exact
/// drift recordings exist to catch).
type RawLine = std::result::Result<String, String>;

/// One inner item: the parsed event (or error) plus the raw payload bytes the
/// [recorder](Stream::record) writes, when there are any.
type Recorded = (Result<Event, Error>, Option<RawLine>);

/// Stream of [`Event`]s or [`Error`]s.
pub struct Stream {
    inner: Pin<Box<dyn futures::Stream<Item = Recorded> + Send + 'static>>,
    recorder: Option<Recorder>,
}

/// The [`Stream::record`] sink and its write-ahead state machine.
struct Recorder {
    writer: Pin<Box<dyn futures::io::AsyncWrite + Send>>,
    state: RecordState,
}

/// Where the recorder is between events. The held item is yielded only once
/// its line is written *and flushed* — everything the consumer has seen is
/// already durably handed to the OS.
enum RecordState {
    /// Nothing pending.
    Idle,
    /// Writing `line` (from `written`) ahead of yielding `item`.
    Writing {
        line: Vec<u8>,
        written: usize,
        item: Option<Result<Event, Error>>,
    },
    /// Flushing the line ahead of yielding `item`.
    Flushing { item: Option<Result<Event, Error>> },
    /// The inner stream ended: closing the writer before yielding `None`.
    Closing,
}

impl Recorder {
    /// Drive the pending write/flush/close. `Ready(Some(item))` hands back the
    /// item now safe to yield; `Ready(None)` means the close finished. An
    /// `Err` carries the failed item (or `None` mid-close) so the caller can
    /// still yield it — a disk problem must not poison the conversation.
    #[allow(clippy::type_complexity)]
    fn poll(
        &mut self,
        cx: &mut std::task::Context,
    ) -> Poll<
        std::result::Result<
            Option<Result<Event, Error>>,
            (std::io::Error, Option<Result<Event, Error>>),
        >,
    > {
        loop {
            match &mut self.state {
                RecordState::Idle => return Poll::Ready(Ok(None)),
                RecordState::Writing {
                    line,
                    written,
                    item,
                } => {
                    while *written < line.len() {
                        match self
                            .writer
                            .as_mut()
                            .poll_write(cx, &line[*written..])
                        {
                            Poll::Ready(Ok(n)) => *written += n,
                            Poll::Ready(Err(e)) => {
                                let item = item.take();
                                self.state = RecordState::Idle;
                                return Poll::Ready(Err((e, item)));
                            }
                            Poll::Pending => return Poll::Pending,
                        }
                    }
                    self.state = RecordState::Flushing { item: item.take() };
                }
                RecordState::Flushing { item } => {
                    match self.writer.as_mut().poll_flush(cx) {
                        Poll::Ready(Ok(())) => {
                            let item = item.take();
                            self.state = RecordState::Idle;
                            return Poll::Ready(Ok(item));
                        }
                        Poll::Ready(Err(e)) => {
                            let item = item.take();
                            self.state = RecordState::Idle;
                            return Poll::Ready(Err((e, item)));
                        }
                        Poll::Pending => return Poll::Pending,
                    }
                }
                RecordState::Closing => {
                    return match self.writer.as_mut().poll_close(cx) {
                        Poll::Ready(Ok(())) => Poll::Ready(Ok(None)),
                        Poll::Ready(Err(e)) => Poll::Ready(Err((e, None))),
                        Poll::Pending => Poll::Pending,
                    };
                }
            }
        }
    }

    /// Queue `raw` (wrapped, one jsonl line) ahead of yielding `item`.
    fn enqueue(&mut self, raw: RawLine, item: Result<Event, Error>) {
        // Pure text concatenation, mirroring `test/data/capture.sh`.
        let (tag, data) = match &raw {
            Ok(data) => ("{\"Ok\":", data),
            Err(data) => ("{\"Err\":", data),
        };
        let mut line = Vec::with_capacity(tag.len() + data.len() + 2);
        line.extend_from_slice(tag.as_bytes());
        line.extend_from_slice(data.as_bytes());
        line.extend_from_slice(b"}\n");
        self.state = RecordState::Writing {
            line,
            written: 0,
            item: Some(item),
        };
    }
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

                    // The raw `data:` bytes ride alongside the parsed item
                    // for the recorder. The success arm *moves* them (the
                    // wire event is dropped anyway — recording or not, the
                    // cost is zero); the error arms clone because the wire
                    // event is preserved inside the `Error`.
                    match serde_json::from_str::<ApiResult>(&event.data) {
                        Ok(ApiResult::Event { event: parsed }) => {
                            (Ok(parsed), Some(Ok(event.data)))
                        }
                        Ok(ApiResult::Error(ErrorEvent { error, .. })) => {
                            let raw = event.data.clone();
                            (
                                Err(Error::Anthropic { error, event }),
                                Some(Err(raw)),
                            )
                        }
                        // A payload our types can't parse is the single most
                        // valuable thing to record — wrapped `Ok`, it's still
                        // an event line, just one we don't model (yet).
                        Err(error) => {
                            let raw = event.data.clone();
                            (Err(Error::Parse { error, event }), Some(Ok(raw)))
                        }
                    }
                }
                Err(error) => {
                    #[cfg(feature = "log")]
                    log::error!("Stream error: {:?}", error);
                    // Transport errors have no payload line — `capture.sh`
                    // wouldn't have one either.
                    (Err(Error::Stream { error }), None)
                }
            })),
            recorder: None,
        }
    }

    /// Tee every raw `data:` payload into `writer` as one wrapped
    /// `{"Ok": <event>}` / `{"Err": <error event>}` jsonl line — the
    /// `*.sse.stream.jsonl` fixture format (see `test/data/README.md`),
    /// written by pure text concatenation, never re-serialized through the
    /// crate's own types. Point a long-running app at a file and every rare
    /// wire shape it ever encounters lands on disk already fixture-shaped.
    ///
    /// Each line is written **and flushed before its event is yielded**, so
    /// everything the consumer has seen is already handed to the OS — a
    /// recording survives the process dying mid-stream, which is exactly
    /// when the interesting shapes appear. (Flush is not fsync: it's a
    /// userspace-to-kernel push, microseconds against network-paced events.)
    /// The writer is closed when the stream ends.
    ///
    /// A write error stops the recording (logged on the `log` feature) but
    /// never poisons the stream itself: the conversation outlives a full
    /// disk. Payloads our types fail to parse are still recorded — those
    /// are the lines the fixtures exist to catch.
    pub fn record<W>(mut self, writer: W) -> Self
    where
        W: futures::io::AsyncWrite + Send + 'static,
    {
        self.recorder = Some(Recorder {
            writer: Box::pin(writer),
            state: RecordState::Idle,
        });
        self
    }

    /// Replay a [`record`](Self::record)ing (or a checked-in
    /// `*.sse.stream.jsonl` fixture) through the **real** parse path, exactly
    /// as the live wire would arrive: each line is unwrapped back to its raw
    /// `data:` payload bytes and fed through the same machinery as
    /// [`new`](Self::new). Total on purpose — a line that doesn't match the
    /// wrapped format passes through as raw payload and surfaces downstream
    /// as a typed [`Error::Parse`], so a corrupt recording fails loudly
    /// in-stream rather than silently dropping lines.
    pub fn replay(jsonl: &str) -> Self {
        let events: Vec<_> = jsonl
            .lines()
            .map(|line| {
                let data = line
                    .strip_prefix("{\"Ok\":")
                    .or_else(|| line.strip_prefix("{\"Err\":"))
                    .and_then(|rest| rest.strip_suffix('}'))
                    .unwrap_or(line);
                Ok(eventsource_stream::Event {
                    event: String::new(),
                    data: data.to_string(),
                    id: String::new(),
                    retry: None,
                })
            })
            .collect();
        Self::new(futures::stream::iter(events))
    }
}

impl futures::Stream for Stream {
    type Item = Result<Event, Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context,
    ) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            // Drive any pending write/flush/close ahead of new work — the
            // held item yields only once its line is durably written.
            if let Some(recorder) = this.recorder.as_mut() {
                let closing = matches!(recorder.state, RecordState::Closing);
                match recorder.poll(cx) {
                    Poll::Pending => return Poll::Pending,
                    Poll::Ready(Ok(Some(item))) => {
                        return Poll::Ready(Some(item));
                    }
                    Poll::Ready(Ok(None)) if closing => {
                        // The close finished: the recording is complete.
                        this.recorder = None;
                        return Poll::Ready(None);
                    }
                    Poll::Ready(Ok(None)) => {} // idle: poll the inner stream
                    Poll::Ready(Err((_error, item))) => {
                        #[cfg(feature = "log")]
                        log::error!("stream recording failed: {_error}");
                        // A disk problem must not poison the conversation:
                        // stop recording, keep streaming.
                        this.recorder = None;
                        match item {
                            Some(item) => return Poll::Ready(Some(item)),
                            None if closing => return Poll::Ready(None),
                            None => {}
                        }
                    }
                }
            }

            match this.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => match this.recorder.as_mut() {
                    // Close the writer before reporting the end.
                    Some(recorder) => {
                        recorder.state = RecordState::Closing;
                    }
                    None => return Poll::Ready(None),
                },
                Poll::Ready(Some((item, raw))) => {
                    match (this.recorder.as_mut(), raw) {
                        (Some(recorder), Some(raw)) => {
                            recorder.enqueue(raw, item);
                            // Loop: drive the write before yielding.
                        }
                        (_, _) => return Poll::Ready(Some(item)),
                    }
                }
            }
        }
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

/// Close a tool input the turn ended without closing — a `max_tokens` clip
/// — as the non-streaming path does: the completed members only, their open
/// containers closed. The wire stops at a member boundary, so normally only
/// brackets are missing; a member cut mid-value is dropped whole. `None`
/// when nothing completed.
fn close_partial(json: &str) -> Option<serde_json::Value> {
    /// An open container. A string ending in an object's `key` position is
    /// a key, not a completed member.
    enum Frame {
        Object { key: bool },
        Array,
    }

    if let Ok(value) = serde_json::from_str(json) {
        return Some(value);
    }

    let closers = |stack: &[Frame]| -> String {
        stack
            .iter()
            .rev()
            .map(|frame| match frame {
                Frame::Object { .. } => '}',
                Frame::Array => ']',
            })
            .collect()
    };
    // Whether a value ending here is a member value or an array element.
    let completes = |stack: &[Frame]| {
        matches!(
            stack.last(),
            Some(Frame::Object { key: false } | Frame::Array)
        )
    };

    let mut stack = Vec::new();
    // The last offset where every value before it is complete, and the
    // closers its open containers need.
    let mut cut: Option<(usize, String)> = None;
    let (mut in_string, mut escaped) = (false, false);
    // Start of a number or literal in progress.
    let mut scalar: Option<usize> = None;

    for (i, b) in json.bytes().enumerate() {
        if in_string {
            match (escaped, b) {
                (true, _) => escaped = false,
                (false, b'\\') => escaped = true,
                (false, b'"') => {
                    in_string = false;
                    if completes(&stack) {
                        cut = Some((i + 1, closers(&stack)));
                    }
                }
                _ => {}
            }
            continue;
        }
        // A scalar ends at the first delimiter: until then, a number could
        // still grow.
        let delimits =
            b.is_ascii_whitespace() || matches!(b, b',' | b'}' | b']');
        if delimits && scalar.take().is_some() && completes(&stack) {
            cut = Some((i, closers(&stack)));
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                stack.push(match b {
                    b'{' => Frame::Object { key: true },
                    _ => Frame::Array,
                });
                cut = Some((i + 1, closers(&stack)));
            }
            b'}' | b']' => {
                stack.pop();
                if completes(&stack) {
                    cut = Some((i + 1, closers(&stack)));
                }
            }
            b':' | b',' => {
                if let Some(Frame::Object { key }) = stack.last_mut() {
                    *key = b == b',';
                }
            }
            b if b.is_ascii_whitespace() => {}
            _ => {
                let start = *scalar.get_or_insert(i);
                // A literal can't grow: complete on its last byte. Bytes,
                // not `str`: `i` may sit inside a multi-byte char.
                let literal = &json.as_bytes()[start..=i];
                if matches!(literal, b"true" | b"false" | b"null")
                    && completes(&stack)
                {
                    scalar = None;
                    cut = Some((i + 1, closers(&stack)));
                }
            }
        }
    }

    // `end` follows an ASCII byte or sits on one: a char boundary.
    let (end, closers) = cut?;
    serde_json::from_str(&format!("{}{closers}", &json[..end])).ok()
}

/// What [`assemble_tool_uses`] yields.
enum Assembled {
    /// An event to pass on, a completed call included.
    Event(Result<Event, Error>),
    /// A call its turn ended without closing, its input closed by
    /// [`close_partial`]. Never shown as a completed call.
    Open(Block),
}

/// Assemble tool calls from their deltas: [`FilterExt::with_tool_use`]'s
/// engine, which also flushes a call left open for
/// [`FilterExt::with_message_ip`] to seat.
fn assemble_tool_uses<S>(stream: S) -> impl futures::Stream<Item = Assembled>
where
    S: futures::Stream<Item = Result<Event, Error>> + Send,
{
    async_stream::stream! {
        let mut call: Option<tool::Use> = None;
        // Whether the block being assembled is a server tool use, so we
        // emit the matching `Event` variant at the block's end.
        let mut is_server = false;
        let mut input = String::new();

        // A call still open when its turn moves on: flushed with whatever
        // completed. No deltas means the start's input stands.
        let open = |mut call: tool::Use, input: &str, is_server: bool| {
            if let Some(input) = close_partial(input) {
                call.input = input;
            }
            Assembled::Open(if is_server {
                Block::ServerToolUse { call }
            } else {
                Block::ToolUse { call }
            })
        };

        pin_mut!(stream);

        while let Some(result) = stream.next().await {
            if let Ok(
                Event::ContentBlockStart { .. }
                | Event::MessageDelta { .. }
                | Event::MessageStop,
            ) = &result
                && let Some(call) = call.take()
            {
                yield open(call, &input, is_server);
            }

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
                                    yield Assembled::Event(Err(Error::MessageAssembly {
                                        message: format!("Failed to parse JSON: {}", err).into(),
                                        delta: None,
                                    }));
                                    continue;
                                }
                            };
                        }

                        yield Assembled::Event(Ok(if is_server {
                            Event::ServerToolUse { tool_use: call }
                        } else {
                            Event::ToolUse { tool_use: call }
                        }));
                    }
                }
                event => yield Assembled::Event(event),
            }
        }

        if let Some(call) = call.take() {
            yield open(call, &input, is_server);
        }
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
    /// - Implies [`with_tool_use`], and seats a call the turn leaves open
    ///   (see [`Event::Message`]).
    ///
    /// [`with_tool_use`]: FilterExt::with_tool_use
    /// [`with_message`]: FilterExt::with_message
    fn with_message_ip(
        self,
        message: &mut Option<response::Message>,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        async_stream::stream! {
            let stream = assemble_tool_uses(self);

            pin_mut!(stream);

            // reset the message if it's not already None.
            *message = None;

            while let Some(assembled) = stream.next().await {
                let result = match assembled {
                    Assembled::Event(result) => result,
                    // Seated, not shown: `with_tool_use` never yields it.
                    Assembled::Open(block) => {
                        if let Some(message) = message.as_mut() {
                            message.inner.content.push(block);
                        } else {
                            yield Err(Error::MessageAssembly {
                                message: "Tool use received before message start.".into(),
                                delta: None,
                            });
                        }
                        continue;
                    }
                };

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
                                message.usage.apply_delta(usage.clone());
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
    ///
    /// # Note
    /// A call is yielded when its block closes, *before* the
    /// [`MessageDelta`] carrying the turn's [`StopReason`]. For display only:
    /// a [`Refusal`], [`StopSequence`] or [`MaxTokens`] stop can still
    /// follow, and such a call must not run — a stop sequence matched inside
    /// a string argument closes the block, the argument cut at the match. A
    /// clip at `max_tokens` streams a call's input only through its last
    /// completed argument and never closes the block: such a call is never
    /// yielded here, and never errors ([`with_message`] seats it, closed).
    /// To dispatch, use [`with_message`] and take
    /// [`response::Message::tool_uses`] from the final [`Event::Message`].
    ///
    /// [`Refusal`]: StopReason::Refusal
    /// [`StopSequence`]: StopReason::StopSequence
    /// [`MaxTokens`]: StopReason::MaxTokens
    /// [`with_message`]: FilterExt::with_message
    fn with_tool_use(
        self,
    ) -> impl futures::Stream<Item = Result<Event, Error>> + Send {
        assemble_tool_uses(self).filter_map(|assembled| async move {
            match assembled {
                Assembled::Event(result) => Some(result),
                Assembled::Open(_) => None,
            }
        })
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
    /// The serialized shapes, pinned to literal JSON so a change to them is
    /// a deliberate edit here.
    #[test]
    fn error_serializes_to_tagged_shapes() {
        let shape = |err: Error| serde_json::to_value(err).unwrap();
        let literal = |json: &str| {
            serde_json::from_str::<serde_json::Value>(json).unwrap()
        };
        let event = || eventsource_stream::Event {
            event: "error".into(),
            data: "{".into(),
            id: "7".into(),
            retry: None,
        };

        let error = serde_json::from_str::<u8>("{").unwrap_err();
        let message = format!("JSON error: {error}");
        assert_eq!(
            shape(Error::Parse {
                error,
                event: event()
            }),
            literal(&format!(
                r#"{{"type":"parse","message":"{message}",
                    "event":{{"event":"error","data":"{{","id":"7","retry":null}}}}"#
            ))
        );

        assert_eq!(
            shape(Error::Anthropic {
                error: AnthropicError::Overloaded {
                    message: "busy".into(),
                    retry_after: None,
                },
                event: event(),
            }),
            literal(
                r#"{"type":"anthropic","message":"API error: overloaded (529): busy",
                    "error":{"type":"overloaded_error","message":"busy"},
                    "event":{"event":"error","data":"{","id":"7","retry":null}}"#
            )
        );

        assert_eq!(
            shape(Error::MessageAssembly {
                message: "no start".into(),
                delta: None,
            }),
            literal(
                r#"{"type":"message_assembly",
                    "message":"Message assembly error: no start","delta":null}"#
            )
        );

        let delta = Delta::Text { text: "x".into() };
        let mismatch = DeltaError::ContentMismatch {
            error: ContentMismatch {
                from: delta,
                to: "Block::Thought",
            },
        };
        let message = Error::from(DeltaError::Parse {
            error: String::new(),
        })
        .to_string();
        assert!(message.starts_with("Delta error"));
        assert_eq!(
            shape(mismatch.into())["error"],
            literal(
                r#"{"ContentMismatch":{"error":{
                    "from":{"type":"text_delta","text":"x"},
                    "to":"Block::Thought"}}}"#
            )
        );

        assert_eq!(
            shape(Error::JsonAssembly {
                message: "cut off".into(),
                index: 2,
            }),
            literal(
                r#"{"type":"json_assembly",
                    "message":"JSON assembly error: cut off","index":2}"#
            )
        );
    }

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

    /// A raw SSE fixture's `data:` payloads in the wrapped jsonl format — the
    /// same pure text transform as `test/data/capture.sh` — so
    /// [`roundtrip_sse`](crate::utils::roundtrip_sse) can gate it per event.
    pub(crate) fn sse_jsonl(sse: &str) -> String {
        sse.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(|payload| match payload.starts_with(r#"{"type":"error""#) {
                true => format!("{{\"Err\":{payload}}}\n"),
                false => format!("{{\"Ok\":{payload}}}\n"),
            })
            .collect()
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

    /// Assemble a captured SSE fixture (see [`mock_stream_jsonl`]) into its
    /// response, as a streaming client would.
    pub(crate) fn assembled(jsonl: &'static str) -> crate::response::Message {
        assembled_from(mock_stream_jsonl(jsonl))
    }

    /// [`assembled`], from a raw SSE fixture (see [`mock_stream`]).
    pub(crate) fn assembled_sse(sse: &'static str) -> crate::response::Message {
        assembled_from(mock_stream(sse))
    }

    /// The last [`Event::Message`] `events` assemble into.
    fn assembled_from(
        events: impl futures::Stream<Item = Result<Event, Error>> + Send,
    ) -> crate::response::Message {
        use futures::StreamExt;

        let events = events.with_message();
        futures::executor::block_on(
            events
                .filter_map(async |event| match event {
                    Ok(Event::Message { message }) => Some(message),
                    _ => None,
                })
                .collect::<Vec<_>>(),
        )
        .pop()
        .expect("the fixture assembles a message")
    }

    /// An SSE `error` event arrives after the HTTP 200, so there is no
    /// status to take: an unrecognized `type` mid-stream stays
    /// `Unknown { code: None }`, while known types keep their implied one.
    #[tokio::test]
    async fn test_mid_stream_error_has_no_http_status() {
        use futures::StreamExt;

        const SSE: &str = "event: error\n\
            data: {\"type\":\"error\",\"error\":{\"type\":\"unknown\",\"message\":\"resample\"}}\n\
            \n\
            event: error\n\
            data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}\n\
            \n";

        let errors: Vec<_> = mock_stream(SSE)
            .map(|r| match r {
                Err(Error::Anthropic { error, .. }) => error,
                other => panic!("expected Error::Anthropic, got {other:?}"),
            })
            .collect()
            .await;

        assert_eq!(
            errors[0],
            AnthropicError::Unknown {
                code: None,
                message: "unknown: resample".to_string(),
            }
        );
        assert_eq!(errors[0].status(), None);
        assert_eq!(errors[1].status(), std::num::NonZeroU16::new(529));
    }

    /// `record(replay(fixture))` reproduces the file byte-for-byte: the
    /// recorder's wrapper concatenation is the exact inverse of `replay`'s
    /// unwrap slicing, both pure text — no trip through the crate's own
    /// types in either direction. Covers both halves of the round trip.
    #[test]
    fn test_record_reproduces_fixture() {
        use futures::StreamExt;

        const FIXTURE: &str = include_str!(
            "../test/data/system_after_server_tool.sse.stream.jsonl"
        );

        /// A clonable sink so the bytes survive `record` consuming it.
        #[derive(Clone, Default)]
        struct SharedBuf(std::sync::Arc<std::sync::Mutex<(Vec<u8>, bool)>>);
        impl futures::io::AsyncWrite for SharedBuf {
            fn poll_write(
                self: Pin<&mut Self>,
                _cx: &mut std::task::Context,
                buf: &[u8],
            ) -> Poll<std::io::Result<usize>> {
                self.0.lock().unwrap().0.extend_from_slice(buf);
                Poll::Ready(Ok(buf.len()))
            }
            fn poll_flush(
                self: Pin<&mut Self>,
                _cx: &mut std::task::Context,
            ) -> Poll<std::io::Result<()>> {
                Poll::Ready(Ok(()))
            }
            fn poll_close(
                self: Pin<&mut Self>,
                _cx: &mut std::task::Context,
            ) -> Poll<std::io::Result<()>> {
                self.0.lock().unwrap().1 = true;
                Poll::Ready(Ok(()))
            }
        }

        let sink = SharedBuf::default();
        let stream = Stream::replay(FIXTURE).record(sink.clone());

        let n = futures::executor::block_on(
            stream.fold(0usize, |n, _| async move { n + 1 }),
        );
        assert_eq!(n, FIXTURE.lines().count());

        let (bytes, closed) = {
            let guard = sink.0.lock().unwrap();
            (guard.0.clone(), guard.1)
        };
        assert!(closed, "the writer must be closed at end of stream");
        assert_eq!(std::str::from_utf8(&bytes).unwrap(), FIXTURE);
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

    /// `Event::ToolUse` fires before `message_delta` says why the turn
    /// stopped; only the assembled message's `tool_uses` knows whether the
    /// call may run. The refusal is synthetic — `sse.stream.txt` (from the
    /// API docs) with its stop reason swapped, no live capture — standing in
    /// for a refusal landing after a closed `tool_use` block.
    #[tokio::test]
    async fn tool_use_dispatch_waits_for_stop_reason() {
        async fn events(sse: &'static str) -> Vec<Event> {
            mock_stream(sse)
                .with_message()
                .filter_map(|result| async move { result.ok() })
                .collect()
                .await
        }
        async fn dispatchable(sse: &'static str) -> (usize, Vec<String>) {
            let events = events(sse).await;
            let early = events.iter().filter(|e| e.is_tool_use()).count();
            let calls = events
                .iter()
                .find_map(|event| match event {
                    Event::Message { message } => Some(
                        message
                            .tool_uses()
                            .map(|call| call.name.to_string())
                            .collect(),
                    ),
                    _ => None,
                })
                .expect("with_message yields the whole turn");
            (early, calls)
        }

        let captured = include_str!("../test/data/sse.stream.txt");
        let refused: &'static str = captured
            .replace(
                r#""stop_reason":"tool_use""#,
                r#""stop_reason":"refusal""#,
            )
            .leak();
        assert_ne!(captured, refused);

        assert_eq!(
            dispatchable(captured).await,
            (1, vec!["get_weather".into()])
        );
        // The block-close event still fires; the gate is what holds.
        assert_eq!(dispatchable(refused).await, (1, vec![]));
    }

    /// Synthetic: `sse.stream.txt` (minus its error events) cut off mid
    /// tool input — no closing `input_json_delta` or `content_block_stop` —
    /// then stopped for `max_tokens`. Nothing is dispatchable or shown, and
    /// the open block neither panics nor surfaces an error: it assembles
    /// closed, with its completed arguments.
    #[tokio::test]
    async fn truncated_tool_input_is_not_dispatchable() {
        let captured = include_str!("../test/data/sse.stream.txt");
        let cut = captured
            .find(r#"\"unit\""#)
            .expect("fixture has the second input chunk");
        let head = &captured[..captured[..cut].rfind("event: ").unwrap()];
        let truncated: &'static str = head
            .split_terminator("\n\n")
            .filter(|event| !event.starts_with("event: error"))
            .chain([
                "event: message_delta\n\
                 data: {\"type\":\"message_delta\",\"delta\":\
                 {\"stop_reason\":\"max_tokens\",\"stop_sequence\":null},\
                 \"usage\":{\"output_tokens\":89}}",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}",
            ])
            .map(|event| format!("{event}\n\n"))
            .collect::<String>()
            .leak();

        let results: Vec<_> =
            mock_stream(truncated).with_message().collect().await;
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        assert!(!results.iter().flatten().any(Event::is_tool_use));
        let message = results
            .into_iter()
            .flatten()
            .find_map(|event| match event {
                Event::Message { message } => Some(message),
                _ => None,
            })
            .expect("with_message yields the whole turn");
        assert!(message.stop_reason.unwrap().is_max_tokens());
        assert_eq!(message.disposition(), response::Disposition::Clipped);
        // The unclosed call assembles closed, the trailing `, ` dropped.
        let calls: Vec<_> = message.inner.content.tool_uses().collect();
        assert_eq!(calls.len(), 1);
        let expected = r#"{"location": "San Francisco, CA"}"#;
        let expected: serde_json::Value =
            serde_json::from_str(expected).unwrap();
        assert_eq!(calls[0].input, expected);
        assert_eq!(message.tool_uses().count(), 0);
        assert!(message.tool_use().is_none());
    }

    /// The `write_file` arguments of the `test/data/stop/` captures.
    #[derive(Debug, PartialEq, serde::Deserialize)]
    struct WriteFile {
        path: String,
        contents: String,
    }

    impl WriteFile {
        fn of(call: &tool::Use) -> Self {
            serde_json::from_value(call.input.clone()).unwrap()
        }
    }

    /// Replay a raw `test/data/stop/` capture: gate every event's exact
    /// round-trip, then return what [`FilterExt::with_message`] yields —
    /// every result, and the calls shown on block close.
    async fn replay_stop(
        sse: &'static str,
    ) -> (Vec<Result<Event, Error>>, Vec<tool::Use>) {
        crate::utils::roundtrip_sse(&sse_jsonl(sse)).assert_round_trips();
        let results: Vec<_> = mock_stream(sse).with_message().collect().await;
        let shown = results
            .iter()
            .flatten()
            .filter_map(|event| match event {
                Event::ToolUse { tool_use } => Some(tool_use.clone()),
                _ => None,
            })
            .collect();
        (results, shown)
    }

    /// Live (Haiku 4.5): a stop sequence (`print(`) matched inside a forced
    /// call's input. The API still closes the block — the input truncated at
    /// the match, but valid, closed JSON — so the call is *shown* on block
    /// close, and only the stop reason keeps it from dispatch.
    #[tokio::test]
    async fn stop_sequence_in_tool_input() {
        const SSE: &str =
            include_str!("../test/data/stop/stop_sequence_tool.sse.stream.txt");
        let (results, shown) = replay_stop(SSE).await;
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        let message = assembled_sse(SSE);

        assert_eq!(message.stop_reason, Some(StopReason::StopSequence));
        assert_eq!(message.stop_sequence.as_deref(), Some("print("));
        assert_eq!(message.disposition(), response::Disposition::Done);
        assert_eq!(message.tool_uses().count(), 0);
        assert!(message.tool_use().is_none());
        let raw: Vec<_> = message.inner.content.tool_uses().collect();
        assert_eq!(raw, shown.iter().collect::<Vec<_>>());

        // The same cut the non-streaming twin made.
        let twin: response::Message = serde_json::from_str(include_str!(
            "../test/data/stop/stop_sequence_tool.response.json"
        ))
        .unwrap();
        let twin_call = twin.inner.content.tool_uses().next().unwrap();
        assert_eq!(WriteFile::of(raw[0]), WriteFile::of(twin_call));
        assert_eq!(message.usage, twin.usage);
        assert_eq!(
            WriteFile::of(raw[0]),
            WriteFile {
                path: "hello.py".into(),
                contents: "import datetime\n".into(),
            }
        );
    }

    /// Live (Haiku 4.5): text, then a call cut by the stop sequence
    /// (`auto` tool choice). Same shape as a forced call: closed, truncated,
    /// shown, never dispatchable.
    #[tokio::test]
    async fn stop_sequence_after_text() {
        const SSE: &str = include_str!(
            "../test/data/stop/stop_sequence_text_tool.sse.stream.txt"
        );
        let (results, shown) = replay_stop(SSE).await;
        assert!(results.iter().all(Result::is_ok), "{results:?}");
        let message = assembled_sse(SSE);

        assert_eq!(message.stop_reason, Some(StopReason::StopSequence));
        assert_eq!(message.stop_sequence.as_deref(), Some("print("));
        assert_eq!(message.disposition(), response::Disposition::Done);
        assert_eq!(message.tool_uses().count(), 0);
        let [Block::Text { text, .. }, Block::ToolUse { call }] =
            &message.inner.content[..]
        else {
            panic!("text, then the call: {:?}", message.inner.content);
        };
        assert!(text.starts_with("I will create"), "{text}");
        assert_eq!(shown, std::slice::from_ref(call));
        assert_eq!(
            WriteFile::of(call),
            WriteFile {
                path: "hello.py".into(),
                contents: "".into(),
            }
        );

        // The twin is a separate generation: same input, its own output.
        let twin: response::Message = serde_json::from_str(include_str!(
            "../test/data/stop/stop_sequence_text_tool.response.json"
        ))
        .unwrap();
        let mut expected = twin.usage;
        expected.output_tokens = 72;
        assert_eq!(message.usage, expected);
    }

    /// Live (Haiku 4.5): forced calls clipped at `max_tokens` — early, and
    /// 140 tokens deep into `contents`. The input chunks stop at the last
    /// completed member (the one in progress is never sent) and the wire
    /// never closes the block — no `content_block_stop` — then
    /// `message_delta` and `message_stop` follow. No error and no call
    /// surface, but the turn assembles *with* the call, closed: exactly the
    /// non-streaming twin's content, valid JSON holding only the completed
    /// members. Either way it is [`Clipped`], so nothing dispatches.
    ///
    /// [`Clipped`]: response::Disposition::Clipped
    #[tokio::test]
    async fn clip_closes_the_open_call() {
        let cases = [
            (
                include_str!("../test/data/stop/clip_tool.sse.stream.txt"),
                include_str!("../test/data/stop/clip_tool.response.json"),
                r#"{"path": "hello.py""#,
            ),
            (
                include_str!("../test/data/stop/clip_long_tool.sse.stream.txt"),
                include_str!("../test/data/stop/clip_long_tool.response.json"),
                r#"{"path": "story.txt""#,
            ),
        ];
        for (sse, twin, streamed_input) in cases {
            let (results, shown) = replay_stop(sse).await;
            assert!(results.iter().all(Result::is_ok), "{results:?}");
            assert!(shown.is_empty());
            let messages = results.iter().flatten().filter(|e| e.is_message());
            assert_eq!(messages.count(), 1, "one turn assembles");
            let wire: Vec<_> = mock_stream(sse).try_collect().await.unwrap();
            assert!(!wire.iter().any(Event::is_content_block_stop));
            let input: String = wire
                .iter()
                .filter_map(|event| match event {
                    Event::ContentBlockDelta {
                        delta: Delta::Json { partial_json },
                        ..
                    } => Some(partial_json.as_ref()),
                    _ => None,
                })
                .collect();
            assert_eq!(input, streamed_input, "completed members only");
            let message = assembled_sse(sse);

            assert_eq!(message.stop_reason, Some(StopReason::MaxTokens));
            assert_eq!(message.disposition(), response::Disposition::Clipped);
            assert_eq!(message.tool_uses().count(), 0);
            assert!(message.tool_use().is_none());

            // The twin is a separate generation: only the call's id differs.
            let twin: response::Message = serde_json::from_str(twin).unwrap();
            let [Block::ToolUse { call }] = &message.inner.content[..] else {
                panic!("the closed call: {:?}", message.inner.content);
            };
            let mut expected = twin.inner.content.clone();
            if let Some(Block::ToolUse { call: twin_call }) =
                expected.first_mut()
            {
                twin_call.id = call.id.clone();
            }
            assert_eq!(message.inner.content, expected);
            assert_eq!(message.usage, twin.usage);

            // `with_tool_use` alone: the open call is never shown.
            let events: Vec<_> =
                mock_stream(sse).with_tool_use().collect().await;
            assert!(events.iter().all(Result::is_ok), "{events:?}");
            assert!(!events.iter().flatten().any(Event::is_tool_use));
        }
    }

    /// [`close_partial`] keeps completed members only, closing their
    /// containers: whatever is mid-value is dropped whole.
    #[test]
    fn close_partial_keeps_completed_members() {
        let cases = [
            // Complete input passes through.
            (r#"{"a": 1}"#, Some(r#"{"a": 1}"#)),
            // The wire's shape: only brackets missing.
            (r#"{"path": "story.txt""#, Some(r#"{"path": "story.txt"}"#)),
            (
                r#"{"a": {"b": [1, "x"], "c": "y""#,
                Some(r#"{"a": {"b": [1, "x"], "c": "y"}}"#),
            ),
            (
                r#"{"a": [{"b": null}, {"#,
                Some(r#"{"a": [{"b": null}, {}]}"#),
            ),
            (r#"{"a": true"#, Some(r#"{"a": true}"#)),
            // Mid-string, mid-escape, mid-key, mid-number: dropped whole.
            (
                r#"{"path": "story.txt", "contents": "Once"#,
                Some(r#"{"path": "story.txt"}"#),
            ),
            (r#"{"a": "x\"#, Some("{}")),
            (r#"{"a": 1, "b"#, Some(r#"{"a": 1}"#)),
            (r#"{"a": 1, "b": "#, Some(r#"{"a": 1}"#)),
            (r#"{"a": [1, 2"#, Some(r#"{"a": [1]}"#)),
            (r#"{"a": 12 "#, Some(r#"{"a": 12}"#)),
            (r#"{"a": {"#, Some(r#"{"a": {}}"#)),
            (r#"{"#, Some("{}")),
            ("", None),
            ("tru", None),
            // Non-ASCII outside a string: dropped, never sliced mid-char.
            (r#"{"a": é"#, Some("{}")),
            (r#"{"a": 1é"#, Some("{}")),
            (r#"{"a": tré"#, Some("{}")),
            (r#"{é"#, Some("{}")),
            (r#"[é"#, Some("[]")),
            ("é", None),
            // Inside a string it's just text.
            (r#"{"a": "é""#, Some(r#"{"a": "é"}"#)),
        ];
        for (partial, closed) in cases {
            let closed = closed.map(|c| serde_json::from_str(c).unwrap());
            assert_eq!(close_partial(partial), closed, "{partial}");
        }
    }

    /// [`close_partial`] never panics: every prefix of realistic inputs
    /// (non-ASCII, escapes, nesting), and every short string over a
    /// structural alphabet. A whole input closes to itself.
    #[test]
    fn close_partial_never_panics() {
        let payloads = [
            r#"{"path": "café/naïve.txt", "line": 42, "ok": true}"#,
            r#"{"s": "a\"b\\cé\n", "n": [1.5e3, -2, null, false]}"#,
            r#"{"a": {"b": [{"c": "日本語"}, "🦀"]}, "d": {}}"#,
            r#"[{"é": 1}, ["x", [ ]], "\\"]"#,
        ];
        for payload in payloads {
            let whole: serde_json::Value =
                serde_json::from_str(payload).unwrap();
            assert_eq!(close_partial(payload), Some(whole), "{payload}");
            payload
                .char_indices()
                .map(|(i, _)| &payload[..i])
                .for_each(|prefix| drop(close_partial(prefix)));
        }

        // Every string up to four chars over the bytes that steer the scan.
        let alphabet =
            ['{', '}', '[', ']', '"', '\\', ':', ',', ' ', '1', 't', 'é'];
        let mut strings = vec![String::new()];
        for _ in 0..4 {
            strings = strings
                .iter()
                .flat_map(|s| alphabet.iter().map(move |c| format!("{s}{c}")))
                .collect();
            strings.iter().for_each(|s| drop(close_partial(s)));
        }
    }

    /// Every captured stream assembles its turn's *final* usage: the
    /// cumulative `message_delta` report, over `message_start`'s for any
    /// counter it omits — never the two summed. (`thinking.sse.stream.txt`,
    /// from the docs, reports no usage.)
    #[test]
    fn assembled_usage_is_the_final_report() {
        use crate::response::message::{
            CacheCreation, ServerToolUsage, TokenCounts,
        };

        // A current capture: cache counters, and `message_start`'s TTL
        // breakdown (deltas carry none).
        let counts = |input, output| TokenCounts {
            cache_creation_input_tokens: Some(0),
            cache_creation: Some(CacheCreation::default()),
            cache_read_input_tokens: Some(0),
            ..TokenCounts::new(input, output)
        };
        // A server-tool turn, counting (searches, fetches).
        let tools = |input, output, (searches, fetches)| TokenCounts {
            server_tool_use: Some(ServerToolUsage {
                web_search_requests: searches,
                web_fetch_requests: fetches,
                tool_search_requests: None,
            }),
            ..counts(input, output)
        };

        let cases = [
            // An old delta reporting only `output_tokens`.
            (
                "sse",
                assembled_sse(include_str!("../test/data/sse.stream.txt")),
                TokenCounts::new(472, 89),
            ),
            // No delta usage: `message_start`'s stands.
            (
                "redacted_thought",
                assembled(include_str!(
                    "../test/data/redacted_thought.sse.stream.jsonl"
                )),
                TokenCounts {
                    cache_creation_input_tokens: Some(0),
                    cache_read_input_tokens: Some(0),
                    ..TokenCounts::new(92, 3)
                },
            ),
            (
                "text",
                assembled(include_str!("../test/data/text.sse.stream.jsonl")),
                counts(11, 4),
            ),
            (
                "structured_items",
                assembled(include_str!(
                    "../test/data/incremental/structured_items.sse.stream.jsonl"
                )),
                counts(284, 47),
            ),
            (
                "tool_items",
                assembled(include_str!(
                    "../test/data/incremental/tool_items.sse.stream.jsonl"
                )),
                counts(739, 91),
            ),
            (
                "system_after_server_tool",
                assembled(include_str!(
                    "../test/data/system_after_server_tool.sse.stream.jsonl"
                )),
                TokenCounts {
                    output_tokens_details: Some(
                        crate::response::message::OutputTokensDetails {
                            thinking_tokens: 0,
                        },
                    ),
                    ..counts(1305, 27)
                },
            ),
            // Server tools: the delta's input grows past `message_start`'s.
            (
                "code_execution",
                assembled(include_str!(
                    "../test/data/server_tools/code_execution.sse.stream.jsonl"
                )),
                tools(13375, 534, (0, 0)),
            ),
            (
                "code_execution_result",
                assembled(include_str!(
                    "../test/data/server_tools/code_execution_result.sse.stream.jsonl"
                )),
                tools(3363, 113, (0, 0)),
            ),
            (
                "pause_turn",
                assembled(include_str!(
                    "../test/data/server_tools/pause_turn.sse.stream.jsonl"
                )),
                tools(22682, 902, (0, 10)),
            ),
            (
                "pause_turn_resume",
                assembled(include_str!(
                    "../test/data/server_tools/pause_turn_resume.sse.stream.jsonl"
                )),
                tools(6058, 362, (0, 2)),
            ),
            (
                "ptc",
                assembled(include_str!(
                    "../test/data/server_tools/ptc.sse.stream.jsonl"
                )),
                tools(3164, 153, (0, 0)),
            ),
            // A pre-populated `message_start`, no delta.
            (
                "ptc_resume",
                assembled(include_str!(
                    "../test/data/server_tools/ptc_resume.sse.stream.jsonl"
                )),
                TokenCounts {
                    server_tool_use: Some(ServerToolUsage::default()),
                    ..TokenCounts::default()
                },
            ),
            (
                "tool_search",
                assembled(include_str!(
                    "../test/data/server_tools/tool_search.sse.stream.jsonl"
                )),
                tools(1641, 163, (0, 0)),
            ),
            (
                "web_fetch",
                assembled(include_str!(
                    "../test/data/server_tools/web_fetch.sse.stream.jsonl"
                )),
                tools(5956, 140, (0, 1)),
            ),
            (
                "web_search",
                assembled(include_str!(
                    "../test/data/server_tools/web_search.sse.stream.jsonl"
                )),
                tools(12116, 126, (1, 0)),
            ),
            (
                "stop_sequence_tool",
                assembled_sse(include_str!(
                    "../test/data/stop/stop_sequence_tool.sse.stream.txt"
                )),
                counts(685, 34),
            ),
            (
                "stop_sequence_text_tool",
                assembled_sse(include_str!(
                    "../test/data/stop/stop_sequence_text_tool.sse.stream.txt"
                )),
                counts(595, 72),
            ),
            (
                "clip_tool",
                assembled_sse(include_str!(
                    "../test/data/stop/clip_tool.sse.stream.txt"
                )),
                counts(685, 30),
            ),
            (
                "clip_long_tool",
                assembled_sse(include_str!(
                    "../test/data/stop/clip_long_tool.sse.stream.txt"
                )),
                counts(687, 140),
            ),
        ];
        for (name, message, expected) in cases {
            assert_eq!(message.usage.counts, expected, "{name}");
        }

        // Where a non-streaming twin exists, the whole `Usage` matches it.
        let twin: response::Message = serde_json::from_str(include_str!(
            "../test/data/system_after_server_tool.response.json"
        ))
        .unwrap();
        let streamed = assembled(include_str!(
            "../test/data/system_after_server_tool.sse.stream.jsonl"
        ));
        assert_eq!(streamed.usage, twin.usage);
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
