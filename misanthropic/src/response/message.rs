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
#[non_exhaustive]
pub struct Message {
    /// Unique `id` for the message.
    pub id: Cow<'static, str>,
    /// Object-type discriminator the wire sends inside `message_start` (and
    /// non-streaming responses): always `"message"`. Absent on older API
    /// versions, so optional and skipped when `None`.
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<Kind>,
    /// Inner [`prompt::message`].
    #[serde(flatten)]
    pub inner: prompt::AssistantMessage,
    /// [`crate::model::Model`] that generated the message.
    pub model: model::Model,
    /// The reason the model stopped generating tokens.
    pub stop_reason: Option<StopReason>,
    /// If the [`StopReason`] was [`StopSequence`], this is the sequence that
    /// triggered it.
    ///
    /// [`StopSequence`]: StopReason::StopSequence
    pub stop_sequence: Option<Cow<'static, str>>,
    /// Structured detail about why the model stopped — populated on
    /// [`Refusal`](StopReason::Refusal), explicitly `null` otherwise. See
    /// [`StopDetails`].
    ///
    /// Boxed because it is absent on the vast majority of turns (only
    /// refusals populate it), keeping [`Message`] small.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_details: Option<Box<StopDetails>>,
    /// Usage statistics for the message.
    #[serde(default)]
    pub usage: Usage,
    /// The [code execution] container backing this turn, present when the
    /// request used the [`code_execution`] tool. Pass its
    /// `id` to [`Prompt::container`] to resume the *same*
    /// container — required when a [programmatic tool call] paused the turn and
    /// you are sending its [`tool::Result`](crate::tool::Result) back.
    ///
    /// [code execution]: crate::tool::ServerMethodDef::code_execution
    /// [`code_execution`]: crate::tool::ServerMethodDef::code_execution
    /// [`Prompt::container`]: crate::Prompt::container
    /// [programmatic tool call]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    ///
    /// Boxed because it is absent on the vast majority of turns (only code
    /// execution populates it), keeping [`Message`] small.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub container: Option<Box<Container>>,
}

/// The [code execution] container backing a response turn (the `container`
/// field). Its [`id`](Self::id) is what you pass to
/// [`Prompt::container`](crate::Prompt::container) to resume the same sandbox.
///
/// [code execution]: crate::tool::ServerMethodDef::code_execution
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[non_exhaustive]
pub struct Container {
    /// The container id (`container_…`), reused via [`Prompt::container`].
    ///
    /// [`Prompt::container`]: crate::Prompt::container
    pub id: Cow<'static, str>,
    /// When the container expires (RFC 3339). Idle containers are reaped after
    /// ~4.5 minutes; respond before then to keep a paused turn alive.
    pub expires_at: Cow<'static, str>,
}

impl Message {
    /// Apply a [`MessageDelta`] with metadata to the message.
    pub fn apply_delta(&mut self, delta: MessageDelta) {
        if let Some(stop_reason) = delta.stop_reason {
            self.stop_reason = Some(stop_reason);
        }
        if let Some(stop_sequence) = delta.stop_sequence {
            self.stop_sequence = Some(stop_sequence);
        }
        if let Some(stop_details) = delta.stop_details {
            self.stop_details = Some(stop_details);
        }
        if let Some(container) = delta.container {
            self.container = Some(container);
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

    /// Remove an incomplete thought from the message. If after removal, the
    /// message is empty, `None` is returned.
    ///
    /// See also [`prompt::Message::remove_incomplete_thought`].
    pub fn remove_incomplete_thought(self) -> Option<Self> {
        let inner = self.inner.remove_incomplete_thought()?;
        Some(Self { inner, ..self })
    }

    /// Construction path for inference *providers* — local engines and
    /// proxies that synthesize a response rather than deserialize one off
    /// the wire. The [`Client`] never needs this. Everything not set
    /// defaults: a fresh UUID [`id`], [`Kind::Message`], `None` stop
    /// metadata, [`Usage::default`].
    ///
    /// [`Client`]: crate::Client
    /// [`id`]: Message::id
    pub fn builder(
        model: impl Into<model::Model>,
        inner: prompt::AssistantMessage,
    ) -> Builder {
        Builder {
            message: Message {
                id: Cow::Owned(uuid::Uuid::new_v4().to_string()),
                kind: Some(Kind::Message),
                inner,
                model: model.into(),
                stop_reason: None,
                stop_sequence: None,
                stop_details: None,
                usage: Usage::default(),
                container: None,
            },
        }
    }
}

/// Builder for a provider-side [`Message`] — see [`Message::builder`].
#[derive(Clone, Debug)]
pub struct Builder {
    message: Message,
}

impl Builder {
    /// Override the generated UUID [`id`](Message::id) — e.g. to correlate
    /// the response with an out-of-band stream.
    pub fn id(mut self, id: impl Into<Cow<'static, str>>) -> Self {
        self.message.id = id.into();
        self
    }

    /// The reason the model stopped. Accepts a bare [`StopReason`] or an
    /// `Option` passed straight through.
    pub fn stop_reason(
        mut self,
        stop_reason: impl Into<Option<StopReason>>,
    ) -> Self {
        self.message.stop_reason = stop_reason.into();
        self
    }

    /// The stop sequence that fired, if [`stop_reason`](Self::stop_reason)
    /// is [`StopSequence`](StopReason::StopSequence).
    pub fn stop_sequence(
        mut self,
        stop_sequence: impl Into<Option<Cow<'static, str>>>,
    ) -> Self {
        self.message.stop_sequence = stop_sequence.into();
        self
    }

    /// Structured stop detail — see [`StopDetails`].
    pub fn stop_details(
        mut self,
        stop_details: impl Into<Option<Box<StopDetails>>>,
    ) -> Self {
        self.message.stop_details = stop_details.into();
        self
    }

    /// Usage statistics. Accepts [`Usage`] or bare [`TokenCounts`].
    pub fn usage(mut self, usage: impl Into<Usage>) -> Self {
        self.message.usage = usage.into();
        self
    }

    /// The code-execution container backing the turn — see
    /// [`Container`].
    pub fn container(
        mut self,
        container: impl Into<Option<Box<Container>>>,
    ) -> Self {
        self.message.container = container.into();
        self
    }

    /// Finish. Infallible — every field has a sane provider-side default.
    pub fn build(self) -> Message {
        self.message
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
    /// A long-running [`ServerMethodDef`] (e.g. web search) paused the turn. Send
    /// the response's content back as an [`Assistant`] message in a follow-up
    /// request — keeping the same tools — to let the model continue. See
    /// [server tools].
    ///
    /// [`ServerMethodDef`]: crate::tool::ServerMethodDef
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    /// [server tools]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/server-tools>
    PauseTurn,
    /// The model refused to produce output, typically due to a conflict
    /// between the request and its safety constraints. When this occurs
    /// with [`Prompt::output_config`], the response body may not match
    /// the requested schema; see [Anthropic docs on invalid outputs].
    ///
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    /// [Anthropic docs on invalid outputs]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#invalid-outputs>
    Refusal,
}

/// Object-type discriminator on a response [`Message`]. Always
/// [`Kind::Message`].
#[derive(
    Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Kind {
    /// A message.
    #[default]
    Message,
}

/// Structured detail about why a [`Message`] stopped — the `stop_details`
/// field, populated when [`Message::stop_reason`] is
/// [`Refusal`](StopReason::Refusal).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[non_exhaustive]
pub struct StopDetails {
    /// Refusal category, e.g. `"cyber"` or `"bio"`; `null` when the API
    /// doesn't classify the refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub category: Option<Cow<'static, str>>,
    /// Human-readable explanation of the refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<Cow<'static, str>>,
}

/// Usage statistics from the API. This is used in multiple contexts, not just
/// for messages.
///
/// The numeric token counters live in [`counts`](Self::counts) (flattened on
/// the wire) and are re-exposed here via `Deref`, so `usage.input_tokens`
/// reads through. `Usage` itself is not `Copy` — the API also reports
/// `service_tier`/`inference_geo` strings here, which arguably belong on the
/// parent response but are out of our control — so downstream code that wants
/// cheap copyable counters should take `usage.counts` ([`TokenCounts`]).
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    Default,
    derive_more::Deref,
    derive_more::DerefMut,
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
#[non_exhaustive]
pub struct Usage {
    /// The numeric token counters, `Copy` — see [`TokenCounts`].
    #[serde(flatten)]
    #[deref]
    #[deref_mut]
    pub counts: TokenCounts,
    /// Capacity tier that served the request: `"standard"`, `"priority"`, or
    /// `"batch"`. Distinct from the *requested*
    /// [`ServiceTier`](crate::prompt::ServiceTier) (`auto`/`standard_only`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<Cow<'static, str>>,
    /// Region that served the request, e.g. `"us"`, `"eu"`, or
    /// `"not_available"`. Distinct from the *requested*
    /// [`InferenceGeo`](crate::prompt::InferenceGeo) constraint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<Cow<'static, str>>,
}

/// The numeric token counters of [`Usage`], flattened on the wire and
/// re-exposed on `Usage` via `Deref`. Split out so the counters stay `Copy`
/// for cheap accumulation even though `Usage` carries strings.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Default)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
#[non_exhaustive]
pub struct TokenCounts {
    /// Number of input tokens used.
    pub input_tokens: u64,
    /// Number of input tokens used to create the cache entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation_input_tokens: Option<u64>,
    /// Cache-write breakdown by TTL. Sent on message-level usage (e.g.
    /// `message_start`); absent on `message_delta` usage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<CacheCreation>,
    /// Number of input tokens read from the cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read_input_tokens: Option<u64>,
    /// Number of output tokens generated.
    pub output_tokens: u64,
    /// Breakdown of [`output_tokens`](Self::output_tokens) — see
    /// [`OutputTokensDetails`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
    /// Server-tool invocation counts (e.g. web searches), when any server tool
    /// ran. See [`ServerMethodDef`](crate::tool::ServerMethodDef).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub server_tool_use: Option<ServerToolUsage>,
}

impl TokenCounts {
    /// The two counters every completion has. The optional detail fields
    /// stay [`Default`] — all fields are `pub`, so assign the ones you have
    /// (e.g. `cache_read_input_tokens`) and [`Usage`] is a `.into()` away.
    pub fn new(input_tokens: u64, output_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            ..Default::default()
        }
    }
}

/// The `output_tokens_details` object in [`TokenCounts`]: how many of the
/// output tokens were thinking. On the wire in both paths (the non-streaming
/// `usage` and the final `message_delta` usage; captured live on Opus 4.8,
/// 2026-06-12) even with thinking off, as an explicit `{"thinking_tokens": 0}`.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
#[non_exhaustive]
pub struct OutputTokensDetails {
    /// Number of output tokens spent thinking.
    pub thinking_tokens: u64,
}

impl std::ops::Add<OutputTokensDetails> for OutputTokensDetails {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            thinking_tokens: self.thinking_tokens + rhs.thinking_tokens,
        }
    }
}

/// Cache-write token counts broken down by TTL — the `cache_creation` object
/// in [`Usage`], the per-TTL detail behind
/// [`TokenCounts::cache_creation_input_tokens`].
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
#[non_exhaustive]
pub struct CacheCreation {
    /// Input tokens written to the 5-minute-TTL cache.
    pub ephemeral_5m_input_tokens: u64,
    /// Input tokens written to the 1-hour-TTL cache.
    pub ephemeral_1h_input_tokens: u64,
}

/// Per-request counts of [`ServerMethodDef`](crate::tool::ServerMethodDef) invocations,
/// reported in [`Usage::server_tool_use`].
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
#[non_exhaustive]
pub struct ServerToolUsage {
    /// Number of web searches performed.
    pub web_search_requests: u64,
    /// Number of [web fetches](crate::tool::ServerMethodDef::web_fetch) performed.
    #[serde(default)]
    pub web_fetch_requests: u64,
    /// Number of [tool-search](crate::tool::ServerMethodDef::tool_search_regex)
    /// queries performed.
    ///
    /// **As of 2026-06-10 the API does not populate this.** The documented
    /// `tool_search_requests` key is absent from the wire entirely (verified
    /// by `curl`, re-confirmed by the #78 stream captures), so this is `None`
    /// even on a turn where a tool search demonstrably ran. Count
    /// [`ToolSearchToolResult`](crate::prompt::message::Block::ToolSearchToolResult)
    /// blocks for a reliable signal. See [#72]. (Filed upstream as a docs/wire
    /// discrepancy; the field stays so it Just Works if the API starts sending
    /// it.)
    ///
    /// [#72]: <https://github.com/mdegans/misanthropic/issues/72>
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_search_requests: Option<u64>,
}

impl std::ops::Add<ServerToolUsage> for ServerToolUsage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            web_search_requests: self.web_search_requests
                + rhs.web_search_requests,
            web_fetch_requests: self.web_fetch_requests
                + rhs.web_fetch_requests,
            tool_search_requests: self
                .tool_search_requests
                .map(|c| c + rhs.tool_search_requests.unwrap_or(0))
                .or(rhs.tool_search_requests),
        }
    }
}

impl std::ops::Add<Usage> for Usage {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            counts: self.counts + rhs.counts,
            // Not counts — later (more final) values win.
            service_tier: rhs.service_tier.or(self.service_tier),
            inference_geo: rhs.inference_geo.or(self.inference_geo),
        }
    }
}

impl std::ops::Add<TokenCounts> for TokenCounts {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            cache_creation_input_tokens: self
                .cache_creation_input_tokens
                .map(|c| c + rhs.cache_creation_input_tokens.unwrap_or(0))
                .or(rhs.cache_creation_input_tokens),
            cache_creation: match (self.cache_creation, rhs.cache_creation) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            },
            cache_read_input_tokens: self
                .cache_read_input_tokens
                .map(|c| c + rhs.cache_read_input_tokens.unwrap_or(0))
                .or(rhs.cache_read_input_tokens),
            output_tokens: self.output_tokens + rhs.output_tokens,
            output_tokens_details: match (
                self.output_tokens_details,
                rhs.output_tokens_details,
            ) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            },
            server_tool_use: match (self.server_tool_use, rhs.server_tool_use) {
                (Some(a), Some(b)) => Some(a + b),
                (a, b) => a.or(b),
            },
        }
    }
}

impl std::ops::AddAssign<TokenCounts> for TokenCounts {
    fn add_assign(&mut self, rhs: TokenCounts) {
        *self = *self + rhs;
    }
}

impl From<TokenCounts> for Usage {
    fn from(counts: TokenCounts) -> Self {
        Self {
            counts,
            ..Default::default()
        }
    }
}

impl std::ops::Add<CacheCreation> for CacheCreation {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            ephemeral_5m_input_tokens: self.ephemeral_5m_input_tokens
                + rhs.ephemeral_5m_input_tokens,
            ephemeral_1h_input_tokens: self.ephemeral_1h_input_tokens
                + rhs.ephemeral_1h_input_tokens,
        }
    }
}

impl std::ops::AddAssign<Usage> for Usage {
    fn add_assign(&mut self, rhs: Usage) {
        *self = std::mem::take(self) + rhs;
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Message {
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
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

    /// A captured non-streaming response (Opus 4.8, 2026-06-12) round-trips
    /// exactly — guards `usage.output_tokens_details` (sent even with thinking
    /// off) and the full response envelope. Streaming twin:
    /// `system_after_server_tool.sse.stream.jsonl`, gated by wire coverage.
    #[test]
    fn captured_response_roundtrip() {
        let message: Message = crate::utils::roundtrip(include_str!(
            "../../test/data/system_after_server_tool.response.json"
        ));
        assert_eq!(
            message
                .usage
                .output_tokens_details
                .expect("wire sends output_tokens_details")
                .thinking_tokens,
            0
        );
    }

    #[test]
    fn deserialize_response_message() {
        let message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        assert_eq!(message.inner.content.len(), 1); // single block
        assert_eq!(message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(message.model, crate::Id::Sonnet35_20240620);
        assert!(matches!(message.stop_reason, Some(StopReason::EndTurn)));
        assert_eq!(message.stop_sequence, None);
    }

    #[test]
    fn test_apply_delta() {
        let mut message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        let delta = MessageDelta {
            stop_reason: Some(StopReason::MaxTokens),
            stop_sequence: Some("sequence".into()),
            stop_details: None,
            container: None,
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

        message.inner.content.push(
            crate::tool::Use::new("name", serde_json::json!({})).with_id("id"),
        );
        assert!(message.tool_use().is_some());
    }

    #[test]
    fn test_deserialize_response_message() {
        let static_message: Message =
            serde_json::from_str(RESPONSE_JSON).unwrap();

        assert_eq!(static_message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(static_message.model, crate::Id::Sonnet35_20240620);
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
        content: prompt::message::Content,
    ) -> Message {
        Message {
            id: "id".into(),
            kind: None,
            inner: prompt::AssistantMessage {
                role: prompt::message::markers::Assistant,
                content,
            },
            model: crate::Id::Sonnet35.into(),
            stop_reason,
            stop_sequence: None,
            stop_details: None,
            usage: Usage::default(),
            container: None,
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
            call: crate::tool::Use::new("x", serde_json::json!({}))
                .with_id("u"),
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
    fn usage_roundtrips_without_cache_fields() {
        // The wire omits the cache-token fields when no caching is involved
        // (e.g. a plain `message_start` usage). Re-serializing must not invent
        // explicit `null`s for them. See issue #93.
        let usage: Usage = crate::utils::roundtrip(
            r#"{"input_tokens":472,"output_tokens":2}"#,
        );
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_markdown() {
        use crate::markdown::ToMarkdown;

        let message = Message {
            id: "id".into(),
            kind: None,
            inner: prompt::AssistantMessage {
                role: prompt::message::markers::Assistant,
                content: prompt::message::Content::text("Hello, **world**!"),
            },
            model: crate::Id::Sonnet35.into(),
            stop_reason: None,
            stop_sequence: None,
            stop_details: None,
            usage: TokenCounts {
                input_tokens: 1,
                cache_creation_input_tokens: Some(2),
                cache_read_input_tokens: Some(3),
                output_tokens: 4,
                ..Default::default()
            }
            .into(),
            container: None,
        };

        let expected = "### Assistant\n\nHello, **world**!";
        let markdown = message.markdown();
        assert_eq!(markdown.as_ref(), expected);
    }

    /// The builder reproduces a wire-deserialized message exactly — the
    /// provider-side construction path (#134) is field-for-field equivalent
    /// to the deserialize path the `Client` uses.
    #[test]
    fn builder_matches_deserialize_path() {
        let wire: Message = serde_json::from_str(RESPONSE_JSON).unwrap();

        let built =
            Message::builder(crate::Id::Sonnet35_20240620, wire.inner.clone())
                .id("msg_013Zva2CMHLNnXjNJJKqJ2EF")
                .stop_reason(StopReason::EndTurn)
                .usage(TokenCounts::new(2095, 503))
                .build();

        assert_eq!(built, wire);
    }

    /// Builder defaults: fresh non-empty UUID id, `Kind::Message`, custom
    /// model id, default usage — and the result serde round-trips.
    #[test]
    fn builder_defaults_round_trip() {
        let wire: Message = serde_json::from_str(RESPONSE_JSON).unwrap();

        let built = Message::builder("local-model", wire.inner).build();

        assert!(!built.id.is_empty());
        assert!(matches!(built.kind, Some(Kind::Message)));
        assert_eq!(built.model.name(), "local-model");
        assert!(built.stop_reason.is_none());

        let json = serde_json::to_string(&built).unwrap();
        let round: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(round, built);
    }
}
