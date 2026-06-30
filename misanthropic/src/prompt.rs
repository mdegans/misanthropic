//! [Anthropic Messages API] `Request` type. We call it [`Prompt`] since in
//! actual usage this makes the code more readable.
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use std::{
    borrow::Cow,
    num::{NonZeroU16, NonZeroU32},
    vec,
};

use crate::{
    model,
    stream::{self, DeltaError},
    tool::{self, CustomMethodDef},
};
use message::Content;

use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

pub mod citation;
pub use citation::Citation;

pub mod message;
pub use message::{
    AssistantMessage, Message, RoleMessage, SystemMessage, UserMessage,
    WrongRole,
};

pub mod thinking;
pub use thinking::Display as ThinkingDisplay;
pub use thinking::Thinking;

pub mod cached;
pub use cached::CachedPrompt;

pub mod output;
pub use output::{Effort, Items, JsonSchemaFormat, OutputConfig, OutputFormat};

pub mod index;
pub use index::{BlockIndex, Index, IndexMut, IndexRef, MethodIndex};

/// Request for the [Anthropic Messages API].
///
/// [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Prompt {
    /// [`Model`](model::Model) to use for inference.
    pub model: model::Model,
    /// Input [`prompt::message`]s. If this ends with an [`Assistant`]
    /// [`Message`], the completion will be constrained by that last message.
    /// Otherwise a new [`Assistant`] [`Message`] will be generated.
    ///
    /// See [Anthropic docs] for more information.
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    /// [`prompt::message`]: crate::prompt::message
    /// [Anthropic docs]: <https://docs.anthropic.com/en/api/messages>
    pub messages: Vec<Message>,
    /// Max tokens to generate. See Anthropic [docs] for the maximum number of
    /// tokens for each model.
    ///
    /// [docs]: <https://docs.anthropic.com/en/docs/about-claude/models>
    pub max_tokens: NonZeroU32,
    /// Optional info about the request, for example, `user_id` to help
    /// Anthropic detect and prevent abuse. Do not use PII here (email, phone).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Optional stop sequences. If the model generates any of these sequences,
    /// the completion will stop with [`StopReason::StopSequence`].
    ///
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<Cow<'static, str>>>,
    /// If `true`, the response will be a stream of [`Event`]s. If `false`, the
    /// response will be a single [`response::Message`].
    ///
    /// [`Event`]: crate::stream::Event
    /// [`response::Message`]: crate::response::Message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// System prompt as [`Content`].
    ///
    /// [`Content`]: message::Content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<message::Content>,
    /// Temperature for sampling. Must be between 0 and 1. Higher values mean
    /// more randomness. Note that 0.0 is not fully deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// [`tool::Choice`] for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<tool::Choice>,
    /// Tool definitions for the model — the [`CustomMethodDef`]s you execute
    /// and [`ServerMethodDef`]s the API runs, intermixed via [`MethodDef`].
    ///
    /// [`ServerMethodDef`]: crate::tool::ServerMethodDef
    /// [`CustomMethodDef`]: crate::tool::CustomMethodDef
    /// [`MethodDef`]: crate::tool::MethodDef
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<tool::MethodDef>>,
    /// Top K tokens to consider for each token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<NonZeroU16>,
    /// Top P nucleus sampling. The probabilities of each token are added in
    /// order from most to least likely until the probability mass exceeds
    /// `top_p`. A token is then sampled from this reduced distribution.
    ///
    /// This is a float between 0 and 1 where higher values mean more
    /// randomness.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Extended thinking support, for Anthropic's built-in chain-of-thought on
    /// Sonnet 3.7 and later. Use [`Thinking::adaptive`] on current models. The
    /// `cot` feature works with all models instead, provided the system prompt
    /// instructs the Assistant to use `<thinking>` tags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
    /// Structured output configuration. When set, the response is
    /// constrained by grammar-based decoding to a single [`Text`] [`Block`]
    /// whose body matches the configured schema.
    ///
    /// Changing [`OutputConfig::format`] invalidates the [prompt cache] for
    /// the conversation thread. Incompatible with message prefilling and
    /// [`citations`]; combinable with [`strict`] [tool use], [streaming],
    /// and [batching]. A [`Refusal`] [`StopReason`] can occur here when the
    /// model declines to produce structured output.
    ///
    /// [`Text`]: crate::prompt::message::Block::Text
    /// [`Block`]: crate::prompt::message::Block
    /// [`strict`]: crate::tool::CustomMethodDef::strict
    /// [`Refusal`]: crate::response::StopReason::Refusal
    /// [`StopReason`]: crate::response::StopReason
    /// [`citations`]: <https://docs.anthropic.com/en/docs/build-with-claude/citations>
    /// [tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    /// [streaming]: <https://docs.anthropic.com/en/docs/build-with-claude/streaming>
    /// [batching]: <https://docs.anthropic.com/en/docs/build-with-claude/batch-processing>
    /// [prompt cache]: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_config: Option<OutputConfig>,
    /// Capacity tier for the request. See [`ServiceTier`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<ServiceTier>,
    /// Geographic region constraint for inference. See [`InferenceGeo`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub inference_geo: Option<InferenceGeo>,
    /// Container ID to reuse across requests (used with code execution).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<Cow<'static, str>>,
}

impl std::fmt::Debug for Prompt {
    /// The debug repr of a [`Prompt`] hides the `messages` (the chat history)
    /// — only their count is shown. Two reasons: it's the field most likely to
    /// carry user data into logs, and dumping a full conversation is *huge*.
    /// Every other field is request configuration and is shown in full.
    ///
    /// `metadata` is shown too, so don't put PII there. If you do, somewhere in
    /// your design you've made a mistake. Rethink your design.
    ///
    /// Fields are listed in declaration order so this stays easy to reconcile
    /// against the struct when new ones are added.
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Prompt")
            .field("model", &self.model)
            .field(
                "messages",
                &format_args!("<{} hidden>", self.messages.len()),
            )
            .field("max_tokens", &self.max_tokens)
            .field("metadata", &self.metadata)
            .field("stop_sequences", &self.stop_sequences)
            .field("stream", &self.stream)
            .field("system", &self.system)
            .field("temperature", &self.temperature)
            .field("tool_choice", &self.tool_choice)
            .field("tools", &self.tools)
            .field("top_k", &self.top_k)
            .field("top_p", &self.top_p)
            .field("thinking", &self.thinking)
            .field("output_config", &self.output_config)
            .field("service_tier", &self.service_tier)
            .field("inference_geo", &self.inference_geo)
            .field("container", &self.container)
            .finish()
    }
}

impl Default for Prompt {
    fn default() -> Self {
        Self {
            max_tokens: NonZeroU32::new(4096).unwrap(),
            messages: Default::default(),
            metadata: Default::default(),
            model: Default::default(),
            stop_sequences: Default::default(),
            stream: Default::default(),
            system: Default::default(),
            temperature: Default::default(),
            tool_choice: Default::default(),
            tools: Default::default(),
            top_k: Default::default(),
            top_p: Default::default(),
            thinking: Default::default(),
            output_config: Default::default(),
            service_tier: Default::default(),
            inference_geo: Default::default(),
            container: Default::default(),
        }
    }
}

/// Capacity tier for a request. Set via [`Prompt::service_tier`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
#[serde(rename_all = "snake_case")]
pub enum ServiceTier {
    /// Let the API choose the tier (priority when available).
    Auto,
    /// Use only the standard tier — never priority capacity.
    StandardOnly,
}

/// Geographic region constraint for inference. Set via
/// [`Prompt::inference_geo`].
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
#[serde(rename_all = "snake_case")]
pub enum InferenceGeo {
    /// United States.
    Us,
    /// European Union.
    Eu,
}

/// Message turn order is incorrect. A pure prompt-construction fault the caller
/// can fix before any request is sent.
///
/// [`User`] and [`Assistant`] turns must alternate. A [`System`] turn may not
/// open the conversation; it must follow a user turn and either end the array
/// or immediately precede an assistant turn.
///
/// **Server-tool exception:** two adjacent [`Assistant`] turns are permitted
/// when the first contains a [`ServerToolUse`] block. Anthropic pauses a
/// long-running server-tool turn with [`StopReason::PauseTurn`]; you continue
/// it by appending the paused turn back and resending, which yields adjacent
/// assistant turns that Anthropic merges server-side. Backends that emit
/// server-tool blocks accept this relaxed ordering, so the presence of such a
/// block is treated as evidence the backend allows it. This is a heuristic,
/// not a guarantee: a backend that emits `server_tool_use` yet enforces strict
/// alternation would be wrongly permitted here — but the failure surfaces as a
/// backend-side error, not silent corruption.
///
/// [`User`]: crate::prompt::message::Role::User
/// [`Assistant`]: crate::prompt::message::Role::Assistant
/// [`System`]: crate::prompt::message::Role::System
/// [`ServerToolUse`]: crate::prompt::message::Block::ServerToolUse
/// [`StopReason::PauseTurn`]: crate::response::StopReason::PauseTurn
// Deliberately NOT Serialize: serde's blanket `impl Serialize for Result`
// would otherwise let an un-unwrapped builder Result pass `impl Serialize`
// bounds (e.g. `client.message(prompt.messages(…))`) and reach the wire as
// `{"Ok": {…}}` — a silent 400. Without the derive that's a compile error.
#[derive(Debug, thiserror::Error)]
pub enum TurnOrderError {
    /// The first message must be from the user — assistant and system turns
    /// cannot open a conversation. Use the top-level [`Prompt::system`] field
    /// for from-the-start instructions.
    ///
    /// [`Prompt::system`]: Prompt::system
    #[error("the first message must be from the user, but it is a {} turn", .message.role)]
    BadFirst {
        /// The offending first message.
        message: Message,
    },
    /// `second` is not a legal turn after `first`. Either two same-role turns
    /// are adjacent, or a [`System`] turn is misplaced — it must follow a user
    /// turn or an assistant turn
    /// [ending in a server-tool result](crate::prompt::message::Message::ends_in_server_tool_result),
    /// and must precede an assistant turn or end the array.
    ///
    /// [`System`]: crate::prompt::message::Role::System
    #[error("a {} turn may not immediately follow a {} turn", .second.role, .first.role)]
    BadTransition {
        /// The earlier message.
        first: Message,
        /// The message that may not follow it.
        second: Message,
    },
    /// A turn places a [`ToolResult`] after non-result content. The wire
    /// requires `tool_result` blocks to *lead* their (user) message — a
    /// contiguous leading run. `[text, tool_result]` is a 400; `[tool_result,
    /// text]` is accepted (verified live, 2026-06-12).
    ///
    /// [`ToolResult`]: crate::prompt::message::Block::ToolResult
    #[error("tool_result blocks must lead their message, but a {} turn places one after other content", .message.role)]
    ToolResultNotLeading {
        /// The offending message.
        message: Message,
    },
    /// An [`Assistant`] turn emitted client [`ToolUse`] blocks the immediately
    /// following user turn does not answer: each client `tool_use` must have a
    /// matching leading [`ToolResult`] (by [`tool_use_id`]) in the next
    /// message.
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    /// [`ToolUse`]: crate::prompt::message::Block::ToolUse
    /// [`ToolResult`]: crate::prompt::message::Block::ToolResult
    /// [`tool_use_id`]: crate::tool::Result::tool_use_id
    #[error("{} client tool_use block(s) were not answered by the next turn: {}", .unanswered.len(), .unanswered.join(", "))]
    UnansweredToolUse {
        /// The assistant message whose `tool_use` blocks went unanswered.
        message: Message,
        /// The `tool_use` ids with no matching leading `tool_result` next.
        unanswered: Vec<String>,
    },
}
static_assertions::assert_impl_all!(TurnOrderError: Send, Sync);

/// Error from [`Prompt::add_examples`]: an exemplar failed to serialize, or
/// inserting the example pairs violated [turn order].
///
/// Kept separate from the runtime [`Error`](crate::client::Error) (reqwest, Anthropic, …)
/// because both arms are pure prompt-construction faults the caller can fix
/// before any request is sent.
///
/// [turn order]: TurnOrderError
#[derive(Debug, thiserror::Error)]
pub enum ExamplesError {
    /// An exemplar could not be serialized to JSON.
    #[error("failed to serialize example to JSON: {0}")]
    Serialize(#[from] serde_json::Error),
    /// Appending the example pairs collided with a preceding turn.
    #[error(transparent)]
    TurnOrder(#[from] TurnOrderError),
}
static_assertions::assert_impl_all!(ExamplesError: Send, Sync);

impl Prompt {
    /// Turn streaming on.
    ///
    /// **Note**: [`Client::stream`] and [`Client::message`] are more ergonomic
    /// and will overwrite this setting.
    ///
    /// [`Client::stream`]: crate::Client::stream
    /// [`Client::message`]: crate::Client::message
    pub fn stream(mut self) -> Self {
        self.stream = Some(true);
        self
    }

    /// Turn streaming off.
    ///
    /// **Note**: [`Client::stream`] and [`Client::message`] are more ergonomic
    /// and will overwrite this setting.
    ///
    /// [`Client::stream`]: crate::Client::stream
    /// [`Client::message`]: crate::Client::message
    pub fn no_stream(mut self) -> Self {
        self.stream = Some(false);
        self
    }

    /// Set the [`model`] to a [`Model`].
    ///
    /// [`model`]: Prompt::model
    pub fn model<M>(mut self, model: M) -> Self
    where
        M: Into<model::Model>,
    {
        self.model = model.into();
        self
    }

    /// Pick the first of `preferred` [`Role`]s the current [`model`] supports,
    /// for seating a pushed [`Notification`](crate::tool::Notification). Only
    /// [`Role::System`] is capability-gated (see [`supports_system_role`]);
    /// [`User`] and [`Assistant`] are always available. An empty list (or one
    /// whose every entry is unsupported) falls back to [`User`].
    ///
    /// [`supports_system_role`]: crate::model::Model::supports_system_role
    ///
    /// [`model`]: Prompt::model
    /// [`Role`]: message::Role
    /// [`Role::System`]: message::Role::System
    /// [`User`]: message::Role::User
    /// [`Assistant`]: message::Role::Assistant
    pub fn resolve_role(&self, preferred: &[message::Role]) -> message::Role {
        self.model.resolve_role(preferred)
    }

    /// Replace the [`messages`] from an iterable of [`Message`]s. To append,
    /// use [`add_messages`] or [`push_messages`].
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    /// [`add_messages`]: Prompt::add_messages
    /// [`push_messages`]: Prompt::push_messages
    pub fn messages<M, Ms>(
        mut self,
        messages: Ms,
    ) -> Result<Self, TurnOrderError>
    where
        M: Into<Message>,
        Ms: IntoIterator<Item = M>,
    {
        self.messages = messages.into_iter().map(Into::into).collect();
        self.check_turn_order()?;
        Ok(self)
    }

    /// Check the turn order of [`messages`]. Returns the **first** placement
    /// violation found.
    ///
    /// [`messages`]: Prompt::messages
    pub fn check_turn_order(&self) -> Result<(), TurnOrderError> {
        if let Some(first) = self.messages.first().filter(|m| !m.role.is_user())
        {
            return Err(TurnOrderError::BadFirst {
                message: first.clone(),
            });
        }
        for message in &self.messages {
            Self::check_results_lead(message)?;
        }
        for pair in self.messages.windows(2) {
            pair[0].may_precede(&pair[1])?;
        }
        Ok(())
    }

    /// A turn's [`ToolResult`](message::Block::ToolResult) blocks must lead
    /// (the within-message half of #102; the pairwise half lives in
    /// [`Message::may_precede`]).
    fn check_results_lead(message: &Message) -> Result<(), TurnOrderError> {
        if message.results_lead() {
            Ok(())
        } else {
            Err(TurnOrderError::ToolResultNotLeading {
                message: message.clone(),
            })
        }
    }

    /// Add a [`Message`] to [`messages`]. When adding multiple messages, use
    /// [`add_messages`] or [`push_messages`] for better performance.
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    /// [`add_messages`]: Prompt::add_messages
    /// [`push_messages`]: Prompt::push_messages
    pub fn add_message<M>(mut self, message: M) -> Result<Self, TurnOrderError>
    where
        M: Into<Message>,
    {
        let message: Message = message.into();
        self.push_message(message)?;
        Ok(self)
    }

    /// Push a [`Message`] to [`messages`]. Like [`add_message`] but in place.
    /// When adding multiple messages, use [`push_messages`] or [`add_messages`]
    /// for better performance.
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    ///
    /// [`add_message`]: Prompt::add_message
    /// [`messages`]: Prompt::messages
    /// [`push_messages`]: Prompt::push_messages
    /// [`add_messages`]: Prompt::add_messages
    pub fn push_message<M>(
        &mut self,
        message: M,
    ) -> Result<&mut Self, TurnOrderError>
    where
        M: Into<Message>,
    {
        let message: Message = message.into();
        Self::check_results_lead(&message)?;
        match self.messages.last() {
            Some(last) => {
                last.may_precede(&message)?;
            }
            None => {
                // The first message must be a user message.
                if !message.role.is_user() {
                    return Err(TurnOrderError::BadFirst {
                        message: message.clone(),
                    });
                }
            }
        }
        self.messages.push(message);
        Ok(self)
    }

    /// Extend the [`messages`] from an iterable. For an in-place version, see
    /// [`push_messages`].
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    /// [`push_messages`]: Prompt::push_messages
    pub fn add_messages<M, Ms>(
        mut self,
        messages: Ms,
    ) -> Result<Self, TurnOrderError>
    where
        M: Into<Message>,
        Ms: IntoIterator<Item = M>,
    {
        self.push_messages(messages)?;
        Ok(self)
    }

    /// Push many [`Message`]s to [`messages`]. Like [`add_messages`] but in
    /// place.
    ///
    /// # Errors
    /// - If the turn order is incorrect (and leaves self unmodified).
    ///
    /// [`add_messages`]: Prompt::add_messages
    /// [`messages`]: Prompt::messages
    pub fn push_messages<M, Ms>(
        &mut self,
        messages: Ms,
    ) -> Result<&mut Self, TurnOrderError>
    where
        M: Into<Message>,
        Ms: IntoIterator<Item = M>,
    {
        let mut count = 0;
        self.messages.extend(messages.into_iter().map(|m| {
            count += 1;
            m.into()
        }));
        if let Err(e) = self.check_turn_order() {
            // Undo our changes.
            self.messages.truncate(self.messages.len() - count);

            Err(e)
        } else {
            Ok(self)
        }
    }

    /// Set the [`max_tokens`]. If this is reached, the [`StopReason`] will be
    /// [`MaxTokens`] in the [`response::Message::stop_reason`].
    ///
    /// [`max_tokens`]: Prompt::max_tokens
    /// [`StopReason`]: crate::response::StopReason
    /// [`MaxTokens`]: crate::response::StopReason::MaxTokens
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    pub fn max_tokens(mut self, max_tokens: NonZeroU32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Replace the [`metadata`] from an iterable of key-value pairs.
    /// The values must be serializable to JSON. To add a single pair, use
    /// [`add_metadata`].
    ///
    /// [`metadata`]: Prompt::metadata
    /// [`add_metadata`]: Prompt::add_metadata
    pub fn metadata<S, V, Vs>(
        mut self,
        metadata: Vs,
    ) -> Result<Self, serde_json::Error>
    where
        S: Into<String>,
        V: Serialize,
        Vs: IntoIterator<Item = (S, V)>,
    {
        let mut map = serde_json::Map::new();

        for (k, v) in metadata {
            map.insert(k.into(), serde_json::to_value(v)?);
        }

        self.metadata = map;

        Ok(self)
    }

    /// Insert a key-value pair into the metadata. Replace the value if the key
    /// already exists.
    pub fn add_metadata<S, V>(
        mut self,
        key: S,
        value: V,
    ) -> Result<Self, serde_json::Error>
    where
        S: Into<String>,
        V: Serialize,
    {
        self.metadata
            .insert(key.into(), serde_json::to_value(value)?);
        Ok(self)
    }

    /// Set the [`stop_sequences`]. If one is generated, the completion will
    /// stop with [`StopReason::StopSequence`] in the
    /// [`response::Message::stop_reason`].
    ///
    /// [`stop_sequences`]: Prompt::stop_sequences
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    pub fn stop_sequences<S, Ss>(mut self, stop_sequences: Ss) -> Self
    where
        S: Into<Cow<'static, str>>,
        Ss: IntoIterator<Item = S>,
    {
        self.stop_sequences =
            Some(stop_sequences.into_iter().map(Into::into).collect());
        self
    }

    /// Add a stop sequence to [`stop_sequences`]. If one is generated, the
    /// completion will stop with [`StopReason::StopSequence`] in the
    /// [`response::Message::stop_reason`].
    ///
    /// [`stop_sequences`]: Prompt::stop_sequences
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    pub fn add_stop_sequence<S>(mut self, stop_sequence: S) -> Self
    where
        S: Into<Cow<'static, str>>,
    {
        self.stop_sequences
            .get_or_insert_with(Default::default)
            .push(stop_sequence.into());
        self
    }

    /// Extend the [`stop_sequences`] from an iterable. If one is generated, the
    /// completion will stop with [`StopReason::StopSequence`] in the
    /// [`response::Message::stop_reason`].
    ///
    /// [`stop_sequences`]: Prompt::stop_sequences
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    pub fn add_stop_sequences<S, Ss>(mut self, stop_sequences: Ss) -> Self
    where
        S: Into<Cow<'static, str>>,
        Ss: IntoIterator<Item = S>,
    {
        self.stop_sequences
            .get_or_insert_with(Default::default)
            .extend(stop_sequences.into_iter().map(Into::into));
        self
    }

    /// Replace the [`system`] prompt [`Content`]. This is content that the
    /// model will give special attention to. Instructions should be placed
    /// here. To append a [`Block`], use [`add_system`].
    ///
    /// [`system`]: Prompt::system
    /// [`Block`]: message::Block
    /// [`add_system`]: Prompt::add_system
    pub fn system<S>(mut self, system: S) -> Self
    where
        S: Into<message::Content>,
    {
        self.system = Some(system.into());
        self
    }

    /// Add a [`Block`] to the [`system`] prompt [`Content`]. If there is no
    /// [`system`] prompt, one will be created with the supplied `block`.
    ///
    /// Among the types that can convert to a [`Block`] are:
    /// * [`str`] slices
    /// * [`String`]
    /// * [`message::Image`] base64-encoded images
    ///
    /// With the `image` feature flag:
    /// * [`image::RgbaImage`] images (they will be encoded as PNG)
    /// * [`image::DynamicImage`] images (they will be converted to RGBA and
    ///   encoded as PNG)
    ///
    /// For other image formats, see the [`message::Image::encode`] method,
    /// the [`MediaType`] enum, and the image codec feature flags.
    ///
    /// [`system`]: Prompt::system
    /// [`Block`]: message::Block
    /// [`MediaType`]: message::MediaType
    pub fn add_system<B>(mut self, block: B) -> Self
    where
        B: Into<message::Block>,
    {
        match self.system {
            Some(mut content) => {
                content.push(block);
                self.system = Some(content);
            }
            None => {
                self.system = Some(Content(vec![block.into()]));
            }
        }
        self
    }

    /// Set the [`temperature`] — accepts a bare `f32` or [`None`] to use the
    /// default. Anthropic has deprecated this knob upstream, but third-party
    /// endpoints still honor it; it is omitted from the wire when unset.
    ///
    /// [`temperature`]: Prompt::temperature
    pub fn temperature(mut self, temperature: impl Into<Option<f32>>) -> Self {
        self.temperature = temperature.into();
        self
    }

    /// Set the [`service_tier`] (capacity tier) for the request.
    ///
    /// [`service_tier`]: Prompt::service_tier
    pub fn service_tier(mut self, tier: ServiceTier) -> Self {
        self.service_tier = Some(tier);
        self
    }

    /// Set the [`inference_geo`] (region constraint) for the request.
    ///
    /// [`inference_geo`]: Prompt::inference_geo
    pub fn inference_geo(mut self, geo: InferenceGeo) -> Self {
        self.inference_geo = Some(geo);
        self
    }

    /// Set the [`container`] ID to reuse across requests (code execution).
    ///
    /// [`container`]: Prompt::container
    pub fn container(mut self, id: impl Into<Cow<'static, str>>) -> Self {
        self.container = Some(id.into());
        self
    }

    /// Set the [`tool::Choice`]. This constrains how the model uses tools.
    ///
    /// [`tool::Choice`]: crate::tool::Choice
    pub fn tool_choice(mut self, choice: tool::Choice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the available [`tools`] — each item anything [`Into`] a
    /// [`MethodDef`], like [`add_tool`], so it composes with a [`Tool`]'s
    /// [`definitions`] and server defs survive as-is. When the [`Model`] uses
    /// a [`Tool`], the [`StopReason`] will be [`ToolUse`] in the
    /// [`response::Message::stop_reason`] and the final [`Content`] [`Block`]
    /// will be [`Block::ToolUse`] with a unique [`tool::Use::id`].
    ///
    /// The response may then be provided in a [`Message`] with a [`Role`] of
    /// [`User`] and [`Content`] [`Block`] of [`tool::Result`] with matching
    /// [`tool_use_id`] to the [`tool::Use::id`].
    ///
    /// For a fallible version, see [`try_tools`].
    ///
    /// [`tools`]: Prompt::tools
    /// [`add_tool`]: Prompt::add_tool
    /// [`MethodDef`]: crate::tool::MethodDef
    /// [`definitions`]: crate::Tool::definitions
    /// [`Model`]: crate::model::Model
    /// [`Tool`]: crate::Tool
    /// [`StopReason`]: crate::response::StopReason
    /// [`ToolUse`]: crate::response::StopReason::ToolUse
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    /// [`Block::ToolUse`]: crate::prompt::message::Block::ToolUse
    /// [`Role`]: crate::prompt::message::Role
    /// [`User`]: crate::prompt::message::Role::User
    /// [`Block`]: crate::prompt::message::Block
    /// [`tool_use_id`]: tool::Result::tool_use_id
    /// [`try_tools`]: Prompt::try_tools
    pub fn tools<T, Ts>(mut self, tools: Ts) -> Self
    where
        T: Into<tool::MethodDef>,
        Ts: IntoIterator<Item = T>,
    {
        self.tools = Some(tools.into_iter().map(Into::into).collect());
        self
    }

    /// Try to set the [`tools`]. When the [`Model`] uses a [`Tool`], the
    /// [`StopReason`] will be [`ToolUse`] in the
    /// [`response::Message::stop_reason`] and the final [`Content`] [`Block`]
    /// will be [`Block::ToolUse`] with a unique [`tool::Use::id`].
    ///
    /// The response may then be provided in a [`Message`] with a [`Role`] of
    /// [`User`] and [`Content`] [`Block`] of [`tool::Result`] with matching
    /// [`tool_use_id`] to the [`tool::Use::id`].
    ///
    /// [`tools`]: Prompt::tools
    /// [`Model`]: crate::model::Model
    /// [`Tool`]: crate::Tool
    /// [`StopReason`]: crate::response::StopReason
    /// [`ToolUse`]: crate::response::StopReason::ToolUse
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    /// [`Block::ToolUse`]: message::Block::ToolUse
    /// [`id`]: crate::tool::Use::id
    /// [`Role`]: message::Role
    /// [`User`]: message::Role::User
    /// [`Block`]: message::Block
    /// [`ToolResult`]: message::Block::ToolResult
    /// [`tool_use_id`]: crate::tool::Result::tool_use_id
    pub fn try_tools<T, E, Ts>(mut self, tools: Ts) -> Result<Self, E>
    where
        T: TryInto<CustomMethodDef, Error = E>,
        Ts: IntoIterator<Item = T>,
    {
        self.tools = Some(
            tools
                .into_iter()
                .map(|t| t.try_into().map(tool::MethodDef::Custom))
                .collect::<Result<_, _>>()?,
        );
        Ok(self)
    }

    /// Add a tool to the request — anything [`Into`] a [`MethodDef`]: a
    /// [`CustomMethodDef`], a [`ServerMethodDef`], or (with the `memory`
    /// feature) a memory tool. The right [`MethodDef`] variant is chosen for
    /// you.
    ///
    /// ```
    /// # use misanthropic::{Prompt, tool::{CustomMethodDef, ServerMethodDef}};
    /// let prompt = Prompt::default()
    ///     .add_tool(CustomMethodDef::simple("get_weather", "Get the weather."))
    ///     .add_tool(ServerMethodDef::web_search(Default::default()));
    /// ```
    ///
    /// [`MethodDef`]: crate::tool::MethodDef
    /// [`CustomMethodDef`]: crate::tool::CustomMethodDef
    /// [`ServerMethodDef`]: crate::tool::ServerMethodDef
    pub fn add_tool<T>(mut self, tool: T) -> Self
    where
        T: Into<tool::MethodDef>,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .push(tool.into());
        self
    }

    /// Try to add a custom tool to the request. Returns an error if the value
    /// cannot be converted into a [`CustomMethodDef`].
    pub fn try_add_tool<T, E>(mut self, tool: T) -> Result<Self, E>
    where
        T: TryInto<CustomMethodDef, Error = E>,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .push(tool::MethodDef::Custom(tool.try_into()?));
        Ok(self)
    }

    /// Add several tools at once — the plural of [`add_tool`]. Each item is
    /// anything [`Into`] a [`MethodDef`], so this composes with a tool's
    /// [`definitions`] just as well as with hand-built [`CustomMethodDef`]s:
    ///
    /// ```
    /// # use misanthropic::{Prompt, tool::CustomMethodDef};
    /// let prompt = Prompt::default().add_tools([
    ///     CustomMethodDef::simple("get_weather", "Get the weather."),
    ///     CustomMethodDef::simple("get_time", "Get the time."),
    /// ]);
    /// ```
    ///
    /// [`add_tool`]: Self::add_tool
    /// [`MethodDef`]: crate::tool::MethodDef
    /// [`definitions`]: crate::Tool::definitions
    pub fn add_tools<T, Ts>(mut self, tools: Ts) -> Self
    where
        T: Into<tool::MethodDef>,
        Ts: IntoIterator<Item = T>,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .extend(tools.into_iter().map(Into::into));
        self
    }

    /// Register every [`MethodDef`] a [`Tool`] exposes through its
    /// [`definitions`] — the one-liner for the common case of handing the
    /// model a typed tool:
    ///
    /// ```no_run
    /// # use misanthropic::{Prompt, Tool};
    /// # fn demo(weather: &impl Tool) {
    /// let prompt = Prompt::default().register_tool(weather);
    /// # }
    /// ```
    ///
    /// Unlike [`add_tools`](Self::add_tools), this preserves each def's variant:
    /// a tool that contributes a [`Server`] def (e.g. the client-executed
    /// `memory` backend) installs it as-is rather than forcing it into
    /// [`Custom`].
    ///
    /// [`Tool`]: crate::Tool
    /// [`definitions`]: crate::Tool::definitions
    /// [`MethodDef`]: crate::tool::MethodDef
    /// [`Server`]: crate::tool::MethodDef::Server
    /// [`Custom`]: crate::tool::MethodDef::Custom
    pub fn register_tool<T>(mut self, tool: &T) -> Self
    where
        T: tool::Tool + ?Sized,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .extend(tool.definitions());
        self
    }

    /// Mark every custom tool's [`CustomMethodDef`] as
    /// [`defer_loading`](CustomMethodDef::defer_loading), so the API loads their
    /// schemas only when the model discovers them through the [tool-search
    /// tool](tool::ServerMethodDef::tool_search_regex). Server tools are left
    /// untouched (they are never deferred).
    ///
    /// The Messages API requires at least one non-deferred tool, so pair this
    /// with a tool-search server tool — which stays non-deferred and gives the
    /// model a way to find the deferred ones:
    ///
    /// ```
    /// # use misanthropic::{Prompt, tool::{CustomMethodDef, ServerMethodDef}};
    /// let prompt = Prompt::default()
    ///     .add_tool(CustomMethodDef::simple("get_weather", "Get the weather."))
    ///     .add_tool(ServerMethodDef::tool_search_regex())
    ///     .defer_tools();
    /// ```
    pub fn defer_tools(mut self) -> Self {
        if let Some(methods) = self.tools.as_mut() {
            for method in methods
                .iter_mut()
                .filter_map(tool::MethodDef::as_method_mut)
            {
                method.defer_loading = Some(true);
            }
        }
        self
    }

    // No extend for tools because it's not very common or useful. If somebody
    // really wants this they can submit a PR.

    /// Set the top K tokens to consider for each token — accepts a bare
    /// [`NonZeroU16`] or `None` for the default. Deprecated upstream like
    /// [`temperature`], but kept for third-party endpoints.
    ///
    /// [`temperature`]: Prompt::temperature
    pub fn top_k(mut self, top_k: impl Into<Option<NonZeroU16>>) -> Self {
        self.top_k = top_k.into();
        self
    }

    /// Set the top P for nucleus sampling — accepts a bare `f32` or [`None`]
    /// for the default. Deprecated upstream like [`temperature`], but kept
    /// for third-party endpoints.
    ///
    /// [`temperature`]: Prompt::temperature
    pub fn top_p(mut self, top_p: impl Into<Option<f32>>) -> Self {
        self.top_p = top_p.into();
        self
    }

    /// Set the [`Thinking`] support.
    pub fn thinking(mut self, thinking: Thinking) -> Self {
        self.thinking = Some(thinking);
        self
    }

    /// Set [`output_config`] wholesale. See [`OutputConfig`] for construction
    /// helpers ([`OutputConfig::json_schema`], [`OutputConfig::for_type`],
    /// [`OutputConfig::effort`]).
    ///
    /// Unlike the granular [`json_schema`] / [`structured_output`] / [`effort`]
    /// builders — which each touch a single knob and preserve the rest — this
    /// **replaces** any existing config. Reach for it when you want exact
    /// control; otherwise prefer the granular setters so format and effort
    /// compose.
    ///
    /// [`output_config`]: Prompt::output_config
    /// [`json_schema`]: Prompt::json_schema
    /// [`structured_output`]: Prompt::structured_output
    /// [`effort`]: Prompt::effort
    pub fn output_config<C>(mut self, config: C) -> Self
    where
        C: Into<OutputConfig>,
    {
        self.output_config = Some(config.into());
        self
    }

    /// Merge `config`'s set fields into [`output_config`], preserving any knob
    /// it leaves unset.
    ///
    /// [`output_config`]: Prompt::output_config
    fn merge_output_config(&mut self, config: OutputConfig) {
        match &mut self.output_config {
            Some(existing) => existing.overlay(config),
            none => *none = Some(config),
        }
    }

    /// Sugar: constrain output to a raw [JSON Schema] value, preserving any
    /// [`effort`](Self::effort) already set.
    ///
    /// [JSON Schema]: <https://json-schema.org/>
    pub fn json_schema(mut self, schema: serde_json::Value) -> Self {
        self.merge_output_config(OutputConfig::json_schema(schema));
        self
    }

    /// Sugar: constrain output to the schema derived from `T`, preserving any
    /// [`effort`](Self::effort) already set.
    pub fn structured_output<T: schemars::JsonSchema>(mut self) -> Self {
        self.merge_output_config(OutputConfig::for_type::<T>());
        self
    }

    /// Set the [`Effort`] the model spends — text, tool calls, and thinking —
    /// preserving any output [`format`] already set. The recommended
    /// thinking-depth control on Claude 4 when paired with
    /// [`Thinking::adaptive`]; no beta header required.
    ///
    /// [`format`]: OutputConfig::format
    /// [`Thinking::adaptive`]: crate::prompt::Thinking::adaptive
    pub fn effort(mut self, effort: Effort) -> Self {
        self.merge_output_config(OutputConfig::effort(effort));
        self
    }

    /// Append schema-conformant few-shot examples for structured output.
    ///
    /// Each `(input, output)` pair becomes a [`Role::User`](message::Role::User)
    /// turn followed by a [`Role::Assistant`](message::Role::Assistant) turn
    /// whose single text block is `output` serialized
    /// to JSON — exactly the form the model emits under [`output_config`]. One
    /// or two well-populated exemplars before the real prompt nudge the model
    /// toward the desired depth of field population.
    ///
    /// The exemplar type `A` is also the *schema* type: this sets
    /// [`output_config`]'s [`format`] from `A`'s schema (via
    /// [`OutputConfig::for_type`]) — preserving any [`effort`] already set — so
    /// the constraint and the examples can never drift apart. Set any custom
    /// [`output_config`] / [`json_schema`] *after* this call if you need one.
    ///
    /// [`format`]: OutputConfig::format
    /// [`effort`]: Prompt::effort
    ///
    /// `U: Into<UserMessage>` accepts `&str`, `String`, or [`Content`] — so
    /// image exemplars (e.g. classification) work, not just text.
    ///
    /// # Errors
    /// - [`ExamplesError::Serialize`] if an exemplar will not serialize.
    /// - [`ExamplesError::TurnOrder`] if the first pair's user turn collides
    ///   with a preceding user turn (the prompt is left unmodified).
    ///
    /// [`output_config`]: Prompt::output_config
    /// [`json_schema`]: Prompt::json_schema
    pub fn add_examples<I, U, A>(
        mut self,
        examples: I,
    ) -> Result<Self, ExamplesError>
    where
        I: IntoIterator<Item = (U, A)>,
        U: Into<UserMessage>,
        A: Serialize + schemars::JsonSchema,
    {
        self.push_examples(examples)?;
        Ok(self)
    }

    /// Append schema-conformant few-shot examples in place. Like
    /// [`add_examples`] but `&mut`, for prompts already seated in a larger
    /// struct.
    ///
    /// [`add_examples`]: Prompt::add_examples
    pub fn push_examples<I, U, A>(
        &mut self,
        examples: I,
    ) -> Result<&mut Self, ExamplesError>
    where
        I: IntoIterator<Item = (U, A)>,
        U: Into<UserMessage>,
        A: Serialize + schemars::JsonSchema,
    {
        self.merge_output_config(OutputConfig::for_type::<A>());

        // Serialize every exemplar first so a failure leaves `messages`
        // untouched, then push the flattened User/Assistant turns through the
        // turn-order check in one all-or-nothing batch.
        let pairs = examples
            .into_iter()
            .map(|(input, output)| {
                let user: UserMessage = input.into();
                let json = serde_json::to_string(&output)?;
                Ok([user.into(), AssistantMessage::text(json).into()])
            })
            .collect::<Result<Vec<[Message; 2]>, serde_json::Error>>()?;

        self.push_messages(pairs.into_iter().flatten())?;

        Ok(self)
    }

    /// Add a cache breakpoint to the end of the prompt, setting `cache_control`
    /// to `Ephemeral`.
    ///
    /// # Notes
    /// * Cache breakpoints apply to the full prefix in the order of [`tools`],
    ///   [`system`], and [`messages`]. To effectively use this method, call it
    ///   after setting [`tools`] and [`system`] if you have no examples or
    ///   after setting [`messages`] if you do.
    /// * For [`Sonnet35`] and [`Opus30`] models, the prompt must have at least
    ///   1024 tokens for this to have an effect. For [`Haiku30`], the minimum
    ///   is 2048 tokens.
    /// * Since this is a beta feature, the API may change in the future, likely
    ///   to include another form of `cache_control`.
    ///
    /// [`tools`]: Prompt::tools
    /// [`system`]: Prompt::system
    /// [`messages`]: Prompt::messages
    /// [`Sonnet35`]: crate::Id::Sonnet35
    /// [`Opus30`]: crate::Id::Opus30
    /// [`Haiku30`]: crate::Id::Haiku30
    pub fn cache(self) -> Self {
        self.cache_with(crate::prompt::message::CacheControl::ephemeral())
    }

    /// Add a 1-hour cache breakpoint on the last cacheable block.
    ///
    /// Behaves identically to [`cache`](Prompt::cache) but uses
    /// [`CacheControl::one_hour`](crate::prompt::message::CacheControl::one_hour).
    /// Useful when the priming write and the real requests may be
    /// separated by more than the default 5-minute window.
    pub fn cache_1h(self) -> Self {
        self.cache_with(crate::prompt::message::CacheControl::one_hour())
    }

    /// Add a cache breakpoint with a caller-provided
    /// [`CacheControl`](message::CacheControl) on
    /// the last cacheable block. Shared implementation for
    /// [`cache`](Prompt::cache) and [`cache_1h`](Prompt::cache_1h).
    pub fn cache_with(
        mut self,
        cache_control: crate::prompt::message::CacheControl,
    ) -> Self {
        // If there are messages, add a cache breakpoint to the last one.
        if let Some(last) = self.messages.last_mut() {
            last.content.cache_with(cache_control);
            return self;
        }

        // If there are no messages, add a cache breakpoint to the system prompt
        // if it exists.
        if let Some(system) = self.system.as_mut() {
            system.cache_with(cache_control);
            return self;
        }

        // If there are no messages or system prompt, add a cache breakpoint to
        // the tools if they exist.
        if let Some(tool) =
            self.tools.as_mut().and_then(|tools| tools.last_mut())
        {
            tool.cache_with(cache_control);
            return self;
        }

        self
    }

    /// Apply a [`stream::Event`] to the [`Prompt`]. This is useful for
    /// appending to a [`Prompt`] in a streaming context.
    ///
    /// # Note
    /// - If the `partial-eq` feature is enabled, this will check for equality
    ///   for `Event::Message` and `Event::ToolUse` events, checking for the
    ///   consistency of the final message or tool use. Otherwise these messages
    ///   are ignored.
    // `ApplyEventError` embeds the offending message/block (~216 B), so the
    // `Result<(), _>` is ~216 B. Permanent allow: boxing is breaking, this
    // isn't stored in bulk (an apply failure is fatal, only the last kept),
    // and a per-event move is negligible next to per-event JSON parsing.
    #[allow(clippy::result_large_err)]
    pub fn handle_stream_event(
        &mut self,
        event: stream::Event,
    ) -> Result<(), ApplyEventError> {
        use stream::Event;

        match event {
            Event::ContentBlockDelta { index, delta } => {
                if let Some(last) = self.messages.last_mut() {
                    // There is a last message. Is it the correct index?
                    if index == last.content.len() - 1 {
                        // The last content block has the correct index.
                        if let Err(e) = last.content.push_delta(delta) {
                            return Err(e.into());
                        }
                    } else {
                        return Err(ApplyEventError::UnexpectedIndex {
                            event: Event::ContentBlockDelta { index, delta },
                            actual: index,
                            max: last.content.len() - 1,
                        });
                    }
                } else {
                    return Err(ApplyEventError::EmptyPrompt {
                        event: Event::ContentBlockDelta { index, delta },
                    });
                }
            }
            stream::Event::MessageStart { message } => {
                self.push_message(message)?;
            }
            stream::Event::ContentBlockStart {
                index,
                content_block,
            } => {
                if let Some(last) = self.messages.last_mut() {
                    if index == last.content.len() {
                        // The last content block has the correct index. It
                        // belongs pushed onto the end of the last message.
                        last.content.push(content_block);
                    } else {
                        return Err(ApplyEventError::UnexpectedIndex {
                            event: Event::ContentBlockStart {
                                index,
                                content_block,
                            },
                            actual: index,
                            max: last.content.len(),
                        });
                    }
                }
            }
            stream::Event::ContentBlockStop { index } => {
                if let Some(last) = self.messages.last_mut() {
                    if index == last.content.len() - 1 {
                        // The last content block has the correct index. There
                        // is nothing to do here.
                    } else {
                        // Either Anthropic screwed up or somebody mutated the
                        // prompt in between events.
                        return Err(ApplyEventError::UnexpectedIndex {
                            event: Event::ContentBlockStop { index },
                            actual: index,
                            max: last.content.len(),
                        });
                    }
                }
            }
            #[cfg_attr(not(feature = "partial-eq"), allow(unused_variables))]
            Event::Message { message } => {
                // The complete message should be identical to the last message
                // or there is a logic error in the caller.
                if let Some(last) = self.messages.last_mut() {
                    *last = message.inner.into();
                } else {
                    return Err(ApplyEventError::EmptyPrompt {
                        event: Event::Message { message },
                    });
                }
            }
            #[cfg_attr(not(feature = "partial-eq"), allow(unused_variables))]
            stream::Event::ToolUse { tool_use } => {
                // The last content block of the last message should be a tool
                // use. This is the final, assembled tool use.
                if let Some(last) = self.messages.last_mut() {
                    // If `with_tool_use` and `with_message` are both on, it's
                    // possible there is already a tool use block, in that case
                    // there is nothing to do.
                    if let Some(existing) = last.tool_use() {
                        if existing.id == tool_use.id {
                            // The tool use is already present.
                            return Ok(());
                        } else {
                            return Err(ApplyEventError::UnexpectedMessage {
                                event: Event::ToolUse { tool_use },
                                last: last.clone(),
                            });
                        }
                    } else {
                        last.content.push(tool_use);
                    }
                } else {
                    return Err(ApplyEventError::EmptyPrompt {
                        event: Event::ToolUse { tool_use },
                    });
                }
            }
            #[cfg_attr(not(feature = "partial-eq"), allow(unused_variables))]
            stream::Event::ServerToolUse { tool_use } => {
                // Like `ToolUse`, but the final block is a `ServerToolUse`. The
                // API ran it; we just record the assembled call.
                if let Some(last) = self.messages.last_mut() {
                    // Idempotent when `with_tool_use` + `with_message` both run.
                    if let Some(existing) = last.server_tool_use() {
                        if existing.id == tool_use.id {
                            return Ok(());
                        } else {
                            return Err(ApplyEventError::UnexpectedMessage {
                                event: Event::ServerToolUse { tool_use },
                                last: last.clone(),
                            });
                        }
                    } else {
                        last.content.push(message::Block::ServerToolUse {
                            call: tool_use,
                        });
                    }
                } else {
                    return Err(ApplyEventError::EmptyPrompt {
                        event: Event::ServerToolUse { tool_use },
                    });
                }
            }
            stream::Event::Ping
            | stream::Event::MessageStop
            | stream::Event::MessageDelta { .. }
            // Synthetic and derived from deltas already applied above; the
            // element is part of the block's text/input, not its own block.
            | stream::Event::JsonObject { .. } => {
                // Can't merge MessageDelta because a prompt contains
                // `prompt::Message` not `response::Message` which contains
                // `Usage`. But also I don't like throwing this away since it's
                // useful for debugging. Adding a field on the Prompt would be
                // messy because it's not part of the API. We'd need to test to
                // see if the API rejects it. I'm not writing two serialization
                // functions for this.
            }
        }

        Ok(())
    }

    /// Extend a prompt with an [`Extendable`](ExtendOntoPrompt) object. This also functions as a
    /// append. This is useful for streaming prompts. This is async because some
    /// of the extendables, like [`stream::FilterExt`], are async.
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    /// - If the stream of events cannot be applied to the prompt.
    pub async fn extend<E>(
        &mut self,
        extendable: E,
    ) -> Result<&mut Self, ExtendError>
    where
        E: ExtendOntoPrompt,
    {
        extendable.extend_onto(self).await
    }

    /// Helper for the above.
    pub async fn extend_stream<T>(
        &mut self,
        mut stream: std::pin::Pin<Box<T>>,
    ) -> Result<&mut Self, ExtendError>
    where
        T: futures::stream::Stream<Item = Result<stream::Event, stream::Error>>
            + Sized
            + Send,
    {
        loop {
            match stream.try_next().await? {
                Some(event) => self.handle_stream_event(event)?,
                None => break Ok(self),
            }
        }
    }

    /// Initialize a [`Tool`](crate::tool::Tool) with this [`Prompt`] asynchronously.
    /// This will call [`Tool::on_init`](crate::tool::Tool::on_init) to set up the tool's initial context.
    pub async fn init_tool<T>(
        mut self,
        tool: &mut T,
    ) -> std::result::Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        T: ?Sized + crate::tool::Tool,
    {
        tool.on_init(&mut self).await?;
        Ok(self)
    }

    /// Update turn context for a tool.
    /// Call this before each conversation turn to refresh dynamic content.
    pub async fn update_tool_context<T>(
        &mut self,
        tool: &mut T,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: ?Sized + crate::tool::Tool,
    {
        tool.on_turn(self).await
    }

    /// Tear down a [`Tool`](crate::tool::Tool), releasing resources it acquired in
    /// [`Tool::on_init`](crate::tool::Tool::on_init). Call this when the conversation ends; it invokes
    /// [`Tool::on_teardown`](crate::tool::Tool::on_teardown).
    pub async fn teardown_tool<T>(
        &mut self,
        tool: &mut T,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
    where
        T: ?Sized + crate::tool::Tool,
    {
        tool.on_teardown(self).await
    }
}

/// Error when [`extend`]ing a [`Prompt`].
///
/// [`extend`]: Prompt::extend
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub enum ExtendError {
    /// Turn Order is incorrect.
    TurnOrder(#[from] TurnOrderError),
    /// Error when applying a stream event to a prompt. Boxed to keep the
    /// error enum small ([`ApplyEventError`] carries a whole event).
    ApplyEvent(Box<ApplyEventError>),
    /// Stream error.
    Stream(#[from] stream::Error),
    /// Other error.
    Other(Box<dyn std::error::Error + Send + Sync>),
}

impl From<ApplyEventError> for ExtendError {
    fn from(error: ApplyEventError) -> Self {
        Self::ApplyEvent(Box::new(error))
    }
}

/// Object that can be appended to a [`Prompt`].
#[async_trait::async_trait]
pub trait ExtendOntoPrompt {
    /// Extend the prompt with the extendable object.
    async fn extend_onto(
        self,
        prompt: &mut Prompt,
    ) -> Result<&mut Prompt, ExtendError>;
}

#[async_trait::async_trait]
impl ExtendOntoPrompt for Message {
    async fn extend_onto(
        self,
        prompt: &mut Prompt,
    ) -> Result<&mut Prompt, ExtendError> {
        prompt.push_message(self).map_err(ExtendError::TurnOrder)?;
        Ok(prompt)
    }
}

#[async_trait::async_trait]
impl ExtendOntoPrompt for stream::Event {
    async fn extend_onto(
        self,
        prompt: &mut Prompt,
    ) -> Result<&mut Prompt, ExtendError> {
        prompt.handle_stream_event(self)?;
        Ok(prompt)
    }
}

#[async_trait::async_trait]
impl<T> ExtendOntoPrompt for T
where
    T: futures::stream::Stream<Item = Result<stream::Event, stream::Error>>
        + Sized
        + Send,
{
    async fn extend_onto(
        self,
        prompt: &mut Prompt,
    ) -> Result<&mut Prompt, ExtendError> {
        prompt.extend_stream(Box::pin(self)).await
    }
}

/// Reason for the error when applying a [`stream::Event`] to a [`Prompt`].
#[derive(Debug, thiserror::Error, derive_more::IsVariant)]
pub enum ApplyEventError {
    /// The [`Event`] is not supported by the [`Prompt`]. It cannot logically be
    /// applied to a [`Prompt`] at all (e.g. a [`Ping`] event).
    ///
    /// [`Event`]: stream::Event
    /// [`Ping`]: stream::Event::Ping
    #[error("This `Event` cannot be appended to a `Prompt`.")]
    Unsupported {
        /// The unsupported [`Event`](stream::Event).
        event: stream::Event,
    },
    /// Turn Order is incorrect.
    #[error(transparent)]
    TurnOrderError {
        /// The cause of the error.
        #[from]
        error: TurnOrderError,
    },
    /// Expected the last message to be an [`Assistant`](message::Role::Assistant). Similar to
    /// TurnOrderError but more specific and does not originate from
    /// `push_message` or `add_message`.
    #[error(
        "`Role::Assistant` must be the final message role in the prompt to apply this `Event`."
    )]
    ExpectedAssistant {
        /// The [`Event`](stream::Event) that caused the error.
        event: stream::Event,
        /// The role of the last message.
        last: message::Role,
    },
    /// Delta application error.
    #[error(transparent)]
    Delta(#[from] DeltaError),
    /// Unexpected index. Not necessarily out of bounds, but applying this event
    /// would be incorrect.
    #[error("Index {actual} is unexpected.")]
    UnexpectedIndex {
        /// The [`Event`](stream::Event) that caused the error.
        event: stream::Event,
        /// The actual index.
        actual: usize,
        /// The maximum index.
        max: usize,
    },
    /// Complete message did not match the last message.
    #[error("The complete message did not match the last message.")]
    UnexpectedMessage {
        /// The complete message.
        event: stream::Event,
        /// The last message.
        last: Message,
    },
    /// Event cannot be applied to an empty prompt.
    #[error("The prompt is empty and cannot accept this `Event`.")]
    EmptyPrompt {
        /// The [`Event`](stream::Event) that caused the error.
        event: stream::Event,
    },
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Prompt {
    /// Format the [`Prompt`] as markdown in OpenAI style. H3 headings are used
    /// for "System", "Tool", "User", and "Assistant" messages even though
    /// technically there are only [`User`] and [`Assistant`] [`Role`]s.
    ///
    /// [`User`]: message::Role::User
    /// [`Assistant`]: message::Role::Assistant
    /// [`Role`]: message::Role
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{Event, HeadingLevel::H3, Tag, TagEnd};

        // TODO: Add the title if there is metadata for it. Also add a metadata
        // option to Options to include arbitrary metadata. In my use case I am
        // feeding the markdown to another model that will make use of this data
        // so it does need to be included.

        let system: Box<dyn Iterator<Item = Event<'_>>> = if let Some(system) =
            self.system
                .as_ref()
                .map(|s| s.markdown_events_custom(options))
        {
            if options.system {
                let heading_level = options.heading_level.unwrap_or(H3);

                let header = [
                    Event::Start(Tag::Heading {
                        level: heading_level,
                        id: None,
                        classes: vec![],
                        attrs: if options.attrs {
                            vec![("role".into(), Some("system".into()))]
                        } else {
                            vec![]
                        },
                    }),
                    Event::Text("System".into()),
                    Event::End(TagEnd::Heading(heading_level)),
                ];

                Box::new(header.into_iter().chain(system))
            } else {
                Box::new(std::iter::empty())
            }
        } else {
            Box::new(std::iter::empty())
        };

        let messages = self
            .messages
            .iter()
            .flat_map(move |m| m.markdown_events_custom(options));

        Box::new(system.chain(messages))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::num::{NonZeroU16, NonZeroU32};

    use crate::{Id, prompt::message::Role};

    const STOP_SEQUENCES: [&str; 2] = ["stop1", "stop2"];

    #[test]
    fn defer_tools_marks_only_custom_tools() {
        let prompt = Prompt::default()
            .add_tool(crate::tool::CustomMethodDef::simple("a", "Tool A."))
            .add_tool(crate::tool::CustomMethodDef::simple("b", "Tool B."))
            .add_tool(crate::tool::ServerMethodDef::tool_search_regex())
            .defer_tools();

        let methods = prompt.tools.as_ref().unwrap();
        // Both custom tools are now deferred...
        for def in methods.iter().filter_map(|d| d.as_method()) {
            assert_eq!(def.defer_loading, Some(true));
        }
        // ...and the server tool-search tool is untouched (never deferred),
        // satisfying the API's "at least one non-deferred tool" rule.
        let json = serde_json::to_value(&prompt).unwrap();
        let tools = json["tools"].as_array().unwrap();
        let search = tools
            .iter()
            .find(|t| t["type"] == "tool_search_tool_regex_20251119")
            .unwrap();
        assert!(search.get("defer_loading").is_none());
    }

    #[test]
    fn add_tools_appends_each_method() {
        use crate::tool::CustomMethodDef;
        // Appends to any tools already present rather than replacing them.
        let prompt = Prompt::default()
            .add_tool(CustomMethodDef::simple("a", "Tool A."))
            .add_tools([
                CustomMethodDef::simple("b", "Tool B."),
                CustomMethodDef::simple("c", "Tool C."),
            ]);

        let names: Vec<_> = prompt
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|d| d.as_method())
            .map(|m| m.name.as_ref())
            .collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn register_tool_pulls_definitions() {
        use crate::tool::{CustomMethodDef, MethodDef, Tool, Use};

        struct PairTool;

        #[async_trait::async_trait]
        impl Tool for PairTool {
            fn name(&self) -> &str {
                "PairTool"
            }
            fn definitions(&self) -> Vec<MethodDef> {
                vec![
                    CustomMethodDef::simple("PairTool__a", "A.").into(),
                    CustomMethodDef::simple("PairTool__b", "B.").into(),
                ]
            }
            async fn call(&mut self, call: Use) -> crate::tool::Result {
                crate::tool::Result::new(call.id, "ok")
            }
        }

        // A short-lived `Prompt` accepts the tool's `MethodDef`s.
        let prompt = Prompt::default().register_tool(&PairTool);
        let names: Vec<_> = prompt
            .tools
            .as_ref()
            .unwrap()
            .iter()
            .filter_map(|d| d.as_method())
            .map(|m| m.name.as_ref())
            .collect();
        assert_eq!(names, ["PairTool__a", "PairTool__b"]);
    }

    #[test]
    fn debug_hides_messages_shows_config() {
        let prompt = Prompt {
            top_p: Some(0.9),
            container: Some("container-xyz".into()),
            service_tier: Some(ServiceTier::Auto),
            inference_geo: Some(InferenceGeo::Us),
            ..Default::default()
        }
        .add_message((Role::User, "SECRET-USER-DATA"))
        .unwrap();

        let dbg = format!("{prompt:?}");
        // The chat history is the privacy-sensitive part: content hidden, only
        // the count shown.
        assert!(!dbg.contains("SECRET-USER-DATA"), "messages leaked: {dbg}");
        assert!(dbg.contains("messages: <1 hidden>"), "{dbg}");
        // Request configuration is shown in full — these fields were all
        // missing from the Debug impl before #73.
        assert!(dbg.contains("container-xyz"), "container hidden: {dbg}");
        assert!(dbg.contains("top_p: Some(0.9)"), "top_p hidden: {dbg}");
        assert!(dbg.contains("service_tier"), "service_tier hidden: {dbg}");
        assert!(dbg.contains("inference_geo"), "inference_geo hidden: {dbg}");
    }

    // Credit to GitHub Copilot for the following tests.

    #[test]
    fn test_default_request() {
        let request = Prompt::default();
        assert_eq!(request.model, crate::model::Model::default());
        assert!(request.messages.is_empty());
        assert_eq!(request.max_tokens, NonZeroU32::new(4096).unwrap());
        assert!(request.metadata.is_empty());
        assert!(request.stop_sequences.is_none());
        assert!(request.stream.is_none());
        assert!(request.system.is_none());
        assert!(request.temperature.is_none());
        assert!(request.tool_choice.is_none());
        assert!(request.tools.is_none());
        assert!(request.top_k.is_none());
        assert!(request.top_p.is_none());
    }

    #[test]
    fn test_stream_on() {
        let request = Prompt::default().stream();
        assert_eq!(request.stream, Some(true));
    }

    #[test]
    fn test_stream_off() {
        let request = Prompt::default().no_stream();
        assert_eq!(request.stream, Some(false));
    }

    #[test]
    fn test_prompt_debug_hides_messages() {
        let request = Prompt::default().add_message((Role::User, "Hello"));
        let debug = format!("{:?}", request);
        assert!(!debug.contains("Hello"));
    }

    #[test]
    fn test_set_model() {
        let model = Id::default();
        let request = Prompt::default().model(model); // Id is Copy
        assert_eq!(request.model, crate::model::Model::default());
    }

    fn create_test_messages() -> [Message; 2] {
        let message = Message {
            role: Role::User,
            content: Content::text("Hello"),
        };

        let message2 = Message {
            role: Role::Assistant,
            content: Content::text("Hi"),
        };

        [message, message2]
    }

    #[test]
    fn test_messages() {
        let request =
            Prompt::default().messages(create_test_messages()).unwrap();
        assert_eq!(request.messages, create_test_messages());
    }

    #[test]
    fn test_add_message() {
        let prompt = Prompt::default()
            .add_message((Role::User, "Hello"))
            .unwrap()
            .add_message((Role::Assistant, "Hi"))
            .unwrap();
        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(prompt.messages[0], (Role::User, "Hello").into());
        assert_eq!(prompt.messages[1], (Role::Assistant, "Hi").into());
    }

    #[test]
    #[should_panic]
    fn test_add_message_turn_order() {
        let prompt = Prompt::default()
            .add_message((Role::User, "Hello"))
            .unwrap();
        prompt.add_message((Role::User, "Hi")).unwrap();
    }

    #[test]
    #[should_panic]
    fn test_add_messages_turn_order() {
        Prompt::default()
            .add_messages([(Role::User, "Hello"), (Role::User, "And boom!")])
            .unwrap();
    }

    #[test]
    fn test_push_message() {
        let mut prompt = Prompt::default();
        prompt.push_message((Role::User, "Hello")).unwrap();
        prompt.push_message((Role::Assistant, "Hi")).unwrap();
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    #[should_panic]
    fn test_push_message_turn_order() {
        let mut prompt = Prompt::default();
        prompt.push_message((Role::User, "Hello")).unwrap();
        prompt.push_message((Role::User, "Hi")).unwrap();
    }

    #[test]
    fn test_push_messages() {
        let mut prompt = Prompt::default();
        prompt
            .push_messages([(Role::User, "Hello"), (Role::Assistant, "Hi")])
            .unwrap();
        assert_eq!(prompt.messages.len(), 2);
    }

    #[test]
    fn test_push_messages_turn_order() {
        let mut prompt = Prompt::default();
        let result =
            prompt.push_messages([(Role::User, "Hello"), (Role::User, "Hi")]);
        assert!(result.is_err());
        assert!(prompt.messages.is_empty());
    }

    #[test]
    fn test_system_cannot_be_first() {
        let err = Prompt::default()
            .add_message((Role::System, "be terse"))
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::BadFirst { .. }));
    }

    #[test]
    fn test_system_follows_user_and_ends_array() {
        // user → system, with system as the last entry, is legal.
        let prompt = Prompt::default()
            .add_message((Role::User, "hi"))
            .unwrap()
            .add_message((Role::System, "be terse"))
            .unwrap();
        assert_eq!(prompt.messages.len(), 2);
        prompt.check_turn_order().unwrap();
    }

    #[test]
    fn test_system_between_user_and_assistant() {
        // user → system → assistant is legal.
        let prompt = Prompt::default()
            .add_messages([
                (Role::User, "hi"),
                (Role::System, "be terse"),
                (Role::Assistant, "ok."),
            ])
            .unwrap();
        prompt.check_turn_order().unwrap();
    }

    #[test]
    fn test_system_cannot_follow_assistant() {
        // assistant → system is rejected unless the assistant turn ends in a
        // server-tool result (see test_system_after_server_tool_tails).
        let err = Prompt::default()
            .add_message((Role::User, "hi"))
            .unwrap()
            .add_message((Role::Assistant, "hello!"))
            .unwrap()
            .add_message((Role::System, "be terse"))
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::BadTransition { .. }));
    }

    /// Offline mirror of the live `count_tokens` placement probes
    /// (`client::tests::test_count_tokens_validates_system_placement`):
    /// assistant → system is legal iff the assistant turn ends in a
    /// server-tool *result* — strictly the last block, and a *use* (the
    /// paused-turn tail) does not qualify. The fixture blocks keep the
    /// shapes wire-sourced.
    #[test]
    fn test_system_after_server_tool_tails() {
        use crate::prompt::message::{Block, Content};

        let fetch_use: Block = serde_json::from_str(include_str!(
            "../test/data/server_tools/server_tool_use.json"
        ))
        .unwrap();
        let fetch_result: Block = serde_json::from_str(include_str!(
            "../test/data/server_tools/web_fetch_result.json"
        ))
        .unwrap();
        let text = Block::text("done.");

        // (assistant tail blocks, may a system turn follow?)
        let cases = [
            (vec![text.clone(), fetch_use.clone()], false), // paused turn
            (
                vec![fetch_use.clone(), fetch_result.clone(), text.clone()],
                false, // "ending in" is strict on the last block
            ),
            (vec![fetch_use, fetch_result], true),
        ];

        for (blocks, legal) in cases {
            let outcome = Prompt::default()
                .add_message((Role::User, "fetch it"))
                .unwrap()
                .add_message((Role::Assistant, Content(blocks)))
                .unwrap()
                .add_message((Role::System, "note"));
            assert_eq!(outcome.is_ok(), legal);
        }
    }

    #[test]
    fn test_system_must_be_followed_by_assistant() {
        // system → user is rejected: a system turn must end the array or be
        // immediately followed by an assistant turn.
        let err = Prompt::default()
            .add_messages([
                (Role::User, "hi"),
                (Role::System, "be terse"),
                (Role::User, "still there?"),
            ])
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::BadTransition { .. }));
    }

    #[test]
    fn test_consecutive_system_rejected() {
        let err = Prompt::default()
            .add_messages([
                (Role::User, "hi"),
                (Role::System, "be terse"),
                (Role::System, "and polite"),
            ])
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::BadTransition { .. }));
    }

    #[test]
    fn test_adjacent_assistant_allowed_after_server_tool_use() {
        // A paused server-tool turn followed by its continuation: two adjacent
        // assistant turns are legal because the first carries a server-tool-use
        // block (the `pause_turn` continuation case).
        let paused: message::Message =
            serde_json::from_value(serde_json::json!({
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "searching..." },
                    {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": { "query": "anthropic products" }
                    }
                ]
            }))
            .unwrap();

        let prompt = Prompt::default()
            .add_message((Role::User, "name an Anthropic product"))
            .unwrap()
            .add_message(paused)
            .unwrap()
            .add_message((Role::Assistant, "Claude Code."))
            .unwrap();

        // Both the incremental check (in `push_message`) and the whole-array
        // check must accept it.
        prompt.check_turn_order().unwrap();
        assert_eq!(prompt.messages.len(), 3);
    }

    #[test]
    fn test_adjacent_assistant_rejected_without_server_tool_use() {
        // Two plain assistant turns remain a turn-order error: the exception is
        // gated on a server-tool-use block, so ordinary back-to-back assistant
        // turns are still caught as the programmer error they usually are.
        let err = Prompt::default()
            .add_message((Role::User, "hi"))
            .unwrap()
            .add_message((Role::Assistant, "hello"))
            .unwrap()
            .add_message((Role::Assistant, "hello again"))
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::BadTransition { .. }));
    }

    #[test]
    fn test_tool_result_must_lead_user_turn() {
        // The wire requires `tool_result` blocks to lead their user message
        // (#102). `[result, text]` is accepted; `[text, result]` is a 400.
        use crate::prompt::message::{Block, Content};

        let tool_use: Block = serde_json::from_value(serde_json::json!({
            "type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}
        }))
        .unwrap();
        let result: Block = serde_json::from_value(serde_json::json!({
            "type": "tool_result", "tool_use_id": "toolu_1", "content": "12:00"
        }))
        .unwrap();
        let text = Block::text("here you go");

        let head = || {
            Prompt::default()
                .add_message((Role::User, "what time is it?"))
                .unwrap()
                .add_message((Role::Assistant, Content(vec![tool_use.clone()])))
                .unwrap()
        };

        // Results lead → accepted.
        head()
            .add_message((
                Role::User,
                Content(vec![result.clone(), text.clone()]),
            ))
            .unwrap();

        // A result trailing other content → rejected.
        let err = head()
            .add_message((Role::User, Content(vec![text, result])))
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::ToolResultNotLeading { .. }));
    }

    #[test]
    fn test_client_tool_use_must_be_answered() {
        // A client `tool_use` must be answered by a matching leading
        // `tool_result` in the immediately following user turn (#102).
        use crate::prompt::message::{Block, Content};

        let mk_use = |id: &str| -> Block {
            serde_json::from_value(serde_json::json!({
                "type": "tool_use", "id": id, "name": "get_time", "input": {}
            }))
            .unwrap()
        };
        let mk_result = |id: &str| -> Block {
            serde_json::from_value(serde_json::json!({
                "type": "tool_result", "tool_use_id": id, "content": "ok"
            }))
            .unwrap()
        };

        // Answered → accepted.
        Prompt::default()
            .add_message((Role::User, "go"))
            .unwrap()
            .add_message((Role::Assistant, Content(vec![mk_use("toolu_1")])))
            .unwrap()
            .add_message((Role::User, Content(vec![mk_result("toolu_1")])))
            .unwrap();

        // A plain-text user turn answers nothing → rejected.
        let err = Prompt::default()
            .add_message((Role::User, "go"))
            .unwrap()
            .add_message((Role::Assistant, Content(vec![mk_use("toolu_1")])))
            .unwrap()
            .add_message((Role::User, "no results here"))
            .unwrap_err();
        assert!(matches!(err, TurnOrderError::UnansweredToolUse { .. }));

        // Partial coverage → the error names the unanswered id.
        let err = Prompt::default()
            .add_message((Role::User, "go"))
            .unwrap()
            .add_message((
                Role::Assistant,
                Content(vec![mk_use("toolu_1"), mk_use("toolu_2")]),
            ))
            .unwrap()
            .add_message((Role::User, Content(vec![mk_result("toolu_1")])))
            .unwrap_err();
        match err {
            TurnOrderError::UnansweredToolUse { unanswered, .. } => {
                assert_eq!(unanswered, vec!["toolu_2".to_string()]);
            }
            other => panic!("expected UnansweredToolUse, got {other:?}"),
        }
    }

    #[test]
    fn test_trailing_client_tool_use_is_allowed() {
        // An assistant `tool_use` as the *last* turn has no following turn to
        // answer it, so the pairwise rule does not fire. (Whether a trailing
        // unanswered `tool_use` is itself a wire error is a separate, as-yet
        // unverified question — see the #102 live probe.)
        use crate::prompt::message::{Block, Content};

        let tool_use: Block = serde_json::from_value(serde_json::json!({
            "type": "tool_use", "id": "toolu_1", "name": "get_time", "input": {}
        }))
        .unwrap();

        Prompt::default()
            .add_message((Role::User, "go"))
            .unwrap()
            .add_message((Role::Assistant, Content(vec![tool_use])))
            .unwrap()
            .check_turn_order()
            .unwrap();
    }

    #[test]
    fn test_request_params_serde() {
        let prompt = Prompt::default()
            .service_tier(ServiceTier::StandardOnly)
            .inference_geo(InferenceGeo::Eu)
            .container("ctr_123");

        let json = serde_json::to_value(&prompt).unwrap();
        assert_eq!(json["service_tier"], "standard_only");
        assert_eq!(json["inference_geo"], "eu");
        assert_eq!(json["container"], "ctr_123");

        // Omitted entirely when unset.
        let bare = serde_json::to_value(Prompt::default()).unwrap();
        assert!(bare.get("service_tier").is_none());
        assert!(bare.get("inference_geo").is_none());
        assert!(bare.get("container").is_none());

        // Round-trip.
        let back: Prompt = serde_json::from_value(json).unwrap();
        assert_eq!(back.service_tier, Some(ServiceTier::StandardOnly));
        assert_eq!(back.inference_geo, Some(InferenceGeo::Eu));
        assert_eq!(back.container.as_deref(), Some("ctr_123"));
    }

    #[test]
    fn test_system_message_serde_roundtrip() {
        let message: message::Message =
            (Role::System, "operator policy").into();
        let json = serde_json::to_value(&message).unwrap();
        assert_eq!(json["role"], "system");
        let back: message::Message = serde_json::from_value(json).unwrap();
        assert_eq!(back.role, Role::System);
    }

    #[test]
    fn test_add_messages() {
        let mut request = Prompt::default();
        request = request.add_messages(create_test_messages()).unwrap();
        assert_eq!(request.messages, create_test_messages());
    }

    #[test]
    fn test_set_max_tokens() {
        let max_tokens = NonZeroU32::new(1024).unwrap();
        let request = Prompt::default().max_tokens(max_tokens);
        assert_eq!(request.max_tokens, max_tokens);
    }

    #[test]
    fn test_metadata() {
        let request = Prompt::default()
            .metadata([("key", "value"), ("key2", "value2")])
            .unwrap();
        assert_eq!(request.metadata.get("key").unwrap(), "value");
        assert_eq!(request.metadata.get("key2").unwrap(), "value2");
    }

    #[test]
    fn test_add_metadata() {
        let request = Prompt::default().add_metadata("key", "value").unwrap();
        assert_eq!(request.metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_set_stop_sequences() {
        let request = Prompt::default().stop_sequences(STOP_SEQUENCES);
        assert_eq!(request.stop_sequences.unwrap(), STOP_SEQUENCES);
    }

    #[test]
    fn test_add_stop_sequence() {
        let mut request = Prompt::default();
        request = request.add_stop_sequence(STOP_SEQUENCES[0]);
        assert_eq!(request.stop_sequences.as_ref().unwrap().len(), 1);
        assert_eq!(request.stop_sequences.unwrap()[0], STOP_SEQUENCES[0]);
    }

    #[test]
    fn test_add_stop_sequences() {
        let mut request = Prompt::default();
        request = request.add_stop_sequences(STOP_SEQUENCES);
        assert_eq!(request.stop_sequences.unwrap().len(), 2);
    }

    #[test]
    fn test_system() {
        let request = Prompt::default().system("system");
        assert_eq!(request.system.unwrap().to_string(), "system");
    }

    // End of GitHub Copilot tests.

    #[test]
    fn test_add_system_block() {
        // Test with a system prompt. The call to cache should affect the final
        // Block in the system prompt.
        let request = Prompt::default()
            .add_system("Do this.") // Will add a system Content block
            .add_system("And then do this.");

        assert_eq!(
            request.system.as_ref().unwrap().to_string(),
            "Do this.\n\nAnd then do this."
        );
    }

    #[test]
    fn test_cache() {
        // Test with nothing to cache. This should be a no-op.
        let request = Prompt::default().cache();
        assert!(request == Prompt::default());

        // Test with no system prompt or messages that the call to cache affects
        // the tools.
        let request = Prompt::default().add_tool(CustomMethodDef {
            name: "ping".into(),
            description: "Ping a server.".into(),
            schema: json!({}),
            cache_control: None,
            strict: None,
            defer_loading: None,
            allowed_callers: None,
        });

        assert!(!request.tools.as_ref().unwrap().last().unwrap().is_cached());

        let mut request = request.cache();

        assert!(request.tools.as_ref().unwrap().last().unwrap().is_cached());

        // remove the cache breakpoint
        // TODO: add an un_cache method? set_cache?
        request
            .tools
            .as_mut()
            .unwrap()
            .last_mut()
            .unwrap()
            .as_method_mut()
            .unwrap()
            .cache_control = None;

        // Test with a system prompt. The call to cache should affect the final
        // Block in the system prompt.
        let request = request
            .add_system("Do this.") // Will add a system Content block
            .add_system("And then do this.")
            .cache();

        assert!(request.system.as_ref().unwrap().last().unwrap().is_cached());
        // ensure the tools are not affected
        assert!(!request.tools.as_ref().unwrap().last().unwrap().is_cached());

        // Test with messages. The call to cache should affect the last message.
        let request = request
            .add_message(Message {
                role: Role::User,
                content: Content::text("Hello"),
            })
            .unwrap()
            .add_message(Message {
                role: Role::Assistant,
                content: Content::text("Hi"),
            })
            .unwrap()
            .cache();

        // The first message should not be cached — cache() only touches the
        // last message.
        assert!(
            !request
                .messages
                .first()
                .unwrap()
                .content
                .last()
                .unwrap()
                .is_cached()
        );

        // The last message's final block should now be cached.
        assert!(
            request
                .messages
                .last()
                .unwrap()
                .content
                .last()
                .unwrap()
                .is_cached()
        );
    }

    #[test]
    fn test_serde() {
        // Test default deserialization.
        const JSON: &str = r#"{}"#;

        let defaults = serde_json::from_str::<Prompt>(JSON).unwrap();

        // Another round trip to ensure serialization works.
        let json = serde_json::to_string(&defaults).unwrap();
        let _ = serde_json::from_str::<Prompt>(&json).unwrap();

        // TODO: impl Default and PartialEq when `cfg(test)`
    }

    #[test]
    fn test_serde_json_fields() {
        let default = Prompt::default();
        let json = dbg!(serde_json::to_string_pretty(&default).unwrap());
        let value = serde_json::from_str::<serde_json::Value>(&json).unwrap();

        if let serde_json::Value::Object(map) = value {
            assert_eq!(map.len(), 3);
            assert!(map.contains_key("model"));
            assert!(map.contains_key("max_tokens"));
            assert!(map.contains_key("messages"));
        } else {
            panic!("Expected an object.");
        }
    }

    #[test]
    fn test_output_config_defaults_to_none() {
        let prompt = Prompt::default();
        assert!(prompt.output_config.is_none());
        // And is elided from serialized form.
        let json = serde_json::to_string(&prompt).unwrap();
        assert!(!json.contains("output_config"));
    }

    #[test]
    fn test_output_config_builder_and_roundtrip() {
        let schema = json!({
            "type": "object",
            "properties": { "support": { "type": "boolean" } },
            "required": ["support"],
            "additionalProperties": false,
        });
        let prompt = Prompt::default().json_schema(schema.clone());
        let cfg = prompt.output_config.as_ref().unwrap();
        assert!(cfg.format.as_ref().unwrap().is_json_schema());

        // Wire shape matches Anthropic's `output_config.format` exactly.
        let value = serde_json::to_value(&prompt).unwrap();
        assert_eq!(
            value["output_config"],
            json!({
                "format": {
                    "type": "json_schema",
                    "schema": schema,
                }
            })
        );

        // Roundtrip.
        let back = serde_json::from_value::<Prompt>(value).unwrap();
        assert_eq!(back.output_config, prompt.output_config);
    }

    #[test]
    fn test_output_config_accepts_into_impls() {
        // From<serde_json::Value>
        let from_value: Prompt =
            Prompt::default().output_config(json!({"type": "object"}));
        assert!(from_value.output_config.is_some());

        // From<JsonSchemaFormat>
        let from_format = Prompt::default().output_config(JsonSchemaFormat {
            schema: json!({"type": "object"}),
        });
        assert!(from_format.output_config.is_some());

        // Explicit OutputConfig.
        let explicit = Prompt::default().output_config(
            OutputConfig::json_schema(json!({"type": "object"})),
        );
        assert!(explicit.output_config.is_some());
    }

    #[test]
    fn test_structured_output_from_type() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct VoteIntent {
            post_id: String,
            support: bool,
            rationale: String,
        }

        let prompt = Prompt::default().structured_output::<VoteIntent>();
        let cfg = prompt.output_config.as_ref().unwrap();
        let Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema })) =
            &cfg.format
        else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        assert_eq!(
            schema.get("additionalProperties"),
            Some(&serde_json::Value::Bool(false))
        );
        let props = schema.get("properties").unwrap().as_object().unwrap();
        for name in ["post_id", "support", "rationale"] {
            assert!(props.contains_key(name), "missing property: {name}");
        }
    }

    #[test]
    fn test_effort_composes_with_format_either_order() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Out {
            answer: String,
        }

        // effort then format.
        let a = Prompt::default()
            .effort(Effort::Low)
            .structured_output::<Out>();
        let cfg = a.output_config.as_ref().unwrap();
        assert_eq!(cfg.effort, Some(Effort::Low));
        assert!(cfg.format.is_some(), "format clobbered by effort-first");

        // format then effort.
        let b = Prompt::default()
            .structured_output::<Out>()
            .effort(Effort::Max);
        let cfg = b.output_config.as_ref().unwrap();
        assert_eq!(cfg.effort, Some(Effort::Max));
        assert!(cfg.format.is_some(), "format clobbered by effort-last");

        // effort-only: no format, serializes without a format key.
        let c = Prompt::default().effort(Effort::Medium);
        let json = serde_json::to_value(&c).unwrap();
        assert_eq!(json["output_config"], json!({ "effort": "medium" }));
    }

    #[derive(serde::Serialize, schemars::JsonSchema)]
    struct Triage {
        component: String,
        is_regression: bool,
    }

    #[test]
    fn test_add_examples_sets_config_and_pairs() {
        let ex = Triage {
            component: "auth-ui".into(),
            is_regression: true,
        };
        // Serialize before the move so we can assert the assistant turn.
        let expected = serde_json::to_string(&ex).unwrap();

        let prompt = Prompt::default()
            .add_examples([("login broken on safari", ex)])
            .unwrap();

        // output_config is seeded from the exemplar type `A`.
        let cfg = prompt.output_config.as_ref().unwrap();
        let Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema })) =
            &cfg.format
        else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        let props = schema.get("properties").unwrap().as_object().unwrap();
        assert!(props.contains_key("component"));

        // Exactly one (User input, Assistant JSON) pair, in order.
        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(
            prompt.messages[0],
            (Role::User, "login broken on safari").into()
        );
        assert_eq!(
            prompt.messages[1],
            (Role::Assistant, expected.as_str()).into()
        );
    }

    #[test]
    fn test_add_examples_turn_order_error() {
        // A user turn already at the tail collides with the first example's
        // user turn, and the prompt is left unmodified.
        let err = Prompt::default()
            .add_message((Role::User, "real question"))
            .unwrap()
            .add_examples([(
                "example input",
                Triage {
                    component: "x".into(),
                    is_regression: false,
                },
            )])
            .unwrap_err();
        assert!(matches!(err, ExamplesError::TurnOrder(_)));
    }

    #[test]
    fn test_add_examples_clobbers_output_config() {
        // An explicitly-set config is overwritten by the exemplar's schema,
        // even when no examples are supplied.
        let prompt = Prompt::default()
            .json_schema(serde_json::json!({ "type": "object" }))
            .add_examples(std::iter::empty::<(&str, Triage)>())
            .unwrap();

        let cfg = prompt.output_config.as_ref().unwrap();
        let Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema })) =
            &cfg.format
        else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        let props = schema.get("properties").unwrap().as_object().unwrap();
        assert!(props.contains_key("is_regression"));
        assert!(prompt.messages.is_empty());
    }

    #[test]
    fn test_tools() {
        // A tool can be added from a json object. This is fallible. It must
        // deserialize into a Tool.
        let json_tool = json!({
            "name": "ping2",
            "description": "Ping a server. Part deux.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "The host to ping."
                    }
                },
                "required": ["host"]
            }
        });

        let schema = json_tool["input_schema"].clone();

        // A tool can be created from a Tool itself. This is infallible, however
        // the API might reject the request if the tool is invalid. There is
        // currently no schema validation in this crate.
        let tool = CustomMethodDef {
            name: "ping".into(),
            description: "Ping a server.".into(),
            schema: schema.clone(),
            cache_control: None,
            strict: None,
            defer_loading: None,
            allowed_callers: None,
        };

        let request = Prompt::default()
            .tools([tool])
            .try_add_tool(json_tool)
            .unwrap();

        let methods = request.tools.as_ref().unwrap();
        let method = |i: usize| methods[i].as_method().unwrap();
        assert_eq!(methods.len(), 2);
        assert_eq!(method(0).name, "ping");
        assert_eq!(method(1).name, "ping2");
        assert_eq!(method(0).description, "Ping a server.");
        assert_eq!(method(1).description, "Ping a server. Part deux.");
        assert_eq!(method(0).schema, schema);

        // Test with a fallible tool. This should fail.

        let invalid = json!({
            "potato": "ping3",
            "description": "Ping a server. Part trois.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "The host to ping."
                    }
                },
                "required": ["host"]
            }
        });
        let err = Prompt::default().try_add_tool(invalid.clone());
        if let Err(e) = err {
            assert_eq!(e.to_string(), "missing field `name`");
        } else {
            panic!("Expected an error.");
        }

        let err = Prompt::default().try_tools([invalid]);
        if let Err(e) = err {
            assert_eq!(e.to_string(), "missing field `name`");
        } else {
            panic!("Expected an error.");
        }
    }

    #[test]
    fn test_temperature() {
        let request = Prompt::default().temperature(0.5);
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    #[allow(unused_variables)] // because the compiler is silly sometimes
    fn test_tool_choice() {
        let choice = tool::Choice::any();
        let request = Prompt::default().tool_choice(choice);
        assert!(matches!(
            request.tool_choice,
            Some(tool::Choice::Any { .. })
        ));
    }

    #[test]
    fn test_top_k() {
        let request = Prompt::default().top_k(NonZeroU16::new(5).unwrap());
        assert_eq!(request.top_k, Some(NonZeroU16::new(5).unwrap()));
    }

    #[test]
    fn test_top_p() {
        let request = Prompt::default().top_p(0.5);
        assert_eq!(request.top_p, Some(0.5));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_markdown() {
        use crate::markdown::{Markdown, ToMarkdown};

        let request = Prompt::default()
            .tools([CustomMethodDef {
                name: "ping".into(),
                description: "Ping a server.".into(),
                schema: json!({
                    "type": "object",
                    "properties": {
                        "host": {
                            "type": "string",
                            "description": "The host to ping."
                        }
                    },
                    "required": ["host"]
                }),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            }])
            .system("You are a very succinct assistant.")
            .messages([
                Message {
                    role: Role::User,
                    content: Content::text("Hello"),
                },
                Message {
                    role: Role::Assistant,
                    content: Content::text("Hi"),
                },
                Message {
                    role: Role::User,
                    content: Content::text("Call a tool."),
                },
                tool::Use::new(
                    "ping",
                    json!({
                        "host": "example.com"
                    }),
                )
                .with_id("abc123")
                .into(),
                tool::Result::new("abc123", "Pinging example.com.").into(),
                Message {
                    role: Role::Assistant,
                    content: Content::text("Done."),
                },
            ])
            .unwrap();

        let markdown: Markdown = request.markdown_verbose();

        // OpenAI format. Anthropic doesn't have a "system" or "tool" role but
        // we generate markdown like this because it's easier to read. The user
        // does not submit a tool result, so it's confusing if the header is
        // "User".
        let expected = "### System { role=system }\n\nYou are a very succinct assistant.\n\n### User { role=user }\n\nHello\n\n### Assistant { role=assistant }\n\nHi\n\n### User { role=user }\n\nCall a tool.\n\n### Assistant { role=assistant }\n\n````json\n{\"type\":\"tool_use\",\"id\":\"abc123\",\"name\":\"ping\",\"input\":{\"host\":\"example.com\"}}\n````\n\n### Tool { role=tool }\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"abc123\",\"content\":[{\"type\":\"text\",\"text\":\"Pinging example.com.\"}],\"is_error\":false}\n````\n\n### Assistant { role=assistant }\n\nDone.";

        assert_eq!(markdown.as_ref(), expected);
    }
}
