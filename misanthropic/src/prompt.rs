//! [Anthropic Messages API] `Request` type. We call it [`Prompt`] since in
//! actual usage this makes the code more readable.
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use std::{borrow::Cow, num::NonZeroU16, vec};

use crate::{
    stream::{self, DeltaError},
    tool, Id, Tool,
};
use message::Content;

use futures::TryStreamExt;
use serde::{Deserialize, Serialize};

pub mod message;
pub use message::{AssistantMessage, Message, UserMessage};

pub mod thinking;
pub use thinking::Thinking;

/// Request for the [Anthropic Messages API].
///
/// [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Prompt<'a> {
    /// [`Model`] to use for inference.
    pub model: Id<'a>,
    /// Input [`prompt::message`]s. If this ends with an [`Assistant`]
    /// [`Message`], the completion will be constrained by that last message.
    /// Otherwise a new [`Assistant`] [`Message`] will be generated.
    ///
    /// See [Anthropic docs] for more information.
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    /// [`prompt::message`]: crate::prompt::message
    /// [Anthropic docs]: <https://docs.anthropic.com/en/api/messages>
    pub messages: Vec<Message<'a>>,
    /// Max tokens to generate. See Anthropic [docs] for the maximum number of
    /// tokens for each model.
    ///
    /// [docs]: <https://docs.anthropic.com/en/docs/about-claude/models>
    pub max_tokens: NonZeroU16,
    /// Optional info about the request, for example, `user_id` to help
    /// Anthropic detect and prevent abuse. Do not use PII here (email, phone).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
    /// Optional stop sequences. If the model generates any of these sequences,
    /// the completion will stop with [`StopReason::StopSequence`].
    ///
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<Cow<'a, str>>>,
    /// If `true`, the response will be a stream of [`Event`]s. If `false`, the
    /// response will be a single [`response::Message`].
    ///
    /// [`Event`]: crate::stream::Event
    /// [`response::Message`]: crate::response::Message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// System prompt as [`SinglePart`] or [`MultiPart`] [`Content`].
    ///
    /// [`SinglePart`]: message::Content::SinglePart
    /// [`MultiPart`]: message::Content::MultiPart
    /// [`Content`]: message::Content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<message::Content<'a>>,
    /// Temperature for sampling. Must be between 0 and 1. Higher values mean
    /// more randomness. Note that 0.0 is not fully deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// [`tool::Choice`] for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<tool::Choice>,
    /// Tool definitions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool<'a>>>,
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
    /// Thinking support. Note that this is only required for using Anthropic's
    /// built-in COT support with Sonnet 3.7 and later models. The `cot` feature
    /// can be used with all models, provided the system prompt instructs the
    /// Assistant to use `<thiking>` tags.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<Thinking>,
}

impl std::fmt::Debug for Prompt<'_> {
    /// For the sake of user privacy, the debug repr of a [`Prompt`] will hide
    /// the user's chat history. Otherwise it's likely to end up in logs.
    ///
    /// Metadata is still shown, so don't put PII in there. If you do, somewhere
    /// in your design you've made a mistake. Rethink your design.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Prompt")
            .field("metadata", &self.metadata)
            .field("stop_sequences", &self.stop_sequences)
            .field("stream", &self.stream)
            .field("system", &self.system)
            .field("temperature", &self.temperature)
            .field("tool_choice", &self.tool_choice)
            .field("tools", &self.tools)
            .field("top_k", &self.top_k)
            .field("...", &"...")
            .finish()
    }
    // For all sorts of reasons like user privacy we are going to hide the
    // contents of the prompt as in `Prompts`
}

impl Default for Prompt<'_> {
    fn default() -> Self {
        Self {
            max_tokens: NonZeroU16::new(4096).unwrap(),
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
        }
    }
}

/// Message turn order is incorrect.
#[derive(Debug, thiserror::Error, Serialize, Deserialize)]
#[error("The message turn order must alternate between User and Assistant. The first message is {first:?} and the second message is {second:?}.")]
pub struct TurnOrderError {
    /// First message in the pair of duplicate roles.
    pub first: Message<'static>,
    /// Second message in the pair of duplicate roles.
    pub second: Message<'static>,
}
static_assertions::assert_impl_all!(TurnOrderError: Send, Sync);

impl<'a> Prompt<'a> {
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
        M: Into<Id<'a>>,
    {
        self.model = model.into();
        self
    }

    /// Set the [`messages`] from an iterable of [`Message`]s.
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    pub fn set_messages<M, Ms>(
        mut self,
        messages: Ms,
    ) -> Result<Self, TurnOrderError>
    where
        M: Into<Message<'a>>,
        Ms: IntoIterator<Item = M>,
    {
        self.messages = messages.into_iter().map(Into::into).collect();
        self.check_turn_order()?;
        Ok(self)
    }

    /// Check the turn order of [`messages`]. Returns the **first** pair of
    /// messages that are the same role.
    pub fn check_turn_order(&self) -> Result<(), TurnOrderError> {
        for pair in self.messages.windows(2) {
            if pair[0].role == pair[1].role {
                return Err(TurnOrderError {
                    first: pair[0].clone().into_static(),
                    second: pair[1].clone().into_static(),
                });
            }
        }
        Ok(())
    }

    /// Add a [`Message`] to [`messages`]. When adding multiple messages, use
    /// [`add_messages`] or [`push_messages`] for better performance.
    ///
    /// # Panics
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    /// [`add_messages`]: Prompt::add_messages
    /// [`push_messages`]: Prompt::push_messages
    // So we don't break the API, but in version 1.0.0 this will be removed.
    #[deprecated(
        since = "0.6.0",
        note = "Use `add_message` or `push_message` instead."
    )]
    pub fn message<M>(self, message: M) -> Self
    where
        M: Into<Message<'a>>,
    {
        self.add_message(message).unwrap()
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
        M: Into<Message<'a>>,
    {
        let message: Message<'a> = message.into();
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
    pub fn push_message<M>(&mut self, message: M) -> Result<(), TurnOrderError>
    where
        M: Into<Message<'a>>,
    {
        let message: Message<'a> = message.into();
        if let Some(last) = self.messages.last() {
            if last.role == message.role {
                return Err(TurnOrderError {
                    first: last.clone().into_static(),
                    second: message.clone().into_static(),
                });
            }
        } else {
            // The first message must be a user message.
            if message.role.is_assistant() {
                return Err(TurnOrderError {
                    first: message.clone().into_static(),
                    second: message.clone().into_static(),
                });
            }
        }
        self.messages.push(message);
        Ok(())
    }

    /// Extend the [`messages`] from an iterable. For an in-place version, see
    /// [`push_messages`]. For a fallible version, see [`add_messages`].
    ///
    /// # Panics
    /// - If the turn order is incorrect.
    ///
    /// [`messages`]: Prompt::messages
    /// [`push_messages`]: Prompt::push_messages
    // So we don't break the API, but in version 1.0.0 this will be removed.
    #[deprecated(
        since = "0.6.0",
        note = "Use `add_message` or `push_message` instead."
    )]
    pub fn messages<M, Ms>(self, messages: Ms) -> Self
    where
        M: Into<Message<'a>>,
        Ms: IntoIterator<Item = M>,
    {
        self.add_messages(messages).unwrap()
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
        M: Into<Message<'a>>,
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
    ) -> Result<(), TurnOrderError>
    where
        M: Into<Message<'a>>,
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
            Ok(())
        }
    }

    /// Set the [`max_tokens`]. If this is reached, the [`StopReason`] will be
    /// [`MaxTokens`] in the [`response::Message::stop_reason`].
    ///
    /// [`max_tokens`]: Prompt::max_tokens
    /// [`StopReason`]: crate::response::StopReason
    /// [`MaxTokens`]: crate::response::StopReason::MaxTokens
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    pub fn max_tokens(mut self, max_tokens: NonZeroU16) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set the [`metadata`] from an iterable of key-value pairs.
    /// The values must be serializable to JSON.
    ///
    /// # Panics
    /// - if a value cannot be serialized to JSON.
    ///
    /// See [`try_metadata`] for a fallible version.
    ///
    /// [`metadata`]: Prompt::metadata
    /// [`try_metadata`]: Prompt::try_metadata
    pub fn metadata<S, V, Vs>(mut self, metadata: Vs) -> Self
    where
        S: Into<String>,
        V: Serialize,
        Vs: IntoIterator<Item = (S, V)>,
    {
        self.metadata = metadata
            .into_iter()
            .map(|(k, v)| (k.into(), serde_json::to_value(v).unwrap()))
            .collect();
        self
    }

    /// Set the [`metadata`] from an iterable of key-value pairs.
    /// The values must be serializable to JSON.
    ///
    /// [`metadata`]: Prompt::metadata
    pub fn try_metadata<S, V, Vs>(
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
    pub fn insert_metadata<S, V>(
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
    pub fn stop_sequence<S>(mut self, stop_sequence: S) -> Self
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
    pub fn extend_stop_sequences<S, Ss>(mut self, stop_sequences: Ss) -> Self
    where
        S: Into<Cow<'a, str>>,
        Ss: IntoIterator<Item = S>,
    {
        self.stop_sequences
            .get_or_insert_with(Default::default)
            .extend(stop_sequences.into_iter().map(Into::into));
        self
    }

    /// Set the [`system`] prompt [`Content`]. This is content that the model
    /// will give special attention to. Instructions should be placed here.
    ///
    /// [`system`]: Prompt::system
    // So we don't break the API, but in version 1.0.0 this will be removed.
    #[deprecated(
        since = "0.6.0",
        note = "Use `set_system` or `add_system` instead."
    )]
    pub fn system<S>(mut self, system: S) -> Self
    where
        S: Into<message::Content<'a>>,
    {
        self.system = Some(system.into());
        self
    }

    /// Set the [`system`] prompt [`Content`]. This is content that the model
    /// will give special attention to. Instructions should be placed here.
    ///
    /// [`system`]: Prompt::system
    pub fn set_system<S>(mut self, system: S) -> Self
    where
        S: Into<message::Content<'a>>,
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
        B: Into<message::Block<'a>>,
    {
        match self.system {
            Some(mut content) => {
                content.push(block);
                self.system = Some(content);
            }
            None => {
                // MultiPart doesn't actually need to have multiple parts.
                self.system = Some(Content::MultiPart(vec![block.into()]));
            }
        }
        self
    }

    /// Set the [`temperature`] to `Some(value)` or [`None`] to use the default.
    ///
    /// [`temperature`]: Prompt::temperature
    pub fn temperature(mut self, temperature: Option<f32>) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set the [`tool::Choice`]. This constrains how the model uses tools.
    ///
    /// [`tool::Choice`]: crate::tool::Choice
    pub fn tool_choice(mut self, choice: tool::Choice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Set the available [`tools`]. When the [`Model`] uses a [`Tool`], the
    /// [`StopReason`] will be [`ToolUse`] in the
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
        T: Into<Tool<'a>>,
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
    /// [`Tool`]: crate::Tool
    /// [`StopReason`]: crate::response::StopReason
    /// [`ToolUse`]: crate::response::StopReason::ToolUse
    /// [`response::Message::stop_reason`]: crate::response::Message::stop_reason
    /// [`Block::ToolUse`]: message::Block::ToolUse
    /// [`id`]: tool::use::id
    /// [`Role`]: message::Role
    /// [`User`]: message::Role::User
    /// [`Block`]: message::Block
    /// [`ToolResult`]: message::Block::ToolResult
    /// [`tool_use_id`]: crate::tool::Result::tool_use_id
    pub fn try_tools<T, E, Ts>(mut self, tools: Ts) -> Result<Self, E>
    where
        T: TryInto<Tool<'a>, Error = E>,
        Ts: IntoIterator<Item = T>,
    {
        self.tools = Some(
            tools
                .into_iter()
                .map(TryInto::try_into)
                .collect::<Result<_, _>>()?,
        );
        Ok(self)
    }

    /// Add a tool to the request.
    pub fn add_tool<T>(mut self, tool: T) -> Self
    where
        T: Into<Tool<'a>>,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .push(tool.into());
        self
    }

    /// Try to add a tool to the request. Returns an error if the value cannot
    /// be converted into a [`Tool`].
    pub fn try_add_tool<T, E>(mut self, tool: T) -> Result<Self, E>
    where
        T: TryInto<Tool<'a>, Error = E>,
    {
        self.tools
            .get_or_insert_with(Default::default)
            .push(tool.try_into()?);
        Ok(self)
    }

    // No extend for tools because it's not very common or useful. If somebody
    // really wants this they can submit a PR.

    /// Set the top K tokens to consider for each token. Set to `None` to use
    /// the default value.
    pub fn top_k(mut self, top_k: Option<NonZeroU16>) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the top P for nucleus sampling. Set to [`None`] to use the default
    /// value.
    pub fn top_p(mut self, top_p: Option<f32>) -> Self {
        self.top_p = top_p;
        self
    }

    /// Set the [`Thinking`] support.
    pub fn thinking(mut self, thinking: Thinking) -> Self {
        self.thinking = Some(thinking);
        self
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
    /// [`Sonnet35`]: crate::Model::Sonnet35
    /// [`Opus30`]: crate::Model::Opus30
    /// [`Haiku30`]: crate::Model::Haiku30
    #[cfg(feature = "prompt-caching")]
    pub fn cache(mut self) -> Self {
        // If there are messages, add a cache breakpoint to the last one.
        if let Some(last) = self.messages.last_mut() {
            last.content.cache();
            return self;
        }

        // If there are no messages, add a cache breakpoint to the system prompt
        // if it exists.
        if let Some(system) = self.system.as_mut() {
            system.cache();
            return self;
        }

        // If there are no messages or system prompt, add a cache breakpoint to
        // the tools if they exist.
        if let Some(tool) =
            self.tools.as_mut().and_then(|tools| tools.last_mut())
        {
            tool.cache();
            return self;
        }

        self
    }

    /// Convert to static lifetime by taking ownership of the [`Cow`] fields.
    pub fn into_static(self) -> Prompt<'static> {
        Prompt {
            model: self.model.into_static(),
            messages: self
                .messages
                .into_iter()
                .map(Message::into_static)
                .collect(),
            max_tokens: self.max_tokens,
            metadata: self.metadata,
            stop_sequences: self.stop_sequences.map(|s| {
                s.into_iter().map(Cow::into_owned).map(Cow::Owned).collect()
            }),
            stream: self.stream,
            system: self.system.map(Content::into_static),
            temperature: self.temperature,
            tool_choice: self.tool_choice,
            tools: self
                .tools
                .map(|t| t.into_iter().map(Tool::into_static).collect()),
            top_k: self.top_k,
            top_p: self.top_p,
            thinking: self.thinking,
        }
    }

    /// Apply a [`stream::Event`] to the [`Prompt`]. This is useful for
    /// appending to a [`Prompt`] in a streaming context.
    ///
    /// # Note
    /// - If the `partial-eq` feature is enabled, this will check for equality
    ///   for `Event::Message` and `Event::ToolUse` events, checking for the
    ///   consistency of the final message or tool use. Otherwise these messages
    ///   are ignored.
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
                            return Err(e.into_static().into());
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
                #[cfg(feature = "partial-eq")]
                if let Some(last) = self.messages.last() {
                    if *last == **&message.inner {
                        return Ok(());
                    }

                    return Err(ApplyEventError::UnexpectedMessage {
                        event: Event::Message { message },
                        last: last.clone().into_static(),
                    });
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
                #[cfg(feature = "partial-eq")]
                if let Some(last) = self.messages.last() {
                    if let Content::MultiPart(blocks) = &last.content {
                        if let Some(message::Block::ToolUse { call }) =
                            blocks.last()
                        {
                            if call == &tool_use {
                                return Ok(());
                            }
                        }
                    }

                    return Err(ApplyEventError::UnexpectedMessage {
                        event: Event::ToolUse { tool_use },
                        last: last.clone().into_static(),
                    });
                } else {
                    return Err(ApplyEventError::EmptyPrompt {
                        event: Event::ToolUse { tool_use },
                    });
                }
            }
            stream::Event::Ping
            | stream::Event::MessageStop
            | stream::Event::MessageDelta { .. } => {
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

    /// Extend a prompt with an [`Extendable`] object. This also functions as a
    /// append. This is useful for streaming prompts. This is async because some
    /// of the extendables, like [`stream::FilterExt`], are async.
    ///
    /// # Errors
    /// - If the turn order is incorrect.
    /// - If the stream of events cannot be applied to the prompt.
    pub async fn extend<E>(
        &'a mut self,
        extendable: E,
    ) -> Result<&'a mut Self, ExtendError>
    where
        E: ExtendOntoPrompt<'a>,
    {
        extendable.extend_onto(self).await
    }

    /// Helper for the above.
    pub async fn extend_stream<T>(
        &'a mut self,
        mut stream: std::pin::Pin<Box<T>>,
    ) -> Result<&'a mut Self, ExtendError>
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
}

/// Error when [`extend`]ing a [`Prompt`].
///
/// [`extend`]: Prompt::extend
#[derive(Debug, thiserror::Error)]
#[error(transparent)]
pub enum ExtendError {
    /// Turn Order is incorrect.
    TurnOrder(#[from] TurnOrderError),
    /// Error when applying a stream event to a prompt.
    ApplyEvent(#[from] ApplyEventError),
    /// Stream error.
    Stream(#[from] stream::Error),
    /// Other error.
    Other(Box<dyn std::error::Error + Send + Sync>),
}

/// Object that can be appended to a [`Prompt`].
#[async_trait::async_trait]
pub trait ExtendOntoPrompt<'a> {
    /// Extend the prompt with the extendable object.
    async fn extend_onto(
        self,
        prompt: &'a mut Prompt<'a>,
    ) -> Result<&'a mut Prompt<'a>, ExtendError>;
}

#[async_trait::async_trait]
impl<'a> ExtendOntoPrompt<'a> for Message<'a> {
    async fn extend_onto(
        self,
        prompt: &'a mut Prompt<'a>,
    ) -> Result<&'a mut Prompt<'a>, ExtendError> {
        prompt.push_message(self).map_err(ExtendError::TurnOrder)?;
        Ok(prompt)
    }
}

#[async_trait::async_trait]
impl<'a> ExtendOntoPrompt<'a> for stream::Event {
    async fn extend_onto(
        self,
        prompt: &'a mut Prompt<'a>,
    ) -> Result<&'a mut Prompt<'a>, ExtendError> {
        prompt.handle_stream_event(self)?;
        Ok(prompt)
    }
}

#[async_trait::async_trait]
impl<'a, T> ExtendOntoPrompt<'a> for T
where
    T: futures::stream::Stream<Item = Result<stream::Event, stream::Error>>
        + Sized
        + Send,
{
    async fn extend_onto(
        self,
        prompt: &'a mut Prompt<'a>,
    ) -> Result<&'a mut Prompt<'a>, ExtendError> {
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
        /// The unsupported [`Event`].
        event: stream::Event,
    },
    /// Turn Order is incorrect.
    #[error(transparent)]
    TurnOrderError {
        /// The cause of the error.
        #[from]
        error: TurnOrderError,
    },
    /// Expected the last message to be an [`Assistant`]. Similar to
    /// TurnOrderError but more specific and does not originate from
    /// `push_message` or `add_message`.
    #[error("`Role::Assistant` must be the final message role in the prompt to apply this `Event`.")]
    ExpectedAssistant {
        /// The [`Event`] that caused the error.
        event: stream::Event,
        /// The role of the last message.
        last: message::Role,
    },
    /// Delta application error.
    #[error(transparent)]
    Delta(#[from] DeltaError<'static>),
    /// Unexpected index. Not necessarily out of bounds, but applying this event
    /// would be incorrect.
    #[error("Index {actual} is unexpected.")]
    UnexpectedIndex {
        /// The [`Event`] that caused the error.
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
        last: Message<'static>,
    },
    /// Event cannot be applied to an empty prompt.
    #[error("The prompt is empty and cannot accept this `Event`.")]
    EmptyPrompt {
        /// The [`Event`] that caused the error.
        event: stream::Event,
    },
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Prompt<'a> {
    /// Format the [`Prompt`] as markdown in OpenAI style. H3 headings are used
    /// for "System", "Tool", "User", and "Assistant" messages even though
    /// technically there are only [`User`] and [`Assistant`] [`Role`]s.
    ///
    /// [`User`]: message::Role::User
    /// [`Assistant`]: message::Role::Assistant
    /// [`Role`]: message::Role
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
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
    use std::num::NonZeroU16;

    use crate::{prompt::message::Role, AnthropicModel};

    const STOP_SEQUENCES: [&str; 2] = ["stop1", "stop2"];

    // Credit to GitHub Copilot for the following tests.

    #[test]
    fn test_default_request() {
        let request = Prompt::default();
        assert_eq!(request.model, Id::default());
        assert!(request.messages.is_empty());
        assert_eq!(request.max_tokens, NonZeroU16::new(4096).unwrap());
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
        let model = AnthropicModel::default();
        let request = Prompt::default().model(model); // AnthropicModel is Copy
        assert_eq!(request.model, Id::default());
    }

    fn create_test_messages() -> [Message<'static>; 2] {
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
    fn test_set_messages() {
        let request = Prompt::default()
            .set_messages(create_test_messages())
            .unwrap();
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
    fn test_add_messages() {
        let mut request = Prompt::default();
        request = request.add_messages(create_test_messages()).unwrap();
        assert_eq!(request.messages, create_test_messages());
    }

    #[test]
    fn test_set_max_tokens() {
        let max_tokens = NonZeroU16::new(1024).unwrap();
        let request = Prompt::default().max_tokens(max_tokens);
        assert_eq!(request.max_tokens, max_tokens);
    }

    #[test]
    fn test_set_metadata() {
        let metadata = vec![("key".to_string(), json!("value"))];
        let request = Prompt::default().metadata(metadata);
        assert_eq!(request.metadata.get("key").unwrap(), "value");
    }

    #[test]
    fn test_try_metadata() {
        let request = Prompt::default()
            .try_metadata([("key", "value"), ("key2", "value2")])
            .unwrap();
        assert_eq!(request.metadata.get("key").unwrap(), "value");
        assert_eq!(request.metadata.get("key2").unwrap(), "value2");
    }

    #[test]
    fn test_insert_metadata() {
        let request =
            Prompt::default().insert_metadata("key", "value").unwrap();
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
        request = request.stop_sequence(STOP_SEQUENCES[0]);
        assert_eq!(request.stop_sequences.as_ref().unwrap().len(), 1);
        assert_eq!(request.stop_sequences.unwrap()[0], STOP_SEQUENCES[0]);
    }

    #[test]
    fn test_extend_stop_sequences() {
        let mut request = Prompt::default();
        request = request.extend_stop_sequences(STOP_SEQUENCES);
        assert_eq!(request.stop_sequences.unwrap().len(), 2);
    }

    #[test]
    fn test_set_system() {
        let request = Prompt::default().set_system("system");
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
    #[cfg(feature = "prompt-caching")]
    fn test_cache() {
        // Test with nothing to cache. This should be a no-op.
        let request = Prompt::default().cache();
        assert!(request == Prompt::default());

        // Test with no system prompt or messages that the call to cache affects
        // the tools.
        let request = Prompt::default().add_tool(Tool {
            name: "ping".into(),
            description: "Ping a server.".into(),
            input_schema: json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
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

        // The first message should still be a single part string.
        assert!(request.messages.first().unwrap().content.last().is_none());

        // By now the final part should be a multi part string, since only
        // Block has `cache_control`
        assert!(request
            .messages
            .last()
            .unwrap()
            .content
            .last()
            .unwrap()
            .is_cached());
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
        let tool = Tool {
            name: "ping".into(),
            description: "Ping a server.".into(),
            input_schema: schema.clone(),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let request = Prompt::default()
            .tools([tool])
            .try_add_tool(json_tool)
            .unwrap();

        assert_eq!(request.tools.as_ref().unwrap().len(), 2);
        assert_eq!(request.tools.as_ref().unwrap()[0].name, "ping");
        assert_eq!(request.tools.as_ref().unwrap()[1].name, "ping2");
        assert_eq!(
            request.tools.as_ref().unwrap()[0].description,
            "Ping a server."
        );
        assert_eq!(
            request.tools.as_ref().unwrap()[1].description,
            "Ping a server. Part deux."
        );
        assert_eq!(request.tools.as_ref().unwrap()[0].input_schema, schema);

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
        let request = Prompt::default().temperature(Some(0.5));
        assert_eq!(request.temperature, Some(0.5));
    }

    #[test]
    #[allow(unused_variables)] // because the compiler is silly sometimes
    fn test_tool_choice() {
        let choice = tool::Choice::Any;
        let request = Prompt::default().tool_choice(choice);
        assert!(matches!(request.tool_choice, Some(choice)));
    }

    #[test]
    fn test_top_k() {
        let request =
            Prompt::default().top_k(Some(NonZeroU16::new(5).unwrap()));
        assert_eq!(request.top_k, Some(NonZeroU16::new(5).unwrap()));
    }

    #[test]
    fn test_top_p() {
        let request = Prompt::default().top_p(Some(0.5));
        assert_eq!(request.top_p, Some(0.5));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_markdown() {
        use crate::markdown::{Markdown, ToMarkdown};

        let request = Prompt::default()
            .tools([Tool {
                name: "ping".into(),
                description: "Ping a server.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "host": {
                            "type": "string",
                            "description": "The host to ping."
                        }
                    },
                    "required": ["host"]
                }),
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }])
            .set_system("You are a very succinct assistant.")
            .set_messages([
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
                tool::Use {
                    id: "abc123".into(),
                    name: "ping".into(),
                    input: json!({
                        "host": "example.com"
                    }),
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                }
                .into(),
                tool::Result {
                    tool_use_id: "abc123".into(),
                    content: "Pinging example.com.".into(),
                    is_error: false,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                }
                .into(),
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
