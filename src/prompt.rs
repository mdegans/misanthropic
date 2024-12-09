//! [Anthropic Messages API] `Request` type. We call it [`Prompt`] since in
//! actual usage this makes the code more readable.
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use std::{borrow::Cow, num::NonZeroU16, vec};

use crate::{tool, Model, Tool};
use message::Content;
use serde::{Deserialize, Serialize};

pub mod message;
pub use message::Message;

/// Request for the [Anthropic Messages API].
///
/// [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
#[derive(Serialize, Deserialize, Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Prompt<'a> {
    /// [`Model`] to use for inference.
    pub model: Model<'a>,
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
}

impl Default for Prompt<'_> {
    fn default() -> Self {
        Self {
            model: Default::default(),
            messages: Default::default(),
            max_tokens: NonZeroU16::new(4096).unwrap(),
            metadata: Default::default(),
            stop_sequences: Default::default(),
            stream: Default::default(),
            system: Default::default(),
            temperature: Default::default(),
            tool_choice: Default::default(),
            tools: Default::default(),
            top_k: Default::default(),
            top_p: Default::default(),
        }
    }
}

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
        M: Into<Model<'a>>,
    {
        self.model = model.into();
        self
    }

    /// Set the [`messages`] from an iterable of [`Message`]s.
    ///
    /// [`messages`]: Prompt::messages
    pub fn messages<M, Ms>(mut self, messages: Ms) -> Self
    where
        M: Into<Message<'a>>,
        Ms: IntoIterator<Item = M>,
    {
        self.messages = messages.into_iter().map(Into::into).collect();
        self
    }

    /// Add a [`Message`] to [`messages`].
    ///
    /// [`messages`]: Prompt::messages
    pub fn add_message<M>(mut self, message: M) -> Self
    where
        M: Into<Message<'a>>,
    {
        self.messages.push(message.into());
        self
    }

    /// Extend the [`messages`] from an iterable.
    ///
    /// [`messages`]: Prompt::messages
    pub fn extend_messages<M, Ms>(mut self, messages: Ms) -> Self
    where
        M: Into<Message<'a>>,
        Ms: IntoIterator<Item = M>,
    {
        self.messages.extend(messages.into_iter().map(Into::into));
        self
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
    pub fn system<S>(mut self, system: S) -> Self
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
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Prompt<'_> {
    /// Format the [`Prompt`] as markdown in OpenAI style. H3 headings are used
    /// for "System", "Tool", "User", and "Assistant" messages even though
    /// technically there are only [`User`] and [`Assistant`] [`Role`]s.
    ///
    /// [`User`]: message::Role::User
    /// [`Assistant`]: message::Role::Assistant
    /// [`Role`]: message::Role
    fn markdown_events_custom<'a>(
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

    const STOP_SEQUENCES: [&'static str; 2] = ["stop1", "stop2"];

    // Credit to GitHub Copilot for the following tests.

    #[test]
    fn test_default_request() {
        let request = Prompt::default();
        assert_eq!(request.model, Model::default());
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
    fn test_set_model() {
        let model = AnthropicModel::default();
        let request = Prompt::default().model(model); // AnthropicModel is Copy
        assert_eq!(request.model, Model::default());
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
        let request = Prompt::default().messages(create_test_messages());
        assert_eq!(request.messages, create_test_messages());
    }

    #[test]
    fn test_add_message() {
        let prompt = Prompt::default()
            .add_message((Role::User, "Hello"))
            .add_message((Role::Assistant, "Hi"));
        assert_eq!(prompt.messages.len(), 2);
        assert_eq!(prompt.messages[0], (Role::User, "Hello").into());
        assert_eq!(prompt.messages[1], (Role::Assistant, "Hi").into());
    }

    #[test]
    fn test_extend_messages() {
        let mut request = Prompt::default();
        request = request.extend_messages(create_test_messages());
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
            .add_message(Message {
                role: Role::Assistant,
                content: Content::text("Hi"),
            })
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
            ]);

        let markdown: Markdown = request.markdown_verbose();

        // OpenAI format. Anthropic doesn't have a "system" or "tool" role but
        // we generate markdown like this because it's easier to read. The user
        // does not submit a tool result, so it's confusing if the header is
        // "User".
        let expected = "### System { role=system }\n\nYou are a very succinct assistant.\n\n### User { role=user }\n\nHello\n\n### Assistant { role=assistant }\n\nHi\n\n### User { role=user }\n\nCall a tool.\n\n### Assistant { role=assistant }\n\n````json\n{\"type\":\"tool_use\",\"id\":\"abc123\",\"name\":\"ping\",\"input\":{\"host\":\"example.com\"}}\n````\n\n### Tool { role=tool }\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"abc123\",\"content\":[{\"type\":\"text\",\"text\":\"Pinging example.com.\"}],\"is_error\":false}\n````\n\n### Assistant { role=assistant }\n\nDone.";

        assert_eq!(markdown.as_ref(), expected);
    }
}
