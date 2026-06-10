//! A [`prompt::Message`] and associated types. The API will return a
//! [`response::Message`] with the same type plus additional metadata.
//!
//! [`response::Message`]: crate::response::Message
//! [`prompt::Message`]: crate::prompt::Message

use std::borrow::Cow;

use base64::engine::{Engine as _, general_purpose};
use serde::{Deserialize, Serialize};

use crate::{
    prompt::Citation,
    response,
    stream::{ContentMismatch, Delta, DeltaError},
    tool,
    utils::cold_path,
};

/// Role of the [`Message`] author.
#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    PartialEq,
    Hash,
    derive_more::IsVariant,
)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// From the user.
    User,
    /// From the AI.
    Assistant,
    /// An operator-authoritative instruction injected *within* the
    /// conversation, distinct from the top-level [`Prompt::system`] field.
    ///
    /// Unlike a [`User`] turn, a system turn is treated as authoritative: when
    /// instructions conflict, system outranks user. It also overrides the
    /// top-level system prompt for the turns that follow it, and — appended
    /// after the cached prefix — does not bust the prompt cache.
    ///
    /// Placement is constrained; see [turn order]. Available on
    /// [Opus 4.8](crate::model::Id::Opus48) and later.
    ///
    /// Never place untrusted content (raw tool output, retrieved documents,
    /// web content) in a system turn — keep it in [`tool::Result`] blocks.
    ///
    /// [`User`]: Role::User
    /// [`Prompt::system`]: crate::Prompt::system
    /// [`tool::Result`]: crate::tool::Result
    /// [turn order]: crate::prompt::TurnOrderError
    System,
}

impl Role {
    /// Get the string representation of the role.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::User => "User",
            Self::Assistant => "Assistant",
            Self::System => "System",
        }
    }

    /// Convenience method for lowercase role.
    pub const fn as_lowercase(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::System => "system",
        }
    }

    /// Toggle the role between [`Role::User`] and [`Role::Assistant`].
    ///
    /// [`Role::System`] is not part of the user/assistant alternation, so it
    /// toggles to itself.
    pub const fn toggle(&self) -> Self {
        match self {
            Self::User => Self::Assistant,
            Self::Assistant => Self::User,
            Self::System => Self::System,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A message in a [`Request`]. See [`response::Message`] for the version with
/// additional metadata.
///
/// A message is [`Display`]ed as markdown with a heading indicating the
/// [`Role`] of the author. [`Image`]s are supported and will be rendered as
/// markdown images with embedded base64 data.
///
/// [`Display`]: std::fmt::Display
/// [`Request`]: crate::prompt
/// [`response::Message`]: crate::response::Message
#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("{}{}{}{}", Self::HEADING, role, Content::SEP, content)
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Message {
    /// Who is the message from.
    pub role: Role,
    /// The [`Content`] of the message as a sequence of [`Block`]s.
    pub content: Content,
}

impl Message {
    /// Heading for the message when rendered as markdown using [`Display`].
    ///
    /// [`Display`]: std::fmt::Display
    #[cfg(not(feature = "markdown"))]
    pub const HEADING: &'static str = "### ";

    /// Returns the number of [`Content`] [`Block`]s in the message.
    pub fn len(&self) -> usize {
        self.content.len()
    }

    /// Returns true if self has no parts.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// Returns Some([`tool::Use`]) if the final [`Content`] [`Block`] is a
    /// [`Block::ToolUse`].
    pub fn tool_use(&self) -> Option<&crate::tool::Use> {
        self.content.last()?.tool_use()
    }

    /// Returns Some([`tool::Use`]) if the final [`Content`] [`Block`] is a
    /// [`Block::ServerToolUse`] (a server tool the API ran itself).
    pub fn server_tool_use(&self) -> Option<&crate::tool::Use> {
        match self.content.last()? {
            Block::ServerToolUse { call } => Some(call),
            _ => None,
        }
    }

    /// Returns Some([`tool::Result`]) if the first [`Content`] [`Block`] is a
    /// [`Block::ToolResult`].
    pub fn tool_result(&self) -> Option<&crate::tool::Result> {
        if let Some(Block::ToolResult { result }) = self.content.first() {
            Some(result)
        } else {
            None
        }
    }

    /// Whether this turn may be immediately followed by `next` in the
    /// `messages` array.
    ///
    /// Encodes the user/assistant alternation plus the placement rules for a
    /// mid-conversation [`System`](Role::System) turn (it must follow a user
    /// turn and immediately precede an assistant turn), with one
    /// content-sensitive exception: two adjacent [`Assistant`](Role::Assistant)
    /// turns are allowed when the first carries a
    /// [`ServerToolUse`](Block::ServerToolUse) block — see [`TurnOrderError`]
    /// for the rationale.
    ///
    /// [`TurnOrderError`]: crate::prompt::TurnOrderError
    pub(crate) fn may_precede(&self, next: &Self) -> bool {
        use Role::{Assistant, System, User};
        matches!(
            (self.role, next.role),
            (User, Assistant)
                | (Assistant, User)
                | (User, System)
                | (System, Assistant)
        ) || (self.role == Assistant
            && next.role == Assistant
            && self.has_server_tool_use())
    }

    /// Whether any [`Content`] [`Block`] is a
    /// [`ServerToolUse`](Block::ServerToolUse).
    fn has_server_tool_use(&self) -> bool {
        self.content
            .iter()
            .any(|b| matches!(b, Block::ServerToolUse { .. }))
    }

    /// A convenience method to fix an incomplete [`Block::Thought`] in the case
    /// of interruption in a streaming context.
    ///
    /// Such messages must be removed or the API will reject the request if this
    /// is sent in a new request (because the signature will be absent).
    ///
    /// If, after removing the last block, there are no more blocks, None will
    /// be returned.
    pub fn remove_incomplete_thought(mut self) -> Option<Self> {
        if self.role != Role::Assistant {
            // There cannot be thinking content from the user.
            return Some(self);
        }

        if let Some(Block::Thought { signature, .. }) = self.content.last()
            && signature.is_empty()
        {
            self.content.pop();
        }

        if self.is_empty() { None } else { Some(self) }
    }
}

impl From<response::Message> for Message {
    fn from(message: response::Message) -> Self {
        message.inner.inner
    }
}

impl From<response::Message> for AssistantMessage {
    fn from(message: response::Message) -> Self {
        message.inner
    }
}

impl<T> From<(Role, T)> for Message
where
    T: Into<Content>,
{
    fn from((role, content): (Role, T)) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

impl From<tool::Use> for Message {
    fn from(call: tool::Use) -> Self {
        Message {
            role: Role::Assistant,
            content: call.into(),
        }
    }
}

impl From<tool::Result> for Message {
    fn from(result: tool::Result) -> Self {
        Message {
            role: Role::User,
            content: result.into(),
        }
    }
}

impl IntoIterator for Message {
    type Item = Block;
    type IntoIter = std::vec::IntoIter<Block>;

    fn into_iter(self) -> Self::IntoIter {
        self.content.into_iter()
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Message {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{Event, HeadingLevel::H3, Tag};

        let content = self.content.markdown_events_custom(options);
        let role = match self.content.last() {
            Some(Block::ToolResult {
                result: tool::Result { is_error, .. },
            }) => {
                if !options.tool_results {
                    return Box::new(std::iter::empty());
                }

                if *is_error { "Error" } else { "Tool" }
            }
            Some(Block::ToolUse { .. }) => {
                if !options.tool_use {
                    return Box::new(std::iter::empty());
                }

                self.role.as_str()
            }
            _ => self.role.as_str(),
        };
        let heading_tag = Tag::Heading {
            level: options.heading_level.unwrap_or(H3),
            id: None,
            classes: vec![],
            attrs: if options.attrs {
                vec![("role".into(), Some(role.to_lowercase().into()))]
            } else {
                vec![]
            },
        };
        let heading_end = heading_tag.to_end();
        let heading = [
            Event::Start(heading_tag),
            Event::Text(role.into()),
            Event::End(heading_end),
        ];

        Box::new(heading.into_iter().chain(content))
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

/// A message guaranteed to be from the assistant.
#[derive(
    Debug,
    Serialize,
    Clone,
    derive_more::Deref,
    Deserialize,
    derive_more::Display,
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(try_from = "Message", into = "Message")]
#[display("{}", inner)]
pub struct AssistantMessage {
    pub(crate) inner: Message, // Invariant: role == Role::Assistant
}

impl AssistantMessage {
    /// An assistant turn whose [`Content`] is a single [`Block::Text`]. Handy
    /// for prefill and for hand-authored examples (see
    /// [`Prompt::add_examples`](crate::Prompt::add_examples)).
    pub fn text<T>(text: T) -> Self
    where
        T: Into<crate::CowStr>,
    {
        Content::text(text).into()
    }

    /// Get the inner [`Content`].
    pub fn content(&self) -> &Content {
        &self.inner.content
    }

    /// Get the inner [`Content`] mutably.
    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.inner.content
    }

    /// Remove an incomplete [`Block::Thought`] from the end of the message.
    /// See [`Message::remove_incomplete_thought`] for more information.
    pub fn remove_incomplete_thought(self) -> Option<Self> {
        self.inner
            .remove_incomplete_thought()
            .map(|inner| AssistantMessage { inner })
    }
}

impl From<Content> for AssistantMessage {
    fn from(content: Content) -> Self {
        Self {
            inner: Message {
                role: Role::Assistant,
                content,
            },
        }
    }
}

impl From<String> for AssistantMessage {
    fn from(string: String) -> Self {
        Content::text(string).into()
    }
}

impl From<&str> for AssistantMessage {
    fn from(string: &str) -> Self {
        Content::text(string.to_owned()).into()
    }
}

impl From<AssistantMessage> for Message {
    fn from(val: AssistantMessage) -> Self {
        val.inner
    }
}

impl From<AssistantMessage> for Content {
    fn from(val: AssistantMessage) -> Self {
        val.inner.content
    }
}

impl<T> FromIterator<T> for AssistantMessage
where
    T: Into<Block>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Content::from_iter(iter))
    }
}

impl TryFrom<Message> for AssistantMessage {
    type Error = NotTheAssistant;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        if message.role == Role::Assistant {
            Ok(Self { inner: message })
        } else {
            Err(NotTheAssistant)
        }
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for AssistantMessage {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        self.inner.markdown_events_custom(options)
    }
}

/// Error message when conversion to [`AssistantMessage`] fails.
#[derive(Debug, Serialize, Deserialize, thiserror::Error)]
#[error("Message is not from the assistant.")]
pub struct NotTheAssistant;

/// A message guaranteed to be from the user.
#[derive(
    Clone,
    Debug,
    derive_more::Deref,
    derive_more::Display,
    Deserialize,
    Serialize,
)]
#[serde(try_from = "Message", into = "Message")]
#[display("{}", inner)]
pub struct UserMessage {
    inner: Message, // Invariant: role == Role::User
}

impl UserMessage {
    /// Get the inner [`Content`].
    pub fn content(&self) -> &Content {
        &self.inner.content
    }

    /// Get the inner [`Content`] mutably.
    pub fn content_mut(&mut self) -> &mut Content {
        &mut self.inner.content
    }
}

impl From<Content> for UserMessage {
    fn from(content: Content) -> Self {
        Self {
            inner: Message {
                role: Role::User,
                content,
            },
        }
    }
}

impl From<UserMessage> for Content {
    fn from(message: UserMessage) -> Self {
        message.inner.content
    }
}

impl From<String> for UserMessage {
    fn from(string: String) -> Self {
        UserMessage {
            inner: Message {
                role: Role::User,
                content: Content::text(string),
            },
        }
    }
}

impl From<&str> for UserMessage {
    fn from(string: &str) -> Self {
        UserMessage {
            inner: Message {
                role: Role::User,
                content: Content::text(string.to_owned()),
            },
        }
    }
}

impl From<tool::Result> for UserMessage {
    fn from(result: tool::Result) -> Self {
        UserMessage {
            inner: result.into(),
        }
    }
}

impl<T> FromIterator<T> for UserMessage
where
    T: Into<Block>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Content::from_iter(iter))
    }
}

impl IntoIterator for UserMessage {
    type Item = Block;
    type IntoIter = std::vec::IntoIter<Block>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.content.into_iter()
    }
}

#[cfg(feature = "dioxus")]
impl From<dioxus::events::FormEvent> for UserMessage {
    fn from(event: dioxus::events::FormEvent) -> Self {
        UserMessage {
            inner: Message {
                role: Role::User,
                content: event.data().value().into(),
            },
        }
    }
}

#[cfg(feature = "dioxus")]
impl From<dioxus::html::FormData> for UserMessage {
    fn from(data: dioxus::html::FormData) -> Self {
        let content = data.into();
        UserMessage {
            inner: Message {
                role: Role::User,
                content,
            },
        }
    }
}

impl TryFrom<Message> for UserMessage {
    type Error = NotTheUser;

    fn try_from(message: Message) -> Result<Self, Self::Error> {
        if message.role == Role::User {
            Ok(Self { inner: message })
        } else {
            Err(NotTheUser)
        }
    }
}

impl From<UserMessage> for Message {
    fn from(message: UserMessage) -> Self {
        message.inner
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for UserMessage {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        self.inner.markdown_events_custom(options)
    }
}

/// Error message when conversion to [`UserMessage`] fails.
#[derive(Debug, Serialize, Deserialize, thiserror::Error)]
#[error("Message is not from the user.")]
pub struct NotTheUser;

impl From<NotTheUser> for Cow<'static, str> {
    fn from(_: NotTheUser) -> Self {
        "Message is not from the user.".into()
    }
}

/// Content of a [`Message`], stored as a sequence of [`Block`]s.
///
/// [`Content`] derefs to `Vec<Block>`, so the usual slice/`Vec` accessors
/// (`get`, `get_mut`, `iter`, `iter_mut`, `len`, indexing, ...) are available
/// directly. On the wire it always serializes as an array of blocks; a bare
/// JSON string is still accepted when deserializing and becomes a single
/// [`Block::Text`].
#[derive(
    Clone, Debug, Hash, Serialize, derive_more::Deref, derive_more::DerefMut,
)]
#[serde(transparent)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Content(pub Vec<Block>);

impl<'de> Deserialize<'de> for Content {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Mirror the API's "content may be a bare string or an array of blocks"
        // wire form with the untagged derive, then normalize to blocks. A
        // hand-written visitor would be more code and harder to reason about.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Wire {
            Text(crate::CowStr),
            Blocks(Vec<Block>),
        }

        Ok(Content(match Wire::deserialize(deserializer)? {
            Wire::Text(text) => vec![Block::Text {
                text,
                citations: None,
                cache_control: None,
            }],
            Wire::Blocks(blocks) => blocks,
        }))
    }
}

impl Content {
    /// Text content as a single [`Block::Text`].
    pub fn text<T>(text: T) -> Self
    where
        T: Into<crate::CowStr>,
    {
        Self(vec![Block::text(text)])
    }

    /// Add a [`Block`] to the [`Content`], returning the index of the inserted
    /// block.
    pub fn push<P>(&mut self, part: P) -> usize
    where
        P: Into<Block>,
    {
        let index = self.0.len();
        self.0.push(part.into());
        index
    }

    /// Add a cache breakpoint to the final [`Block`].
    ///
    /// Uses the default 5-minute ephemeral TTL. For a 1-hour TTL, use
    /// [`cache_1h`](Content::cache_1h).
    pub fn cache(&mut self) {
        self.cache_with(CacheControl::ephemeral());
    }

    /// Add a 1-hour cache breakpoint to the final [`Block`].
    ///
    /// Behaves identically to [`cache`](Content::cache) but uses
    /// [`CacheControl::one_hour`].
    pub fn cache_1h(&mut self) {
        self.cache_with(CacheControl::one_hour());
    }

    /// Add a cache breakpoint with a caller-provided [`CacheControl`] to the
    /// final [`Block`]. Does nothing if the content is empty.
    pub fn cache_with(&mut self, cache_control: CacheControl) {
        if let Some(block) = self.0.last_mut() {
            block.cache_with(cache_control);
        }
    }

    /// Remove all cache breakpoints from all blocks in this content.
    pub fn uncache(&mut self) {
        for block in &mut self.0 {
            block.uncache();
        }
    }

    /// Returns `true` if any block in this content has a cache breakpoint.
    pub fn has_cache(&self) -> bool {
        self.0.iter().any(|b| b.is_cached())
    }

    /// Push a [`Delta`] into the final [`Block`]. The types must be compatible
    /// or this will return a [`ContentMismatch`] error.
    ///
    /// It is an error to try to merge a single json delta into a content block.
    pub fn push_delta(&mut self, delta: Delta) -> Result<(), DeltaError> {
        if let Delta::Json { .. } = &delta {
            // It isn't possible to merge a single json delta into a content
            // block because ToolUse::input is a serde_json::Value and not a
            // string. Instead. FilterExt::with_tool_use should be used to
            // assemble tool use blocks.
            return Err(DeltaError::ContentMismatch {
                error: ContentMismatch {
                    from: delta.clone(),
                    to: stringify!(Content),
                },
            });
        }

        self.0
            .last_mut()
            .unwrap()
            .merge_deltas(std::iter::once(delta))?;

        Ok(())
    }

    /// Drains the blocks from the content.
    pub fn drain(&mut self) -> impl Iterator<Item = Block> + '_ {
        self.0.drain(..)
    }
}

impl IntoIterator for Content {
    type Item = Block;
    type IntoIter = std::vec::IntoIter<Block>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Content {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::Event;

        let it: Box<dyn Iterator<Item = Event<'_>> + '_> = Box::new(
            self.0
                .iter()
                .flat_map(move |part| part.markdown_events_custom(options)),
        );

        it
    }
}

#[cfg(not(feature = "markdown"))]
impl std::fmt::Display for Content {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // This could be derived but the `Join` trait is not stable. Neither is
        // `Iterator::intersperse`. This also has fewer allocations.
        let mut iter = self.0.iter();
        if let Some(part) = iter.next() {
            write!(f, "{}", part)?;
            for part in iter {
                write!(f, "{}{}", Self::SEP, part)?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "dioxus")]
impl From<dioxus::html::FormData> for Content {
    fn from(data: dioxus::html::FormData) -> Self {
        data.value().into()
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Content {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl Content {
    /// Separator for multi-part content.
    #[cfg(not(feature = "markdown"))]
    pub const SEP: &'static str = "\n\n";
}

impl<T> From<T> for Content
where
    T: Into<Block>,
{
    fn from(block: T) -> Self {
        Self(vec![block.into()])
    }
}

impl<T> FromIterator<T> for Content
where
    T: Into<Block>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(iter.into_iter().map(Into::into).collect())
    }
}

impl<T> Extend<T> for Content
where
    T: Into<Block>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        self.0.extend(iter.into_iter().map(Into::into));
    }
}

// I would love to have a conversion method form IntoIterator<Item = T> but
// that conflicts for str because in the future str might implement IntoIterator
// and Iterator. This is a workaround for now.

// I don't really like this because the generics mean a new function for every
// array size. But in most cases the array size is between 1 and 3 so it's not
// a big deal.
impl<T, const N: usize> From<[T; N]> for Content
where
    T: Into<Block>,
{
    fn from(blocks: [T; N]) -> Self {
        Self(blocks.into_iter().map(|t| t.into()).collect())
    }
}

impl From<&[&str]> for Content {
    fn from(text: &[&str]) -> Self {
        Self(text.iter().map(|t| (*t).into()).collect())
    }
}

impl<T> From<Vec<T>> for Content
where
    T: Into<Block>,
{
    fn from(blocks: Vec<T>) -> Self {
        Self(blocks.into_iter().map(Into::into).collect())
    }
}

/// A [`Content`] [`Block`] of a [`Message`].
#[derive(
    Clone, Debug, Serialize, Deserialize, Hash, derive_more::IsVariant,
)]
#[cfg_attr(not(feature = "markdown"), derive(derive_more::Display))]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
// `BlockKind`: a fieldless mirror of this enum's variants, for the wire-coverage
// gate (`tests::wire_coverage`). `EnumIter` enumerates every variant; adding one
// here forces it through `is_wire` and, if wire-sourced, into a captured fixture.
// Both derive lines are `cfg_attr(test)`-gated, so `strum` stays a dev-dep and
// `BlockKind` exists only under test.
#[cfg_attr(test, derive(strum::EnumDiscriminants))]
#[cfg_attr(
    test,
    strum_discriminants(name(BlockKind), derive(strum::EnumIter, Hash))
)]
pub enum Block {
    /// Text content.
    #[serde(alias = "text_delta")]
    #[cfg_attr(not(feature = "markdown"), display("{text}"))]
    Text {
        /// The actual text content.
        text: crate::CowStr,
        /// Citations referencing source [`Document`]s, populated by the API on
        /// response [`Text`] blocks when a document had citations enabled.
        ///
        /// [`Document`]: Block::Document
        /// [`Text`]: Block::Text
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation>>,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Thinking content. Only the Assistant can send this. Should be submitted
    /// with each request but does not count against the token budget uness the
    /// Assistant refers to a past thought.
    #[serde(rename = "thinking")]
    #[cfg_attr(not(feature = "markdown"), display("{thought}"))]
    Thought {
        /// Complete Thought.
        ///
        /// # Security
        ///
        /// The `langsan` feature is not available for this field. This is
        /// because if we sanitize the thought and it is resubmitted the
        /// signature will not match and the API will reject the request. So it
        /// is up to the caller to handle user facing thought sanitization, the
        /// easiest solution to which is just not to show the thought to the
        /// user. This is only for the developer and the Assistant. If there is
        /// a need to show the thought, a Cow<'static, str> is convertable to a
        /// `langsan::CowStr`.
        #[serde(rename = "thinking")]
        thought: Cow<'static, str>,
        /// Signature. Guarantees thought was not tampered with. It's up to the
        /// caller to not mix up the thought signatures. Anthropic will reject
        /// the request if the signature is invalid.
        #[serde(default)]
        signature: Cow<'static, str>,
    },
    /// Redacted thinking. Sometimes the system will redact the thinking content
    /// for safety reasons. The Assistant can still see the redacted content.
    #[cfg_attr(not(feature = "markdown"), display("[REDACTED]"))]
    #[serde(rename = "redacted_thinking")]
    RedactedThought {
        /// Allows the Assistant to see the redacted thought if it is provided.
        #[serde(rename = "data")]
        signature: Cow<'static, str>,
    },
    /// Image content.
    #[cfg_attr(not(feature = "markdown"), display("{}", image))]
    Image {
        #[serde(rename = "source")]
        /// An base64 encoded image.
        image: Image,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Document content (PDF, plain text, or custom content). Enable
    /// [`citations`] to have the model return [`Citation`]s on its response
    /// [`Text`] blocks.
    ///
    /// [`citations`]: Block::document_with_citations
    /// [`Text`]: Block::Text
    #[cfg_attr(not(feature = "markdown"), display("{}", source))]
    Document {
        /// The document source.
        #[serde(rename = "source")]
        source: DocumentSource,
        /// Optional title (passed to the model, not citable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<Cow<'static, str>>,
        /// Optional context (passed to the model, not citable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<Cow<'static, str>>,
        /// Enable citations for this document.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// [`Tool`] call. This should only be used with the [`Assistant`] role.
    ///
    /// [`Assistant`]: Role::Assistant
    /// [`Tool`]: crate::Tool
    // Default display is to hide this from the user.
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ToolUse {
        /// Tool use input.
        #[serde(flatten)]
        call: tool::Use,
    },
    /// Result of a [`Tool`] call. This should only be used with the [`User`]
    /// role.
    ///
    /// [`User`]: Role::User
    /// [`Tool`]: crate::Tool
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ToolResult {
        /// Tool result
        #[serde(flatten)]
        result: tool::Result,
    },
    /// A server tool invocation the API executed itself (`server_tool_use`),
    /// e.g. a [`web_search`]. Like [`Block::ToolUse`], but Anthropic ran it: its
    /// result block follows in the same assistant turn and you never return a
    /// [`tool::Result`]. The [`id`](tool::Use::id) carries a `srvtoolu_` prefix.
    ///
    /// [`web_search`]: crate::tool::ServerMethodDef::web_search
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ServerToolUse {
        /// The server tool call.
        #[serde(flatten)]
        call: tool::Use,
    },
    /// Result of a [`web_search`] server tool call (`web_search_tool_result`),
    /// appearing in the assistant turn right after its
    /// [`ServerToolUse`](Block::ServerToolUse) block.
    ///
    /// [`web_search`]: crate::tool::ServerMethodDef::web_search
    #[cfg_attr(not(feature = "markdown"), display(""))]
    WebSearchToolResult {
        /// The [`id`](tool::Use::id) of the
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The search results, or an error.
        content: WebSearchToolResultContent,
        /// Who invoked the search — set by the API on responses (e.g.
        /// [`Direct`](tool::KnownCaller::Direct)), omitted on the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
    /// Result of a [`web_fetch`] server tool call (`web_fetch_tool_result`),
    /// appearing in the assistant turn right after its
    /// [`ServerToolUse`](Block::ServerToolUse) block. Carries the fetched
    /// document (text or base64 PDF) and its source URL, or an error.
    ///
    /// [`web_fetch`]: crate::tool::ServerMethodDef::web_fetch
    #[cfg_attr(not(feature = "markdown"), display(""))]
    WebFetchToolResult {
        /// The [`id`](tool::Use::id) of the
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The fetched document, or an error.
        content: WebFetchToolResultContent,
        /// Who invoked the fetch — set by the API on responses (e.g.
        /// [`Direct`](tool::KnownCaller::Direct)), omitted on the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
    /// Result of a [tool-search] server tool call (`tool_search_tool_result`),
    /// appearing in the assistant turn right after its
    /// [`ServerToolUse`](Block::ServerToolUse) block. Carries the
    /// [`tool_reference`](ToolReference)s the API discovered (and expands into
    /// full definitions automatically), or an error.
    ///
    /// [tool-search]: crate::tool::ServerMethodDef::tool_search_regex
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ToolSearchToolResult {
        /// The [`id`](tool::Use::id) of the
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The discovered tool references, or an error.
        content: ToolSearchToolResultContent,
        /// Who invoked the search — set by the API on responses (e.g.
        /// [`Direct`](tool::KnownCaller::Direct)), omitted on the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
    /// A `tool_reference` block naming a [`defer_loading`] tool to expand. Used
    /// to implement [custom client-side tool search]: a custom tool returns
    /// these in its [`tool::Result`] content (a [`User`] turn) and the API
    /// expands each into the matching tool's full definition. (The server-side
    /// tool-search tool instead nests [`ToolReference`]s in a
    /// [`ToolSearchToolResult`](Block::ToolSearchToolResult).)
    ///
    /// [`defer_loading`]: crate::tool::CustomMethodDef::defer_loading
    /// [custom client-side tool search]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool#custom-tool-search-implementation>
    /// [`tool::Result`]: crate::tool::Result
    /// [`User`]: Role::User
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ToolReference {
        /// The [`name`](crate::tool::MethodDef::name) of the tool to expand.
        tool_name: Cow<'static, str>,
    },
    /// Result of a [code execution] server tool call
    /// (`code_execution_tool_result`) — the captured `stdout`/`stderr`/exit
    /// code after the container finished running the model's code, including
    /// any [programmatic tool calls] it made. Appears in the assistant turn
    /// after its `code_execution` [`ServerToolUse`](Block::ServerToolUse) block.
    ///
    /// [code execution]: crate::tool::ServerMethodDef::code_execution
    /// [programmatic tool calls]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    #[cfg_attr(not(feature = "markdown"), display(""))]
    CodeExecutionToolResult {
        /// The [`id`](tool::Use::id) of the `code_execution`
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The execution outcome.
        content: CodeExecutionResult,
        /// Who invoked the execution — set by the API on responses, omitted on
        /// the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
    /// Result of the `bash_code_execution` sub-tool
    /// (`bash_code_execution_tool_result`) — the captured `stdout`/`stderr`/exit
    /// code of a shell command run inside a [code execution] container, or a
    /// tool-level [error](BashCodeExecutionResultContent::Error). Unlocked
    /// alongside [`TextEditorCodeExecutionToolResult`](Self::TextEditorCodeExecutionToolResult)
    /// whenever the [code execution] tool is enabled; appears in the assistant
    /// turn after its `server_tool_use` block.
    ///
    /// [code execution]: crate::tool::ServerMethodDef::code_execution
    #[cfg_attr(not(feature = "markdown"), display(""))]
    BashCodeExecutionToolResult {
        /// The [`id`](tool::Use::id) of the `bash_code_execution`
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The command output, or a tool-level error.
        content: BashCodeExecutionResultContent,
        /// Who invoked the execution — set by the API on responses, omitted on
        /// the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
    /// Result of the `text_editor_code_execution` sub-tool
    /// (`text_editor_code_execution_tool_result`) — a
    /// [view](TextEditorCodeExecutionResultContent::View),
    /// [create](TextEditorCodeExecutionResultContent::Create), or
    /// [str_replace](TextEditorCodeExecutionResultContent::StrReplace) outcome
    /// from a [code execution] container, or a tool-level
    /// [error](TextEditorCodeExecutionResultContent::Error). Unlocked alongside
    /// [`BashCodeExecutionToolResult`](Self::BashCodeExecutionToolResult)
    /// whenever the [code execution] tool is enabled.
    ///
    /// [code execution]: crate::tool::ServerMethodDef::code_execution
    #[cfg_attr(not(feature = "markdown"), display(""))]
    TextEditorCodeExecutionToolResult {
        /// The [`id`](tool::Use::id) of the `text_editor_code_execution`
        /// [`ServerToolUse`](Block::ServerToolUse) this answers.
        tool_use_id: Cow<'static, str>,
        /// The file operation outcome, or a tool-level error.
        content: TextEditorCodeExecutionResultContent,
        /// Who invoked the execution — set by the API on responses, omitted on
        /// the input side.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<tool::Caller>,
    },
}

/// The `content` of a [`Block::CodeExecutionToolResult`]: the captured output
/// of the container run (the `code_execution_result` object). A failure is
/// reported in-band via [`return_code`](Self::return_code) / `stderr` /
/// [`abort_reason`](Self::abort_reason) rather than a separate error shape.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "code_execution_result")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct CodeExecutionResult {
    /// Captured standard output.
    pub stdout: Cow<'static, str>,
    /// Captured standard error (a Python traceback, `TimeoutError`, …).
    pub stderr: Cow<'static, str>,
    /// Process exit code — `0` on success.
    pub return_code: i64,
    /// Rich outputs the run produced (e.g. files). Empty for plain
    /// `stdout`/`stderr` runs; left as raw JSON pending a captured shape.
    #[serde(default)]
    pub content: Vec<serde_json::Value>,
    /// Why the run aborted, if it did. Undocumented but always present on the
    /// wire (explicit `null` when it ran to completion), so it is serialized
    /// even when [`None`] to round-trip the captured bytes.
    #[serde(default)]
    pub abort_reason: Option<Cow<'static, str>>,
}

/// The `content` of a [`Block::BashCodeExecutionToolResult`]: the captured
/// output of a `bash_code_execution` command, or a tool-level error. A command
/// that *ran* but exited non-zero is still a [`Result`](Self::Result) with the
/// failure reported in-band via [`return_code`](Self::Result::return_code) /
/// `stderr`; the [`Error`](Self::Error) variant is for the sub-tool itself
/// failing (e.g. `unavailable`, `output_file_too_large`).
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum BashCodeExecutionResultContent {
    /// The command ran (its own exit code is in [`return_code`](Self::Result::return_code)).
    #[serde(rename = "bash_code_execution_result")]
    Result {
        /// Captured standard output.
        stdout: Cow<'static, str>,
        /// Captured standard error.
        stderr: Cow<'static, str>,
        /// Process exit code — `0` on success.
        return_code: i64,
        /// Files the command wrote to the sandbox's `$OUTPUT_DIR`, each a
        /// [`file_id`](BashCodeExecutionOutput::file_id) to fetch via the Files
        /// API (#32). Empty for plain `stdout`/`stderr` runs — a file only
        /// surfaces here when the command places it in `$OUTPUT_DIR` (writing
        /// to `/tmp` or the cwd does *not* register it).
        #[serde(default)]
        content: Vec<BashCodeExecutionOutput>,
    },
    /// The sub-tool itself failed before (or instead of) running the command.
    #[serde(rename = "bash_code_execution_tool_result_error")]
    Error {
        /// The error code (e.g. `unavailable`, `execution_time_exceeded`,
        /// `output_file_too_large`, `too_many_requests`, `container_expired`).
        error_code: Cow<'static, str>,
        /// A human-readable message. Undocumented but present on the wire when
        /// the sandbox has one to give.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<Cow<'static, str>>,
    },
}

/// A file a `bash_code_execution` command emitted into the sandbox's
/// `$OUTPUT_DIR` (the `bash_code_execution_output` object), referenced by
/// [`file_id`](Self::file_id). Fetch its bytes via the Files API (#32).
///
/// The container only registers a file here when the command writes it under
/// `$OUTPUT_DIR`; files left in `/tmp` or the working directory do not appear.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "bash_code_execution_output")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct BashCodeExecutionOutput {
    /// The Files API id (`file_…`) of the emitted file.
    pub file_id: Cow<'static, str>,
}

/// The `content` of a [`Block::TextEditorCodeExecutionToolResult`]: the outcome
/// of a `text_editor_code_execution` operation. The wire `type` discriminates
/// per command — a [`View`](Self::View), [`Create`](Self::Create), or
/// [`StrReplace`](Self::StrReplace) success, or an [`Error`](Self::Error).
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum TextEditorCodeExecutionResultContent {
    /// A `view` of a file's contents.
    #[serde(rename = "text_editor_code_execution_view_result")]
    View {
        /// The kind of file viewed (e.g. `text`).
        file_type: Cow<'static, str>,
        /// The file contents shown.
        content: Cow<'static, str>,
        /// Number of lines returned.
        num_lines: i64,
        /// The 1-based line the view starts at.
        start_line: i64,
        /// Total number of lines in the file.
        total_lines: i64,
    },
    /// A `create` outcome.
    #[serde(rename = "text_editor_code_execution_create_result")]
    Create {
        /// Whether the file already existed (an overwrite rather than a new
        /// file).
        is_file_update: bool,
    },
    /// A `str_replace` outcome — the unified-diff hunk of the edit. Field names
    /// are the wire's snake_case (`old_start` …), *not* the camelCase the docs
    /// show.
    #[serde(rename = "text_editor_code_execution_str_replace_result")]
    StrReplace {
        /// The 1-based start line of the replaced region.
        old_start: i64,
        /// The number of lines replaced.
        old_lines: i64,
        /// The 1-based start line of the new region.
        new_start: i64,
        /// The number of new lines.
        new_lines: i64,
        /// The diff lines (`-`/`+` prefixed, plus `\ No newline …` markers).
        #[serde(default)]
        lines: Vec<Cow<'static, str>>,
    },
    /// The sub-tool failed (e.g. `file_not_found`, `string_not_found`).
    #[serde(rename = "text_editor_code_execution_tool_result_error")]
    Error {
        /// The error code (e.g. `file_not_found`, `string_not_found`,
        /// `unavailable`).
        error_code: Cow<'static, str>,
        /// A human-readable message. Undocumented but present on the wire when
        /// the sandbox has one to give.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error_message: Option<Cow<'static, str>>,
    },
}

/// The `content` of a [`Block::WebSearchToolResult`]: either the search results
/// or an error.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum WebSearchToolResultContent {
    /// Successful search results.
    Results(Vec<WebSearchResult>),
    /// The search failed.
    Error(WebSearchToolError),
}

/// A single result in a [`Block::WebSearchToolResult`], cited on the model's
/// response [`Text`](Block::Text) blocks via
/// [`Citation::WebSearchResultLocation`].
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "web_search_result")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct WebSearchResult {
    /// The result URL.
    pub url: Cow<'static, str>,
    /// The result title.
    pub title: Cow<'static, str>,
    /// Opaque content the model uses to cite this result. Pass it back verbatim
    /// when echoing the turn (e.g. to continue a [`pause_turn`]).
    ///
    /// [`pause_turn`]: crate::response::StopReason::PauseTurn
    pub encrypted_content: Cow<'static, str>,
    /// Approximate age of the page, e.g. `"3 days ago"`. The API sends this on
    /// every result, explicitly `null` when unknown — so it is always
    /// serialized (not skipped) to round-trip the wire exactly.
    #[serde(default)]
    pub page_age: Option<Cow<'static, str>>,
}

/// An error reported in a [`Block::WebSearchToolResult`].
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "web_search_tool_result_error")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct WebSearchToolError {
    /// The error code, e.g. `"max_uses_exceeded"`, `"too_many_requests"`,
    /// `"query_too_long"`, `"invalid_input"`, or `"unavailable"`.
    pub error_code: Cow<'static, str>,
}

impl WebSearchToolResultContent {}

impl WebSearchResult {}

impl WebSearchToolError {}

/// The `content` of a [`Block::WebFetchToolResult`]: either the fetched
/// document or an error. Tagged on `type` (both arms carry one).
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum WebFetchToolResultContent {
    /// The fetch succeeded.
    #[serde(rename = "web_fetch_result")]
    Result {
        /// The URL that was fetched.
        url: Cow<'static, str>,
        /// The fetched content as a document block.
        content: FetchedDocument,
        /// ISO-8601 timestamp of when the content was retrieved, e.g.
        /// `"2025-08-25T10:30:00Z"`.
        retrieved_at: Cow<'static, str>,
    },
    /// The fetch failed.
    #[serde(rename = "web_fetch_tool_result_error")]
    Error {
        /// The error code, e.g. `"url_not_accessible"`, `"url_not_allowed"`,
        /// `"url_too_long"`, `"too_many_requests"`,
        /// `"unsupported_content_type"`, `"max_uses_exceeded"`,
        /// `"invalid_input"`, or `"unavailable"`.
        error_code: Cow<'static, str>,
    },
}

/// The `document` block nested in a successful [`Block::WebFetchToolResult`].
/// Mirrors a [`Block::Document`] but only carries the fields the API returns
/// for a fetch: the [`source`](FetchedDocument::source) (plain text or a
/// base64 PDF), an optional [`title`](FetchedDocument::title), and whether
/// [`citations`](FetchedDocument::citations) are enabled.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "document")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct FetchedDocument {
    /// The fetched content: [`text/plain`] for web pages, base64
    /// [`application/pdf`] for PDFs.
    ///
    /// [`text/plain`]: DocumentSource::PlainText
    /// [`application/pdf`]: DocumentSource::Base64
    pub source: DocumentSource,
    /// The document title, when the page provided one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<Cow<'static, str>>,
    /// Whether citations were enabled (mirrors the [`WebFetch`] tool's
    /// [`citations`] config).
    ///
    /// [`WebFetch`]: crate::tool::WebFetch
    /// [`citations`]: crate::tool::WebFetch::citations
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citations: Option<CitationsConfig>,
}

impl WebFetchToolResultContent {}

impl FetchedDocument {}

/// The `content` of a [`Block::ToolSearchToolResult`]: either the discovered
/// tool references or an error. Tagged on `type` (unlike the untagged
/// [`WebSearchToolResultContent`], both arms here are objects carrying a
/// `type`).
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum ToolSearchToolResultContent {
    /// The tools the search discovered. The API expands each
    /// [`ToolReference`] into the matching deferred tool's full definition
    /// before the model sees it.
    #[serde(rename = "tool_search_tool_search_result")]
    Results {
        /// The discovered references, in relevance order (3–5 per search).
        tool_references: Vec<ToolReference>,
    },
    /// The search failed.
    #[serde(rename = "tool_search_tool_result_error")]
    Error {
        /// The error code, e.g. `"too_many_requests"`, `"invalid_pattern"`,
        /// `"pattern_too_long"`, or `"unavailable"`.
        error_code: Cow<'static, str>,
    },
}

/// A pointer to a deferred tool discovered by the [tool-search
/// tool](crate::tool::ServerMethodDef::tool_search_regex), naming a tool whose full
/// definition lives in the request's tools array with
/// [`defer_loading`](crate::tool::CustomMethodDef::defer_loading) set. Appears in
/// [`ToolSearchToolResultContent::Results`].
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type", rename = "tool_reference")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct ToolReference {
    /// The [`name`](crate::tool::MethodDef::name) of the discovered tool.
    pub tool_name: Cow<'static, str>,
}

impl ToolSearchToolResultContent {}

impl ToolReference {}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl Block {
    /// Const constructor for text content. Only available without the `langsan`
    /// feature.
    // TODO: rename this to `text` which is more consistent with the other
    // constructors? Or the other way around?
    #[cfg(not(feature = "langsan"))]
    pub const fn const_text(text: &'static str) -> Self {
        Self::Text {
            text: std::borrow::Cow::Borrowed(text),
            citations: None,
            cache_control: None,
        }
    }

    /// Text content.
    pub fn text<T>(text: T) -> Self
    where
        T: Into<crate::CowStr>,
    {
        Self::Text {
            text: text.into(),
            citations: None,
            cache_control: None,
        }
    }

    /// A [`tool_reference`](Block::ToolReference) block naming a
    /// [`defer_loading`] tool. Return these in a custom tool's
    /// [`tool::Result`] content to implement [custom client-side tool search].
    ///
    /// [`defer_loading`]: crate::tool::CustomMethodDef::defer_loading
    /// [`tool::Result`]: crate::tool::Result
    /// [custom client-side tool search]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool#custom-tool-search-implementation>
    pub fn tool_reference<T>(tool_name: T) -> Self
    where
        T: Into<Cow<'static, str>>,
    {
        Self::ToolReference {
            tool_name: tool_name.into(),
        }
    }

    /// [`Document`] content block. Use [`document_with_citations`] to enable
    /// [`Citation`]s.
    ///
    /// [`Document`]: Block::Document
    /// [`document_with_citations`]: Block::document_with_citations
    pub fn document(source: DocumentSource) -> Self {
        Self::Document {
            source,
            title: None,
            context: None,
            citations: None,
            cache_control: None,
        }
    }

    /// [`Document`] content block with citations enabled, so the model returns
    /// [`Citation`]s referencing it on its response [`Text`] blocks.
    ///
    /// [`Document`]: Block::Document
    /// [`Text`]: Block::Text
    pub fn document_with_citations(source: DocumentSource) -> Self {
        Self::Document {
            source,
            title: None,
            context: None,
            citations: Some(CitationsConfig { enabled: true }),
            cache_control: None,
        }
    }

    /// Is a [`Thought`] and also complete (signature is not empty).
    ///
    /// [`Thought`]: Block::Thought
    pub fn is_complete_thought(&self) -> bool {
        matches!(self, Self::Thought { signature, .. } if !signature.is_empty())
    }

    /// Merge [`Delta`]s into a [`Block`]. The types must be compatible or this
    /// will return a [`ContentMismatch`] error. In the case of a
    /// [`ToolUse`](Block::ToolUse) block, the deltas, together, must form a
    /// complete json object.
    pub fn merge_deltas<Ds>(&mut self, deltas: Ds) -> Result<(), DeltaError>
    where
        Ds: IntoIterator<Item = Delta>,
    {
        let mut it = deltas.into_iter();

        // Get the first delta so we can try to fold the rest into it.
        let acc: Delta = match it.next() {
            Some(delta) => delta,
            // Empty iterator, nothing to merge.
            None => return Ok(()),
        };

        // Merge the rest of the deltas into the first one. (there isn't a
        // `try_reduce` method yet)
        let acc: Delta = it.try_fold(acc, |acc, delta| acc.merge(delta))?;

        // Apply the merged delta to the block.
        match (self, acc) {
            (Block::Text { text, .. }, Delta::Text { text: delta }) => {
                #[cfg(not(feature = "langsan"))]
                {
                    text.to_mut().push_str(&delta);
                }
                #[cfg(feature = "langsan")]
                {
                    text.push_str(&delta);
                }
            }
            (
                Block::ToolUse {
                    call: tool::Use { input, .. },
                }
                | Block::ServerToolUse {
                    call: tool::Use { input, .. },
                },
                Delta::Json { partial_json },
            ) => {
                use serde_json::Value::Object;
                // Parse the partial json as an object and merge it into the
                // input.
                let partial_json: serde_json::Value =
                    serde_json::from_str(&partial_json).map_err(|e| {
                        DeltaError::Parse {
                            error: format!(
                        "Could not merge partial json `{}` into `{}` because {}",
                        partial_json, input, e
                    ),
                        }
                    })?;
                if let (Object(new), Object(old)) = (partial_json, input) {
                    old.extend(new);
                }
            }
            (
                Block::Thought { thought, signature },
                Delta::Thought {
                    thinking: delta_thinking,
                    signature: delta_signature,
                },
            ) => {
                if let Some(delta_signature) = delta_signature {
                    cold_path();
                    // This is legal because it's possible to merge a bunch of
                    // deltas outside this function in which case this would be
                    // considered a complete thought. So we re-assign.
                    *thought = delta_thinking;
                    *signature = delta_signature;
                } else {
                    // Normal case, partial thought. Simple append.
                    thought.to_mut().push_str(&delta_thinking);
                }
            }
            (
                Block::RedactedThought { signature },
                Delta::RedactedThought {
                    signature: signature_delta,
                },
            ) => {
                if !signature.is_empty() {
                    // It is not possible to merge signatures because every
                    // signature is already complete. However there is a case
                    // where a block is just created and is empty.
                    return Err(ContentMismatch {
                        from: Delta::RedactedThought {
                            signature: signature_delta,
                        },
                        to: "RedactedThought",
                    }
                    .into());
                }

                // Lhs is empty, so we just assign.
                *signature = signature_delta;
            }
            (
                Block::Thought { signature, .. },
                Delta::Signature {
                    signature: delta_signature,
                },
            ) => {
                if !signature.is_empty() {
                    // It is not possible to merge signatures because every
                    // signature is already complete. However there is a case
                    // where a block is just created and is empty.
                    return Err(ContentMismatch {
                        from: Delta::Signature {
                            signature: delta_signature,
                        },
                        to: "Thought",
                    }
                    .into());
                }

                // Lhs is empty, so we just assign. Thought is now complete.
                *signature = delta_signature;
            }
            // A citations delta appends a single citation to a text block.
            (
                Block::Text { citations, .. },
                Delta::CitationsDelta { citation },
            ) => {
                citations.get_or_insert_with(Vec::new).push(citation);
            }
            (this, acc) => {
                let variant_name = match this {
                    Block::Text { .. } => stringify!(Block::Text),
                    Block::Thought { .. } => stringify!(Block::Thinking),
                    Block::RedactedThought { .. } => {
                        stringify!(Block::RedactedThinking)
                    }
                    Block::ToolUse { .. } => stringify!(Block::ToolUse),
                    Block::ToolResult { .. } => stringify!(Block::ToolResult),
                    Block::ServerToolUse { .. } => {
                        stringify!(Block::ServerToolUse)
                    }
                    Block::WebSearchToolResult { .. } => {
                        stringify!(Block::WebSearchToolResult)
                    }
                    Block::WebFetchToolResult { .. } => {
                        stringify!(Block::WebFetchToolResult)
                    }
                    Block::ToolSearchToolResult { .. } => {
                        stringify!(Block::ToolSearchToolResult)
                    }
                    Block::CodeExecutionToolResult { .. } => {
                        stringify!(Block::CodeExecutionToolResult)
                    }
                    Block::BashCodeExecutionToolResult { .. } => {
                        stringify!(Block::BashCodeExecutionToolResult)
                    }
                    Block::TextEditorCodeExecutionToolResult { .. } => {
                        stringify!(Block::TextEditorCodeExecutionToolResult)
                    }
                    Block::ToolReference { .. } => {
                        stringify!(Block::ToolReference)
                    }
                    Block::Image { .. } => stringify!(Block::Image),
                    Block::Document { .. } => stringify!(Block::Document),
                };

                return Err(ContentMismatch {
                    from: acc,
                    to: variant_name,
                }
                .into());
            }
        }

        Ok(())
    }

    /// Create a cache breakpoint at this block. See [`Prompt::cache`] for more
    /// information. Returns true if the block was cached. This alwasy succeeds,
    /// however some blocks are automatically cached and will return false.
    ///
    /// Uses the default 5-minute ephemeral TTL. For a 1-hour TTL, use
    /// [`cache_1h`](Block::cache_1h).
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    pub fn cache(&mut self) -> bool {
        self.cache_with(CacheControl::ephemeral())
    }

    /// Create a 1-hour cache breakpoint at this block.
    ///
    /// Behaves identically to [`cache`](Block::cache) but uses
    /// [`CacheControl::one_hour`]. See [`Prompt::cache_1h`] for usage.
    ///
    /// [`Prompt::cache_1h`]: crate::Prompt::cache_1h
    pub fn cache_1h(&mut self) -> bool {
        self.cache_with(CacheControl::one_hour())
    }

    /// Create a cache breakpoint at this block with a caller-provided
    /// [`CacheControl`]. Returns true if the block was cached; returns
    /// false for thought blocks (which are automatically cached).
    pub fn cache_with(&mut self, cache_control_value: CacheControl) -> bool {
        use crate::tool;

        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Document { cache_control, .. }
            | Self::ToolUse {
                call: tool::Use { cache_control, .. },
            }
            | Self::ToolResult {
                result: tool::Result { cache_control, .. },
            }
            | Self::ServerToolUse {
                call: tool::Use { cache_control, .. },
            } => {
                *cache_control = Some(cache_control_value);

                true
            }
            // These are automatically cached or carry no cache_control.
            // https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#using-extended-thinking-with-prompt-caching
            Self::Thought { .. }
            | Self::RedactedThought { .. }
            | Self::WebSearchToolResult { .. }
            | Self::WebFetchToolResult { .. }
            | Self::ToolSearchToolResult { .. }
            | Self::CodeExecutionToolResult { .. }
            | Self::BashCodeExecutionToolResult { .. }
            | Self::TextEditorCodeExecutionToolResult { .. }
            | Self::ToolReference { .. } => false,
        }
    }

    /// Remove the cache breakpoint from this block. Returns `true` if a
    /// breakpoint was removed.
    pub fn uncache(&mut self) -> bool {
        use crate::tool;

        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Document { cache_control, .. }
            | Self::ToolUse {
                call: tool::Use { cache_control, .. },
            }
            | Self::ToolResult {
                result: tool::Result { cache_control, .. },
            }
            | Self::ServerToolUse {
                call: tool::Use { cache_control, .. },
            } => {
                let was_cached = cache_control.is_some();
                *cache_control = None;
                was_cached
            }
            Self::Thought { .. }
            | Self::RedactedThought { .. }
            | Self::WebSearchToolResult { .. }
            | Self::WebFetchToolResult { .. }
            | Self::ToolSearchToolResult { .. }
            | Self::CodeExecutionToolResult { .. }
            | Self::BashCodeExecutionToolResult { .. }
            | Self::TextEditorCodeExecutionToolResult { .. }
            | Self::ToolReference { .. } => false,
        }
    }

    /// Returns true if the block has a `cache_control` breakpoint.
    pub const fn is_cached(&self) -> bool {
        use crate::tool;

        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::Document { cache_control, .. }
            | Self::ToolUse {
                call: tool::Use { cache_control, .. },
            }
            | Self::ToolResult {
                result: tool::Result { cache_control, .. },
            }
            | Self::ServerToolUse {
                call: tool::Use { cache_control, .. },
            } => cache_control.is_some(),
            Self::Thought { .. }
            | Self::RedactedThought { .. }
            | Self::WebSearchToolResult { .. }
            | Self::WebFetchToolResult { .. }
            | Self::ToolSearchToolResult { .. }
            | Self::CodeExecutionToolResult { .. }
            | Self::BashCodeExecutionToolResult { .. }
            | Self::TextEditorCodeExecutionToolResult { .. }
            | Self::ToolReference { .. } => false,
        }
    }

    /// Returns the [`tool::Use`] if this is a [`Block::ToolUse`]. See also
    /// [`response::Message::tool_use`].
    pub fn tool_use(&self) -> Option<&crate::tool::Use> {
        match self {
            Self::ToolUse { call, .. } => Some(call),
            _ => None,
        }
    }

    /// Returns the number of bytes in the block. Does not include tool use or
    /// other metadata. Does include the base64 encoded image data length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len(),
            Self::Thought {
                thought: thinking, ..
            } => thinking.len(),
            Self::Image { image, .. } => image.len(),
            Self::Document { source, .. } => source.len(),
            Self::RedactedThought { .. }
            | Self::ToolUse { .. }
            | Self::ToolResult { .. }
            | Self::ServerToolUse { .. }
            | Self::WebSearchToolResult { .. }
            | Self::WebFetchToolResult { .. }
            | Self::ToolSearchToolResult { .. }
            | Self::CodeExecutionToolResult { .. }
            | Self::BashCodeExecutionToolResult { .. }
            | Self::TextEditorCodeExecutionToolResult { .. }
            | Self::ToolReference { .. } => 0,
        }
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Block {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        let it: Box<dyn Iterator<Item = Event<'_>> + '_> = match self {
            Self::Text { text, .. } => {
                // We'll parse the inner text as markdown.
                Box::new(pulldown_cmark::Parser::new_ext(text, options.inner))
            }
            Block::Image { image, .. } => {
                // We use Event::Text for images because they are rendered as
                // markdown images with embedded base64 data.
                Box::new([Event::Text(image.to_string().into())].into_iter())
            }
            Block::ToolUse { .. } | Block::ServerToolUse { .. } => {
                if options.tool_use {
                    Box::new(
                        [
                            Event::Start(Tag::CodeBlock(
                                CodeBlockKind::Fenced("json".into()),
                            )),
                            Event::Text(
                                serde_json::to_string(self).unwrap().into(),
                            ),
                            Event::End(TagEnd::CodeBlock),
                        ]
                        .into_iter(),
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Block::Thought {
                thought: thinking, ..
            } => Box::new(Box::new(pulldown_cmark::Parser::new_ext(
                thinking,
                options.inner,
            ))),
            // Anthropic says to be transparent with the user but I think this
            // is naive. Users do not need to know if a thought was redacted.
            // A `tool_reference` is tool-search plumbing, not user content.
            Block::RedactedThought { .. } | Block::ToolReference { .. } => {
                Box::new(std::iter::empty())
            }
            Block::ToolResult { .. }
            | Block::WebSearchToolResult { .. }
            | Block::WebFetchToolResult { .. }
            | Block::ToolSearchToolResult { .. }
            | Block::CodeExecutionToolResult { .. }
            | Block::BashCodeExecutionToolResult { .. }
            | Block::TextEditorCodeExecutionToolResult { .. } => {
                if options.tool_results {
                    Box::new(
                        [
                            Event::Start(Tag::CodeBlock(
                                CodeBlockKind::Fenced("json".into()),
                            )),
                            Event::Text(
                                serde_json::to_string(self).unwrap().into(),
                            ),
                            Event::End(TagEnd::CodeBlock),
                        ]
                        .into_iter(),
                    )
                } else {
                    Box::new(std::iter::empty())
                }
            }
            Block::Document { source, .. } => {
                Box::new([Event::Text(source.to_string().into())].into_iter())
            }
        };

        it
    }
}

impl From<&str> for Block {
    fn from(text: &str) -> Self {
        Self::text(text.to_owned())
    }
}

impl From<String> for Block {
    fn from(text: String) -> Self {
        Self::Text {
            text: text.into(),
            citations: None,
            cache_control: None,
        }
    }
}

impl From<crate::CowStr> for Block {
    fn from(text: crate::CowStr) -> Self {
        Self::Text {
            text,
            citations: None,
            cache_control: None,
        }
    }
}

impl From<Image> for Block {
    fn from(image: Image) -> Self {
        Self::Image {
            image,
            cache_control: None,
        }
    }
}

impl From<DocumentSource> for Block {
    fn from(source: DocumentSource) -> Self {
        Self::document(source)
    }
}

impl From<tool::Use> for Block {
    fn from(call: tool::Use) -> Self {
        Self::ToolUse { call }
    }
}

impl From<tool::Result> for Block {
    fn from(result: tool::Result) -> Self {
        Self::ToolResult { result }
    }
}

#[cfg(feature = "png")]
impl From<image::RgbaImage> for Block {
    fn from(image: image::RgbaImage) -> Self {
        #[allow(unused_variables)] // for the `e` variable
        Image::encode(MediaType::Png, image)
            // Unwrap can never panic unless the PNG encoding fails, which
            // should really never happen, but no matter what we don't panic.
            .unwrap_or_else(|e| {
                #[cfg(feature = "log")]
                log::error!("Error encoding image: {}", e);
                Image::from_parts(MediaType::Png, String::new().into())
            })
            .into()
    }
}

#[cfg(feature = "png")]
impl From<image::DynamicImage> for Block {
    fn from(image: image::DynamicImage) -> Self {
        image.to_rgba8().into()
    }
}

/// Time-to-live for prompt cache entries.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum CacheTtl {
    /// Cache for 5 minutes — the default. Equivalent to omitting `ttl`; this
    /// variant exists so an explicit `"5m"` round-trips.
    #[serde(rename = "5m")]
    FiveMinutes,
    /// Cache for 1 hour. Costs 2x base input token price.
    #[serde(rename = "1h")]
    OneHour,
}

impl std::fmt::Display for CacheTtl {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CacheTtl::FiveMinutes => write!(f, "5m"),
            CacheTtl::OneHour => write!(f, "1h"),
        }
    }
}

/// Cache control for prompt caching.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum CacheControl {
    /// Ephemeral cache. Default TTL is 5 minutes; set `ttl` for longer
    /// durations.
    Ephemeral {
        /// Optional TTL override. When `None`, the default 5-minute cache
        /// is used. Set to [`CacheTtl::OneHour`] for a 1-hour cache at 2x
        /// base input token price.
        #[serde(skip_serializing_if = "Option::is_none")]
        ttl: Option<CacheTtl>,
    },
}

impl Default for CacheControl {
    fn default() -> Self {
        CacheControl::Ephemeral { ttl: None }
    }
}

impl CacheControl {
    /// Create an ephemeral cache control with the default 5-minute TTL.
    pub fn ephemeral() -> Self {
        CacheControl::Ephemeral { ttl: None }
    }

    /// Create an ephemeral cache control with a 1-hour TTL.
    ///
    /// This costs 2x base input token price compared to the default
    /// 5-minute cache.
    pub fn one_hour() -> Self {
        CacheControl::Ephemeral {
            ttl: Some(CacheTtl::OneHour),
        }
    }
}

/// Configuration to enable citations on a [`Document`] block.
///
/// [`Document`]: Block::Document
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct CitationsConfig {
    /// Whether citations are enabled for this document.
    pub enabled: bool,
}

/// Media type for PDF [`Document`]s.
///
/// [`Document`]: Block::Document
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum DocumentMediaType {
    /// `application/pdf`
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl std::fmt::Display for DocumentMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Pdf => write!(f, "application/pdf"),
        }
    }
}

/// Media type for plain text [`Document`]s.
///
/// [`Document`]: Block::Document
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum PlainTextMediaType {
    /// `text/plain`
    #[serde(rename = "text/plain")]
    Plain,
}

impl std::fmt::Display for PlainTextMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Plain => write!(f, "text/plain"),
        }
    }
}

/// A text chunk for a custom-content [`DocumentSource`].
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type", rename = "text")]
pub struct ContentText {
    /// The text content of this chunk.
    pub text: Cow<'static, str>,
}

impl ContentText {}

/// Source of a [`Document`] content block. Analogous to [`Image`] for image
/// content.
///
/// [`Document`]: Block::Document
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DocumentSource {
    /// Base64-encoded document (PDF).
    Base64 {
        /// Document encoding format.
        media_type: DocumentMediaType,
        /// Base64-encoded document data.
        data: Cow<'static, str>,
    },
    /// URL to a hosted document.
    Url {
        /// The URL.
        url: Cow<'static, str>,
    },
    /// Plain text document (auto-chunked into sentences for citations).
    #[serde(rename = "text")]
    PlainText {
        /// Always `text/plain`.
        media_type: PlainTextMediaType,
        /// The plain text content.
        data: Cow<'static, str>,
    },
    /// Custom content blocks (the caller controls citation granularity).
    Content {
        /// The content blocks.
        content: Vec<ContentText>,
    },
    /// Reference to a file uploaded via the Files API.
    File {
        /// The file ID.
        file_id: Cow<'static, str>,
    },
}

impl DocumentSource {
    /// Create a base64-encoded PDF document source.
    pub fn from_base64(data: impl Into<Cow<'static, str>>) -> Self {
        Self::Base64 {
            media_type: DocumentMediaType::Pdf,
            data: data.into(),
        }
    }

    /// Create a URL document source.
    pub fn from_url(url: impl Into<Cow<'static, str>>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Create a plain text document source.
    pub fn from_text(data: impl Into<Cow<'static, str>>) -> Self {
        Self::PlainText {
            media_type: PlainTextMediaType::Plain,
            data: data.into(),
        }
    }

    /// Create a custom content document source from text chunks.
    pub fn from_content(blocks: Vec<ContentText>) -> Self {
        Self::Content { content: blocks }
    }

    /// Create a Files API reference document source.
    pub fn from_file_id(id: impl Into<Cow<'static, str>>) -> Self {
        Self::File { file_id: id.into() }
    }

    /// Read a file, base64-encode it, and create a PDF document source.
    pub fn from_file(
        path: impl AsRef<std::path::Path>,
    ) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        let encoded = general_purpose::STANDARD.encode(&data);
        Ok(Self::Base64 {
            media_type: DocumentMediaType::Pdf,
            data: Cow::Owned(encoded),
        })
    }

    /// Returns the byte length of the source data.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Base64 { data, .. } | Self::PlainText { data, .. } => {
                data.len()
            }
            Self::Url { url } => url.len(),
            Self::Content { content } => {
                content.iter().map(|c| c.text.len()).sum()
            }
            Self::File { file_id } => file_id.len(),
        }
    }
}

impl std::fmt::Display for DocumentSource {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Base64 { media_type, .. } => {
                write!(f, "[Document ({media_type})]")
            }
            Self::Url { url } => {
                write!(f, "[Document ({url})]")
            }
            Self::PlainText { .. } => {
                write!(f, "[Document (text/plain)]")
            }
            Self::Content { content } => {
                write!(f, "[Document ({} blocks)]", content.len())
            }
            Self::File { file_id } => {
                write!(f, "[Document (file:{file_id})]")
            }
        }
    }
}

/// Image content [`Block`] of a [`Message`].
#[derive(Clone, Debug, Serialize, Deserialize, derive_more::Display, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum Image {
    /// Base64 encoded image data. When displayed, it will be rendered as a
    /// markdown image with embedded data.
    #[display("![Image](data:{media_type};base64,{data})")]
    Base64 {
        /// Image encoding format.
        media_type: MediaType,
        /// Base64 encoded compressed image data.
        data: Cow<'static, str>,
    },
    /// URL to a hosted image. Anthropic fetches it server-side; the crate never
    /// downloads it. When displayed, rendered as a markdown image to the URL.
    #[display("![Image]({url})")]
    Url {
        /// The image URL.
        url: Cow<'static, str>,
    },
}

impl Image {
    /// From raw parts. The data is expected to be base64 encoded compressed
    /// image data or the API will reject it.
    pub fn from_parts(media_type: MediaType, data: Cow<'static, str>) -> Self {
        Self::Base64 { media_type, data }
    }

    /// From a URL to a hosted image. Anthropic fetches it server-side; the
    /// crate never downloads it.
    pub fn from_url(url: impl Into<Cow<'static, str>>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Encode from compressed image data (not base64 encoded). This cannot fail
    /// but if the data is invalid, the API will reject it.
    pub fn from_compressed<D>(format: MediaType, data: D) -> Self
    where
        D: AsRef<[u8]>,
    {
        let data: &[u8] = data.as_ref();
        let encoder = general_purpose::STANDARD;

        Self::Base64 {
            media_type: format,
            data: encoder.encode(data).into(),
        }
    }

    /// Encode an [`Image`] from any type that can be converted into an
    /// [`image::RgbaImage`].
    #[cfg(feature = "image")]
    pub fn encode<I>(
        format: MediaType,
        image: I,
    ) -> Result<Self, image::ImageError>
    where
        I: Into<image::RgbaImage>,
    {
        let mut cursor = std::io::Cursor::new(Vec::new());
        let rgba: image::RgbaImage = image.into();
        rgba.write_to(&mut cursor, format.into())?;
        Ok(Self::from_compressed(format, cursor.into_inner()))
    }

    /// Decode the image data into an [`image::RgbaImage`].
    ///
    /// # Note:
    /// - There is also a [`TryInto`] implementation for this.
    #[cfg(feature = "image")]
    pub fn decode(&self) -> Result<image::RgbaImage, ImageDecodeError> {
        match self {
            Self::Base64 { data, .. } => {
                let data = general_purpose::STANDARD.decode(data.as_bytes())?;
                Ok(image::load_from_memory(&data)?.to_rgba8())
            }
            Self::Url { .. } => Err(ImageDecodeError::Url),
        }
    }

    /// Returns the number of bytes in the image data (base64 encoded). Call
    /// [`decode`](Self::decode) to get the actual image size.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Base64 { data, .. } => data.len(),
            Self::Url { url } => url.len(),
        }
    }
}

/// Errors that can occur when decoding an [`Image`].
#[cfg(feature = "image")]
#[derive(Debug, thiserror::Error)]
pub enum ImageDecodeError {
    /// Invalid base64 encoding.
    #[error("Base64 decode error: {0}")]
    Base64(#[from] base64::DecodeError),
    /// Invalid image data.
    #[error("Image decode error: {0}")]
    Image(#[from] image::ImageError),
    /// The source is a URL, not embedded data — fetch the URL yourself first.
    #[error("cannot decode a URL image source; fetch the URL first")]
    Url,
}

#[cfg(feature = "image")]
impl TryInto<image::RgbaImage> for Image {
    type Error = ImageDecodeError;

    /// An [`Image`] can be decoded into an [`image::RgbaImage`] if it is valid
    /// base64 encoded compressed image data and the image format is supported.
    fn try_into(self) -> Result<image::RgbaImage, Self::Error> {
        self.decode()
    }
}

/// Encoding format for [`Image`]s.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum MediaType {
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[serde(rename = "image/png")]
    Png,
    #[serde(rename = "image/gif")]
    Gif,
    #[serde(rename = "image/webp")]
    Webp,
}

impl MediaType {
    /// Supported [`MediaType`]s.
    pub const SUPPORTED: &'static [Self] =
        &[Self::Jpeg, Self::Png, Self::Gif, Self::Webp];

    /// Extensions supported by the [`MediaType`].
    pub const fn exts(&self) -> &'static [&'static str] {
        match self {
            Self::Jpeg => &["jpeg", "jpg"],
            Self::Png => &["png"],
            Self::Gif => &["gif"],
            Self::Webp => &["webp"],
        }
    }

    /// Returns true if `filename` has a supported extension.
    pub fn is_supported(filename: &str) -> bool {
        Self::detect(filename).is_some()
    }

    /// Detects the [`MediaType`] from the `filename` extension.
    pub fn detect(filename: &str) -> Option<Self> {
        for mt in Self::SUPPORTED {
            if mt.exts().iter().any(|ext| filename.ends_with(ext)) {
                return Some(*mt);
            }
        }

        None
    }
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // Use serde to get the string representation.
        write!(
            f,
            "{}",
            serde_json::to_string(self).unwrap().trim_matches('"')
        )
    }
}

#[cfg(feature = "image")]
impl From<MediaType> for image::ImageFormat {
    /// A [`MediaType`] can always be converted into an [`image::ImageFormat`].
    fn from(value: MediaType) -> image::ImageFormat {
        match value {
            MediaType::Jpeg => image::ImageFormat::Jpeg,
            MediaType::Png => image::ImageFormat::Png,
            MediaType::Gif => image::ImageFormat::Gif,
            MediaType::Webp => image::ImageFormat::WebP,
        }
    }
}

/// An [`ImageFormat`] is unsupported. See [`MediaType`] for supported formats.
///
/// [`ImageFormat`]: image::ImageFormat
#[cfg(feature = "image")]
#[derive(Debug, thiserror::Error)]
#[error("Unsupported image format: {0:?}")]
pub struct UnsupportedImageFormat(image::ImageFormat);

#[cfg(feature = "image")]
impl TryFrom<image::ImageFormat> for MediaType {
    type Error = UnsupportedImageFormat;

    /// An [`image::ImageFormat`] can only be converted into a [`MediaType`] if
    /// the feature for the format is enabled. Otherwise, it will return an
    /// [`UnsupportedImageFormat`] error.
    fn try_from(value: image::ImageFormat) -> Result<Self, Self::Error> {
        match value {
            image::ImageFormat::Jpeg => Ok(Self::Jpeg),
            image::ImageFormat::Png => Ok(Self::Png),
            image::ImageFormat::Gif => Ok(Self::Gif),
            image::ImageFormat::WebP => Ok(Self::Webp),
            _ => Err(UnsupportedImageFormat(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "markdown")]
    use crate::markdown::ToMarkdown;

    use super::*;

    // The server-tool block round-trips replay captured wire fixtures from
    // `test/data/server_tools/` through `utils::roundtrip`, which asserts an
    // exact serialize-back. See `test/data/README.md` for the capture
    // discipline and per-fixture provenance.

    #[test]
    fn server_tool_use_block_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/server_tool_use.json"
        ));
        assert!(block.is_server_tool_use());
        let Block::ServerToolUse { call } = &block else {
            panic!("expected ServerToolUse");
        };
        assert_eq!(call.id, "srvtoolu_01XAxdGfRL2vypN6SF17MJXT");
        assert_eq!(call.name, "web_search");
    }

    #[test]
    fn tool_use_block_with_caller_roundtrip() {
        // A programmatic-tool-calling `tool_use`: a code-execution container
        // called `query_sales` on the model's behalf, so the block carries a
        // `caller` of `code_execution_20260120` with the `srvtoolu_` id of the
        // code-execution call. Captured live on Sonnet 4.6 (PTC is not
        // available on Haiku); see `test/data/README.md`.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/ptc_tool_use.json"
        ));
        let Block::ToolUse { call } = &block else {
            panic!("expected ToolUse");
        };
        assert_eq!(call.id, "toolu_01Ep3muNAqgo6WcHSNzL7cYK");
        assert_eq!(call.name, "query_sales");
        assert_eq!(
            call.caller,
            Some(crate::tool::Caller::code_execution_20260120(
                "srvtoolu_01EnSeFfRxcsNTUgLjYHD5XG"
            ))
        );
    }

    #[test]
    fn memory_tool_use_block_roundtrip() {
        // The `memory` tool is *client-executed*: it comes back as an ordinary
        // `tool_use` (not `server_tool_use`) carrying a `direct` caller — both
        // verified here against bytes captured live on Haiku 4.5. See
        // `test/data/README.md`.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/memory_tool_use.json"
        ));
        let Block::ToolUse { call } = &block else {
            panic!(
                "expected ToolUse (memory is client-executed, not a server tool)"
            );
        };
        assert_eq!(call.name, "memory");
        assert_eq!(call.caller, Some(crate::tool::Caller::direct()));
        assert_eq!(call.input["command"], "view");
    }

    #[test]
    fn web_search_tool_result_block_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/web_search_result.json"
        ));
        let Block::WebSearchToolResult {
            tool_use_id,
            content,
            caller,
        } = &block
        else {
            panic!("expected WebSearchToolResult");
        };
        assert_eq!(tool_use_id, "srvtoolu_01XAxdGfRL2vypN6SF17MJXT");
        assert!(matches!(
            content,
            WebSearchToolResultContent::Results(r) if r.len() == 2
        ));
        // The API reports who called the tool; this one was a direct call.
        assert_eq!(
            caller.as_ref(),
            Some(&crate::tool::Caller::Known(
                crate::tool::KnownCaller::Direct
            ))
        );
    }

    #[test]
    fn web_search_tool_result_error_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/web_search_error.json"
        ));
        let Block::WebSearchToolResult { content, .. } = &block else {
            panic!("expected WebSearchToolResult");
        };
        assert!(matches!(
            content,
            WebSearchToolResultContent::Error(e)
                if e.error_code == "max_uses_exceeded"
        ));
    }

    #[test]
    fn web_fetch_tool_result_block_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/web_fetch_result.json"
        ));
        let Block::WebFetchToolResult {
            tool_use_id,
            content,
            caller,
        } = &block
        else {
            panic!("expected WebFetchToolResult");
        };
        assert_eq!(tool_use_id, "srvtoolu_018ABFMzLzbdKgAKd3MFBtiN");
        let WebFetchToolResultContent::Result {
            url,
            content,
            retrieved_at,
        } = content
        else {
            panic!("expected a successful fetch");
        };
        assert_eq!(url, "https://www.rust-lang.org");
        assert_eq!(retrieved_at, "2026-06-04T15:49:37.688665");
        assert_eq!(content.title.as_deref(), Some("Rust Programming Language"));
        assert!(matches!(content.source, DocumentSource::PlainText { .. }));
        // A plain fetch (no citations requested) omits the citations config.
        assert!(content.citations.is_none());
        assert_eq!(
            caller.as_ref(),
            Some(&crate::tool::Caller::Known(
                crate::tool::KnownCaller::Direct
            ))
        );
    }

    #[test]
    fn web_fetch_tool_result_pdf_roundtrip() {
        // PDFs come back as base64 application/pdf with no title.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/web_fetch_pdf.json"
        ));
        let Block::WebFetchToolResult { content, .. } = &block else {
            panic!("expected WebFetchToolResult");
        };
        let WebFetchToolResultContent::Result { content, .. } = content else {
            panic!("expected a successful fetch");
        };
        assert!(content.title.is_none());
        assert!(matches!(
            content.source,
            DocumentSource::Base64 {
                media_type: DocumentMediaType::Pdf,
                ..
            }
        ));
    }

    #[test]
    fn web_fetch_tool_result_error_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/web_fetch_error.json"
        ));
        let Block::WebFetchToolResult { content, .. } = &block else {
            panic!("expected WebFetchToolResult");
        };
        assert!(matches!(
            content,
            WebFetchToolResultContent::Error { error_code }
                if error_code == "url_not_accessible"
        ));
    }

    #[test]
    fn tool_search_tool_result_block_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/tool_search_result.json"
        ));
        let Block::ToolSearchToolResult {
            tool_use_id,
            content,
            ..
        } = &block
        else {
            panic!("expected ToolSearchToolResult");
        };
        assert_eq!(tool_use_id, "srvtoolu_01ABC123");
        let ToolSearchToolResultContent::Results { tool_references } = content
        else {
            panic!("expected Results");
        };
        assert_eq!(tool_references[0].tool_name, "get_weather");
    }

    #[test]
    fn tool_search_tool_result_error_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/tool_search_error.json"
        ));
        let Block::ToolSearchToolResult { content, .. } = &block else {
            panic!("expected ToolSearchToolResult");
        };
        assert!(matches!(
            content,
            ToolSearchToolResultContent::Error { error_code }
                if error_code == "invalid_pattern"
        ));
    }

    #[test]
    fn code_execution_tool_result_block_roundtrip() {
        // The completion of a programmatic-tool-calling run: the container's
        // captured stdout/exit code. Note the undocumented `abort_reason`
        // (explicit `null` on the wire), which the round-trip would catch if
        // dropped. Captured live on Sonnet 4.6; see `test/data/README.md`.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/code_execution_result.json"
        ));
        let Block::CodeExecutionToolResult {
            tool_use_id,
            content,
            ..
        } = &block
        else {
            panic!("expected CodeExecutionToolResult");
        };
        assert_eq!(tool_use_id, "srvtoolu_01EnSeFfRxcsNTUgLjYHD5XG");
        assert_eq!(content.return_code, 0);
        assert!(content.abort_reason.is_none());
        assert!(content.stdout.contains("Highest: West"));
    }

    #[test]
    fn bash_code_execution_result_block_roundtrip() {
        // A `bash_code_execution` command's captured output. Live, Haiku 4.5.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/bash_code_execution_result.json"
        ));
        let Block::BashCodeExecutionToolResult { content, .. } = &block else {
            panic!("expected BashCodeExecutionToolResult");
        };
        let BashCodeExecutionResultContent::Result {
            stdout,
            return_code,
            ..
        } = content
        else {
            panic!("expected a ran command, not a tool error");
        };
        assert_eq!(*return_code, 0);
        assert!(stdout.contains("hello"));
    }

    #[test]
    fn bash_code_execution_file_output_roundtrip() {
        // A command that wrote a file to `$OUTPUT_DIR`: the result `content`
        // carries a `bash_code_execution_output` block with the Files API
        // `file_id`. Live, Sonnet 4.6 (captured via the `$OUTPUT_DIR` route;
        // files in `/tmp`/cwd never surface). See #32.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/bash_code_execution_file_output.json"
        ));
        let Block::BashCodeExecutionToolResult { content, .. } = &block else {
            panic!("expected BashCodeExecutionToolResult");
        };
        let BashCodeExecutionResultContent::Result { content: files, .. } =
            content
        else {
            panic!("expected a ran command");
        };
        assert_eq!(files.len(), 1);
        assert!(files[0].file_id.starts_with("file_"));
    }

    #[test]
    fn text_editor_code_execution_create_result_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/text_editor_code_execution_create_result.json"
        ));
        let Block::TextEditorCodeExecutionToolResult { content, .. } = &block
        else {
            panic!("expected TextEditorCodeExecutionToolResult");
        };
        assert!(matches!(
            content,
            TextEditorCodeExecutionResultContent::Create {
                is_file_update: false
            }
        ));
    }

    #[test]
    fn text_editor_code_execution_view_result_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/text_editor_code_execution_view_result.json"
        ));
        let Block::TextEditorCodeExecutionToolResult { content, .. } = &block
        else {
            panic!("expected TextEditorCodeExecutionToolResult");
        };
        // `num_lines`/`start_line`/`total_lines` are the wire's snake_case, not
        // the docs' camelCase — the round-trip catches that drift.
        let TextEditorCodeExecutionResultContent::View {
            file_type,
            total_lines,
            ..
        } = content
        else {
            panic!("expected a view result");
        };
        assert_eq!(file_type, "text");
        assert_eq!(*total_lines, 3);
    }

    #[test]
    fn text_editor_code_execution_str_replace_result_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/text_editor_code_execution_str_replace_result.json"
        ));
        let Block::TextEditorCodeExecutionToolResult { content, .. } = &block
        else {
            panic!("expected TextEditorCodeExecutionToolResult");
        };
        let TextEditorCodeExecutionResultContent::StrReplace {
            old_start,
            lines,
            ..
        } = content
        else {
            panic!("expected a str_replace result");
        };
        assert_eq!(*old_start, 1);
        assert!(lines.iter().any(|l| l.contains("debug")));
    }

    #[test]
    fn text_editor_code_execution_error_roundtrip() {
        // The error shape (`{error_code, error_message}`) — `error_message` is
        // undocumented (the docs show `error_code` alone) so the round-trip
        // would drop it if unmodeled. Live, Haiku 4.5.
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/text_editor_code_execution_error.json"
        ));
        let Block::TextEditorCodeExecutionToolResult { content, .. } = &block
        else {
            panic!("expected TextEditorCodeExecutionToolResult");
        };
        let TextEditorCodeExecutionResultContent::Error {
            error_code,
            error_message,
        } = content
        else {
            panic!("expected a tool error");
        };
        assert_eq!(error_code, "unavailable");
        assert!(error_message.is_some());
    }

    #[test]
    fn tool_reference_block_roundtrip() {
        let block: Block = crate::utils::roundtrip(include_str!(
            "../../test/data/server_tools/tool_reference.json"
        ));
        let Block::ToolReference { tool_name } = &block else {
            panic!("expected ToolReference");
        };
        assert_eq!(tool_name, "get_weather");
        // The constructor produces the same block.
        assert_eq!(Block::tool_reference("get_weather"), block);
    }

    #[test]
    fn tool_references_nest_in_a_tool_result() {
        // The custom client-side tool-search shape: a `tool_result` whose
        // content is an array of `tool_reference` blocks.
        let json = serde_json::json!({
            "type": "tool_result",
            "tool_use_id": "toolu_your_tool_id",
            "content": [
                { "type": "tool_reference", "tool_name": "get_weather" },
                { "type": "tool_reference", "tool_name": "search_files" },
            ],
            "is_error": false,
        });
        let block: Block = serde_json::from_value(json.clone()).unwrap();
        let Block::ToolResult { result } = &block else {
            panic!("expected ToolResult");
        };
        let refs: Vec<_> = result
            .content
            .iter()
            .map(|b| match b {
                Block::ToolReference { tool_name } => tool_name.as_ref(),
                _ => panic!("expected ToolReference"),
            })
            .collect();
        assert_eq!(refs, ["get_weather", "search_files"]);
        assert_eq!(serde_json::to_value(&block).unwrap(), json);
    }

    pub const CONTENT_SINGLE: &str = "\"Hello, world!\"";
    pub const CONTENT_MULTI: &str = r#"[
    {"type": "text", "text": "Hello, world!"},
    {"type": "text", "text": "How are you?"}
]"#;

    #[test]
    fn test_role_display() {
        assert_eq!(Role::User.to_string(), "User");
        assert_eq!(Role::Assistant.to_string(), "Assistant");
    }

    #[test]
    fn deserialize_content() {
        let content: Content = serde_json::from_str(CONTENT_SINGLE).unwrap();
        assert_eq!(content.to_string(), "Hello, world!");
        let content: Content = serde_json::from_str(CONTENT_MULTI).unwrap();
        assert_eq!(content.to_string(), "Hello, world!\n\nHow are you?");
    }

    pub const MESSAGE_JSON_SINGLE: &str =
        r#"{"role": "user", "content": "Hello, world"}"#;

    #[test]
    fn deserialize_message_single() {
        let message: Message =
            serde_json::from_str(MESSAGE_JSON_SINGLE).unwrap();
        // FIXME: This is really testing the Display impl. There should be a
        // separate test for that.
        assert_eq!(message.to_string(), "### User\n\nHello, world");
    }

    #[test]
    fn test_message_from_role_string_tuple() {
        let message: Message = (Role::User, "Hello, world!".to_string()).into();
        assert_eq!(message.to_string(), "### User\n\nHello, world!");
    }

    #[test]
    fn test_message_from_role_multi_part() {
        let message: Message = (Role::User, ["Hello, world!"]).into();
        assert_eq!(message.to_string(), "### User\n\nHello, world!");
        let content = vec!["Hello, world!", "How are you?"];
        let message: Message = (Role::User, content).into();
        assert_eq!(
            message.to_string(),
            "### User\n\nHello, world!\n\nHow are you?"
        );
    }

    #[test]
    fn test_message_is_empty() {
        let message: Message = (Role::User, "Hello, world!").into();
        assert!(!message.is_empty());
        let message: Message = Message {
            role: Role::User,
            content: Content(vec![]),
        };
        assert!(message.is_empty());
    }

    #[test]
    fn test_message_tool_use() {
        let tool_use: Message = tool::Use::new("tool", serde_json::json!({}))
            .with_id("tool_123")
            .into();

        assert!(tool_use.tool_use().is_some());
    }

    #[test]
    #[cfg(feature = "markdown")]
    // Exercises the `From`/`Into` conversions into Content/Block/Message.
    fn test_block_and_content_conversions() {
        let content: Content = "Hello, world!".into();
        assert_eq!(content.to_string(), "Hello, world!");

        let content = Content::text("Hello, world!");
        assert_eq!(content.to_string(), "Hello, world!");

        let block: Block = "Hello, world!".into();
        assert_eq!(block.to_string(), "Hello, world!");

        let image: Image = Image::from_parts(MediaType::Png, "data".into());
        assert_eq!(image.to_string(), "![Image](data:image/png;base64,data)");

        let tool_use: Block = tool::Use::new("tool", serde_json::json!({}))
            .with_id("tool_123")
            .into();
        assert_eq!(
            tool_use.markdown_verbose().as_ref(),
            "\n````json\n{\"type\":\"tool_use\",\"id\":\"tool_123\",\"name\":\"tool\",\"input\":{}}\n````"
        );

        let _message: Message = (Role::User, "Hello, world!").into();
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_merge_deltas() {
        use crate::markdown::ToMarkdown;

        let mut block: Block = "Hello, world!".into();

        // this is allowed
        block.merge_deltas([]).unwrap();

        let deltas = [
            Delta::Text {
                text: ", how are you?".into(),
            },
            Delta::Text {
                text: " I'm fine.".into(),
            },
        ];

        block.merge_deltas(deltas).unwrap();

        assert_eq!(block.to_string(), "Hello, world!, how are you? I'm fine.");

        // with tool use
        let mut block: Block = Block::ToolUse {
            call: tool::Use::new("tool", serde_json::json!({}))
                .with_id("tool_123"),
        };

        // partial json to apply to the input portion
        let deltas = [Delta::Json {
            partial_json: r#"{"key": "value"}"#.into(),
        }];

        block.merge_deltas(deltas).unwrap();

        // by default tool use is hidden
        let opts = crate::markdown::Options::default().with_tool_use();

        let markdown = block.markdown_custom(opts);

        assert_eq!(
            markdown.as_ref(),
            "\n````json\n{\"type\":\"tool_use\",\"id\":\"tool_123\",\"name\":\"tool\",\"input\":{\"key\":\"value\"}}\n````"
        );

        // test junk json
        let deltas = [Delta::Json {
            partial_json: "blabla".into(),
        }];
        let err = block.merge_deltas(deltas).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Cannot apply delta because deserialization failed because: Could not merge partial json `blabla` into `{\"key\":\"value\"}` because expected value at line 1 column 1"
        );

        // content mismatch
        let deltas = [Delta::Json {
            partial_json: "blabla".into(),
        }];
        let mut block = Block::Text {
            text: "Hello, world!".into(),
            citations: None,
            cache_control: None,
        };

        let err = block.merge_deltas(deltas).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Cannot apply delta because: `Delta::Json { partial_json: \"blabla\" }` canot be applied to `Block::Text`."
        );
    }

    #[test]
    fn test_message_len() {
        let mut message = Message {
            role: Role::User,
            content: Content::text("Hello, world!"),
        };

        assert_eq!(message.len(), 1); // one text block

        message.content.push("How are you?");

        assert_eq!(message.len(), 2); // blocks
    }

    #[test]
    fn test_from_response_message() {
        let response = response::Message {
            inner: Message {
                role: Role::Assistant,
                content: Content::text("Hello, world!"),
            }
            .try_into()
            .unwrap(),
            id: "msg_123".into(),
            kind: None,
            model: crate::Id::Sonnet35.into(),
            stop_reason: None,
            stop_sequence: None,
            stop_details: None,
            usage: Default::default(),
            container: None,
        };

        let message: Message = response.into();

        assert_eq!(message.to_string(), "### Assistant\n\nHello, world!");
    }

    #[test]
    fn test_from_role_cow() {
        let text: crate::CowStr = "Hello, world!".into();
        let message: Message = (Role::User, text).into();

        assert_eq!(message.to_string(), "### User\n\nHello, world!");
    }

    #[test]
    fn test_from_role_str() {
        let message: Message = (Role::User, "Hello, world!").into();

        assert_eq!(message.to_string(), "### User\n\nHello, world!");
    }

    #[test]
    fn test_content_is_empty() {
        let mut content = Content::text("Hello, world!");
        assert!(!content.is_empty());

        content = Content(vec![]);
        assert!(content.is_empty());
    }

    #[test]
    fn test_content_from_string() {
        let content: Content = "Hello, world!".to_string().into();
        assert_eq!(content.to_string(), "Hello, world!");
    }

    #[test]
    fn test_content_from_slice_of_str() {
        let content: Content = ["Hello, world!"].into();
        assert_eq!(content.to_string(), "Hello, world!");
    }

    #[test]
    fn test_content_from_block() {
        let content: Content = Block::text("Hello, world!").into();
        assert_eq!(content.to_string(), "Hello, world!");
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_merge_deltas_error() {
        let mut text_block: Block = "Hello, world!".into();

        let json_deltas = [Delta::Json {
            partial_json: "{\"k\": \"v\"}".into(),
        }];

        let err = text_block.merge_deltas(json_deltas).unwrap_err();

        let mut json_block = Block::ToolUse {
            call: tool::Use::new("tool", serde_json::json!({}))
                .with_id("tool_123"),
        };

        let json_deltas = [Delta::Json {
            partial_json: "{\"k\": \"v\"}".into(),
        }];

        json_block.merge_deltas(json_deltas).unwrap();
        assert_eq!(
            json_block.markdown_verbose().as_ref(),
            "\n````json\n{\"type\":\"tool_use\",\"id\":\"tool_123\",\"name\":\"tool\",\"input\":{\"k\":\"v\"}}\n````"
        );

        assert!(matches!(err, DeltaError::ContentMismatch { .. }));
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_message_markdown() {
        use crate::markdown::ToMarkdown;

        // test user heading, single block
        let message = Message {
            role: Role::User,
            content: Content::text("Hello, world!"),
        };

        let opts = crate::markdown::Options::default()
            .with_tool_use()
            .with_tool_results();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### User\n\nHello, world!"
        );

        // test assistant heading, multiple blocks
        let message = Message {
            role: Role::Assistant,
            content: Content(vec![
                "Hello, world!".into(),
                "How are you?".into(),
            ]),
        };

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Assistant\n\nHello, world!\n\nHow are you?"
        );

        // Test tool result (success)
        let message: Message =
            tool::Result::new("tool_123", Content::text("Hello, world!"))
                .into();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Tool\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}],\"is_error\":false}\n````"
        );

        // Test tool result (error)
        let message: Message =
            tool::Result::new("tool_123", Content::text("Hello, world!"))
                .error()
                .into();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Error\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":[{\"type\":\"text\",\"text\":\"Hello, world!\"}],\"is_error\":true}\n````"
        );
    }

    #[test]
    fn test_block_tool_use() {
        let expected =
            tool::Use::new("tool", serde_json::json!({})).with_id("tool_123");

        let block = Block::ToolUse {
            call: expected.clone(),
        };

        assert_eq!(block.tool_use(), Some(&expected));
    }

    #[test]
    fn test_block_from_str() {
        let block: Block = "Hello, world!".into();
        assert_eq!(block.to_string(), "Hello, world!");
    }

    #[test]
    fn test_block_from_string() {
        let block: Block = "Hello, world!".to_string().into();
        assert_eq!(block.to_string(), "Hello, world!");
    }

    #[test]
    #[cfg(feature = "png")]
    fn test_block_from_image() {
        let image = Image::from_parts(MediaType::Png, "data".into());
        let block: Block = image.into();
        assert_eq!(block.to_string(), "![Image](data:image/png;base64,data)");
    }

    // TODO: Image tests
    #[test]
    #[cfg(feature = "png")]
    fn test_block_from_rgba_image() {
        let image = image::RgbaImage::new(1, 1);
        let block: Block = image.into();
        assert!(matches!(block, Block::Image { .. }));
    }

    #[test]
    #[cfg(feature = "png")]
    fn test_block_from_dynamic_image() {
        let image = image::DynamicImage::new_rgba8(1, 1);
        let block: Block = image.into();
        assert!(matches!(block, Block::Image { .. }));
    }

    #[test]
    #[cfg(feature = "png")]
    fn test_image_from_compressed() {
        use std::io::Cursor;

        // Encode a sample image
        let expected = image::RgbaImage::new(1, 1);
        let mut encoded = Cursor::new(vec![]);
        expected
            .write_to(&mut encoded, image::ImageFormat::Png)
            .unwrap();

        // Decode the image
        let image =
            Image::from_compressed(MediaType::Png, encoded.into_inner());
        let actual: image::RgbaImage = image.try_into().unwrap();

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_image_url_source() {
        // URL sources need no image feature: construct, serde round-trip, and
        // render as a markdown image link via `Display`.
        let image = Image::from_url("https://example.com/cat.png");
        assert!(matches!(image, Image::Url { .. }));
        assert_eq!(image.to_string(), "![Image](https://example.com/cat.png)");
        assert_eq!(image.len(), "https://example.com/cat.png".len());

        let json = serde_json::to_value(&image).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "type": "url",
                "url": "https://example.com/cat.png",
            })
        );

        let back: Image = serde_json::from_value(json).unwrap();
        assert_eq!(back.to_string(), image.to_string());
    }

    #[test]
    #[cfg(feature = "image")]
    fn test_image_url_cannot_decode() {
        let image = Image::from_url("https://example.com/cat.png");
        assert!(matches!(image.decode(), Err(ImageDecodeError::Url)));
    }

    #[test]
    fn test_user_message_try_from_message() {
        let message: Message = (Role::Assistant, "Imitation!").into();
        assert!(UserMessage::try_from(message).is_err());
        let message: Message = (Role::User, "Valid!").into();
        assert!(UserMessage::try_from(message).is_ok());
    }

    #[test]
    fn test_user_message_serde() {
        let message: Message = (Role::Assistant, "Imitation!").into();
        let invalid_json = serde_json::to_string(&message).unwrap();
        let ret: Result<UserMessage, _> = serde_json::from_str(&invalid_json);
        assert!(ret.is_err());
        let message: Message = (Role::User, "Valid!").into();
        let valid_json = serde_json::to_string(&message).unwrap();
        let ret: Result<UserMessage, _> = serde_json::from_str(&valid_json);
        assert!(ret.is_ok());
    }

    #[test]
    fn test_assistant_message_try_from_message() {
        let message: Message = (Role::User, "Imitation!").into();
        assert!(AssistantMessage::try_from(message).is_err());
        let message: Message = (Role::Assistant, "Valid!").into();
        assert!(AssistantMessage::try_from(message).is_ok());
    }

    #[test]
    fn test_assistant_message_serde() {
        let message: Message = (Role::User, "Imitation!").into();
        let invalid_json = serde_json::to_string(&message).unwrap();
        let ret: Result<AssistantMessage, _> =
            serde_json::from_str(&invalid_json);
        assert!(ret.is_err());
        let message: Message = (Role::Assistant, "Valid!").into();
        let valid_json = serde_json::to_string(&message).unwrap();
        let ret: Result<AssistantMessage, _> =
            serde_json::from_str(&valid_json);
        assert!(ret.is_ok());
    }

    #[test]
    fn test_user_message_from_iter() {
        // From an iterator of &str (via blanket Into<Block>).
        let msg: UserMessage = ["Hello,", "world!"].into_iter().collect();
        let blocks = msg.content();
        assert_eq!(blocks.len(), 2);

        // From an iterator of tool::Result.
        let results = vec![
            tool::Result::new("tool_1", "ok"),
            tool::Result::new("tool_2", "parse error: ...").error(),
        ];
        let msg: UserMessage = results.into_iter().collect();
        let blocks = msg.content();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], Block::ToolResult { .. }));
        assert!(matches!(blocks[1], Block::ToolResult { .. }));
    }

    #[test]
    fn test_assistant_message_from_iter() {
        let msg: AssistantMessage =
            ["thinking...", "done."].into_iter().collect();
        let blocks = msg.content();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_cache_control_default_serialization() {
        // Default ephemeral (5-minute) should serialize without ttl field
        let cc = CacheControl::default();
        let json = serde_json::to_string(&cc).unwrap();
        assert_eq!(json, r#"{"type":"ephemeral"}"#);
    }

    #[test]
    fn test_cache_control_ephemeral_convenience() {
        // ephemeral() convenience method should match default
        let cc = CacheControl::ephemeral();
        let json = serde_json::to_string(&cc).unwrap();
        assert_eq!(json, r#"{"type":"ephemeral"}"#);
    }

    #[test]
    fn test_cache_control_one_hour_serialization() {
        // 1-hour TTL should include the ttl field
        let cc = CacheControl::one_hour();
        let json = serde_json::to_string(&cc).unwrap();
        assert_eq!(json, r#"{"type":"ephemeral","ttl":"1h"}"#);
    }

    #[test]
    fn test_cache_control_default_deserialization() {
        // Deserialize without ttl field
        let json = r#"{"type":"ephemeral"}"#;
        let cc: CacheControl = serde_json::from_str(json).unwrap();
        assert_eq!(cc, CacheControl::Ephemeral { ttl: None });
    }

    #[test]
    fn test_cache_control_explicit_5m_deserialization() {
        // An explicit "5m" ttl round-trips (equivalent to omitting it).
        let json = r#"{"type":"ephemeral","ttl":"5m"}"#;
        let cc: CacheControl = serde_json::from_str(json).unwrap();
        assert_eq!(
            cc,
            CacheControl::Ephemeral {
                ttl: Some(CacheTtl::FiveMinutes)
            }
        );
        assert_eq!(serde_json::to_string(&cc).unwrap(), json);
    }

    #[test]
    fn test_cache_control_one_hour_deserialization() {
        // Deserialize with ttl field
        let json = r#"{"type":"ephemeral","ttl":"1h"}"#;
        let cc: CacheControl = serde_json::from_str(json).unwrap();
        assert_eq!(
            cc,
            CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            }
        );
    }

    #[test]
    fn test_cache_control_roundtrip() {
        // Roundtrip for default
        let original = CacheControl::ephemeral();
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CacheControl = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);

        // Roundtrip for 1-hour
        let original = CacheControl::one_hour();
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: CacheControl = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_document_block_wire_shape() {
        // A plain-text document with citations enabled serializes to the
        // wire shape the API expects, and round-trips.
        let block = Block::document_with_citations(DocumentSource::from_text(
            "The sky on planet Zorblax is purple.",
        ));

        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(value["type"], "document");
        assert_eq!(value["source"]["type"], "text");
        assert_eq!(value["source"]["media_type"], "text/plain");
        assert_eq!(value["citations"]["enabled"], true);

        let back: Block = serde_json::from_value(value).unwrap();
        assert!(matches!(
            back,
            Block::Document {
                citations: Some(CitationsConfig { enabled: true }),
                ..
            }
        ));
    }

    #[test]
    fn test_base64_document_source_wire_shape() {
        let value =
            serde_json::to_value(DocumentSource::from_base64("Zm9v")).unwrap();
        assert_eq!(value["type"], "base64");
        assert_eq!(value["media_type"], "application/pdf");
        assert_eq!(value["data"], "Zm9v");
    }

    #[test]
    fn test_text_block_citations_field_roundtrips() {
        // A response text block carrying a CharLocation citation deserializes
        // and the citation survives a round-trip. Mirrors the response wire
        // form (citations omitted when absent, present when the API cites a
        // document).
        let json = r#"{
            "type": "text",
            "text": "The sky is purple.",
            "citations": [{
                "type": "char_location",
                "cited_text": "The sky on planet Zorblax is purple.",
                "document_index": 0,
                "start_char_index": 0,
                "end_char_index": 36
            }]
        }"#;

        let block: Block = serde_json::from_str(json).unwrap();
        let Block::Text {
            citations: Some(cs),
            ..
        } = &block
        else {
            panic!("expected a Text block with citations: {block:?}");
        };
        assert!(matches!(
            cs.as_slice(),
            [Citation::CharLocation {
                end_char_index: 36,
                ..
            }]
        ));

        // Absent citations are elided on the wire (not `"citations": null`).
        let plain = serde_json::to_value(Block::text("hi")).unwrap();
        assert!(plain.get("citations").is_none());
    }
}
