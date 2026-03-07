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
}

impl Role {
    /// Get the string representation of the role.
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::User => "User",
            Self::Assistant => "Assistant",
        }
    }

    /// Convenience method for lowercase role.
    pub const fn as_lowercase(&self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
        }
    }

    /// Toggle the role between [`Role::User`] and [`Role::Assistant`].
    pub const fn toggle(&self) -> Self {
        match self {
            Self::User => Self::Assistant,
            Self::Assistant => Self::User,
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
pub struct Message<'a> {
    /// Who is the message from.
    pub role: Role,
    /// The [`Content`] of the message as [one] or [more] [`Block`]s.
    ///
    /// [one]: Content::SinglePart
    /// [more]: Content::MultiPart
    pub content: Content<'a>,
}

impl Message<'_> {
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
    pub fn tool_use(&self) -> Option<&crate::tool::Use<'_>> {
        self.content.last()?.tool_use()
    }

    /// Returns Some([`tool::Result`]) if the first [`Content`] [`Block`] is a
    /// [`Block::ToolResult`].
    pub fn tool_result(&self) -> Option<&crate::tool::Result<'_>> {
        match &self.content {
            Content::SinglePart(_) => None,
            Content::MultiPart(parts) => {
                if let Some(Block::ToolResult { result }) = parts.first() {
                    Some(result)
                } else {
                    None
                }
            }
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    ///
    /// [`Cow`]: std::borrow::Cow
    pub fn into_static(self) -> Message<'static> {
        Message {
            role: self.role,
            content: self.content.into_static(),
        }
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

        if let Content::MultiPart(parts) = &mut self.content {
            if let Some(Block::Thought { signature, .. }) = parts.last() {
                if signature.is_empty() {
                    parts.pop();
                }
            }
        }

        if self.is_empty() { None } else { Some(self) }
    }
}

impl<'a> From<response::Message<'a>> for Message<'a> {
    fn from(message: response::Message<'a>) -> Self {
        message.inner.inner
    }
}

impl<'a> From<response::Message<'a>> for AssistantMessage<'a> {
    fn from(message: response::Message<'a>) -> Self {
        message.inner
    }
}

impl<'a, T> From<(Role, T)> for Message<'a>
where
    T: Into<Content<'a>>,
{
    fn from((role, content): (Role, T)) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }
}

impl<'a> From<tool::Use<'a>> for Message<'a> {
    fn from(call: tool::Use<'a>) -> Self {
        Message {
            role: Role::Assistant,
            content: call.into(),
        }
    }
}

impl<'a> From<tool::Result<'a>> for Message<'a> {
    fn from(result: tool::Result<'a>) -> Self {
        Message {
            role: Role::User,
            content: result.into(),
        }
    }
}

impl<'a> IntoIterator for Message<'a> {
    type Item = Block<'a>;
    type IntoIter = std::vec::IntoIter<Block<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        match self.content {
            Content::SinglePart(text) => vec![Block::Text {
                text,
                citations: None,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }]
            .into_iter(),
            Content::MultiPart(parts) => parts.into_iter(),
        }
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Message<'a> {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
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
impl std::fmt::Display for Message<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
#[serde(try_from = "Message<'_>", into = "Message<'_>")]
#[display("{}", inner)]
pub struct AssistantMessage<'a> {
    pub(crate) inner: Message<'a>, // Invariant: role == Role::Assistant
}

impl<'a> AssistantMessage<'a> {
    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> AssistantMessage<'static> {
        AssistantMessage::<'static> {
            inner: self.inner.into_static(),
        }
    }

    /// Get the inner [`Content`].
    pub fn content(&self) -> &Content<'a> {
        &self.inner.content
    }

    /// Get the inner [`Content`] mutably.
    pub fn content_mut(&mut self) -> &mut Content<'a> {
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

impl<'a> From<Content<'a>> for AssistantMessage<'a> {
    fn from(content: Content<'a>) -> Self {
        Self {
            inner: Message {
                role: Role::Assistant,
                content,
            },
        }
    }
}

impl<'a> From<AssistantMessage<'a>> for Message<'a> {
    fn from(val: AssistantMessage<'a>) -> Self {
        val.inner
    }
}

impl<'a> From<AssistantMessage<'a>> for Content<'a> {
    fn from(val: AssistantMessage<'a>) -> Self {
        val.inner.content
    }
}

impl<'a> TryFrom<Message<'a>> for AssistantMessage<'a> {
    type Error = NotTheAssistant;

    fn try_from(message: Message<'a>) -> Result<Self, Self::Error> {
        if message.role == Role::Assistant {
            Ok(Self { inner: message })
        } else {
            Err(NotTheAssistant)
        }
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for AssistantMessage<'a> {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        self.inner.markdown_events_custom(options)
    }
}

/// Error message when conversion to [`AgentMessage`] fails.
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
#[serde(try_from = "Message<'_>", into = "Message<'_>")]
#[display("{}", inner)]
pub struct UserMessage<'a> {
    inner: Message<'a>, // Invariant: role == Role::User
}

impl<'a> UserMessage<'a> {
    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> UserMessage<'static> {
        UserMessage::<'static> {
            inner: self.inner.into_static(),
        }
    }

    /// Get the inner [`Content`].
    pub fn content(&self) -> &Content<'a> {
        &self.inner.content
    }

    /// Get the inner [`Content`] mutably.
    pub fn content_mut(&mut self) -> &mut Content<'a> {
        &mut self.inner.content
    }
}

impl<'a> From<Content<'a>> for UserMessage<'a> {
    fn from(content: Content<'a>) -> Self {
        Self {
            inner: Message {
                role: Role::User,
                content,
            },
        }
    }
}

impl<'a> From<UserMessage<'a>> for Content<'a> {
    fn from(message: UserMessage<'a>) -> Self {
        message.inner.content
    }
}

impl From<String> for UserMessage<'_> {
    fn from(string: String) -> Self {
        UserMessage {
            inner: Message {
                role: Role::User,
                content: Content::text(string),
            },
        }
    }
}

impl<'a> From<&'a str> for UserMessage<'a> {
    fn from(string: &'a str) -> Self {
        UserMessage {
            inner: Message {
                role: Role::User,
                content: Content::text(string),
            },
        }
    }
}

impl<'a> From<tool::Result<'a>> for UserMessage<'a> {
    fn from(result: tool::Result<'a>) -> Self {
        UserMessage {
            inner: result.into(),
        }
    }
}

impl<'a> IntoIterator for UserMessage<'a> {
    type Item = Block<'a>;
    type IntoIter = std::vec::IntoIter<Block<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        match self.inner.content {
            Content::SinglePart(text) => vec![Block::Text {
                text,
                citations: None,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }]
            .into_iter(),
            Content::MultiPart(parts) => parts.into_iter(),
        }
    }
}

#[cfg(feature = "dioxus")]
impl From<dioxus::events::FormEvent> for UserMessage<'_> {
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
impl From<dioxus::html::FormData> for UserMessage<'_> {
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

impl<'a> TryFrom<Message<'a>> for UserMessage<'a> {
    type Error = NotTheUser;

    fn try_from(message: Message<'a>) -> Result<Self, Self::Error> {
        if message.role == Role::User {
            Ok(Self { inner: message })
        } else {
            Err(NotTheUser)
        }
    }
}

impl<'a> From<UserMessage<'a>> for Message<'a> {
    fn from(message: UserMessage<'a>) -> Self {
        message.inner
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for UserMessage<'a> {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
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

/// Content of a [`Message`].
#[derive(
    Clone, Debug, Serialize, Deserialize, Hash, derive_more::IsVariant,
)]
#[serde(rename_all = "snake_case")]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Content<'a> {
    /// Single part text-only content.
    SinglePart(crate::CowStr<'a>),
    /// Multiple content [`Block`]s.
    MultiPart(Vec<Block<'a>>),
}

impl<'a> Content<'a> {
    /// Const constructor for static text content. Not available with the
    /// `langsan` feature.
    #[cfg(not(feature = "langsan"))]
    pub const fn const_text(text: &'static str) -> Self {
        Self::SinglePart(std::borrow::Cow::Borrowed(text))
    }

    /// Text content.
    pub fn text<T>(text: T) -> Self
    where
        T: Into<crate::CowStr<'a>>,
    {
        Self::SinglePart(text.into())
    }

    /// Returns the number of [`Block`]s in self.
    pub fn len(&self) -> usize {
        match self {
            Self::SinglePart(_) => 1,
            Self::MultiPart(parts) => parts.len(),
        }
    }

    /// Returns true if `self` is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Unwrap [`Content::SinglePart`] as a [`Block::Text`]. This will panic if
    /// `self` is [`MultiPart`].
    ///
    /// [`SinglePart`]: Content::SinglePart
    /// [`MultiPart`]: Content::MultiPart
    ///
    /// # Panics
    /// - If the content is [`MultiPart`].
    pub fn unwrap_single_part(self) -> Block<'a> {
        match self {
            #[cfg(feature = "prompt-caching")]
            Self::SinglePart(text) => Block::Text {
                text,
                citations: None,
                cache_control: None,
            },
            #[cfg(not(feature = "prompt-caching"))]
            Self::SinglePart(text) => Block::Text {
                text,
                citations: None,
            },
            Self::MultiPart(_) => {
                panic!("Content is MultiPart, not SinglePart");
            }
        }
    }

    /// Add a [`Block`] to the [`Content`]. If the [`Content`] is a
    /// [`SinglePart`], it will be converted to a [`MultiPart`].
    ///
    /// The index of the inserted block is returned.
    ///
    /// [`SinglePart`]: Content::SinglePart
    /// [`MultiPart`]: Content::MultiPart
    pub fn push<P>(&mut self, part: P) -> usize
    where
        P: Into<Block<'a>>,
    {
        // If there is a SinglePart message, convert it to a MultiPart message.
        if self.is_single_part() {
            // the old switcheroo
            let mut old = Content::MultiPart(vec![]);
            std::mem::swap(self, &mut old);
            // This can never loop because we ensure self is a MultiPart which
            // will skip this block.
            self.push(old.unwrap_single_part());
        }

        if let Content::MultiPart(parts) = self {
            let index = parts.len();
            parts.push(part.into());
            return index;
        } else {
            unreachable!()
        }
    }

    /// Add a cache breakpoint to the final [`Block`]. If the [`Content`] is
    /// [`SinglePart`], it will be converted to [`MultiPart`] first.
    ///
    /// [`SinglePart`]: Content::SinglePart
    /// [`MultiPart`]: Content::MultiPart
    #[cfg(feature = "prompt-caching")]
    pub fn cache(&mut self) {
        if self.is_single_part() {
            let mut old = Content::MultiPart(vec![]);
            std::mem::swap(self, &mut old);
            self.push(old.unwrap_single_part());
        }

        if let Content::MultiPart(parts) = self {
            if let Some(block) = parts.last_mut() {
                block.cache();
            }
        }
    }

    /// Get the last [`Block`] in the [`Content`]. Returns [`None`] if the
    /// [`Content`] is empty or [`SinglePart`].
    ///
    /// [`SinglePart`]: Content::SinglePart
    // Because to make it multi-part on access this would have to be &mut
    pub fn last(&self) -> Option<&Block<'_>> {
        match self {
            Self::SinglePart(_) => None,
            Self::MultiPart(parts) => parts.last(),
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    ///
    /// [`Cow`]: std::borrow::Cow
    pub fn into_static(self) -> Content<'static> {
        match self {
            Self::SinglePart(text) => {
                #[cfg(not(feature = "langsan"))]
                {
                    Content::SinglePart(std::borrow::Cow::Owned(
                        text.into_owned(),
                    ))
                }
                #[cfg(feature = "langsan")]
                {
                    Content::SinglePart(text.into_static())
                }
            }
            Self::MultiPart(parts) => Content::MultiPart(
                parts.into_iter().map(Block::into_static).collect(),
            ),
        }
    }

    /// Push a [`Delta`] into the [`Content`]. The types must be compatible or
    /// this will return a [`ContentMismatch`] error.
    ///
    /// It is an error to try to merge a single json delta into a content block.
    pub fn push_delta(
        &mut self,
        delta: Delta<'a>,
    ) -> Result<(), DeltaError<'_>> {
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

        match self {
            Self::SinglePart(_) => {
                let mut old = Content::MultiPart(vec![]);
                std::mem::swap(self, &mut old);
                self.push(old.unwrap_single_part());
                self.push_delta(delta)?;
            }
            Self::MultiPart(parts) => {
                parts
                    .last_mut()
                    .unwrap()
                    .merge_deltas(std::iter::once(delta))?;
            }
        }

        Ok(())
    }

    /// Drains the blocks from the content.
    pub fn drain(&'a mut self) -> impl Iterator<Item = Block<'a>> + 'a {
        let ret: Box<dyn Iterator<Item = Block<'a>>> = match self {
            Self::SinglePart(_) => {
                let mut old = Content::MultiPart(vec![]);
                std::mem::swap(self, &mut old);
                match old {
                    Content::SinglePart(text) => {
                        Box::new(std::iter::once(Block::Text {
                            text,
                            citations: None,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        }))
                    }
                    Content::MultiPart(parts) => Box::new(parts.into_iter()),
                }
            }
            Self::MultiPart(parts) => Box::new(parts.drain(..)),
        };

        ret
    }

    /// Get a block mutably. If this is [`SinglePart`] content, it will be
    /// converted to [`MultiPart`] first.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut Block<'a>> {
        if self.is_single_part() {
            let mut old = Content::MultiPart(vec![]);
            std::mem::swap(self, &mut old);
            self.push(old.unwrap_single_part());
        }
        // Self is now MultiPart

        match self {
            Self::MultiPart(parts) => parts.get_mut(index),
            Self::SinglePart(_) => unreachable!(),
        }
    }

    /// Iterate mutably over the blocks. If this is [`SinglePart`] content, it
    /// will be converted to [`MultiPart`] first.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Block<'a>> {
        if self.is_single_part() {
            let mut old = Content::MultiPart(vec![]);
            std::mem::swap(self, &mut old);
            self.push(old.unwrap_single_part());
        }

        match self {
            Self::MultiPart(parts) => parts.iter_mut(),
            Self::SinglePart(_) => unreachable!(),
        }
    }
}

impl<'a> IntoIterator for Content<'a> {
    type Item = Block<'a>;
    type IntoIter = std::vec::IntoIter<Block<'a>>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Content::SinglePart(text) => vec![Block::Text {
                text,
                citations: None,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }]
            .into_iter(),
            Content::MultiPart(parts) => parts.into_iter(),
        }
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Content<'a> {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::Event;

        let it: Box<dyn Iterator<Item = Event<'a>> + 'a> = match self {
            Self::SinglePart(string) => {
                Box::new(pulldown_cmark::Parser::new(string))
            }
            Self::MultiPart(parts) => Box::new(
                parts
                    .iter()
                    .flat_map(move |part| part.markdown_events_custom(options)),
            ),
        };

        it
    }
}

#[cfg(not(feature = "markdown"))]
impl std::fmt::Display for Content<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SinglePart(string) => write!(f, "{}", string),
            // This could be derived but the `Join` trait is not stable. Neither
            // is `Iterator::intersperse`. This also has fewer allocations.
            Self::MultiPart(parts) => {
                let mut iter = parts.iter();
                if let Some(part) = iter.next() {
                    write!(f, "{}", part)?;
                    for part in iter {
                        write!(f, "{}{}", Self::SEP, part)?;
                    }
                }
                Ok(())
            }
        }
    }
}

#[cfg(feature = "dioxus")]
impl From<dioxus::html::FormData> for Content<'_> {
    fn from(data: dioxus::html::FormData) -> Self {
        data.value().into()
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Content<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl Content<'_> {
    /// Separator for multi-part content.
    #[cfg(not(feature = "markdown"))]
    pub const SEP: &'static str = "\n\n";
}

impl<'a, T> From<T> for Content<'a>
where
    T: Into<Block<'a>>,
{
    fn from(block: T) -> Self {
        Self::MultiPart(vec![block.into()])
    }
}

impl<'a, T> FromIterator<T> for Content<'a>
where
    T: Into<Block<'a>>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::MultiPart(iter.into_iter().map(Into::into).collect())
    }
}

impl<'a, T> Extend<T> for Content<'a>
where
    T: Into<Block<'a>>,
{
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let text = match self {
            Self::SinglePart(old) => {
                let mut text: crate::CowStr<'a> = String::new().into();
                std::mem::swap(old, &mut text);
                text
            }
            Self::MultiPart(parts) => {
                parts.extend(iter.into_iter().map(Into::into));
                return;
            }
        };
        // We have single-part content, so we need to convert it to multi-part
        // and then extend it.

        *self = Self::MultiPart(vec![Block::Text {
            text,
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }]);
        // This can never recurse infinitely because we just converted to
        // MultiPart and that will skip this by returning early.
        self.extend(iter);
    }
}

// I would love to have a conversion method form IntoIterator<Item = T> but
// that conflicts for str because in the future str might implement IntoIterator
// and Iterator. This is a workaround for now.

// I don't really like this because the generics mean a new function for every
// array size. But in most cases the array size is between 1 and 3 so it's not
// a big deal.
impl<'a, T, const N: usize> From<[T; N]> for Content<'a>
where
    T: Into<Block<'a>>,
{
    fn from(blocks: [T; N]) -> Self {
        Self::MultiPart(blocks.into_iter().map(|t| t.into()).collect())
    }
}

impl<'a> From<&'a [&'a str]> for Content<'a> {
    fn from(text: &'a [&'a str]) -> Self {
        Self::MultiPart(text.iter().map(|t| (*t).into()).collect())
    }
}

impl<'a, T> From<Vec<T>> for Content<'a>
where
    T: Into<Block<'a>>,
{
    fn from(blocks: Vec<T>) -> Self {
        Self::MultiPart(blocks.into_iter().map(Into::into).collect())
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
pub enum Block<'a> {
    /// Text content.
    #[serde(alias = "text_delta")]
    #[cfg_attr(not(feature = "markdown"), display("{text}"))]
    Text {
        /// The actual text content.
        text: crate::CowStr<'a>,
        /// Citations referencing source documents.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<Vec<Citation<'a>>>,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[cfg(feature = "prompt-caching")]
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
        /// a need to show the thought, a Cow<'a, str> is convertable to a
        /// `langsan::CowStr`.
        #[serde(rename = "thinking")]
        thought: Cow<'a, str>,
        /// Signature. Guarantees thought was not tampered with. It's up to the
        /// caller to not mix up the thought signatures. Anthropic will reject
        /// the request if the signature is invalid.
        #[serde(default)]
        signature: Cow<'a, str>,
    },
    /// Redacted thinking. Sometimes the system will redact the thinking content
    /// for safety reasons. The Assistant can still see the redacted content.
    #[cfg_attr(not(feature = "markdown"), display("[REDACTED]"))]
    #[serde(rename = "redacted_thinking")]
    RedactedThought {
        /// Allows the Assistant to see the redacted thought if it is provided.
        #[serde(rename = "data")]
        signature: Cow<'a, str>,
    },
    /// Image content.
    #[cfg_attr(not(feature = "markdown"), display("{}", image))]
    Image {
        #[serde(rename = "source")]
        /// An base64 encoded image.
        image: Image<'a>,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[cfg(feature = "prompt-caching")]
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Document content (PDF, plain text, or custom content).
    #[cfg_attr(not(feature = "markdown"), display("{}", source))]
    Document {
        /// The document source.
        #[serde(rename = "source")]
        source: DocumentSource<'a>,
        /// Optional title (passed to model, not citable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<Cow<'a, str>>,
        /// Optional context (passed to model, not citable).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<Cow<'a, str>>,
        /// Enable citations for this document.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        citations: Option<CitationsConfig>,
        /// Use prompt caching.
        #[cfg(feature = "prompt-caching")]
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
        call: tool::Use<'a>,
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
        result: tool::Result<'a>,
    },
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Block<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl<'a> Block<'a> {
    /// Const constructor for text content. Only available without the `langsan`
    /// feature.
    // TODO: rename this to `text` which is more consistent with the other
    // constructors? Or the other way around?
    #[cfg(not(feature = "langsan"))]
    pub const fn const_text(text: &'a str) -> Self {
        Self::Text {
            text: std::borrow::Cow::Borrowed(text),
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Text content.
    pub fn text<T>(text: T) -> Self
    where
        T: Into<crate::CowStr<'a>>,
    {
        Self::Text {
            text: text.into(),
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Document content block.
    pub fn document(source: DocumentSource<'a>) -> Self {
        Self::Document {
            source,
            title: None,
            context: None,
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Document content block with citations enabled.
    pub fn document_with_citations(source: DocumentSource<'a>) -> Self {
        Self::Document {
            source,
            title: None,
            context: None,
            citations: Some(CitationsConfig { enabled: true }),
            #[cfg(feature = "prompt-caching")]
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
    /// will return a [`ContentMismatch`] error. In the case of a [`ToolUse`]
    /// block, the deltas, together, must form a complete json object.
    pub fn merge_deltas<Ds>(&mut self, deltas: Ds) -> Result<(), DeltaError<'_>>
    where
        Ds: IntoIterator<Item = Delta<'a>>,
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
                    *thought = delta_thinking.into();
                    *signature = delta_signature.into();
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
            // Citations delta merges into a text block.
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
                    Block::Image { .. } => stringify!(Block::Image),
                    Block::Document { .. } => {
                        stringify!(Block::Document)
                    }
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
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    pub fn cache(&mut self) -> bool {
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
            } => {
                *cache_control = Some(CacheControl::Ephemeral);

                true
            }
            // These are automatically cached.
            // https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#using-extended-thinking-with-prompt-caching
            Self::Thought { .. } | Self::RedactedThought { .. } => false,
        }
    }

    /// Returns true if the block has a `cache_control` breakpoint.
    #[cfg(feature = "prompt-caching")]
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
            } => cache_control.is_some(),
            Self::Thought { .. } | Self::RedactedThought { .. } => false,
        }
    }

    /// Returns the [`tool::Use`] if this is a [`Block::ToolUse`]. See also
    /// [`response::Message::tool_use`].
    pub fn tool_use(&self) -> Option<&crate::tool::Use<'_>> {
        match self {
            Self::ToolUse { call, .. } => Some(call),
            _ => None,
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    ///
    /// [`Cow`]: std::borrow::Cow
    pub fn into_static(self) -> Block<'static> {
        match self {
            Self::Text {
                text,
                citations,
                #[cfg(feature = "prompt-caching")]
                cache_control,
            } => Block::Text {
                #[cfg(not(feature = "langsan"))]
                text: std::borrow::Cow::Owned(text.into_owned()),
                #[cfg(feature = "langsan")]
                text: text.into_static(),
                citations: citations.map(|cs| {
                    cs.into_iter().map(Citation::into_static).collect()
                }),
                #[cfg(feature = "prompt-caching")]
                cache_control,
            },
            Self::Thought { thought, signature } => Block::Thought {
                thought: thought.into_owned().into(),
                signature: signature.into_owned().into(),
            },
            Self::RedactedThought { signature } => Block::RedactedThought {
                signature: signature.into_owned().into(),
            },
            Self::Image {
                image,
                #[cfg(feature = "prompt-caching")]
                cache_control,
            } => Block::Image {
                image: image.into_static(),
                #[cfg(feature = "prompt-caching")]
                cache_control,
            },
            Self::Document {
                source,
                title,
                context,
                citations,
                #[cfg(feature = "prompt-caching")]
                cache_control,
            } => Block::Document {
                source: source.into_static(),
                title: title.map(|t| Cow::Owned(t.into_owned())),
                context: context.map(|c| Cow::Owned(c.into_owned())),
                citations,
                #[cfg(feature = "prompt-caching")]
                cache_control,
            },
            Self::ToolUse { call } => Block::ToolUse {
                call: call.into_static(),
            },
            Self::ToolResult { result } => Block::ToolResult {
                result: result.into_static(),
            },
        }
    }

    /// Returns the number of bytes in the block. Does not include tool use or
    /// other metadata. Does include the base64 encoded image data length.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.as_bytes().len(),
            Self::Thought {
                thought: thinking, ..
            } => thinking.len(),
            Self::Image { image, .. } => image.len(),
            Self::Document { source, .. } => source.len(),
            Self::RedactedThought { .. }
            | Self::ToolUse { .. }
            | Self::ToolResult { .. } => 0,
        }
    }
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Block<'a> {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        let it: Box<dyn Iterator<Item = Event<'a>> + 'a> = match self {
            Self::Text { text, .. } => {
                // We'll parse the inner text as markdown.
                Box::new(pulldown_cmark::Parser::new_ext(text, options.inner))
            }
            Block::Image { image, .. } => {
                // We use Event::Text for images because they are rendered as
                // markdown images with embedded base64 data.
                Box::new([Event::Text(image.to_string().into())].into_iter())
            }
            Block::ToolUse { .. } => {
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
            Block::RedactedThought { .. } => Box::new(std::iter::empty()),
            Block::ToolResult { .. } => {
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

impl<'a> From<&'a str> for Block<'a> {
    fn from(text: &'a str) -> Self {
        Self::text(text)
    }
}

impl From<String> for Block<'_> {
    fn from(text: String) -> Self {
        Self::Text {
            text: text.into(),
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl<'a> From<crate::CowStr<'a>> for Block<'a> {
    fn from(text: crate::CowStr<'a>) -> Self {
        Self::Text {
            text,
            citations: None,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl<'a> From<Image<'a>> for Block<'a> {
    fn from(image: Image<'a>) -> Self {
        Self::Image {
            image,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl<'a> From<DocumentSource<'a>> for Block<'a> {
    fn from(source: DocumentSource<'a>) -> Self {
        Self::document(source)
    }
}

impl<'a> From<tool::Use<'a>> for Block<'a> {
    fn from(call: tool::Use<'a>) -> Self {
        Self::ToolUse { call }
    }
}

impl<'a> From<tool::Result<'a>> for Block<'a> {
    fn from(result: tool::Result<'a>) -> Self {
        Self::ToolResult { result }
    }
}

#[cfg(feature = "png")]
impl From<image::RgbaImage> for Block<'_> {
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
impl From<image::DynamicImage> for Block<'_> {
    fn from(image: image::DynamicImage) -> Self {
        image.to_rgba8().into()
    }
}

/// Cache control for prompt caching.
#[cfg(feature = "prompt-caching")]
#[derive(Clone, Default, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum CacheControl {
    /// Caches for 5 minutes.
    #[default]
    Ephemeral,
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

/// Media type for PDF documents.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum DocumentMediaType {
    /// `application/pdf`
    #[serde(rename = "application/pdf")]
    Pdf,
}

impl std::fmt::Display for DocumentMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pdf => write!(f, "application/pdf"),
        }
    }
}

/// Media type for plain text documents.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum PlainTextMediaType {
    /// `text/plain`
    #[serde(rename = "text/plain")]
    Plain,
}

impl std::fmt::Display for PlainTextMediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plain => write!(f, "text/plain"),
        }
    }
}

/// A text chunk for custom content [`DocumentSource`]s.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type", rename = "text")]
pub struct ContentText<'a> {
    /// The text content of this chunk.
    pub text: Cow<'a, str>,
}

impl<'a> ContentText<'a> {
    /// Convert to a `'static` lifetime.
    pub fn into_static(self) -> ContentText<'static> {
        ContentText {
            text: Cow::Owned(self.text.into_owned()),
        }
    }
}

/// Source of a [`Document`] content block. Analogous to [`Image`]
/// for image content.
///
/// [`Document`]: Block::Document
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DocumentSource<'a> {
    /// Base64-encoded document (PDF).
    Base64 {
        /// Document encoding format.
        media_type: DocumentMediaType,
        /// Base64-encoded document data.
        data: Cow<'a, str>,
    },
    /// URL to a hosted document.
    Url {
        /// The URL.
        url: Cow<'a, str>,
    },
    /// Plain text document (auto-chunked into sentences for
    /// citations).
    #[serde(rename = "text")]
    PlainText {
        /// Always `text/plain`.
        media_type: PlainTextMediaType,
        /// The plain text content.
        data: Cow<'a, str>,
    },
    /// Custom content blocks (user controls citation granularity).
    Content {
        /// The content blocks.
        content: Vec<ContentText<'a>>,
    },
    /// Reference to a file uploaded via the Files API.
    File {
        /// The file ID.
        file_id: Cow<'a, str>,
    },
}

impl<'a> DocumentSource<'a> {
    /// Create a base64-encoded PDF document source.
    pub fn from_base64(data: impl Into<Cow<'a, str>>) -> Self {
        Self::Base64 {
            media_type: DocumentMediaType::Pdf,
            data: data.into(),
        }
    }

    /// Create a URL document source.
    pub fn from_url(url: impl Into<Cow<'a, str>>) -> Self {
        Self::Url { url: url.into() }
    }

    /// Create a plain text document source.
    pub fn from_text(data: impl Into<Cow<'a, str>>) -> Self {
        Self::PlainText {
            media_type: PlainTextMediaType::Plain,
            data: data.into(),
        }
    }

    /// Create a custom content document source from text chunks.
    pub fn from_content(blocks: Vec<ContentText<'a>>) -> Self {
        Self::Content { content: blocks }
    }

    /// Create a Files API reference document source.
    pub fn from_file_id(id: impl Into<Cow<'a, str>>) -> Self {
        Self::File { file_id: id.into() }
    }

    /// Read a file, base64-encode it, and create a document source.
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

    /// Convert to a `'static` lifetime.
    pub fn into_static(self) -> DocumentSource<'static> {
        match self {
            Self::Base64 { media_type, data } => DocumentSource::Base64 {
                media_type,
                data: Cow::Owned(data.into_owned()),
            },
            Self::Url { url } => DocumentSource::Url {
                url: Cow::Owned(url.into_owned()),
            },
            Self::PlainText { media_type, data } => DocumentSource::PlainText {
                media_type,
                data: Cow::Owned(data.into_owned()),
            },
            Self::Content { content } => DocumentSource::Content {
                content: content
                    .into_iter()
                    .map(ContentText::into_static)
                    .collect(),
            },
            Self::File { file_id } => DocumentSource::File {
                file_id: Cow::Owned(file_id.into_owned()),
            },
        }
    }

    /// Returns the byte length of the source data.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Base64 { data, .. } | Self::PlainText { data, .. } => {
                data.as_bytes().len()
            }
            Self::Url { url } => url.as_bytes().len(),
            Self::Content { content } => {
                content.iter().map(|c| c.text.as_bytes().len()).sum()
            }
            Self::File { file_id } => file_id.as_bytes().len(),
        }
    }
}

impl std::fmt::Display for DocumentSource<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

/// Image content for [`MultiPart`] [`Message`]s.
///
/// [`MultiPart`]: Content::MultiPart
#[derive(Clone, Debug, Serialize, Deserialize, derive_more::Display, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum Image<'a> {
    /// Base64 encoded image data. When displayed, it will be rendered as a
    /// markdown image with embedded data.
    #[display("![Image](data:{media_type};base64,{data})")]
    Base64 {
        /// Image encoding format.
        media_type: MediaType,
        /// Base64 encoded compressed image data.
        data: Cow<'a, str>,
    },
}

impl<'a> Image<'a> {
    /// From raw parts. The data is expected to be base64 encoded compressed
    /// image data or the API will reject it.
    pub fn from_parts(media_type: MediaType, data: Cow<'a, str>) -> Self {
        Self::Base64 { media_type, data }
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
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    ///
    /// [`Cow`]: std::borrow::Cow
    pub fn into_static(self) -> Image<'static> {
        match self {
            Self::Base64 { media_type, data } => Image::Base64 {
                media_type,
                data: std::borrow::Cow::Owned(data.into_owned()),
            },
        }
    }

    /// Returns the number of bytes in the image data (base64 encoded). Call
    /// [`decode`] to get the actual image size.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Base64 { data, .. } => data.as_bytes().len(),
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
}

#[cfg(feature = "image")]
impl TryInto<image::RgbaImage> for Image<'_> {
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
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
    use std::vec;

    #[cfg(feature = "markdown")]
    use crate::markdown::ToMarkdown;

    use super::*;

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
            content: Content::MultiPart(vec![]),
        };
        assert!(message.is_empty());
    }

    #[test]
    fn test_message_tool_use() {
        let tool_use: Message = tool::Use {
            id: "tool_123".into(),
            name: "tool".into(),
            input: serde_json::json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
        .into();

        assert!(tool_use.tool_use().is_some());
    }

    #[test]
    #[cfg(feature = "markdown")]
    // mostly for coverage
    fn test_into_static() {
        let content: Content = "Hello, world!".into();
        let content: Content<'static> = content.into_static();
        assert_eq!(content.to_string(), "Hello, world!");

        let content = Content::SinglePart("Hello, world!".into());
        let content: Content<'static> = content.into_static();
        assert_eq!(content.to_string(), "Hello, world!");

        let block: Block = "Hello, world!".into();
        let block: Block<'static> = block.into_static();
        assert_eq!(block.to_string(), "Hello, world!");

        let image: Image = Image::from_parts(MediaType::Png, "data".into());
        let image: Image<'static> = image.into_static();
        assert_eq!(image.to_string(), "![Image](data:image/png;base64,data)");

        let tool_use: Block = tool::Use {
            id: "tool_123".into(),
            name: "tool".into(),
            input: serde_json::json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
        .into();
        let tool_use: Block<'static> = tool_use.into_static();
        assert_eq!(
            tool_use.markdown_verbose().as_ref(),
            "\n````json\n{\"type\":\"tool_use\",\"id\":\"tool_123\",\"name\":\"tool\",\"input\":{}}\n````"
        );

        let message: Message = (Role::User, "Hello, world!").into();
        let _: Message<'static> = message.into_static();
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
            call: tool::Use {
                id: "tool_123".into(),
                name: "tool".into(),
                input: serde_json::json!({}),
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
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
            #[cfg(feature = "prompt-caching")]
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
            content: Content::SinglePart("Hello, world!".into()),
        };

        assert_eq!(message.len(), 1); // single part

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
            model: crate::AnthropicModel::Sonnet35.into(),
            stop_reason: None,
            stop_sequence: None,
            usage: Default::default(),
        };

        let message: Message = response.into();

        assert_eq!(message.to_string(), "### Assistant\n\nHello, world!");
    }

    #[test]
    fn test_from_role_cow() {
        let text: crate::CowStr<'static> = "Hello, world!".into();
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
        let mut content = Content::SinglePart("Hello, world!".into());
        assert!(!content.is_empty());

        content = Content::MultiPart(vec![]);
        assert!(content.is_empty());
    }

    #[test]
    fn tests_content_unwrap_single_part() {
        let content = Content::SinglePart("Hello, world!".into());
        assert_eq!(content.unwrap_single_part().to_string(), "Hello, world!");
    }

    #[test]
    #[should_panic]
    fn test_content_unwrap_single_part_panics() {
        let content = Content::MultiPart(vec![]);
        content.unwrap_single_part();
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
            call: tool::Use {
                id: "tool_123".into(),
                name: "tool".into(),
                input: serde_json::json!({}),
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
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

        // test user heading, single part
        let message = Message {
            role: Role::User,
            content: Content::SinglePart("Hello, world!".into()),
        };

        let opts = crate::markdown::Options::default()
            .with_tool_use()
            .with_tool_results();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### User\n\nHello, world!"
        );

        // test assistant heading, multi part
        let message = Message {
            role: Role::Assistant,
            content: Content::MultiPart(vec![
                "Hello, world!".into(),
                "How are you?".into(),
            ]),
        };

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Assistant\n\nHello, world!\n\nHow are you?"
        );

        // Test tool result (success)
        let message: Message = tool::Result {
            tool_use_id: "tool_123".into(),
            content: Content::SinglePart("Hello, world!".into()),
            is_error: false,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
        .into();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Tool\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":\"Hello, world!\",\"is_error\":false}\n````"
        );

        // Test tool result (error)
        let message: Message = tool::Result {
            tool_use_id: "tool_123".into(),
            content: Content::SinglePart("Hello, world!".into()),
            is_error: true,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
        .into();

        assert_eq!(
            message.markdown_custom(opts).to_string(),
            "### Error\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":\"Hello, world!\",\"is_error\":true}\n````"
        );
    }

    #[test]
    fn test_block_tool_use() {
        let expected = tool::Use {
            id: "tool_123".into(),
            name: "tool".into(),
            input: serde_json::json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

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
    fn serde_document_base64_pdf() {
        let json = r#"{
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "dGVzdA=="
            },
            "title": "My PDF",
            "citations": {"enabled": true}
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_document());
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn serde_document_plain_text() {
        let json = r#"{
            "type": "document",
            "source": {
                "type": "text",
                "media_type": "text/plain",
                "data": "The grass is green."
            },
            "citations": {"enabled": true}
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_document());
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn serde_document_custom_content() {
        let json = r#"{
            "type": "document",
            "source": {
                "type": "content",
                "content": [
                    {"type": "text", "text": "First chunk"},
                    {"type": "text", "text": "Second chunk"}
                ]
            },
            "title": "Custom Doc",
            "citations": {"enabled": true}
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_document());
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn serde_document_file_id() {
        let json = r#"{
            "type": "document",
            "source": {
                "type": "file",
                "file_id": "file_abc123"
            }
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_document());
    }

    #[test]
    fn serde_document_url() {
        let json = r#"{
            "type": "document",
            "source": {
                "type": "url",
                "url": "https://example.com/doc.pdf"
            }
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_document());
    }

    #[test]
    fn serde_text_with_citations() {
        let json = r#"{
            "type": "text",
            "text": "the grass is green",
            "citations": [
                {
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "My Doc",
                    "start_char_index": 0,
                    "end_char_index": 20
                }
            ]
        }"#;
        let block: Block = serde_json::from_str(json).unwrap();
        assert!(block.is_text());
        if let Block::Text { citations, .. } = &block {
            assert!(citations.is_some());
            assert_eq!(citations.as_ref().unwrap().len(), 1);
        }
        let serialized = serde_json::to_string(&block).unwrap();
        let deserialized: Block = serde_json::from_str(&serialized).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn serde_text_without_citations() {
        let json = r#"{"type": "text", "text": "hello"}"#;
        let block: Block = serde_json::from_str(json).unwrap();
        if let Block::Text { citations, .. } = &block {
            assert!(citations.is_none());
        }
        // citations should be omitted when serialized
        let serialized = serde_json::to_string(&block).unwrap();
        assert!(!serialized.contains("citations"));
    }

    #[test]
    fn document_constructors() {
        let doc = DocumentSource::from_text("hello");
        assert!(matches!(doc, DocumentSource::PlainText { .. }));

        let doc = DocumentSource::from_base64("dGVzdA==");
        assert!(matches!(doc, DocumentSource::Base64 { .. }));

        let doc = DocumentSource::from_url("https://example.com/doc.pdf");
        assert!(matches!(doc, DocumentSource::Url { .. }));

        let doc = DocumentSource::from_content(vec![ContentText {
            text: "chunk".into(),
        }]);
        assert!(matches!(doc, DocumentSource::Content { .. }));

        let doc = DocumentSource::from_file_id("file_abc123");
        assert!(matches!(doc, DocumentSource::File { .. }));
    }

    #[test]
    fn document_into_static() {
        let doc = Block::document_with_citations(DocumentSource::from_text(
            "hello world",
        ));
        let _: Block<'static> = doc.into_static();
    }

    #[test]
    fn document_from_source() {
        let source = DocumentSource::from_text("hello world");
        let block: Block = source.into();
        assert!(block.is_document());
    }

    #[test]
    fn document_len() {
        let block = Block::document(DocumentSource::from_text("hello"));
        assert_eq!(block.len(), 5);
    }

    #[test]
    fn merge_citations_delta() {
        let mut block = Block::text("the grass is green");
        let delta = Delta::CitationsDelta {
            citation: Citation::CharLocation {
                cited_text: "The grass is green.".into(),
                document_index: 0,
                document_title: Some("Doc".into()),
                start_char_index: 0,
                end_char_index: 20,
            },
        };
        block.merge_deltas(std::iter::once(delta)).unwrap();
        if let Block::Text { citations, .. } = &block {
            assert_eq!(citations.as_ref().unwrap().len(), 1);
        } else {
            panic!("Expected text block");
        }
    }
}
