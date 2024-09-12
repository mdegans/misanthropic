//! A [`request::Message`] and associated types. The API will return a
//! [`response::Message`] with the same type plus additional metadata.
//!
//! [`response::Message`]: crate::response::Message
//! [`request::Message`]: crate::request::Message

use std::borrow::Cow;

use base64::engine::{general_purpose, Engine as _};
use serde::{Deserialize, Serialize};

use crate::{
    response,
    stream::{ContentMismatch, Delta, DeltaError},
    tool,
};

/// Role of the [`Message`] author.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
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
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A message in a [`Request`]. See [`response::Message`] for the version with
/// additional metadata.
///
/// A message is [`Display`]ed as markdown with a [heading] indicating the
/// [`Role`] of the author. [`Image`]s are supported and will be rendered as
/// markdown images with embedded base64 data.
///
/// [`Display`]: std::fmt::Display
/// [`Request`]: crate::Request
/// [`response::Message`]: crate::response::Message
/// [heading]: Message::HEADING
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("{}{}{}{}", Self::HEADING, role, Content::SEP, content)
)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
pub struct Message {
    /// Who is the message from.
    pub role: Role,
    /// The [`Content`] of the message as [one] or [more] [`Block`]s.
    ///
    /// [one]: Content::SinglePart
    /// [more]: Content::MultiPart
    pub content: Content,
}

impl Message {
    /// Heading for the message when rendered as markdown using [`Display`].
    ///
    /// [`Display`]: std::fmt::Display
    #[cfg(not(feature = "markdown"))]
    pub const HEADING: &'static str = "### ";
    /// Heading for the message when rendered as markdown using markdown methods
    /// as well as [`Display`].
    ///
    /// [`Display`]: std::fmt::Display
    #[cfg(feature = "markdown")]
    pub const HEADING: pulldown_cmark::Tag<'static> =
        pulldown_cmark::Tag::Heading {
            level: pulldown_cmark::HeadingLevel::H3,
            id: None,
            classes: vec![],
            attrs: vec![],
        };

    /// Returns the number of [`Content`] [`Block`]s in the message.
    pub fn len(&self) -> usize {
        self.content.len()
    }
}

impl From<response::Message> for Message {
    fn from(message: response::Message) -> Self {
        message.message
    }
}

impl From<(Role, Cow<'static, str>)> for Message {
    fn from((role, content): (Role, Cow<'static, str>)) -> Self {
        Self {
            role,
            content: Content::SinglePart(content),
        }
    }
}

impl From<(Role, &'static str)> for Message {
    fn from((role, content): (Role, &'static str)) -> Self {
        Self {
            role,
            content: Content::SinglePart(Cow::Borrowed(content)),
        }
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Message {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`]. This is [`Content`] markdown plus a heading for the
    /// [`Role`].
    ///
    /// [`Options`]: crate::markdown::Options
    fn markdown_events_custom<'a>(
        &'a self,
        options: &'a crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::Event;

        let content = self.content.markdown_events_custom(options);
        let role = match self.content.last() {
            Some(Block::ToolResult { is_error, .. }) => {
                if *is_error {
                    "Error"
                } else {
                    "Tool"
                }
            }
            _ => self.role.as_str(),
        };
        let heading = [
            Event::Start(Self::HEADING),
            Event::Text(role.into()),
            Event::End(Self::HEADING.to_end()),
        ];

        Box::new(heading.into_iter().chain(content))
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

/// Content of a [`Message`].
#[derive(
    Debug, Serialize, Deserialize, derive_more::From, derive_more::IsVariant,
)]
#[serde(rename_all = "snake_case")]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
pub enum Content {
    /// Single part text-only content.
    SinglePart(Cow<'static, str>),
    /// Multiple content [`Block`]s.
    MultiPart(Vec<Block>),
}

impl Content {
    /// Const constructor for static text content.
    pub const fn text(text: &'static str) -> Self {
        Self::SinglePart(Cow::Borrowed(text))
    }

    /// Returns the number of [`Block`]s in `self`.
    pub fn len(&self) -> usize {
        match self {
            Self::SinglePart(_) => 1,
            Self::MultiPart(parts) => parts.len(),
        }
    }

    /// Returns true if the content is empty.
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
    pub fn unwrap_single_part(self) -> Block {
        match self {
            #[cfg(feature = "prompt-caching")]
            Self::SinglePart(text) => Block::Text {
                text,
                cache_control: None,
            },
            #[cfg(not(feature = "prompt-caching"))]
            Self::SinglePart(text) => Block::Text { text },
            Self::MultiPart(_) => {
                panic!("Content is MultiPart, not SinglePart");
            }
        }
    }

    /// Add a [`Block`] to the [`Content`]. If the [`Content`] is a
    /// [`SinglePart`], it will be converted to a [`MultiPart`]. Returns the
    /// index of the added [`Block`].
    ///
    /// [`SinglePart`]: Content::SinglePart
    /// [`MultiPart`]: Content::MultiPart
    pub fn push<P>(&mut self, part: P) -> usize
    where
        P: Into<Block>,
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
            parts.push(part.into());

            parts.len() - 1
        } else {
            unreachable!("Content is not MultiPart");
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
    /// [`Content`] is empty.
    pub fn last(&self) -> Option<&Block> {
        match self {
            Self::SinglePart(_) => None,
            Self::MultiPart(parts) => parts.last(),
        }
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Content {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom<'a>(
        &'a self,
        options: &'a crate::markdown::Options,
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
impl std::fmt::Display for Content {
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

#[cfg(feature = "markdown")]
impl std::fmt::Display for Content {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl Content {
    /// Separator for multi-part content.
    #[cfg(not(feature = "markdown"))]
    pub const SEP: &'static str = "\n\n";
}

impl From<&'static str> for Content {
    fn from(s: &'static str) -> Self {
        Self::SinglePart(s.into())
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Self::SinglePart(s.into())
    }
}

impl From<Block> for Content {
    fn from(block: Block) -> Self {
        Self::MultiPart(vec![block])
    }
}

/// A [`Content`] [`Block`] of a [`Message`].
#[derive(Debug, Serialize, Deserialize)]
#[cfg_attr(not(feature = "markdown"), derive(derive_more::Display))]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
pub enum Block {
    /// Text content.
    #[serde(alias = "text_delta")]
    #[cfg_attr(not(feature = "markdown"), display("{text}"))]
    Text {
        /// The actual text content.
        text: Cow<'static, str>,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[cfg(feature = "prompt-caching")]
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image content.
    Image {
        #[serde(rename = "source")]
        /// An base64 encoded image.
        image: Image,
        /// Use prompt caching. See [`Block::cache`] for more information.
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
        call: crate::tool::Use,
    },
    /// Result of a [`Tool`] call. This should only be used with the [`User`]
    /// role.
    ///
    /// [`User`]: Role::User
    /// [`Tool`]: crate::Tool
    #[cfg_attr(not(feature = "markdown"), display(""))]
    ToolResult {
        /// Unique Id for this tool call.
        tool_use_id: String,
        /// Output of the tool.
        content: Content,
        /// Whether the tool call result was an error.
        is_error: bool,
        /// Use prompt caching. See [`Block::cache`] for more information.
        #[cfg(feature = "prompt-caching")]
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Block {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

impl Block {
    /// Const constructor for static text content.
    pub const fn new_text(text: &'static str) -> Self {
        Self::Text {
            text: Cow::Borrowed(text),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Merge [`Delta`]s into a [`Block`]. The types must be compatible or this
    /// will return a [`ContentMismatch`] error.
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
                text.to_mut().push_str(&delta);
            }
            (
                Block::ToolUse {
                    call: tool::Use { input, .. },
                },
                Delta::Json { partial_json },
            ) => {
                *input = serde_json::from_str(&partial_json)
                    .map_err(|e| e.to_string())?;
            }
            (this, acc) => {
                let variant_name = match this {
                    Block::Text { .. } => stringify!(Block::Text),
                    Block::ToolUse { .. } => stringify!(Block::ToolUse),
                    Block::ToolResult { .. } => stringify!(Block::ToolResult),
                    Block::Image { .. } => stringify!(Block::Image),
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

    /// Create a cache breakpoint at this block. For this to have any effect,
    /// the full prefix before this point needs to be at least 1024 tokens for
    /// [`Sonnet35`] and [`Opus30`] or 2048 tokens for [`Haiku30`].
    ///
    /// Note: The caching feature is in beta, so this is likely to change.
    #[cfg(feature = "prompt-caching")]
    pub fn cache(&mut self) {
        use crate::tool;

        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::ToolUse {
                call: tool::Use { cache_control, .. },
            }
            | Self::ToolResult { cache_control, .. } => {
                *cache_control = Some(CacheControl::Ephemeral);
            }
        }
    }

    /// Returns true if the block has a `cache_control` breakpoint.
    #[cfg(feature = "prompt-caching")]
    pub const fn is_cached(&self) -> bool {
        use crate::tool;

        match self {
            Self::Text { cache_control, .. }
            | Self::Image { cache_control, .. }
            | Self::ToolUse {
                call: tool::Use { cache_control, .. },
            }
            | Self::ToolResult { cache_control, .. } => cache_control.is_some(),
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
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Block {
    /// Returns an iterator over the text as [`pulldown_cmark::Event`]s using
    /// custom [`Options`].
    ///
    /// [`Options`]: crate::markdown::Options
    #[cfg(feature = "markdown")]
    fn markdown_events_custom<'a>(
        &'a self,
        options: &crate::markdown::Options,
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
                Box::new(
                    Some(Event::Text(image.to_string().into())).into_iter(),
                )
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
        };

        it
    }
}

impl From<&'static str> for Block {
    fn from(text: &'static str) -> Self {
        Self::Text {
            text: text.into(),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl From<String> for Block {
    fn from(text: String) -> Self {
        Self::Text {
            text: text.into(),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl From<Image> for Block {
    fn from(image: Image) -> Self {
        Self::Image {
            image,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

#[cfg(feature = "png")]
impl From<image::RgbaImage> for Block {
    fn from(image: image::RgbaImage) -> Self {
        Image::encode(MediaType::Png, image)
            // Unwrap can never panic unless the PNG encoding fails.
            .unwrap_or_else(|e| {
                eprintln!("Error encoding image: {}", e);
                Image::from_parts(MediaType::Png, String::new())
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

/// Cache control for prompt caching.
#[cfg(feature = "prompt-caching")]
#[derive(Clone, Default, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[serde(tag = "type")]
pub enum CacheControl {
    /// Caches for 5 minutes.
    #[default]
    Ephemeral,
}

/// Image content for [`MultiPart`] [`Message`]s.
///
/// [`MultiPart`]: Content::MultiPart
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
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
        data: String,
    },
}

impl Image {
    /// From raw parts. The data is expected to be base64 encoded compressed
    /// image data or the API will reject it.
    pub fn from_parts(media_type: MediaType, data: String) -> Self {
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
            data: encoder.encode(data),
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
                let data = general_purpose::STANDARD.decode(data)?;
                Ok(image::load_from_memory(&data)?.to_rgba8())
            }
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
impl TryInto<image::RgbaImage> for Image {
    type Error = ImageDecodeError;

    /// An [`Image`] can be decoded into an [`image::RgbaImage`] if it is valid
    /// base64 encoded compressed image data and the image format is supported.
    fn try_into(self) -> Result<image::RgbaImage, Self::Error> {
        self.decode()
    }
}

/// Encoding format for [`Image`]s.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
#[allow(missing_docs)]
pub enum MediaType {
    #[cfg(feature = "jpeg")]
    #[serde(rename = "image/jpeg")]
    Jpeg,
    #[cfg(feature = "png")]
    #[serde(rename = "image/png")]
    Png,
    #[cfg(feature = "gif")]
    #[serde(rename = "image/gif")]
    Gif,
    #[cfg(feature = "webp")]
    #[serde(rename = "image/webp")]
    Webp,
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Use serde to get the string representation.
        write!(f, "{}", serde_json::to_string(self).unwrap())
    }
}

#[cfg(feature = "image")]
impl From<MediaType> for image::ImageFormat {
    /// A [`MediaType`] can always be converted into an [`image::ImageFormat`].
    fn from(value: MediaType) -> image::ImageFormat {
        match value {
            #[cfg(feature = "jpeg")]
            MediaType::Jpeg => image::ImageFormat::Jpeg,
            #[cfg(feature = "png")]
            MediaType::Png => image::ImageFormat::Png,
            #[cfg(feature = "gif")]
            MediaType::Gif => image::ImageFormat::Gif,
            #[cfg(feature = "webp")]
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
            #[cfg(feature = "jpeg")]
            image::ImageFormat::Jpeg => Ok(Self::Jpeg),
            #[cfg(feature = "png")]
            image::ImageFormat::Png => Ok(Self::Png),
            #[cfg(feature = "gif")]
            image::ImageFormat::Gif => Ok(Self::Gif),
            #[cfg(feature = "webp")]
            image::ImageFormat::WebP => Ok(Self::Webp),
            _ => Err(UnsupportedImageFormat(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub const CONTENT_SINGLE: &str = "\"Hello, world!\"";
    pub const CONTENT_MULTI: &str = r#"[
    {"type": "text", "text": "Hello, world!"},
    {"type": "text", "text": "How are you?"}
]"#;

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
    #[cfg(feature = "markdown")]
    fn test_merge_deltas() {
        use crate::markdown::ToMarkdown;

        let mut block: Block = "Hello, world!".into();

        // this is allowed
        block.merge_deltas([]).unwrap();

        let deltas = [
            Delta::Text {
                text: ", how are you?".to_string(),
            },
            Delta::Text {
                text: " I'm fine.".to_string(),
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
            partial_json: r#"{"key": "value"}"#.to_string(),
        }];

        block.merge_deltas(deltas).unwrap();

        // by default tool use is hidden
        let opts = crate::markdown::Options::default().with_tool_use();

        let markdown = block.markdown_custom(&opts);

        assert_eq!(
            markdown.as_ref(),
            "\n````json\n{\"type\":\"tool_use\",\"id\":\"tool_123\",\"name\":\"tool\",\"input\":{\"key\":\"value\"}}\n````"
        );

        // content mismatch
        let deltas = [Delta::Json {
            partial_json: "blabla".to_string(),
        }];
        let mut block = Block::Text {
            text: "Hello, world!".into(),
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

        assert_eq!(message.len(), 1);

        message.content.push("How are you?");

        assert_eq!(message.len(), 2);
    }

    #[test]
    fn test_from_response_message() {
        let response = response::Message {
            message: Message {
                role: Role::User,
                content: Content::text("Hello, world!"),
            },
            id: "msg_123".into(),
            model: crate::Model::Sonnet35,
            stop_reason: None,
            stop_sequence: None,
            usage: Default::default(),
        };

        let message: Message = response.into();

        assert_eq!(message.to_string(), "### User\n\nHello, world!");
    }

    #[test]
    fn test_from_role_cow() {
        let text: Cow<'static, str> = "Hello, world!".into();
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
    fn test_content_from_block() {
        let content: Content = Block::new_text("Hello, world!").into();
        assert_eq!(content.to_string(), "Hello, world!");
    }

    #[test]
    fn test_merge_deltas_error() {
        let mut block: Block = "Hello, world!".into();

        let deltas = [Delta::Json {
            partial_json: "blabla".to_string(),
        }];

        let err = block.merge_deltas(deltas).unwrap_err();

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
            message.markdown_custom(&opts).to_string(),
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
            message.markdown_custom(&opts).to_string(),
            "### Assistant\n\nHello, world!\n\nHow are you?"
        );

        // Test tool result (success)
        let message = Message {
            role: Role::User,
            content: Block::ToolResult {
                tool_use_id: "tool_123".into(),
                content: Content::SinglePart("Hello, world!".into()),
                is_error: false,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
            .into(),
        };

        assert_eq!(
            message.markdown_custom(&opts).to_string(),
            "### Tool\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":\"Hello, world!\",\"is_error\":false}\n````"
        );

        // Test tool result (error)
        let message = Message {
            role: Role::User,
            content: Block::ToolResult {
                tool_use_id: "tool_123".into(),
                content: Content::SinglePart("Hello, world!".into()),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
            .into(),
        };

        assert_eq!(
            message.markdown_custom(&opts).to_string(),
            "### Error\n\n````json\n{\"type\":\"tool_result\",\"tool_use_id\":\"tool_123\",\"content\":\"Hello, world!\",\"is_error\":true}\n````"
        );
    }
}
