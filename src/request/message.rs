//! A [`request::Message`] and associated types. The API will return a
//! [`response::Message`] with the same type plus additional metadata.
//!
//! [`response::Message`]: crate::response::Message
//! [`request::Message`]: crate::request::Message

use base64::engine::{general_purpose, Engine as _};
use serde::{Deserialize, Serialize};

use crate::response;

/// Role of the [`Message`] author.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, derive_more::Display)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// From the user.
    User,
    /// From the AI.
    Assistant,
}

/// A message in a [`Request`]. See [`response::Message`] for the version with
/// additional metadata.
///
/// A message is rendered as markdown with a [heading] indicating the [`Role`]
/// of the author. [`Image`]s are supported and will be rendered as markdown
/// images with embedded base64 data. [`Content`] [`Part`]s are separated by
/// [`Content::SEP`].
///
/// [`Request`]: crate::Request
/// [`response::Message`]: crate::response::Message
/// [heading]: Message::HEADING
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[serde(rename_all = "snake_case")]
#[display("{}{}{}{}", Self::HEADING, role, Content::SEP, content)]
pub struct Message {
    /// Who is providing the content.
    pub role: Role,
    /// The content of the message.
    pub content: Content,
}

impl Message {
    /// Heading for the message when rendered as markdown using [`Display`].
    ///
    /// [`Display`]: std::fmt::Display
    pub const HEADING: &'static str = "### ";
}

impl From<response::Message> for Message {
    fn from(message: response::Message) -> Self {
        message.message
    }
}

/// Content of a [`Message`].
#[derive(Debug, Serialize, Deserialize, derive_more::From)]
#[serde(rename_all = "snake_case")]
#[serde(untagged)]
pub enum Content {
    /// Single part text-only content.
    SinglePart(String),
    /// Multiple content [`Part`]s.
    MultiPart(Vec<Part>),
}

impl Content {
    /// Length of the visible content in bytes, not including metadata like the
    /// [`MediaType`] for images, the [`CacheControl`] for text, [`Tool`]
    /// calls, results, or any separators or headers.
    ///
    /// [`Tool`]: crate::tool::Tool
    pub fn len(&self) -> usize {
        match self {
            Self::SinglePart(string) => string.len(),
            Self::MultiPart(parts) => parts.iter().map(Part::len).sum(),
        }
    }

    /// Returns true if the content is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

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

impl Content {
    /// Separator for multi-part content.
    pub const SEP: &'static str = "\n\n";
}

/// A [`Content`] [`Part`] of a [`Message`], either [`Text`] or [`Image`].
///
/// [`Text`]: Part::Text
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
pub enum Part {
    /// Text content.
    #[serde(alias = "text_delta")]
    #[display("{}", text)]
    Text {
        /// The actual text content.
        text: String,
        /// Use prompt caching. The [`text`] needs to be at least 1024 tokens
        /// for Sonnet 3.5 and Opus 3.0 or 2048 for Haiku 3.0 or this will be
        /// ignored.
        ///
        /// [`text`]: Part::Text::text
        #[cfg(feature = "prompt-caching")]
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheControl>,
    },
    /// Image content.
    Image {
        #[serde(rename = "source")]
        /// An base64 encoded image.
        image: Image,
    },
    /// [`Tool`] call. This should only be used with the [`Assistant`] role.
    ///
    /// [`Assistant`]: Role::Assistant
    /// [`Tool`]: crate::Tool
    #[display("")]
    ToolUse {
        /// Unique Id for this tool call.
        id: String,
        /// Name of the tool.
        name: String,
        /// Input for the tool.
        input: serde_json::Value,
    },
    /// Result of a [`Tool`] call. This should only be used with the [`User`]
    /// role.
    ///
    /// [`User`]: Role::User
    /// [`Tool`]: crate::Tool
    #[display("")]
    ToolResult {
        /// Unique Id for this tool call.
        tool_use_id: String,
        /// Output of the tool.
        content: serde_json::Value,
        /// Whether the tool call result was an error.
        is_error: bool,
    },
}

impl Part {
    /// Length of text or image data in bytes not including metadata like
    /// the [`MediaType`] for images or the [`CacheControl`] for text.
    pub fn len(&self) -> usize {
        match self {
            Self::Text { text, .. } => text.len(),
            Self::Image { image } => match image {
                Image::Base64 { data, .. } => data.len(),
            },
            _ => 0,
        }
    }

    /// Returns true if the part is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<&str> for Part {
    fn from(text: &str) -> Self {
        Self::Text {
            text: text.to_string(),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl From<String> for Part {
    fn from(text: String) -> Self {
        Self::Text {
            text,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }
}

impl From<Image> for Part {
    fn from(image: Image) -> Self {
        Self::Image { image }
    }
}

/// Cache control for prompt caching.
#[cfg(feature = "prompt-caching")]
#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CacheControl {
    /// Ephemeral
    #[default]
    Ephemeral,
}

/// Image content for [`MultiPart`] [`Message`]s.
///
/// [`MultiPart`]: Content::MultiPart
#[derive(Debug, Serialize, Deserialize, derive_more::Display)]
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
        assert_eq!(message.to_string(), "### User\n\nHello, world");
    }
}
