use std::ops::Deref;

use pulldown_cmark::HeadingLevel;
use serde::{Deserialize, Serialize};

/// Default [`Options`]
pub const DEFAULT_OPTIONS: Options = Options {
    inner: pulldown_cmark::Options::empty(),
    tool_use: false,
    tool_results: false,
    system: false,
    attrs: false,
    heading_level: None,
};

/// Verbose [`Options`]
pub const VERBOSE_OPTIONS: Options = Options {
    inner: pulldown_cmark::Options::empty(),
    tool_use: true,
    tool_results: true,
    system: true,
    attrs: true,
    heading_level: None,
};

/// A static reference to the default [`Options`].
pub static DEFAULT_OPTIONS_REF: &'static Options = &DEFAULT_OPTIONS;

/// A static reference to the verbose [`Options`].
pub static VERBOSE_OPTIONS_REF: &'static Options = &VERBOSE_OPTIONS;

mod serde_inner {
    use super::*;

    pub fn serialize<S>(
        options: &pulldown_cmark::Options,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        options.bits().serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<pulldown_cmark::Options, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(pulldown_cmark::Options::from_bits_truncate(bits))
    }
}

/// Options for parsing, generating, and rendering [`Markdown`].
#[derive(Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Options {
    /// Inner [`pulldown_cmark::Options`].
    #[serde(with = "serde_inner")]
    pub inner: pulldown_cmark::Options,
    /// Whether to include the system prompt
    pub system: bool,
    /// Whether to include tool uses.
    pub tool_use: bool,
    /// Whether to include tool results.
    pub tool_results: bool,
    /// Whether to include attributes. Useful when converting to HTML.
    ///
    /// This adds:
    /// - `role` attribute to the [`Prompt`] and [`Message`]s. Possible values
    ///   are:
    ///   - `system` - for the system prompt
    ///   - `assistant` - for generated messages
    ///   - `tool` - for tool results
    ///   - `user` - for user messages
    ///   - `error` - for errors
    ///
    /// [`Prompt`]: crate::prompt::Prompt
    /// [`Message`]: crate::prompt::Message
    pub attrs: bool,
    /// Heading level to begin at (optional)
    pub heading_level: Option<HeadingLevel>,
}

impl Options {
    /// Maximum verbosity
    pub fn verbose() -> Self {
        VERBOSE_OPTIONS
    }

    /// Set [`tool_use`] to true
    ///
    /// [`tool_use`]: Options::tool_use
    pub fn with_tool_use(mut self) -> Self {
        self.tool_use = true;
        self
    }

    /// Set [`tool_results`] to true
    ///
    /// [`tool_results`]: Options::tool_results
    pub fn with_tool_results(mut self) -> Self {
        self.tool_results = true;
        self
    }

    /// Set [`system`] to true
    ///
    /// [`system`]: Options::system
    pub fn with_system(mut self) -> Self {
        self.system = true;
        self
    }
}

#[cfg(feature = "markdown")]
impl From<pulldown_cmark::Options> for Options {
    fn from(inner: pulldown_cmark::Options) -> Self {
        Options {
            inner,
            ..Default::default()
        }
    }
}

/// A valid, immutable, Markdown string. It has been parsed and rendered. It can
/// be [`Display`]ed or dereferenced as a [`str`].
///
/// [`Display`]: std::fmt::Display
#[derive(derive_more::Display)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[display("{text}")]
pub struct Markdown {
    text: String,
}

impl Into<String> for Markdown {
    fn into(self) -> String {
        self.text
    }
}

impl AsRef<str> for Markdown {
    fn as_ref(&self) -> &str {
        self.deref().as_ref()
    }
}

impl std::borrow::Borrow<str> for Markdown {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl std::ops::Deref for Markdown {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl<'a, T> From<T> for Markdown
where
    T: Iterator<Item = pulldown_cmark::Event<'a>>,
{
    fn from(events: T) -> Self {
        let mut text = String::new();

        // Unwrap can never panic because the formatter for `String` never
        // returns an error.
        let _ = pulldown_cmark_to_cmark::cmark(events, &mut text).unwrap();

        Markdown { text }
    }
}

#[cfg(any(test, feature = "partial_eq"))]
impl PartialEq<str> for Markdown {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

/// A trait for types that can be converted to [`Markdown`]
///
/// # Note
///
/// - Any of these methods returning an iterator of [`pulldown_cmark::Event`]s
/// can be used to render to html using [`pulldown_cmark::html::push_html`] and
/// other similar functions.
/// - Implementers should guarantee tags are properly closed and nested.
pub trait ToMarkdown {
    /// Render the type to a [`Markdown`] string with [`DEFAULT_OPTIONS`].
    fn markdown(&self) -> Markdown {
        self.markdown_events().into()
    }

    /// Render the type to a [`Markdown`] string with custom [`Options`].
    fn markdown_custom(&self, options: Options) -> Markdown {
        self.markdown_events_custom(options).into()
    }

    /// Render the type to a [`Markdown`] string with maximum verbosity.
    fn markdown_verbose(&self) -> Markdown {
        self.markdown_custom(*VERBOSE_OPTIONS_REF)
    }

    /// Render the markdown to a type implementing [`std::fmt::Write`] with
    /// [`DEFAULT_OPTIONS`].
    fn write_markdown(
        &self,
        writer: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result {
        self.write_markdown_custom(writer, *DEFAULT_OPTIONS_REF)
    }

    /// Render the markdown to a type implementing [`std::fmt::Write`] with
    /// custom [`Options`].
    fn write_markdown_custom(
        &self,
        writer: &mut dyn std::fmt::Write,
        options: Options,
    ) -> std::fmt::Result {
        use pulldown_cmark_to_cmark::cmark;

        let events = self.markdown_events_custom(options);
        let _ = cmark(events, writer)?;
        Ok(())
    }

    /// Return an iterator of [`pulldown_cmark::Event`]s with
    /// [`DEFAULT_OPTIONS`].
    fn markdown_events<'a>(
        &'a self,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        self.markdown_events_custom(*DEFAULT_OPTIONS_REF)
    }

    /// Return an iterator of [`pulldown_cmark::Event`]s with custom
    /// [`Options`].
    fn markdown_events_custom<'a>(
        &'a self,
        options: Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a>;
}

static_assertions::assert_obj_safe!(ToMarkdown);

impl Default for Options {
    fn default() -> Self {
        DEFAULT_OPTIONS
    }
}

#[cfg(test)]
mod tests {
    use crate::prompt::{message::Role, Message};

    use super::*;

    use std::borrow::Borrow;

    #[test]
    fn test_options_serde() {
        let options = Options::default();

        let json = serde_json::to_string(&options).unwrap();
        let options2: Options = serde_json::from_str(&json).unwrap();

        assert!(options == options2);
    }

    #[test]
    fn test_options_from_pulldown() {
        let inner = pulldown_cmark::Options::empty();
        let options: Options = inner.into();
        assert_eq!(options.inner, inner);
    }

    #[test]
    fn test_options_verbose() {
        let options = Options::verbose();
        assert!(options.tool_use);
        assert!(options.tool_results);
        assert!(options.system);
    }

    #[test]
    fn test_markdown() {
        let expected = "Hello, **world**!";
        let events = pulldown_cmark::Parser::new(&expected);
        let markdown: Markdown = events.into();
        let actual: &str = markdown.borrow();
        assert_eq!(actual, expected);
        assert!(&markdown == expected);
        let markdown: String = markdown.into();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn test_message_markdown() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        assert_eq!(
            message.markdown().as_ref(),
            "### User\n\nHello, **world**!"
        );
    }
}
