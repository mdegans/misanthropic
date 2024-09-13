//! [`Tool`] and tool [`Choice`] types for the Anthropic Messages API.
use serde::{Deserialize, Serialize};

use crate::request::message::Content;

/// Choice of [`Tool`] for a specific [`request::Message`].
///
/// [`request::Message`]: crate::request::Message
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
pub enum Choice {
    /// Model chooses which tool to use, or no tool at all.
    Auto,
    /// Model must use at least one of the tools provided.
    Any,
    /// Model must use a specific tool.
    Tool {
        /// Name of the tool.
        name: String,
    },
}

/// A tool a model can use while completing a [`request::Message`].
///
/// [`request::Message`]: crate::request::Message
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[derive(Serialize, Deserialize)]
pub struct Tool {
    /// Name of the tool.
    pub name: String,
    /// Description of the tool. The model will use this as documentation.
    pub description: String,
    /// Input schema for the tool. See [tool use guide] for more information.
    /// The schema is not validated by this crate but should conform to the
    /// [JSON Schema] specification.
    ///
    /// [tool use guide]: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
    /// [JSON Schema]: <https://json-schema.org/>
    pub input_schema: serde_json::Value,
    /// Set a cache breakpoint at this tool. See [`Request::cache`] notes for
    /// more information.
    ///
    /// [`Request::cache`]: crate::request::Request::cache
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::request::message::CacheControl>,
}

impl Tool {
    /// Create a cache breakpoint at this [`Tool`] by setting [`cache_control`]
    /// to [`Ephemeral`] See [`Request::cache`] for more information.
    ///
    /// [`cache_control`]: Self::cache_control
    /// [`Ephemeral`]: crate::request::message::CacheControl::Ephemeral
    /// [`Request::cache`]: crate::request::Request::cache
    #[cfg(feature = "prompt-caching")]
    pub fn cache(&mut self) -> &mut Self {
        self.cache_control =
            Some(crate::request::message::CacheControl::Ephemeral);
        self
    }

    /// Returns true if the [`Tool`] has a cache breakpoint set (if
    /// `cache_control` is [`Some`]).
    #[cfg(feature = "prompt-caching")]
    pub fn is_cached(&self) -> bool {
        self.cache_control.is_some()
    }
}

impl TryFrom<serde_json::Value> for Tool {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

/// A tool call made by the model. This should be handled and a response sent
/// back in a [`Block::ToolRes`]
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("\n````json\n{}\n````\n", serde_json::to_string_pretty(self).unwrap())
)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Use {
    /// Unique Id for this tool call.
    pub id: String,
    /// Name of the tool.
    pub name: String,
    /// Input for the tool.
    pub input: serde_json::Value,
    /// Use prompt caching. See [`Block::cache`] for more information.
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::request::message::CacheControl>,
}

impl TryFrom<serde_json::Value> for Use {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Use {
    fn markdown_events_custom<'a>(
        &'a self,
        options: &'a crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        if options.tool_use {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(serde_json::to_string(self).unwrap().into()),
                    Event::End(TagEnd::CodeBlock),
                ]
                .into_iter(),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Use {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

/// Result of [`Tool`] [`Use`] sent back to the [`Assistant`] as a [`User`]
/// [`Message`].
///
/// [`Assistant`]: crate::request::message::Role::Assistant
/// [`User`]: crate::request::message::Role::User
/// [`Message`]: crate::request::Message
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
// On the one hand this can clash with the `Result` type from the standard
// library, but on the other hand it's what the API uses, and I'm trying to
// be as faithful to the API as possible.
pub struct Result {
    /// Unique Id for this tool call.
    pub tool_use_id: String,
    /// Output of the tool.
    pub content: Content,
    /// Whether the tool call result was an error.
    pub is_error: bool,
    /// Use prompt caching. See [`Block::cache`] for more information.
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::request::message::CacheControl>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn use_try_from_value() {
        let value = serde_json::json!({
            "id": "test_id",
            "name": "test_name",
            "input": {
                "test_key": "test_value"
            }
        });

        let use_ = Use::try_from(value).unwrap();

        assert_eq!(use_.id, "test_id");
        assert_eq!(use_.name, "test_name");
        assert_eq!(
            use_.input,
            serde_json::json!({
                "test_key": "test_value"
            })
        );
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_use_markdown() {
        use crate::markdown::ToMarkdown;

        let use_ = Use {
            id: "test_id".into(),
            name: "test_name".into(),
            input: serde_json::json!({
                "test_key": "test_value"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let markdown = use_.markdown_verbose();

        assert_eq!(
            markdown.as_ref(),
            "\n````json\n{\"id\":\"test_id\",\"name\":\"test_name\",\"input\":{\"test_key\":\"test_value\"}}\n````"
        );

        // By default the tool use is not included in the markdown, however this
        // might change in the future. Really, our Display impl could just
        // return an empty &str but this is more consistent with the rest of the
        // crate.
        assert_eq!(use_.to_string(), "");
    }
}
