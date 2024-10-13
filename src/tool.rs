//! [`Tool`] and tool [`Choice`] types for the Anthropic Messages API.
use std::borrow::Cow;

use crate::prompt::message::Content;
#[allow(unused_imports)]
use crate::Prompt; // without this rustdoc doesn't link to Prompt, even with the
                   // full path and all features enabled. Rustdoc bug?
use serde::{Deserialize, Serialize};

/// Choice of [`Tool`] for a specific [`prompt::message`].
///
/// [`prompt::message`]: crate::prompt::message
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

/// A tool a model can use while completing a [`prompt::Message`].
///
/// [`prompt::Message`]: crate::prompt::Message
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[derive(Serialize, Deserialize)]
pub struct Tool<'a> {
    /// Name of the tool.
    pub name: Cow<'a, str>,
    /// Description of the tool. The model will use this as documentation.
    pub description: Cow<'a, str>,
    /// Input schema for the tool. See [tool use guide] for more information.
    /// The schema is not validated by this crate but should conform to the
    /// [JSON Schema] specification.
    ///
    /// [tool use guide]: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
    /// [JSON Schema]: <https://json-schema.org/>
    pub input_schema: serde_json::Value,
    /// Set a cache breakpoint. See [`Prompt::cache`] for more information.
    ///
    /// [`Prompt::cache`] crate::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

/// A builder for creating a [`Tool`] with some basic validation. See
/// [`Tool::builder`] to create a new builder.
pub struct ToolBuilder<'a> {
    tool: Tool<'a>,
}

impl<'a> ToolBuilder<'a> {
    /// Set the description for the tool.
    pub fn description(mut self, description: impl Into<Cow<'a, str>>) -> Self {
        self.tool.description = description.into();
        self
    }

    /// Set a cache breakpoint at this [`Tool`] by setting [`cache_control`] to
    /// [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// [`cache_control`]: Tool::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    pub fn cache(mut self) -> Self {
        self.tool.cache_control =
            Some(crate::prompt::message::CacheControl::Ephemeral);
        self
    }

    /// Set the [`Tool::input_schema`]. The schema should be a JSON Schema
    /// object conforming to the [JSON Schema] specification like the following
    /// example:
    ///
    /// ```json
    /// {
    ///     "type": "object",
    ///     "properties": {
    ///         "letter": {
    ///             "type": "string",
    ///             "description": "The letter to count",
    ///         },
    ///         "string": {
    ///             "type": "string",
    ///             "description": "The string to count letters in",
    ///         },
    ///     },
    ///     "required": ["letter", "string"],
    /// },
    /// ```
    ///
    /// NOTE: On [`build`], There is some very basic validation done on the
    /// schema to ensure that it is an object with properties and required
    /// fields. This is not exhaustive and does not guarantee that the schema
    /// will be accepted by the API or that the agent will be able to use the
    /// tool.
    ///
    /// [JSON Schema]: <https://json-schema.org/>
    /// [`build`]: ToolBuilder::build
    // TODO: This could be improved by using a JSON Schema library.
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.tool.input_schema = schema;
        self
    }

    /// This will build the [`Tool`] without checking any of the fields. This is
    /// recommended only with static strings.
    pub fn build_unchecked(self) -> Tool<'a> {
        self.tool
    }

    /// Build the tool, validating name, description, and the tool schema.
    fn is_valid_input_schema(
        schema: &serde_json::Value,
    ) -> std::result::Result<(), Cow<'static, str>> {
        let obj = if let Some(obj) = schema.as_object() {
            obj
        } else {
            return Err(format!(
                "Input `schema` not an object: `{}`",
                serde_json::to_string_pretty(schema).unwrap(),
            )
            .into());
        };

        let properties = if let Some(properties) = obj.get("properties") {
            if let Some(o) = properties.as_object() {
                o
            } else {
                return Err("`properties` must be an object.".into());
            }
        } else {
            return Err("Input `schema` must have `properties`.".into());
        };

        let required = if let Some(required) = schema.get("required") {
            if let Some(required) = required.as_array() {
                required
            } else {
                return Err(format!(
                    "Input `schema` `required` not an array: `{}`",
                    serde_json::to_string(required).unwrap()
                )
                .into());
            }
        } else {
            return Err(
                "Input `schema` must have a `required` array of keys.".into()
            );
        };

        for key in required {
            if let Some(key) = key.as_str() {
                if properties.get(key).is_none() {
                    return Err(format!(
                        "`required` key `{key}` not found in `properties.",
                    )
                    .into());
                }
            } else {
                return Err(format!(
                    "`required` key not a string: `{}`",
                    serde_json::to_string(key).unwrap()
                )
                .into());
            }
        }

        Ok(())
    }

    /// This will build the [`Tool`] and do some basic validation on the fields.
    /// This does not guarantee that the tool will be accepted by the API.
    pub fn build(self) -> std::result::Result<Tool<'a>, ToolBuildError> {
        if self.tool.name.is_empty() {
            return Err(ToolBuildError::EmptyName);
        }

        if self.tool.description.is_empty() {
            return Err(ToolBuildError::EmptyDescription);
        }

        if self.tool.input_schema.is_null() {
            return Err(ToolBuildError::EmptyInputSchema);
        }

        if let Err(err_msg) =
            Self::is_valid_input_schema(&self.tool.input_schema)
        {
            return Err(ToolBuildError::InvalidInputSchema {
                message: err_msg,
                schema: self.tool.input_schema,
            });
        }

        Ok(self.tool)
    }
}

/// Errors that can occur when building a [`Tool`] with a [`ToolBuilder`].
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum ToolBuildError {
    #[error("Name unset.")]
    EmptyName,
    #[error("Description unset.")]
    EmptyDescription,
    #[error("Input schema unset.")]
    EmptyInputSchema,
    #[error("Invalid input schema becuase: {message}")]
    InvalidInputSchema {
        schema: serde_json::Value,
        message: Cow<'static, str>,
    },
}

impl<'a> Tool<'a> {
    /// Use a builder to create a new tool with some very basic validation.
    pub fn builder(name: impl Into<Cow<'a, str>>) -> ToolBuilder<'a> {
        ToolBuilder {
            tool: Tool {
                name: name.into(),
                description: Cow::Owned(String::new()),
                input_schema: serde_json::Value::Null,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    /// Create a cache breakpoint at this [`Tool`] by setting [`cache_control`]
    /// to [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// [`cache_control`]: Self::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    pub fn cache(&mut self) -> &mut Self {
        self.cache_control =
            Some(crate::prompt::message::CacheControl::Ephemeral);
        self
    }

    /// Returns true if the [`Tool`] has a cache breakpoint set (if
    /// `cache_control` is [`Some`]).
    #[cfg(feature = "prompt-caching")]
    pub fn is_cached(&self) -> bool {
        self.cache_control.is_some()
    }

    /// Try to convert from a serializable value to a [`Tool`].
    // A blanket impl for TryFrom<T> where T: Serialize would be nice but it
    // would conflict with the blanket impl for TryFrom<Value> where Value:
    // Serialize. This is a bit of a hack but it works.
    pub fn parse<T>(value: T) -> std::result::Result<Self, serde_json::Error>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(value)?;
        value.try_into()
    }
}

impl TryFrom<serde_json::Value> for Tool<'_> {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

/// A [`Tool`] [`Use`] of the model. This should be handled and a response sent
/// back in a [`Block::ToolResult`].
///
/// [`Block::ToolResult`]: crate::prompt::message::Block::ToolResult
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("\n````json\n{}\n````\n", serde_json::to_string_pretty(self).unwrap())
)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Use<'a> {
    /// Unique Id for this tool call.
    ///
    /// ## Notes
    /// - This does not have to be a real id. In your examples you can use any
    ///   string so long as it matches a [`tool::Result::tool_use_id`].
    ///
    /// [`tool::Result::tool_use_id`]: crate::tool::Result::tool_use_id
    pub id: Cow<'a, str>,
    /// Name of the tool.
    pub name: Cow<'a, str>,
    /// Input for the tool.
    pub input: serde_json::Value,
    /// Use prompt caching. See [`Prompt::cache`] for more information.
    ///
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl Use<'_> {
    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> Use<'static> {
        Use {
            id: Cow::Owned(self.id.into_owned()),
            name: Cow::Owned(self.name.into_owned()),
            input: self.input,
            #[cfg(feature = "prompt-caching")]
            cache_control: self.cache_control,
        }
    }
}

impl TryFrom<serde_json::Value> for Use<'_> {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Use<'_> {
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
impl std::fmt::Display for Use<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

/// Result of [`Tool`] [`Use`] sent back to the [`Assistant`] as a [`User`]
/// [`Message`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
/// [`User`]: crate::prompt::message::Role::User
/// [`Message`]: crate::prompt::message
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial_eq", test), derive(PartialEq))]
// On the one hand this can clash with the `Result` type from the standard
// library, but on the other hand it's what the API uses, and I'm trying to
// be as faithful to the API as possible.
pub struct Result<'a> {
    /// Unique Id for this tool call.
    pub tool_use_id: Cow<'a, str>,
    /// Output of the tool.
    pub content: Content<'a>,
    /// Whether the tool call result was an error.
    pub is_error: bool,
    /// Use prompt caching. See [`Prompt::cache`] for more information.
    ///
    /// crate::prompt::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl Result<'_> {
    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> Result<'static> {
        Result {
            tool_use_id: Cow::Owned(self.tool_use_id.into_owned()),
            content: self.content.into_static(),
            is_error: self.is_error,
            #[cfg(feature = "prompt-caching")]
            cache_control: self.cache_control,
        }
    }
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
