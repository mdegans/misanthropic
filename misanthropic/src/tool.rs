//! [`Tool`] [`Use`] and friends.
use std::{borrow::Cow, hash::Hash};

use serde::{Deserialize, Serialize};

use crate::prompt::message::Content;
#[allow(unused_imports)]
use crate::Prompt;

mod toolbox;
pub use toolbox::ToolBox;

#[cfg(feature = "notepad")]
mod notepad;
#[cfg(feature = "notepad")]
pub use notepad::Notepad;

/// Constrain the [`Assistant`]'s choice of [`Method`]s.
///
/// # Note:
/// - Anthropic calls this a "tool" in the API, but since [`Tool`]s can have
///   multiple [`Method`] in this crate, we use "method" instead.
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Choice {
    /// [`Model`] chooses which [`Method`] of a [`Tool`] to use.
    ///
    /// [`Model`]: crate::model::Model
    #[default]
    Auto,
    /// Model must use at least one of the provided [`Method`]s.
    Any,
    /// Model must use a specific [`Method`]`.
    #[serde(rename = "tool")]
    Method {
        /// The [`Method::name`] to use.
        name: String,
    },
}

/// A `Tool` that the [`Assistant`] can [`Use`]. Tools can have multiple
/// [`Method`]s. Tools should generally go in the [`ToolBox`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[async_trait::async_trait]
pub trait Tool: Send {
    /// [`Tool`] name.
    fn name(&self) -> &str;
    /// Get the [`Method`](s) provided by the [`Tool`].
    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_>;
    /// [`Use`] the [`Tool`], returning a [`tool::Result`].
    ///
    /// [`tool::Result`]: Result
    async fn call<'a>(&mut self, call: Use<'a>) -> Result<'a>;
    /// Serialize tool state to json [`Value`]. [`Null`] if not possible.
    ///
    /// # Note:
    ///
    /// Takes &mut self to allow tools to update internal state during
    /// serialization if needed and because of lifetime issues with `&self`.
    ///
    /// [`Value`]: serde_json::Value
    /// [`Null`]: serde_json::Value::Null
    async fn save_json(&mut self) -> serde_json::Value {
        serde_json::Value::Null
    }
    /// Deserialize state from json [`Value`] if possible.
    // String is used for the message because a boxed error is not Send.
    async fn load_json(
        &mut self,
        _json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        Ok(())
    }
    /// Setup the [`Prompt`] with tools. For example, for the [`Notepad`], add
    /// or update a notepad block in the system prompt.
    ///  
    /// # Note:
    /// - Tool implementations should be [idempotent]. A tool should handle the
    ///   case where it has already been called on a prompt. In general, a tool
    ///   should overwrite or update existing state. It should not add without
    ///   checking if it already exists since the frequency of calls is
    ///   undefined and the [`Prompt`] might grow too large. This could be
    ///   called every turn, every message, or only once, depending on the tool.
    ///
    /// # See Also:
    /// - [`Prompt::apply_tool`]
    ///
    /// [idempotent]: https://en.wikipedia.org/wiki/Idempotence
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    fn apply_to_prompt(
        &self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}

static_assertions::assert_obj_safe!(Tool);
// Ensure Tool is Send (but not Sync) for use in async contexts and ToolBox
static_assertions::assert_impl_all!(dyn Tool: Send);

/// `Method` definition for a [`Tool`] a [`Model`] can [`Use`] while
/// completing a [`prompt::Message`].
///
/// [`prompt::Message`]: crate::prompt::Message
/// [`Model`]: crate::model::Model
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(try_from = "MethodBuilder<'a>")]
#[serde(rename = "tool")]
pub struct Method<'a> {
    /// Name of the function. This should be in a `Tool::function` format.
    pub name: Cow<'a, str>,
    /// Description of the tool. The model will use this as documentation.
    pub description: Cow<'a, str>,
    /// Input schema for the tool. See [tool use guide] for more information.
    /// The schema is not validated by this crate but should conform to the
    /// [JSON Schema] specification.
    ///
    /// [tool use guide]: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
    /// [JSON Schema]: <https://json-schema.org/>
    #[serde(rename = "input_schema")]
    pub schema: serde_json::Value,
    /// Set a cache breakpoint. See [`Prompt::cache`] for more information.
    ///
    /// [`Prompt::cache`] crate::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl<'a> TryFrom<MethodBuilder<'a>> for Method<'a> {
    type Error = ToolBuildError;

    fn try_from(
        builder: MethodBuilder<'a>,
    ) -> std::result::Result<Self, Self::Error> {
        builder.build()
    }
}

/// A builder for creating a [`Method`] with some basic validation. See
/// [`Method::builder`] to create one.
pub struct MethodBuilder<'a> {
    tool: Method<'a>,
}

// MethodBuilder must implement Deserialize but we can't derive it because it
// would recursively require Tool to implement Deserialize, so we have to
// implement it manually. This is a bit ugly, but it works and ensures that
// a Tool is always valid when deserialized.
impl<'de> Deserialize<'de> for MethodBuilder<'_> {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Foreign {
            name: Cow<'static, str>,
            description: Cow<'static, str>,
            input_schema: serde_json::Value,
            #[cfg(feature = "prompt-caching")]
            cache_control: Option<crate::prompt::message::CacheControl>,
        }

        let foreign = Foreign::deserialize(deserializer)?;

        let Foreign {
            name,
            description,
            input_schema,
            #[cfg(feature = "prompt-caching")]
            cache_control,
        } = foreign;

        Ok(MethodBuilder {
            tool: Method {
                name,
                description,
                schema: input_schema,
                #[cfg(feature = "prompt-caching")]
                cache_control,
            },
        })
    }
}

impl<'a> MethodBuilder<'a> {
    /// Set the description for the tool.
    pub fn description(mut self, description: impl Into<Cow<'a, str>>) -> Self {
        self.tool.description = description.into();
        self
    }

    /// Set a cache breakpoint at this [`Method`] by setting [`cache_control`] to
    /// [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// [`cache_control`]: Spec::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    #[cfg(feature = "prompt-caching")]
    pub fn cache(mut self) -> Self {
        self.tool.cache_control =
            Some(crate::prompt::message::CacheControl::Ephemeral);
        self
    }

    /// Set the [`Method::input_schema`]. The schema should be a JSON Schema
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
    /// [`build`]: MethodBuilder::build
    // TODO: This could be improved by using a JSON Schema library.
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.tool.schema = schema;
        self
    }

    /// Add a string parameter to the schema.
    pub fn string_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "string", required)
    }

    /// Add a number parameter to the schema.
    pub fn number_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "number", required)
    }

    /// Add a boolean parameter to the schema.
    pub fn boolean_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "boolean", required)
    }

    /// Helper method to add a parameter to the schema.
    fn add_param(
        mut self,
        name: &str,
        description: &str,
        param_type: &str,
        required: bool,
    ) -> Self {
        // Initialize schema if it's null
        if self.tool.schema.is_null() {
            self.tool.schema = serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            });
        }

        // Add the property
        if let Some(properties) = self
            .tool
            .schema
            .get_mut("properties")
            .and_then(|p| p.as_object_mut())
        {
            properties.insert(
                name.to_string(),
                serde_json::json!({
                    "type": param_type,
                    "description": description
                }),
            );
        }

        // Add to required array if needed
        if required {
            if let Some(required_array) = self
                .tool
                .schema
                .get_mut("required")
                .and_then(|r| r.as_array_mut())
            {
                required_array
                    .push(serde_json::Value::String(name.to_string()));
            }
        }

        self
    }

    /// This will build the [`Method`] without checking any of the fields. This is
    /// recommended only with static strings.
    pub fn build_unchecked(self) -> Method<'a> {
        self.tool
    }

    /// Build the tool, validating name, description, and the tool schema.
    fn is_valid_input_schema(
        schema: &serde_json::Value,
    ) -> std::result::Result<(), Cow<'static, str>> {
        let obj = if let Some(obj) = schema.as_object() {
            if obj.is_empty() {
                return Err("Input `schema` is an empty object.".into());
            }

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

    /// This will build the [`Method`] and do some basic validation on the fields.
    /// This does not guarantee that the tool will be accepted by the API.
    pub fn build(self) -> std::result::Result<Method<'a>, ToolBuildError> {
        if self.tool.name.is_empty() {
            return Err(ToolBuildError::EmptyName);
        }

        if self.tool.description.is_empty() {
            return Err(ToolBuildError::EmptyDescription);
        }

        if self.tool.schema.is_null() {
            return Err(ToolBuildError::EmptyInputSchema);
        }

        if let Err(err_msg) = Self::is_valid_input_schema(&self.tool.schema) {
            return Err(ToolBuildError::InvalidInputSchema {
                message: err_msg,
                schema: self.tool.schema,
            });
        }

        Ok(self.tool)
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields. If they are already owned, this is a no-op.
    pub fn into_static(self) -> Method<'static> {
        Method {
            name: Cow::Owned(self.tool.name.into_owned()),
            description: Cow::Owned(self.tool.description.into_owned()),
            schema: self.tool.schema,
            #[cfg(feature = "prompt-caching")]
            cache_control: self.tool.cache_control,
        }
    }
}

/// Errors that can occur when building a [`Method`] with a [`MethodBuilder`].
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

impl<'a> Method<'a> {
    /// Use a builder to create a new tool with some very basic validation.
    pub fn builder(name: impl Into<Cow<'a, str>>) -> MethodBuilder<'a> {
        MethodBuilder {
            tool: Method {
                name: name.into(),
                description: Cow::Owned(String::new()),
                schema: serde_json::Value::Null,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    /// Create a simple method with just a name and description.
    /// Uses an empty object schema with no required fields.
    pub fn simple(
        name: impl Into<Cow<'a, str>>,
        description: impl Into<Cow<'a, str>>,
    ) -> Self {
        Method {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Create a method that takes a single string parameter.
    pub fn with_string_param(
        name: impl Into<Cow<'a, str>>,
        description: impl Into<Cow<'a, str>>,
        param_name: &str,
        param_description: &str,
        required: bool,
    ) -> Self {
        let required_array = if required { vec![param_name] } else { vec![] };

        Method {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    param_name: {
                        "type": "string",
                        "description": param_description
                    }
                },
                "required": required_array
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Create a cache breakpoint at this [`Method`] by setting [`cache_control`]
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

    /// Returns true if the [`Method`] has a cache breakpoint set (if
    /// `cache_control` is [`Some`]).
    #[cfg(feature = "prompt-caching")]
    pub fn is_cached(&self) -> bool {
        self.cache_control.is_some()
    }

    /// Try to convert from a serializable value to a [`Method`].
    // A blanket impl for TryFrom<T> where T: Serialize would be nice but it
    // would conflict with the blanket impl for TryFrom<Value> where Value:
    // Serialize. This is a bit of a hack but it works.
    pub fn from_serializable<T>(
        value: T,
    ) -> std::result::Result<Method<'a>, serde_json::Error>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(value)?;
        value.try_into()
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> Method<'static> {
        Method {
            name: Cow::Owned(self.name.into_owned()),
            description: Cow::Owned(self.description.into_owned()),
            schema: self.schema,
            #[cfg(feature = "prompt-caching")]
            cache_control: self.cache_control,
        }
    }
}

impl TryFrom<serde_json::Value> for Method<'static> {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        let builder: MethodBuilder<'static> = serde_json::from_value(value)?;
        builder
            .build()
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

/// `Method` [`Use`] of the model. This should be handled and a response sent
/// back in a [`Block::ToolResult`].
///
/// [`Block::ToolResult`]: crate::prompt::message::Block::ToolResult
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("\n````json\n{}\n````\n", serde_json::to_string_pretty(self).unwrap())
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
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
impl<'a> crate::markdown::ToMarkdown<'a> for Use<'a> {
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        if options.tool_use {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(self).unwrap().into(),
                    ),
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

/// Result of [`Method`] [`Use`] sent back to the [`Assistant`] as a [`User`]
/// [`Message`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
/// [`User`]: crate::prompt::message::Role::User
/// [`Message`]: crate::prompt::message
#[derive(Clone, Debug, Serialize, Deserialize, Hash, derive_more::Display)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
// FIXME: On the one hand this can clash with the `Result` type from the
// standard library, but on the other hand it's what the API uses. We should
// probably rename this to avoid confusion, since it is confusing.
#[display("{}", self.content)]
pub struct Result<'a> {
    /// Unique Id for this tool call.
    pub tool_use_id: Cow<'a, str>,
    /// Output of the tool. If this is an error message it should be written
    /// with the [`Assistant`]'s perspective in mind. It should tell the
    /// [`Assistant`] what went wrong and how they can try to fix it.
    pub content: Content<'a>,
    /// Is the result an error message?
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

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for Result<'a> {
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        if options.tool_results {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(self).unwrap().into(),
                    ),
                    Event::End(TagEnd::CodeBlock),
                ]
                .into_iter(),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_simple() {
        let method = Method::simple("test_method", "A simple test method");

        assert_eq!(method.name, "test_method");
        assert_eq!(method.description, "A simple test method");
        assert_eq!(
            method.schema,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        );
    }

    #[test]
    fn test_method_with_string_param() {
        let method = Method::with_string_param(
            "get_weather",
            "Get weather for a location",
            "location",
            "The city name",
            true,
        );

        assert_eq!(method.name, "get_weather");
        assert_eq!(method.description, "Get weather for a location");
        assert_eq!(
            method.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            })
        );
    }

    #[test]
    fn test_method_builder_param_helpers() {
        let method = Method::builder("test_method")
            .description("Test method with multiple params")
            .string_param("name", "A person's name", true)
            .number_param("age", "A person's age", false)
            .boolean_param("active", "Whether the person is active", true)
            .build()
            .unwrap();

        assert_eq!(method.name, "test_method");
        assert_eq!(method.description, "Test method with multiple params");

        let expected_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A person's name"
                },
                "age": {
                    "type": "number",
                    "description": "A person's age"
                },
                "active": {
                    "type": "boolean",
                    "description": "Whether the person is active"
                }
            },
            "required": ["name", "active"]
        });

        assert_eq!(method.schema, expected_schema);
    }

    #[test]
    fn test_method_builder_param_helpers_with_existing_schema() {
        // Start with an existing schema and add to it
        let method = Method::builder("test_method")
            .description("Test method")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "existing": {
                        "type": "string",
                        "description": "An existing property"
                    }
                },
                "required": ["existing"]
            }))
            .string_param("new_param", "A new parameter", true)
            .build()
            .unwrap();

        let properties = method.schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("existing"));
        assert!(properties.contains_key("new_param"));

        let required = method.schema["required"].as_array().unwrap();
        assert!(required
            .contains(&serde_json::Value::String("existing".to_string())));
        assert!(required
            .contains(&serde_json::Value::String("new_param".to_string())));
    }

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
            "\n````json\n{\n  \"id\": \"test_id\",\n  \"name\": \"test_name\",\n  \"input\": {\n    \"test_key\": \"test_value\"\n  }\n}\n````"
        );

        // By default the tool use is not included in the markdown, however this
        // might change in the future. Really, our Display impl could just
        // return an empty &str but this is more consistent with the rest of the
        // crate.
        assert_eq!(use_.to_string(), "");
    }

    #[test]
    fn test_tool_schema_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "The letter to count",
                },
                "string": {
                    "type": "string",
                    "description": "The string to count letters in",
                },
            },
            "required": ["letter", "string"],
        });

        assert!(MethodBuilder::is_valid_input_schema(&schema).is_ok());

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "The letter to count",
                },
                "string": {
                    "type": "string",
                    "description": "The string to count letters in",
                },
            },
            "required": "letter",
        });

        assert!(MethodBuilder::is_valid_input_schema(&schema).is_err());
    }

    #[test]
    fn test_build() {
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build()
            .unwrap();

        assert_eq!(tool.name, "test_name");
        assert_eq!(tool.description, "test_description");
        assert_eq!(
            tool.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            })
        );

        // Test error cases
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": "letter",
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // input schema not an object
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::Value::String("blah".into()))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // Properties not an object
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": "blah",
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // Schema does not have properties
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // Schema does not have `required` keys (empty array allowed, but it
        // must be present)
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // required keys not found in properties
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string", "foo"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // required keys not strings
        let tool = Method::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": [1, 2],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // missing schema
        let tool = Method::builder("test_name")
            .description("test_description")
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyInputSchema)));

        // with missing names and descriptions
        let tool = Method::builder("")
            .description("foo")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyName)));

        let tool = Method::builder("foo")
            .description("")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyDescription)));
    }

    #[test]
    fn test_choice_serde() {
        let choice = Choice::Auto;
        let json = serde_json::to_string(&choice).unwrap();
        let choice2: Choice = serde_json::from_str(&json).unwrap();
        assert_eq!(choice, choice2);

        let choice = Choice::Any;
        let json = serde_json::to_string(&choice).unwrap();
        let choice2: Choice = serde_json::from_str(&json).unwrap();
        assert_eq!(choice, choice2);

        let choice = Choice::Method {
            name: "test_name".into(),
        };
        let json = serde_json::to_string(&choice).unwrap();
        let choice2: Choice = serde_json::from_str(&json).unwrap();
        assert_eq!(choice, choice2);
    }

    #[test]
    fn test_result_serde() {
        let result = Result {
            tool_use_id: "test_id".into(),
            content: "test_content".into(),
            is_error: false,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let result2: Result = serde_json::from_str(&json).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn test_result_into_static() {
        let result = Result {
            tool_use_id: "test_id".into(),
            content: "test_content".into(),
            is_error: false,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = result.into_static();

        assert_eq!(result.tool_use_id, "test_id");
        assert_eq!(result.content.to_string(), "test_content");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_from_serializable() {
        let tool = Method::from_serializable(serde_json::json!({
            "name": "test_name",
            "description": "test_description",
            "input_schema": {
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            },
        }))
        .unwrap();

        assert_eq!(tool.name, "test_name");
        assert_eq!(tool.description, "test_description");
        assert_eq!(
            tool.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            })
        );

        // Test invalid schema. Comprehensive testing of this is in the builder
        // tests. This just makes sure that the error is propagated.
        let tool = Method::from_serializable(serde_json::json!({
            "name": "test_name",
            "description": "test_description",
            "input_schema": {
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                // should be an array
                "required": "letter",
            },
        }));

        assert!(tool.is_err());
    }
}
