//! [`Tool`] [`Use`] and friends.
use std::{borrow::Cow, hash::Hash};

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::Prompt;
use crate::prompt::message::Content;

mod toolbox;
pub use toolbox::ToolBox;

mod typed;
pub use typed::{ErasedMethod, Method, Methods, ToolArgs, Typed};

/// `#[derive(ToolArgs)]` — co-located with the [`ToolArgs`] trait (same path,
/// different namespaces) so a single `use misanthropic::tool::ToolArgs;` brings
/// in both, as with `serde`'s `Serialize`.
#[cfg(feature = "derive")]
pub use misanthropic_derive::ToolArgs;

/// `#[tool]` — attribute on an `impl` block that generates the [`Method`] /
/// [`ToolArgs`] / [`Methods`] wiring from `#[method]`-tagged async fns.
#[cfg(feature = "derive")]
pub use misanthropic_derive::tool;

#[cfg(feature = "memory-palace")]
mod memory_palace;
#[cfg(feature = "memory-palace")]
pub use memory_palace::MemoryPalace;
#[cfg(all(test, feature = "memory-palace"))]
mod memory_palace_tests;

#[cfg(feature = "notepad")]
mod notepad;
#[cfg(feature = "notepad")]
pub use notepad::Notepad;

/// Constrain the [`Assistant`]'s choice of [`MethodDef`]s.
///
/// # Note:
/// - Anthropic calls this a "tool" in the API, but since [`Tool`]s can have
///   multiple [`MethodDef`] in this crate, we use "method" instead.
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case", tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Choice {
    /// [`Model`] chooses which [`MethodDef`] of a [`Tool`] to use.
    ///
    /// [`Model`]: crate::model::Model
    #[default]
    Auto,
    /// Model must use at least one of the provided [`MethodDef`]s.
    Any,
    /// Model must use a specific [`MethodDef`]`.
    #[serde(rename = "tool")]
    Method {
        /// The [`MethodDef::name`] to use.
        name: String,
    },
}

/// A `Tool` that the [`Assistant`] can [`Use`]. Tools can have multiple
/// [`MethodDef`]s. Tools should generally go in the [`ToolBox`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[async_trait::async_trait]
pub trait Tool: Send {
    /// [`Tool`] name.
    fn name(&self) -> &str;
    /// Get the [`MethodDef`](s) provided by the [`Tool`].
    fn definitions(&self) -> Vec<MethodDef<'static>>;
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
    /// For tools with external persistence (like databases), this should
    /// only serialize configuration/connection info, not the full state.
    ///
    /// [`Value`]: serde_json::Value
    /// [`Null`]: serde_json::Value::Null
    async fn save_json(&mut self) -> serde_json::Value {
        serde_json::Value::Null
    }
    /// Deserialize state from json [`Value`] if possible.
    ///
    /// For tools with external persistence, this should restore configuration
    /// and ensure the external state is accessible/initialized.
    // String is used for the message because a boxed error is not Send.
    async fn load_json(
        &mut self,
        _json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Called once when the tool is first added to a prompt or toolbox.
    /// Use this to set up initial context, instructions, or static content.
    ///
    /// # Note:
    /// - This is called only once per tool lifetime in a conversation
    /// - Use for setting up tool instructions, initial context blocks
    /// - Should be idempotent in case called multiple times
    async fn on_init(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Called before each turn/message exchange.
    /// Use this to update dynamic context, recent state, or per-turn information.
    ///
    /// # Note:
    /// - This is called before each user message or assistant response
    /// - Use for updating dynamic content like recent memories, current state
    /// - Should efficiently update existing content rather than appending
    async fn on_turn(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

static_assertions::assert_obj_safe!(Tool);
// Ensure Tool is Send (but not Sync) for use in async contexts and ToolBox
static_assertions::assert_impl_all!(dyn Tool: Send);

/// `MethodDef` definition for a [`Tool`] a [`Model`] can [`Use`] while
/// completing a [`prompt::Message`].
///
/// [`prompt::Message`]: crate::prompt::Message
/// [`Model`]: crate::model::Model
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(try_from = "MethodBuilder<'a>")]
#[serde(rename = "tool")]
pub struct MethodDef<'a> {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
    /// When `Some(true)`, enables [strict tool use] — the API uses
    /// grammar-constrained decoding so [`Use::input`] is guaranteed to
    /// validate against [`schema`]. Defaults to `None` (best-effort
    /// adherence only).
    ///
    /// Strict mode is compatible with [`Prompt::output_config`] — the API
    /// accepts both in the same request, but any given response turn
    /// emits either a `tool_use` block or the constrained output text,
    /// not both.
    ///
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    /// [`Use::input`]: crate::tool::Use::input
    /// [`schema`]: MethodDef::schema
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

#[cfg(feature = "markdown")]
impl<'a> crate::markdown::ToMarkdown<'a> for MethodDef<'a> {
    fn markdown_events_custom(
        &'a self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'a>> + 'a> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        // Can't panic because derived Serialize
        let mut payload = serde_json::to_value(self).unwrap();
        // Can't panic because we know it's an object
        payload.as_object_mut().unwrap().remove("cache_control");
        payload.as_object_mut().unwrap().remove("strict");

        if options.tool_use {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(&payload).unwrap().into(),
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

impl<'a> TryFrom<MethodBuilder<'a>> for MethodDef<'a> {
    type Error = ToolBuildError;

    fn try_from(
        builder: MethodBuilder<'a>,
    ) -> std::result::Result<Self, Self::Error> {
        builder.build()
    }
}

/// A builder for creating a [`MethodDef`] with some basic validation. See
/// [`MethodDef::builder`] to create one.
pub struct MethodBuilder<'a> {
    tool: MethodDef<'a>,
}

// `MethodDef` is annotated with `#[serde(try_from = "MethodBuilder<'a>")]`, so
// deserializing a `MethodDef` routes through `MethodBuilder::deserialize` and
// then `MethodBuilder::build`. If we derived `Deserialize` on
// `MethodBuilder`, serde would generate an impl that defers to
// `MethodDef::deserialize`, which in turn calls back into
// `MethodBuilder::deserialize` — an infinite loop. So we hand-roll it via
// a private `Foreign` helper struct that owns the actual field mapping.
// Every public field on `MethodDef` must have a matching entry here.
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
            #[serde(default)]
            cache_control: Option<crate::prompt::message::CacheControl>,
            #[serde(default)]
            strict: Option<bool>,
        }

        let foreign = Foreign::deserialize(deserializer)?;

        let Foreign {
            name,
            description,
            input_schema,
            cache_control,
            strict,
        } = foreign;

        Ok(MethodBuilder {
            tool: MethodDef {
                name,
                description,
                schema: input_schema,
                cache_control,
                strict,
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

    /// Set the [`strict`] flag on the [`MethodDef`], enabling [strict tool
    /// use] (grammar-constrained decoding of tool inputs).
    ///
    /// [`strict`]: MethodDef::strict
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn strict(mut self, strict: bool) -> Self {
        self.tool.strict = Some(strict);
        self
    }

    /// Set a cache breakpoint at this [`MethodDef`] by setting [`cache_control`] to
    /// [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// [`cache_control`]: Spec::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    pub fn cache(mut self) -> Self {
        self.tool.cache_control =
            Some(crate::prompt::message::CacheControl::ephemeral());
        self
    }

    /// Set the [`MethodDef::input_schema`]. The schema should be a JSON Schema
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

    /// This will build the [`MethodDef`] without checking any of the fields. This is
    /// recommended only with static strings.
    pub fn build_unchecked(self) -> MethodDef<'a> {
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

        // `properties` is optional: a no-arg method (e.g. a `clear` with no
        // fields) has none. Validate its shape only when present, treating
        // absence as an empty property set.
        let empty = serde_json::Map::new();
        let properties = match obj.get("properties") {
            Some(serde_json::Value::Object(o)) => o,
            Some(_) => return Err("`properties` must be an object.".into()),
            None => &empty,
        };

        // `required` is optional per JSON Schema. Validate only when present;
        // every listed key must exist in `properties`.
        if let Some(required) = obj.get("required") {
            let required = required.as_array().ok_or_else(|| {
                format!(
                    "Input `schema` `required` not an array: `{}`",
                    serde_json::to_string(required).unwrap()
                )
            })?;

            for key in required {
                match key.as_str() {
                    Some(key) if properties.contains_key(key) => {}
                    Some(key) => {
                        return Err(format!(
                            "`required` key `{key}` not found in `properties`.",
                        )
                        .into());
                    }
                    None => {
                        return Err(format!(
                            "`required` key not a string: `{}`",
                            serde_json::to_string(key).unwrap()
                        )
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// This will build the [`MethodDef`] and do some basic validation on the fields.
    /// This does not guarantee that the tool will be accepted by the API.
    pub fn build(self) -> std::result::Result<MethodDef<'a>, ToolBuildError> {
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
    pub fn into_static(self) -> MethodDef<'static> {
        MethodDef {
            name: Cow::Owned(self.tool.name.into_owned()),
            description: Cow::Owned(self.tool.description.into_owned()),
            schema: self.tool.schema,
            cache_control: self.tool.cache_control,
            strict: self.tool.strict,
        }
    }
}

/// Errors that can occur when building a [`MethodDef`] with a [`MethodBuilder`].
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

impl<'a> MethodDef<'a> {
    /// Use a builder to create a new tool with some very basic validation.
    pub fn builder(name: impl Into<Cow<'a, str>>) -> MethodBuilder<'a> {
        MethodBuilder {
            tool: MethodDef {
                name: name.into(),
                description: Cow::Owned(String::new()),
                schema: serde_json::Value::Null,
                cache_control: None,
                strict: None,
            },
        }
    }

    /// Create a simple method with just a name and description.
    /// Uses an empty object schema with no required fields.
    pub fn simple(
        name: impl Into<Cow<'a, str>>,
        description: impl Into<Cow<'a, str>>,
    ) -> Self {
        MethodDef {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            cache_control: None,
            strict: None,
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

        MethodDef {
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
            cache_control: None,
            strict: None,
        }
    }

    /// Create a cache breakpoint at this [`MethodDef`] by setting [`cache_control`]
    /// to [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// Uses the default 5-minute TTL. For a 1-hour TTL, use
    /// [`cache_1h`](Self::cache_1h).
    ///
    /// [`cache_control`]: Self::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    pub fn cache(&mut self) -> &mut Self {
        self.cache_with(crate::prompt::message::CacheControl::ephemeral())
    }

    /// Create a 1-hour cache breakpoint at this [`MethodDef`]. Behaves
    /// identically to [`cache`](Self::cache) but uses
    /// [`CacheControl::one_hour`](crate::prompt::message::CacheControl::one_hour).
    pub fn cache_1h(&mut self) -> &mut Self {
        self.cache_with(crate::prompt::message::CacheControl::one_hour())
    }

    /// Create a cache breakpoint at this [`MethodDef`] with a caller-provided
    /// [`CacheControl`](crate::prompt::message::CacheControl).
    pub fn cache_with(
        &mut self,
        cache_control: crate::prompt::message::CacheControl,
    ) -> &mut Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Returns true if the [`MethodDef`] has a cache breakpoint set (if
    /// `cache_control` is [`Some`]).
    pub fn is_cached(&self) -> bool {
        self.cache_control.is_some()
    }

    /// Set the [`strict`] flag on the [`MethodDef`], enabling [strict tool
    /// use]. See [`MethodBuilder::strict`] for the builder variant.
    ///
    /// [`strict`]: MethodDef::strict
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn strict(&mut self, strict: bool) -> &mut Self {
        self.strict = Some(strict);
        self
    }

    /// Returns `true` if [strict tool use] is enabled on this [`MethodDef`].
    ///
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn is_strict(&self) -> bool {
        self.strict == Some(true)
    }

    /// Try to convert from a serializable value to a [`MethodDef`].
    // A blanket impl for TryFrom<T> where T: Serialize would be nice but it
    // would conflict with the blanket impl for TryFrom<Value> where Value:
    // Serialize. This is a bit of a hack but it works.
    pub fn from_serializable<T>(
        value: T,
    ) -> std::result::Result<MethodDef<'a>, serde_json::Error>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(value)?;
        value.try_into()
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    /// fields.
    pub fn into_static(self) -> MethodDef<'static> {
        MethodDef {
            name: Cow::Owned(self.name.into_owned()),
            description: Cow::Owned(self.description.into_owned()),
            schema: self.schema,
            cache_control: self.cache_control,
            strict: self.strict,
        }
    }
}

impl TryFrom<serde_json::Value> for MethodDef<'static> {
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

/// `MethodDef` [`Use`] of the model. This should be handled and a response sent
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

/// Result of [`MethodDef`] [`Use`] sent back to the [`Assistant`] as a [`User`]
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
        let method = MethodDef::simple("test_method", "A simple test method");

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
        let method = MethodDef::with_string_param(
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
        let method = MethodDef::builder("test_method")
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
        let method = MethodDef::builder("test_method")
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
        assert!(
            required
                .contains(&serde_json::Value::String("existing".to_string()))
        );
        assert!(
            required
                .contains(&serde_json::Value::String("new_param".to_string()))
        );
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
        let tool = MethodDef::builder("test_name")
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
        let tool = MethodDef::builder("test_name")
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
        let tool = MethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::Value::String("blah".into()))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // Properties not an object
        let tool = MethodDef::builder("test_name")
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

        // `required` lists keys absent from (here, missing) `properties`
        let tool = MethodDef::builder("test_name")
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

        // No `required` array is valid (all-optional / no-arg methods). It is
        // optional per JSON Schema and treated as empty when absent.
        let tool = MethodDef::builder("test_name")
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

        assert!(tool.is_ok());

        // required keys not found in properties
        let tool = MethodDef::builder("test_name")
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
        let tool = MethodDef::builder("test_name")
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
        let tool = MethodDef::builder("test_name")
            .description("test_description")
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyInputSchema)));

        // with missing names and descriptions
        let tool = MethodDef::builder("")
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

        let tool = MethodDef::builder("foo")
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
            cache_control: None,
        };

        let result = result.into_static();

        assert_eq!(result.tool_use_id, "test_id");
        assert_eq!(result.content.to_string(), "test_content");
        assert!(!result.is_error);
    }

    #[test]
    fn test_tool_from_serializable() {
        let tool = MethodDef::from_serializable(serde_json::json!({
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
        let tool = MethodDef::from_serializable(serde_json::json!({
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

    #[test]
    fn test_method_strict_defaults_none_and_elides() {
        let tool = MethodDef::simple("ping", "Ping a server.");
        assert_eq!(tool.strict, None);
        assert!(!tool.is_strict());

        let wire = serde_json::to_value(&tool).unwrap();
        assert!(
            wire.as_object().unwrap().get("strict").is_none(),
            "strict must be elided when None, got {wire:#}",
        );
    }

    #[test]
    fn test_method_builder_strict_flag() {
        let tool = MethodDef::builder("ping")
            .description("Ping a server.")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            }))
            .strict(true)
            .build()
            .unwrap();

        assert_eq!(tool.strict, Some(true));
        assert!(tool.is_strict());
        let wire = serde_json::to_value(&tool).unwrap();
        assert_eq!(wire["strict"], serde_json::Value::Bool(true));
    }

    #[test]
    fn test_method_strict_mut_setter() {
        let mut tool = MethodDef::simple("ping", "Ping a server.");
        tool.strict(true);
        assert_eq!(tool.strict, Some(true));
    }

    #[test]
    fn test_method_strict_roundtrips_through_deserialize() {
        let wire = serde_json::json!({
            "name": "ping",
            "description": "Ping a server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            },
            "strict": true,
        });
        let tool = MethodDef::from_serializable(wire).unwrap();
        assert_eq!(tool.strict, Some(true));
    }

    #[test]
    fn test_method_into_static_preserves_strict() {
        let tool = MethodDef::builder("ping")
            .description("Ping a server.")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            }))
            .strict(true)
            .build()
            .unwrap();
        let owned: MethodDef<'static> = tool.into_static();
        assert_eq!(owned.strict, Some(true));
    }
}
