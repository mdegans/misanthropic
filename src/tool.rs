//! [`Tool`] and tool [`Choice`] types for the Anthropic Messages API.
use serde::{Deserialize, Serialize};

/// Choice of [`Tool`] for a specific [`request::Message`].
///
/// [`request::Message`]: crate::request::Message
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
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
}
