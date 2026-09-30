//! `schema-order-check`: `#[derive(ToolArgs)]` rejects a required field
//! declared after an optional one — `Option<…>` or serde `default`.
#![allow(unused)]
use misanthropic::tool::ToolArgs;

/// Interleaved by `Option`.
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
struct ByOption {
    title: String,
    note: Option<String>,
    body: String,
}

/// Interleaved by `#[serde(default)]`.
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
struct ByDefault {
    #[serde(default)]
    count: u32,
    body: String,
}

/// Fine: `#[schemars(required)]` puts the `Option` back in `required`, and a
/// skipped field isn't in the schema at all.
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
struct Grouped {
    #[schemars(required)]
    title: Option<String>,
    body: String,
    #[serde(skip)]
    cache: u32,
    note: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tag: Option<String>,
}

/// Fine: a container `default` makes every field optional.
#[derive(Default, serde::Deserialize, schemars::JsonSchema, ToolArgs)]
#[serde(default)]
struct AllDefault {
    note: Option<String>,
    body: String,
}

fn main() {}
