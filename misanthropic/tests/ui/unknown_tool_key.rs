//! `#[derive(ToolArgs)]` rejects unknown `#[tool(...)]` keys.
#![allow(unused)]
use misanthropic::tool::ToolArgs;

#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
#[tool(bogus = "x")]
struct Bad {}

fn main() {}
