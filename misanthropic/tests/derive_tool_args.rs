//! Integration tests for `#[derive(ToolArgs)]`.
//!
//! Lives here (not in `misanthropic-derive`) so it exercises the real
//! re-exports through `misanthropic` without a dev-dependency cycle. The
//! single `use` below brings in both the trait and the derive macro (same
//! path, different namespaces) — the serde `Serialize` pattern.
#![cfg(feature = "derive")]

use misanthropic::tool::ToolArgs;

/// Append a note.
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
#[allow(dead_code)]
struct Push {
    note: String,
}

/// This doc comment is overridden by the attribute below.
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
#[tool(name = "clear_all", description = "Erase everything.")]
struct Clear {}

#[test]
fn name_defaults_to_ident_and_description_from_doc() {
    assert_eq!(<Push as ToolArgs>::NAME, "Push");
    assert_eq!(<Push as ToolArgs>::DESCRIPTION, "Append a note.");
}

#[test]
fn attributes_override_name_and_description() {
    assert_eq!(<Clear as ToolArgs>::NAME, "clear_all");
    assert_eq!(<Clear as ToolArgs>::DESCRIPTION, "Erase everything.");
}

#[test]
fn definition_builds_from_derived_consts() {
    let def = <Push as ToolArgs>::definition();
    assert_eq!(def.name, "Push");
    assert_eq!(def.description, "Append a note.");
    assert_eq!(def.schema["type"], "object");
    assert_eq!(def.schema["properties"]["note"]["type"], "string");
}
