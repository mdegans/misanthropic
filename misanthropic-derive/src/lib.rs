//! Derive macros for the [`misanthropic`] typed-tool layer.
//!
//! - [`macro@ToolArgs`] — derive `misanthropic::tool::ToolArgs` on an argument
//!   struct (name from the ident, description from its doc comment).
//!
//! These are re-exported from `misanthropic` behind its `derive` feature; use
//! them from there (`misanthropic::tool::ToolArgs`) rather than depending on
//! this crate directly. Generated code names `misanthropic` items by absolute
//! path (`::misanthropic::…`), so this crate intentionally does **not** depend
//! on `misanthropic` (which would be a cycle).
//!
//! [`misanthropic`]: https://docs.rs/misanthropic
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use proc_macro::TokenStream;

mod tool_args;
mod util;

/// Derive `misanthropic::tool::ToolArgs` for an argument struct.
///
/// The struct must also derive `serde::Deserialize` and `schemars::JsonSchema`
/// (the `ToolArgs` supertraits). `NAME` defaults to the struct ident and
/// `DESCRIPTION` to the struct's doc comment; override either with a
/// `#[tool(name = "…", description = "…")]` attribute.
///
/// ```ignore
/// #[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
/// /// Append a note.
/// struct Push { note: String }
/// // → impl ToolArgs for Push { NAME = "Push"; DESCRIPTION = "Append a note."; }
/// ```
#[proc_macro_derive(ToolArgs, attributes(tool))]
pub fn derive_tool_args(input: TokenStream) -> TokenStream {
    tool_args::derive(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}
