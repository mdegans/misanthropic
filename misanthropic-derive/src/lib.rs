//! Derive macros for the [`misanthropic`] typed-tool layer.
//!
//! - [`macro@ToolArgs`] — derive `misanthropic::tool::ToolArgs` on an argument
//!   struct (name from the ident, description from its doc comment).
//! - [`macro@tool`] — attribute on an `impl` block that generates the
//!   `Method`/`ToolArgs`/`Methods` wiring from `#[method]`-tagged async fns.
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

#[cfg(feature = "schema-order-check")]
mod order;
mod tool;
mod tool_args;
mod util;

/// Derive `misanthropic::tool::ToolArgs` for an argument struct.
///
/// The struct must also derive `serde::Deserialize` and `schemars::JsonSchema`
/// (the `ToolArgs` supertraits). `NAME` defaults to the struct ident and
/// `DESCRIPTION` to the struct's doc comment; override either with a
/// `#[tool(name = "…", description = "…")]` attribute.
///
/// With `misanthropic`'s default `schema-order-check` feature, a required
/// field declared after an optional one (`Option<…>` or serde `default`) is a
/// compile error. Required-first is the one layout every engine generates in
/// the same order, and field order changes what the model generates (reason
/// before you answer) — subtly, and more so on smaller models.
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

/// Generate the typed-tool wiring from an `impl` block. See
/// [`tool::expand`](crate::tool) for the full contract.
///
/// Tag each tool method `#[method]`; it must be an `async fn` taking `&mut
/// self` and exactly one further argument — its `Args` type (deriving
/// `serde::Deserialize` + `schemars::JsonSchema`) — and returning
/// `Result<Content<'static>, Content<'static>>`. The macro keeps your fns as
/// real inherent methods and generates a thin `Method` per fn that delegates
/// to them, plus the `Methods` impl. The tool `NAME` defaults to the self
/// type's ident; override with `#[tool(name = "…")]`. Add `flat` to put
/// method names on the wire bare — no `tool__` segment (sets
/// `Methods::FLAT`; pair with a flat `ToolBox` to drop its segment too).
///
/// The macro can't see the `Args` fields, so with `misanthropic`'s default
/// `schema-order-check` feature a required field after an optional one isn't
/// a compile error here: `ToolArgs::definition` (and so `add_tool`) panics at
/// runtime. To catch it sooner, the macro emits one `#[cfg(test)]` test per
/// method, `__misanthropic_schema_order_{tool}_{method}` (snake-cased), that
/// fails your `cargo test`. Inside a fn body that test can't be collected:
/// `#[allow(unnameable_test_items)]` on the enclosing fn, or move the impl to
/// module level so it runs.
///
/// ```ignore
/// #[tool(name = "Notepad")]
/// impl<'a> Notepad<'a> {
///     /// Take a note.
///     #[method]
///     async fn push(&mut self, args: Push)
///         -> Result<Content<'static>, Content<'static>> { /* … */ }
/// }
/// ```
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    tool::expand(attr.into(), item.into()).into()
}
