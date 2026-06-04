//! Shared helpers for the derive and attribute macros.

use syn::{Attribute, Expr, ExprLit, Ident, Lit, LitBool, Meta, Token};

/// Concatenate the `///` doc comment on `attrs` into a single string: each
/// line trimmed, joined with newlines. Empty when there is no doc comment.
///
/// Rust lowers `/// foo` to `#[doc = " foo"]`; we trim the leading space (and
/// any trailing whitespace) per line so descriptions read cleanly.
pub fn doc_string(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| match &attr.meta {
            Meta::NameValue(nv) if nv.path.is_ident("doc") => match &nv.value {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => Some(s.value().trim().to_string()),
                _ => None,
            },
            _ => None,
        })
        .collect();

    lines.join("\n").trim().to_string()
}

/// Parse a `defer_loading` flag inside `#[tool(…)]` / `#[method(…)]`: accepts a
/// bare path (`defer_loading`, meaning `true`) or `defer_loading = true|false`.
/// `meta` is the entry already matched as `defer_loading`.
pub fn parse_defer_loading(
    meta: &syn::meta::ParseNestedMeta,
) -> syn::Result<bool> {
    if meta.input.peek(Token![=]) {
        Ok(meta.value()?.parse::<LitBool>()?.value())
    } else {
        Ok(true)
    }
}

/// Parse an `allowed_callers(a, b, …)` list inside `#[tool(…)]` / `#[method(…)]`
/// into its caller idents (e.g. `direct`, `code_execution_20260120`). `meta` is
/// the entry already matched as `allowed_callers`. The idents are emitted
/// verbatim as `AllowedCaller::<ident>()` constructor calls, so any
/// `AllowedCaller` const-fn constructor name is valid; an unknown one surfaces
/// as a normal "no associated function" error.
pub fn parse_allowed_callers(
    meta: &syn::meta::ParseNestedMeta,
) -> syn::Result<Vec<Ident>> {
    let mut callers = Vec::new();
    meta.parse_nested_meta(|inner| {
        callers.push(inner.path.require_ident()?.clone());
        Ok(())
    })?;
    if callers.is_empty() {
        return Err(meta.error(
            "`allowed_callers(…)` needs at least one caller, e.g. \
             `allowed_callers(code_execution_20260120)`",
        ));
    }
    Ok(callers)
}
