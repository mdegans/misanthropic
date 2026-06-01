//! Shared helpers for the derive and attribute macros.

use syn::{Attribute, Expr, ExprLit, Lit, Meta};

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
