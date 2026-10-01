//! Compile-time half of `schema-order-check`: reject a `#[derive(ToolArgs)]`
//! struct declaring a required field after an optional one.
//!
//! Required-ness mirrors `schemars`' deserialize contract: a field is optional
//! when it has a serde `default` (field or container) or, lacking
//! `#[schemars(required)]`, is an `Option<…>`. Anything this can't read
//! syntactically — `flatten`, `with`, a container `from`/`transparent`, an
//! unparseable attribute — skips the check; the runtime check in
//! `MethodBuilder::build` still covers it.

use syn::{
    Attribute, Data, DeriveInput, Field, Fields, Meta, PathArguments, Token,
    Type, punctuated::Punctuated,
};

/// Where a field lands in the derived schema's `required`.
#[derive(Clone, Copy, PartialEq)]
enum Presence {
    Required,
    Optional,
    /// Not in the deserialize schema at all (`skip`/`skip_deserializing`).
    Skipped,
}

/// Namespaces `schemars` reads serde-style keys (`default`, `skip`, …) from.
const SERDE: &[&str] = &["serde", "schemars"];
/// Namespaces `schemars` reads validation keys (`required`) from.
const VALIDATE: &[&str] = &["schemars", "validate", "garde"];

/// Error on the first required field declared after an optional one.
pub fn check(input: &DeriveInput) -> syn::Result<()> {
    let Data::Struct(data) = &input.data else {
        return Ok(());
    };
    let Fields::Named(fields) = &data.fields else {
        return Ok(());
    };
    // A container `default` makes every field optional; `from`/`transparent`/
    // `with` swap in another type's schema we can't see.
    let opaque = ["default", "from", "try_from", "transparent", "with"];
    if keys(&input.attrs, SERDE).is_none_or(|keys| {
        keys.iter().any(|(_, k)| opaque.contains(&k.as_str()))
    }) {
        return Ok(());
    }
    let Some(fields) = fields
        .named
        .iter()
        .map(|f| presence(f).map(|p| (f, p)))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(());
    };

    let mut fields =
        fields.into_iter().filter(|(_, p)| *p != Presence::Skipped);
    let Some((optional, _)) =
        fields.by_ref().find(|(_, p)| *p == Presence::Optional)
    else {
        return Ok(());
    };
    match fields.find(|(_, p)| *p == Presence::Required) {
        None => Ok(()),
        Some((late, _)) => Err(syn::Error::new_spanned(
            &late.ident,
            format!(
                "required field `{late}` is declared after optional field \
                 `{optional}`. Declare every required field before any \
                 optional one (or disable misanthropic's \
                 `schema-order-check` feature): it's the one layout every \
                 engine generates in the same order, and field order changes \
                 what the model generates.",
                late = ident(late),
                optional = ident(optional),
            ),
        )),
    }
}

/// A field's [`Presence`], or `None` when it can't be read syntactically.
fn presence(field: &Field) -> Option<Presence> {
    let keys = keys(&field.attrs, &["serde", "schemars", "validate", "garde"])?;
    let has = |namespaces: &[&str], key: &str| {
        keys.iter()
            .any(|(ns, k)| k == key && namespaces.contains(&ns.as_str()))
    };

    if has(SERDE, "flatten") || has(SERDE, "with") || has(SERDE, "schema_with")
    {
        None
    } else if has(SERDE, "skip") || has(SERDE, "skip_deserializing") {
        Some(Presence::Skipped)
    } else if has(SERDE, "default")
        || (!has(VALIDATE, "required") && is_option(&field.ty))
    {
        Some(Presence::Optional)
    } else {
        Some(Presence::Required)
    }
}

/// `(namespace, key)` for each top-level key of every `#[ns(…)]` attribute in
/// `namespaces`, or `None` when one doesn't parse as a meta list.
fn keys(
    attrs: &[Attribute],
    namespaces: &[&str],
) -> Option<Vec<(String, String)>> {
    attrs
        .iter()
        .filter_map(|attr| {
            let ns = attr.path().get_ident()?.to_string();
            namespaces.contains(&ns.as_str()).then_some((ns, attr))
        })
        .map(|(ns, attr)| {
            let metas = attr
                .parse_args_with(
                    Punctuated::<Meta, Token![,]>::parse_terminated,
                )
                .ok()?;
            Some(
                metas
                    .iter()
                    .filter_map(|meta| meta.path().get_ident())
                    .map(|key| (ns.clone(), key.to_string()))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Option<Vec<_>>>()
        .map(|keys| keys.concat())
}

/// Whether `ty` is spelled `Option<…>` (any path prefix). An alias for
/// `Option` reads as required — the runtime check still sees the truth.
fn is_option(ty: &Type) -> bool {
    match ty {
        Type::Group(group) => is_option(&group.elem),
        Type::Paren(paren) => is_option(&paren.elem),
        Type::Path(path) => {
            path.qself.is_none()
                && path.path.segments.last().is_some_and(|seg| {
                    seg.ident == "Option"
                        && matches!(
                            seg.arguments,
                            PathArguments::AngleBracketed(_)
                        )
                })
        }
        _ => false,
    }
}

/// A named field's ident, for diagnostics.
fn ident(field: &Field) -> String {
    field
        .ident
        .as_ref()
        .map(ToString::to_string)
        .unwrap_or_default()
}
