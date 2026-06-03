//! `#[derive(ToolArgs)]` — generate `impl misanthropic::tool::ToolArgs`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, LitStr};

use crate::util::{doc_string, parse_defer_loading};

/// Expand `#[derive(ToolArgs)]` on `input`.
///
/// `NAME` defaults to the struct ident, `DESCRIPTION` to its doc comment; both
/// are overridable via `#[tool(name = "…", description = "…")]`. A bare
/// `#[tool(defer_loading)]` (or `defer_loading = true`) sets
/// [`ToolArgs::DEFER_LOADING`].
pub fn derive(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = syn::parse2(input)?;
    let ident = &input.ident;

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
    let mut defer_loading: Option<bool> = None;
    for attr in &input.attrs {
        if attr.path().is_ident("tool") {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("name") {
                    name = Some(meta.value()?.parse::<LitStr>()?.value());
                    Ok(())
                } else if meta.path.is_ident("description") {
                    description =
                        Some(meta.value()?.parse::<LitStr>()?.value());
                    Ok(())
                } else if meta.path.is_ident("defer_loading") {
                    defer_loading = Some(parse_defer_loading(&meta)?);
                    Ok(())
                } else {
                    Err(meta.error(
                        "unknown `tool` key; expected `name`, `description`, \
                         or `defer_loading`",
                    ))
                }
            })?;
        }
    }

    let name = name.unwrap_or_else(|| ident.to_string());
    let description = description.unwrap_or_else(|| doc_string(&input.attrs));
    // Only emit the const when set, so the trait default (`false`) stands.
    let defer =
        defer_loading.map(|v| quote! { const DEFER_LOADING: bool = #v; });

    let (impl_generics, ty_generics, where_clause) =
        input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::misanthropic::tool::ToolArgs
            for #ident #ty_generics #where_clause
        {
            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
            #defer
        }
    })
}
