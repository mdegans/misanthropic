//! `#[derive(ToolArgs)]` — generate `impl misanthropic::tool::ToolArgs`.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{DeriveInput, LitStr};

use crate::util::doc_string;

/// Expand `#[derive(ToolArgs)]` on `input`.
///
/// `NAME` defaults to the struct ident, `DESCRIPTION` to its doc comment; both
/// are overridable via `#[tool(name = "…", description = "…")]`.
pub fn derive(input: TokenStream) -> syn::Result<TokenStream> {
    let input: DeriveInput = syn::parse2(input)?;
    let ident = &input.ident;

    let mut name: Option<String> = None;
    let mut description: Option<String> = None;
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
                } else {
                    Err(meta.error(
                        "unknown `tool` key; expected `name` or `description`",
                    ))
                }
            })?;
        }
    }

    let name = name.unwrap_or_else(|| ident.to_string());
    let description = description.unwrap_or_else(|| doc_string(&input.attrs));

    let (impl_generics, ty_generics, where_clause) =
        input.generics.split_for_impl();

    Ok(quote! {
        #[automatically_derived]
        impl #impl_generics ::misanthropic::tool::ToolArgs
            for #ident #ty_generics #where_clause
        {
            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;
        }
    })
}
