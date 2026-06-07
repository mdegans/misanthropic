//! `#[tool]` — generate typed-tool wiring from an `impl` block.
//!
//! The author writes their methods as ordinary inherent `async fn`s tagged
//! with marker attributes; this macro keeps those fns untouched and generates,
//! around them:
//! - one zero-sized `Method` wrapper per `#[method]` fn that **delegates** to
//!   it (`state.push(args).await`) — no body rewriting, so nothing is fragile;
//! - an `impl ToolArgs` for each method's `Args` type (name from the fn ident,
//!   description from its doc comment);
//! - one `impl Methods` collecting the wrappers and delegating any tagged
//!   lifecycle hooks
//!   (`#[on_init]`/`#[on_turn]`/`#[on_teardown]`/`#[save_json]`/`#[load_json]`),
//!   plus a `#[connect]` fn spliced straight onto `impl Tool` (it's sync and not
//!   a `Methods` method).
//!
//! Generated code names everything by absolute `::misanthropic::…` path.

use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{
    Attribute, FnArg, Ident, ImplItem, ImplItemFn, ItemImpl, LitStr, Signature,
    Type, parse::Parser,
};

use crate::util::{doc_string, parse_allowed_callers, parse_defer_loading};

/// Marker attributes recognized (and stripped) inside a `#[tool]` impl.
const MARKERS: &[&str] = &[
    "method",
    "connect",
    "on_init",
    "on_turn",
    "on_teardown",
    "save_json",
    "load_json",
];

/// Expand `#[tool(attr)] item`.
///
/// On error we still emit the author's impl (with our markers stripped) next to
/// the `compile_error!`, so their inherent methods keep resolving and the
/// reader sees one root-cause diagnostic instead of a cascade.
pub fn expand(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut item_impl: ItemImpl = match syn::parse2(item) {
        Ok(parsed) => parsed,
        // A syntactically broken impl can't be re-emitted meaningfully.
        Err(err) => return err.into_compile_error(),
    };

    let wiring = build(&item_impl, attr);
    strip_markers(&mut item_impl);

    match wiring {
        Ok(wiring) => quote! { #item_impl #wiring },
        Err(err) => {
            let err = err.into_compile_error();
            quote! { #item_impl #err }
        }
    }
}

/// Build the generated wiring (wrappers + `Methods` impl) for `item_impl`. Does
/// not mutate; [`expand`] strips markers from the re-emitted copy separately.
fn build(item_impl: &ItemImpl, attr: TokenStream) -> syn::Result<TokenStream> {
    let self_ty = (*item_impl.self_ty).clone();
    let self_ident = self_ty_ident(&self_ty)?;
    let tool_name = parse_name(attr)?.unwrap_or_else(|| self_ident.to_string());

    // Scan items for `#[method]`s and lifecycle markers (a fn may carry more
    // than one, e.g. `#[on_init] #[on_turn]`).
    let mut methods: Vec<MethodInfo> = Vec::new();
    let mut lifecycle = Lifecycle::default();
    for item in &item_impl.items {
        if let ImplItem::Fn(f) = item {
            for marker in MARKERS.iter().filter(|m| has_marker(&f.attrs, m)) {
                match *marker {
                    "method" => methods.push(MethodInfo::parse(f)?),
                    kind => lifecycle.set(kind, &f.sig.ident)?,
                }
            }
        }
    }

    let (ig, _tg, wc) = item_impl.generics.split_for_impl();

    let mut wrappers = Vec::new();
    let mut wrapper_idents = Vec::new();
    for m in &methods {
        let wrapper = Ident::new(
            &format!("__{self_ident}_{}", m.ident),
            Span::call_site(),
        );
        let args_ty = &m.args_ty;
        let fn_ident = &m.ident;
        let name = m.ident.to_string();
        let desc = &m.doc;
        // Only emit the const when set, so the trait default (`false`) stands.
        let defer = m
            .defer_loading
            .map(|v| quote! { const DEFER_LOADING: bool = #v; });
        // Emit `ALLOWED_CALLERS` only when the method opts in; the idents are
        // `AllowedCaller` const-fn constructor names, so they compose directly.
        let allowed = (!m.allowed_callers.is_empty()).then(|| {
            let callers = &m.allowed_callers;
            quote! {
                const ALLOWED_CALLERS:
                    &'static [::misanthropic::tool::AllowedCaller] = &[
                    #( ::misanthropic::tool::AllowedCaller::#callers() ),*
                ];
            }
        });

        wrappers.push(quote! {
            #[doc(hidden)]
            #[allow(non_camel_case_types)]
            struct #wrapper;

            #[automatically_derived]
            impl ::misanthropic::tool::ToolArgs for #args_ty {
                const NAME: &'static str = #name;
                const DESCRIPTION: &'static str = #desc;
                #defer
                #allowed
            }

            #[automatically_derived]
            #[::misanthropic::__derive::async_trait::async_trait]
            impl #ig ::misanthropic::tool::Method<#self_ty> for #wrapper #wc {
                type Args = #args_ty;
                async fn run(
                    &self,
                    state: &mut #self_ty,
                    args: #args_ty,
                ) -> ::core::result::Result<
                    ::misanthropic::prompt::message::Content,
                    ::misanthropic::prompt::message::Content,
                > {
                    state.#fn_ident(args).await
                }
            }
        });
        wrapper_idents.push(wrapper);
    }

    let lifecycle_methods = lifecycle.delegations();
    let connect_method = lifecycle.tool_connect();
    // `#[async_trait]` is only needed on the `Methods` impl when it overrides
    // an async lifecycle method; `methods()`/`NAME` alone aren't async.
    let methods_async = if lifecycle_methods.is_empty() {
        quote! {}
    } else {
        quote! { #[::misanthropic::__derive::async_trait::async_trait] }
    };

    let box_err = quote! {
        ::std::boxed::Box<
            dyn ::std::error::Error + ::core::marker::Send + ::core::marker::Sync,
        >
    };

    Ok(quote! {
        #(#wrappers)*

        #[automatically_derived]
        #methods_async
        impl #ig ::misanthropic::tool::Methods for #self_ty #wc {
            const NAME: &'static str = #tool_name;

            fn methods(
                &self,
            ) -> ::std::vec::Vec<
                ::std::boxed::Box<dyn ::misanthropic::tool::ErasedMethod<Self>>,
            > {
                ::std::vec![
                    #(
                        ::std::boxed::Box::new(#wrapper_idents)
                            as ::std::boxed::Box<
                                dyn ::misanthropic::tool::ErasedMethod<Self>,
                            >
                    ),*
                ]
            }

            #(#lifecycle_methods)*
        }

        // Concrete `impl Tool` so the type is usable directly (no `Typed`
        // wrapper). Routing/definitions reuse the shared `Methods` helpers;
        // lifecycle forwards to the `Methods` impl above.
        #[automatically_derived]
        #[::misanthropic::__derive::async_trait::async_trait]
        impl #ig ::misanthropic::tool::Tool for #self_ty #wc {
            #connect_method

            fn name(&self) -> &str {
                <Self as ::misanthropic::tool::Methods>::NAME
            }

            fn definitions(
                &self,
            ) -> ::std::vec::Vec<::misanthropic::tool::MethodDef> {
                ::misanthropic::tool::methods_definitions(self)
            }

            async fn call(
                &mut self,
                call: ::misanthropic::tool::Use,
            ) -> ::misanthropic::tool::Result {
                ::misanthropic::tool::dispatch_methods(self, call).await
            }

            async fn save_json(
                &mut self,
            ) -> ::misanthropic::__derive::serde_json::Value {
                <Self as ::misanthropic::tool::Methods>::save_json(self).await
            }

            async fn load_json(
                &mut self,
                json: ::misanthropic::__derive::serde_json::Value,
            ) -> ::core::result::Result<(), ::std::string::String> {
                <Self as ::misanthropic::tool::Methods>::load_json(self, json)
                    .await
            }

            async fn on_init(
                &mut self,
                prompt: &mut ::misanthropic::Prompt,
            ) -> ::core::result::Result<(), #box_err> {
                <Self as ::misanthropic::tool::Methods>::on_init(self, prompt)
                    .await
            }

            async fn on_turn(
                &mut self,
                prompt: &mut ::misanthropic::Prompt,
            ) -> ::core::result::Result<(), #box_err> {
                <Self as ::misanthropic::tool::Methods>::on_turn(self, prompt)
                    .await
            }

            async fn on_teardown(
                &mut self,
                prompt: &mut ::misanthropic::Prompt,
            ) -> ::core::result::Result<(), #box_err> {
                <Self as ::misanthropic::tool::Methods>::on_teardown(
                    self, prompt,
                )
                .await
            }
        }
    })
}

/// One `#[method]` fn: its name, doc, `Args` type, and optional
/// `defer_loading` / `allowed_callers` overrides (from `#[method(…)]`).
struct MethodInfo {
    ident: Ident,
    doc: String,
    args_ty: Type,
    defer_loading: Option<bool>,
    allowed_callers: Vec<Ident>,
}

impl MethodInfo {
    fn parse(f: &ImplItemFn) -> syn::Result<Self> {
        let MethodArgs {
            defer_loading,
            allowed_callers,
        } = parse_method_args(&f.attrs)?;
        Ok(Self {
            ident: f.sig.ident.clone(),
            doc: doc_string(&f.attrs),
            args_ty: method_arg_type(&f.sig)?,
            defer_loading,
            allowed_callers,
        })
    }
}

/// The parsed `#[method(…)]` keys.
#[derive(Default)]
struct MethodArgs {
    defer_loading: Option<bool>,
    allowed_callers: Vec<Ident>,
}

/// Parse the `#[method(…)]` attribute's keys: `defer_loading` (bare or
/// `= true|false`) and `allowed_callers(…)`. A bare `#[method]` has no keys.
fn parse_method_args(attrs: &[Attribute]) -> syn::Result<MethodArgs> {
    let mut args = MethodArgs::default();
    for attr in attrs.iter().filter(|a| a.path().is_ident("method")) {
        // A bare `#[method]` (path-only) carries no keys to parse.
        if matches!(attr.meta, syn::Meta::Path(_)) {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("defer_loading") {
                args.defer_loading = Some(parse_defer_loading(&meta)?);
                Ok(())
            } else if meta.path.is_ident("allowed_callers") {
                args.allowed_callers = parse_allowed_callers(&meta)?;
                Ok(())
            } else {
                Err(meta.error(
                    "unknown `method` key; expected `defer_loading` or \
                     `allowed_callers`",
                ))
            }
        })?;
    }
    Ok(args)
}

/// The `Args` type of a `#[method]` fn: it must take `&mut self` (or `&self`)
/// and exactly one further typed argument.
fn method_arg_type(sig: &Signature) -> syn::Result<Type> {
    let mut has_receiver = false;
    let mut typed = Vec::new();
    for arg in &sig.inputs {
        match arg {
            FnArg::Receiver(_) => has_receiver = true,
            FnArg::Typed(pt) => typed.push(pt),
        }
    }
    if !has_receiver {
        return Err(syn::Error::new_spanned(
            &sig.inputs,
            "a `#[method]` fn must take a `self` receiver",
        ));
    }
    match typed.as_slice() {
        [pt] => Ok((*pt.ty).clone()),
        _ => Err(syn::Error::new_spanned(
            &sig.inputs,
            "a `#[method]` fn must take exactly one argument besides `self` \
             (its `Args` type, deriving `Deserialize` + `JsonSchema`)",
        )),
    }
}

/// Lifecycle hooks discovered in the impl, each delegating to a tagged fn.
#[derive(Default)]
struct Lifecycle {
    /// `#[connect]` — sync, on `Tool` directly (not routed through `Methods`).
    connect: Option<Ident>,
    on_init: Option<Ident>,
    on_turn: Option<Ident>,
    on_teardown: Option<Ident>,
    save_json: Option<Ident>,
    load_json: Option<Ident>,
}

impl Lifecycle {
    fn set(&mut self, kind: &str, fn_ident: &Ident) -> syn::Result<()> {
        let slot = match kind {
            "connect" => &mut self.connect,
            "on_init" => &mut self.on_init,
            "on_turn" => &mut self.on_turn,
            "on_teardown" => &mut self.on_teardown,
            "save_json" => &mut self.save_json,
            "load_json" => &mut self.load_json,
            _ => unreachable!("kind is one of MARKERS minus `method`"),
        };
        if let Some(existing) = slot {
            return Err(syn::Error::new_spanned(
                fn_ident,
                format!("`#[{kind}]` is already set on `{existing}`"),
            ));
        }
        *slot = Some(fn_ident.clone());
        Ok(())
    }

    fn delegations(&self) -> Vec<TokenStream> {
        let mut out = Vec::new();
        let box_err = quote! {
            ::std::boxed::Box<
                dyn ::std::error::Error
                    + ::core::marker::Send
                    + ::core::marker::Sync,
            >
        };
        if let Some(f) = &self.on_init {
            out.push(quote! {
                async fn on_init(
                    &mut self,
                    prompt: &mut ::misanthropic::Prompt,
                ) -> ::core::result::Result<(), #box_err> {
                    self.#f(prompt).await
                }
            });
        }
        if let Some(f) = &self.on_turn {
            out.push(quote! {
                async fn on_turn(
                    &mut self,
                    prompt: &mut ::misanthropic::Prompt,
                ) -> ::core::result::Result<(), #box_err> {
                    self.#f(prompt).await
                }
            });
        }
        if let Some(f) = &self.on_teardown {
            out.push(quote! {
                async fn on_teardown(
                    &mut self,
                    prompt: &mut ::misanthropic::Prompt,
                ) -> ::core::result::Result<(), #box_err> {
                    self.#f(prompt).await
                }
            });
        }
        if let Some(f) = &self.save_json {
            out.push(quote! {
                async fn save_json(
                    &mut self,
                ) -> ::misanthropic::__derive::serde_json::Value {
                    self.#f().await
                }
            });
        }
        if let Some(f) = &self.load_json {
            out.push(quote! {
                async fn load_json(
                    &mut self,
                    json: ::misanthropic::__derive::serde_json::Value,
                ) -> ::core::result::Result<(), ::std::string::String> {
                    self.#f(json).await
                }
            });
        }
        out
    }

    /// The `Tool::connect` override, when a `#[connect]` fn is present. Unlike
    /// the async hooks, `connect` is sync and lives on `Tool` directly, so it's
    /// spliced into the `impl Tool` block rather than the `Methods` impl.
    fn tool_connect(&self) -> Option<TokenStream> {
        self.connect.as_ref().map(|f| {
            quote! {
                fn connect(
                    &mut self,
                    mailbox: ::misanthropic::tool::Mailbox,
                ) {
                    self.#f(mailbox)
                }
            }
        })
    }
}

fn has_marker(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|a| a.path().is_ident(name))
}

/// Strip our marker attributes from every fn so the impl can be re-emitted as
/// ordinary inherent methods.
fn strip_markers(item_impl: &mut ItemImpl) {
    for item in &mut item_impl.items {
        if let ImplItem::Fn(f) = item {
            f.attrs
                .retain(|a| !MARKERS.iter().any(|m| a.path().is_ident(m)));
        }
    }
}

/// Parse the optional `name = "…"` from the `#[tool(…)]` attribute tokens.
fn parse_name(attr: TokenStream) -> syn::Result<Option<String>> {
    if attr.is_empty() {
        return Ok(None);
    }
    let mut name = None;
    let parser = syn::meta::parser(|meta| {
        if meta.path.is_ident("name") {
            name = Some(meta.value()?.parse::<LitStr>()?.value());
            Ok(())
        } else {
            Err(meta.error("unknown `tool` key; expected `name`"))
        }
    });
    parser.parse2(attr)?;
    Ok(name)
}

/// The last path segment ident of the impl's self type, used to mangle wrapper
/// struct names (always a valid ident, unlike a custom `name = "…"`).
fn self_ty_ident(ty: &Type) -> syn::Result<Ident> {
    if let Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
    {
        return Ok(seg.ident.clone());
    }
    Err(syn::Error::new_spanned(
        ty,
        "`#[tool]` requires a path self type, e.g. `impl Notepad { … }`",
    ))
}
