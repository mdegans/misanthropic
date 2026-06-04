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

/// A deferred tool: `#[tool(defer_loading)]` flips
/// [`ToolArgs::DEFER_LOADING`].
#[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
#[allow(dead_code)]
#[tool(defer_loading)]
struct Lookup {
    query: String,
}

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
fn defer_loading_defaults_false_and_is_overridable() {
    // Default: the const is `false` and the field elides on the wire.
    assert!(!<Push as ToolArgs>::DEFER_LOADING);
    assert_eq!(<Push as ToolArgs>::definition().defer_loading, None);
    // `#[tool(defer_loading)]` flips it and carries onto the `CustomMethodDef`.
    assert!(<Lookup as ToolArgs>::DEFER_LOADING);
    assert_eq!(<Lookup as ToolArgs>::definition().defer_loading, Some(true));
}

#[test]
fn definition_builds_from_derived_consts() {
    let def = <Push as ToolArgs>::definition();
    assert_eq!(def.name, "Push");
    assert_eq!(def.description, "Append a note.");
    assert_eq!(def.schema["type"], "object");
    assert_eq!(def.schema["properties"]["note"]["type"], "string");
}

/// The derive's actual purpose: a **hand-written** [`Method`] whose `Args` use
/// `#[derive(ToolArgs)]` instead of a hand-written `impl ToolArgs`. This is the
/// path `#[tool]` automates; here we drive it manually and dispatch through
/// [`Typed`] to prove the derive wires up end-to-end.
mod hand_written_method {
    use misanthropic::{
        prompt::message::Content,
        tool::{ErasedMethod, Method, Methods, Tool, ToolArgs, Typed, Use},
    };

    /// Greet someone by name.
    #[derive(serde::Deserialize, schemars::JsonSchema, ToolArgs)]
    #[tool(name = "greet")]
    struct Greet {
        name: String,
    }

    struct Greeter;

    struct GreetMethod;

    #[async_trait::async_trait]
    impl Method<Greeter> for GreetMethod {
        type Args = Greet;
        async fn run(
            &self,
            _state: &mut Greeter,
            args: Greet,
        ) -> Result<Content, Content> {
            Ok(format!("Hello, {}!", args.name).into())
        }
    }

    impl Methods for Greeter {
        const NAME: &'static str = "Greeter";
        fn methods(&self) -> Vec<Box<dyn ErasedMethod<Self>>> {
            vec![Box::new(GreetMethod)]
        }
    }

    #[test]
    fn derived_args_carry_name_and_doc() {
        assert_eq!(<Greet as ToolArgs>::NAME, "greet");
        assert_eq!(<Greet as ToolArgs>::DESCRIPTION, "Greet someone by name.");
    }

    #[tokio::test]
    async fn derived_args_drive_a_hand_written_method() {
        let mut greeter = Typed(Greeter);
        let result = greeter
            .call(
                Use::new(
                    "Greeter__greet",
                    serde_json::json!({ "name": "world" }),
                )
                .with_id("id"),
            )
            .await;
        assert!(!result.is_error, "{}", result.content);
        assert_eq!(result.content.to_string(), "Hello, world!");
    }
}
