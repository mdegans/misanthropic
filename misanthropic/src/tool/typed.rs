//! Typed tools: a layer over the object-safe [`Tool`] trait that lets an
//! author declare an `Args` struct per [`Method`] and have the schema derived
//! and the [`Use::input`] deserialized automatically.
//!
//! The pieces:
//! - [`ToolArgs`] — a deserializable, schema-bearing argument struct.
//! - [`Method`] — one typed handler over shared tool state `S`.
//! - [`ErasedMethod`] — object-safe erasure so methods with heterogeneous
//!   `Args` coexist in one `Vec`; blanket-implemented for every [`Method`].
//! - [`Methods`] — the author-facing trait: a tool's [`Method`] set plus the
//!   lifecycle/state hooks mirrored from [`Tool`].
//! - [`Typed`] — newtype adapter that bridges any [`Methods`] to [`Tool`] so
//!   it can live in a [`ToolBox`](super::ToolBox).
//!
//! A blanket `impl<T: Methods> Tool for T` is impossible — Rust coherence
//! would conflict it with [`ToolBox`](super::ToolBox)'s own `impl Tool` (no
//! negative reasoning). [`Typed`] sidesteps that as a distinct type, and keeps
//! the hand-written-`Tool` door open for fully dynamic tools.
//!
//! [`Tool`]: super::Tool
//! [`Use::input`]: super::Use::input
use crate::{
    Prompt,
    prompt::message::Content,
    tool::{self, MethodDef, Tool, Use},
};

/// A deserializable argument struct for a [`Method`]. The schema is derived
/// from `Self` via [`schemars`] and sanitized to Anthropic's accepted subset.
pub trait ToolArgs:
    serde::de::DeserializeOwned + schemars::JsonSchema + Send + Sync + 'static
{
    /// Method name. Must be a single namespace segment (no `__`) and unique
    /// within its tool — [`Typed`] routes by matching this as a suffix of the
    /// (possibly namespaced) called name.
    const NAME: &'static str;
    /// Method description shown to the model.
    const DESCRIPTION: &'static str;

    /// JSON Schema for `Self`, sanitized for Anthropic. See
    /// [`sanitize_for_anthropic`](crate::prompt::output::sanitize_for_anthropic).
    fn schema() -> serde_json::Value {
        let mut schema = serde_json::to_value(schemars::schema_for!(Self))
            .expect("schemars Schema always serializes");
        crate::prompt::output::sanitize_for_anthropic(&mut schema);
        schema
    }

    /// The wire [`MethodDef`] assembled from [`NAME`](Self::NAME),
    /// [`DESCRIPTION`](Self::DESCRIPTION), and [`schema`](Self::schema).
    fn definition() -> MethodDef<'static> {
        MethodDef::builder(Self::NAME)
            .description(Self::DESCRIPTION)
            .schema(Self::schema())
            .build()
            .expect("a ToolArgs-derived schema is valid")
    }
}

/// One typed method over shared tool state `S`. Implement (or derive) this per
/// method; the body receives already-deserialized [`Args`](Self::Args).
///
/// Return `Ok` with anything `Into<Content>` (a `&str`, `String`, image
/// [`Block`](crate::prompt::message::Block), …) for success, or `Err` for an
/// error result — [`Typed`] attaches the `tool_use_id` and sets `is_error`.
#[async_trait::async_trait]
pub trait Method<S: Send>: Send + Sync {
    /// The arguments this method takes.
    type Args: ToolArgs;

    /// Run the method against shared `state` with deserialized `args`.
    async fn run(
        &self,
        state: &mut S,
        args: Self::Args,
    ) -> std::result::Result<Content<'static>, Content<'static>>;
}

/// Object-safe erasure of a [`Method`] so methods with different
/// [`Args`](Method::Args) types share one `Vec<Box<dyn ErasedMethod<S>>>`.
/// Blanket-implemented for every [`Method`]; you never implement this directly.
#[async_trait::async_trait]
pub trait ErasedMethod<S>: Send + Sync {
    /// Method name (cheap; for routing). See [`ToolArgs::NAME`].
    fn name(&self) -> &'static str;
    /// The wire [`MethodDef`] (builds the schema). See [`ToolArgs::definition`].
    fn definition(&self) -> MethodDef<'static>;
    /// Deserialize `input` into the method's `Args` and run it, returning the
    /// result content and whether it is an error.
    async fn dispatch(
        &self,
        state: &mut S,
        input: serde_json::Value,
    ) -> (Content<'static>, bool);
}

#[async_trait::async_trait]
impl<S: Send, M: Method<S>> ErasedMethod<S> for M {
    fn name(&self) -> &'static str {
        <M::Args as ToolArgs>::NAME
    }

    fn definition(&self) -> MethodDef<'static> {
        <M::Args as ToolArgs>::definition()
    }

    async fn dispatch(
        &self,
        state: &mut S,
        input: serde_json::Value,
    ) -> (Content<'static>, bool) {
        // `serde_path_to_error` augments the message with the path to the
        // offending field so the model can correct itself.
        let args: M::Args = match serde_path_to_error::deserialize(input) {
            Ok(args) => args,
            Err(err) => {
                let path = err.path().to_string();
                let message = err.into_inner().to_string();
                let content = if path.is_empty() {
                    format!("Invalid arguments: {message}")
                } else {
                    format!("Invalid arguments at `{path}`: {message}")
                };
                return (content.into(), true);
            }
        };

        match self.run(state, args).await {
            Ok(content) => (content, false),
            Err(content) => (content, true),
        }
    }
}

/// The author-facing typed tool: a [`Method`] set plus the lifecycle/state
/// hooks mirrored from [`Tool`]. Wrap in [`Typed`] to use as a [`Tool`].
#[async_trait::async_trait]
pub trait Methods: Send + Sized {
    /// Tool name.
    const NAME: &'static str;

    /// The methods this tool provides, erased so heterogeneous `Args` coexist.
    fn methods(&self) -> Vec<Box<dyn ErasedMethod<Self>>>;

    /// See [`Tool::save_json`].
    async fn save_json(&mut self) -> serde_json::Value {
        serde_json::Value::Null
    }
    /// See [`Tool::load_json`].
    async fn load_json(
        &mut self,
        _json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        Ok(())
    }
    /// See [`Tool::on_init`].
    async fn on_init(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
    /// See [`Tool::on_turn`].
    async fn on_turn(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

/// Newtype adapter bridging a [`Methods`] tool to the object-safe [`Tool`]
/// trait. Add to a [`ToolBox`](super::ToolBox) with
/// [`ToolBox::add_typed`](super::ToolBox::add_typed).
pub struct Typed<T>(pub T);

#[async_trait::async_trait]
impl<T: Methods + Send> Tool for Typed<T> {
    fn name(&self) -> &str {
        T::NAME
    }

    fn definitions(&self) -> Vec<MethodDef<'static>> {
        self.0.methods().iter().map(|m| m.definition()).collect()
    }

    async fn call<'a>(&mut self, call: Use<'a>) -> tool::Result<'a> {
        let handlers = self.0.methods();
        // `call.name` may be namespaced (`box__tool__method`); match the bare
        // method name as a suffix.
        match handlers.iter().find(|m| call.name.ends_with(m.name())) {
            Some(method) => {
                let (content, is_error) =
                    method.dispatch(&mut self.0, call.input).await;
                tool::Result {
                    tool_use_id: call.id,
                    content,
                    is_error,
                    cache_control: None,
                }
            }
            None => tool::Result {
                tool_use_id: call.id,
                content: format!(
                    "Method `{}` not found on `{}`.",
                    call.name,
                    T::NAME
                )
                .into(),
                is_error: true,
                cache_control: None,
            },
        }
    }

    async fn save_json(&mut self) -> serde_json::Value {
        self.0.save_json().await
    }

    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        self.0.load_json(json).await
    }

    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.0.on_init(prompt).await
    }

    async fn on_turn(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.0.on_turn(prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, schemars::JsonSchema)]
    struct Push {
        note: String,
    }
    impl ToolArgs for Push {
        const NAME: &'static str = "push";
        const DESCRIPTION: &'static str = "Append a note.";
    }

    // A no-arg method: proves the relaxed `build()` accepts an empty-property
    // schema and that heterogeneous `Args` coexist.
    #[derive(Deserialize, schemars::JsonSchema)]
    struct Clear {}
    impl ToolArgs for Clear {
        const NAME: &'static str = "clear";
        const DESCRIPTION: &'static str = "Clear all notes.";
    }

    #[derive(Default)]
    struct Notes {
        notes: Vec<String>,
    }

    struct PushMethod;
    #[async_trait::async_trait]
    impl Method<Notes> for PushMethod {
        type Args = Push;
        async fn run(
            &self,
            state: &mut Notes,
            args: Push,
        ) -> std::result::Result<Content<'static>, Content<'static>> {
            state.notes.push(args.note);
            Ok("noted".into())
        }
    }

    struct ClearMethod;
    #[async_trait::async_trait]
    impl Method<Notes> for ClearMethod {
        type Args = Clear;
        async fn run(
            &self,
            state: &mut Notes,
            _args: Clear,
        ) -> std::result::Result<Content<'static>, Content<'static>> {
            state.notes.clear();
            Ok("cleared".into())
        }
    }

    impl Methods for Notes {
        const NAME: &'static str = "notes";
        fn methods(&self) -> Vec<Box<dyn ErasedMethod<Self>>> {
            vec![
                Box::new(PushMethod) as Box<dyn ErasedMethod<Self>>,
                Box::new(ClearMethod),
            ]
        }
    }

    fn call_with(name: &str, input: serde_json::Value) -> Use<'static> {
        Use {
            id: "id".into(),
            name: name.to_string().into(),
            input,
            cache_control: None,
        }
    }

    #[tokio::test]
    async fn dispatches_heterogeneous_methods() {
        let mut tool = Typed(Notes::default());

        let r = tool
            .call(call_with("push", serde_json::json!({"note": "hi"})))
            .await;
        assert!(!r.is_error, "{}", r.content);
        assert_eq!(tool.0.notes, vec!["hi".to_string()]);

        // Namespaced suffix routing for the no-arg method.
        let r = tool
            .call(call_with("notes__clear", serde_json::json!({})))
            .await;
        assert!(!r.is_error, "{}", r.content);
        assert!(tool.0.notes.is_empty());
    }

    #[tokio::test]
    async fn arg_validation_error_carries_path() {
        let mut tool = Typed(Notes::default());
        let r = tool
            .call(call_with("push", serde_json::json!({"note": 123})))
            .await;
        assert!(r.is_error);
        let msg = r.content.to_string();
        assert!(msg.contains("Invalid arguments"), "got: {msg}");
        assert!(msg.contains("note"), "path should name the field: {msg}");
    }

    #[tokio::test]
    async fn unknown_method_is_error() {
        let mut tool = Typed(Notes::default());
        let r = tool.call(call_with("nope", serde_json::json!({}))).await;
        assert!(r.is_error);
    }

    #[test]
    fn no_arg_definition_builds() {
        let def = <Clear as ToolArgs>::definition();
        assert_eq!(def.name, "clear");
        assert_eq!(def.schema["type"], "object");
    }

    #[test]
    fn definitions_lists_both_methods() {
        let names: Vec<_> = Typed(Notes::default())
            .definitions()
            .into_iter()
            .map(|d| d.name.into_owned())
            .collect();
        assert!(names.contains(&"push".to_string()));
        assert!(names.contains(&"clear".to_string()));
    }
}
