//! Integration tests for the `#[tool]` attribute macro.
//!
//! A two-method tool (one with args, one no-arg) with `on_init` and
//! `save_json`/`load_json` hooks, exercised through the [`Typed`] bridge — the
//! same path a real tool takes in a `ToolBox`.
#![cfg(feature = "derive")]

use misanthropic::{
    Prompt,
    prompt::message::Content,
    tool::{Methods, Tool, ToolArgs, Typed, Use, tool},
};
use schemars::JsonSchema;
use serde::Deserialize;

#[derive(Deserialize, JsonSchema)]
struct Add {
    x: i64,
    y: i64,
}

#[derive(Deserialize, JsonSchema)]
struct Reset {}

#[derive(Default)]
struct Calc {
    acc: i64,
    inited: bool,
}

#[tool(name = "Calc")]
impl Calc {
    /// Add x and y to the accumulator.
    #[method]
    async fn add(
        &mut self,
        args: Add,
    ) -> Result<Content<'static>, Content<'static>> {
        self.acc += args.x + args.y;
        Ok(self.acc.to_string().into())
    }

    /// Reset the accumulator.
    #[method]
    async fn reset(
        &mut self,
        _args: Reset,
    ) -> Result<Content<'static>, Content<'static>> {
        self.acc = 0;
        Ok("reset".into())
    }

    #[on_init]
    async fn init(
        &mut self,
        _prompt: &mut Prompt<'_>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.inited = true;
        Ok(())
    }

    #[save_json]
    async fn save(&mut self) -> serde_json::Value {
        serde_json::json!({ "acc": self.acc })
    }

    #[load_json]
    async fn load(&mut self, json: serde_json::Value) -> Result<(), String> {
        self.acc = json["acc"].as_i64().ok_or("missing `acc`")?;
        Ok(())
    }
}

// A lifetime-parameterized tool with a bare `#[tool]` (no `name`) — proves
// generics threading (`impl<'a> Holder<'a>`) and the ident-derived name
// default, the two things the `Notepad<'a>` port relies on.
#[derive(Deserialize, JsonSchema)]
struct Put {
    v: String,
}

#[derive(Default)]
struct Holder<'a> {
    items: Vec<std::borrow::Cow<'a, str>>,
}

#[tool]
impl<'a> Holder<'a> {
    /// Put a value.
    #[method]
    async fn put(
        &mut self,
        args: Put,
    ) -> Result<Content<'static>, Content<'static>> {
        self.items.push(args.v.into());
        Ok("ok".into())
    }
}

fn use_call(name: &str, input: serde_json::Value) -> Use<'static> {
    Use {
        id: "id".into(),
        name: name.to_string().into(),
        input,
        cache_control: None,
    }
}

#[test]
fn name_and_namespaced_definitions() {
    assert_eq!(<Calc as Methods>::NAME, "Calc");
    let names: Vec<_> = Typed(Calc::default())
        .definitions()
        .into_iter()
        .map(|d| d.name.into_owned())
        .collect();
    assert!(names.contains(&"Calc__add".to_string()));
    assert!(names.contains(&"Calc__reset".to_string()));
}

#[test]
fn derived_tool_args_from_fn() {
    // The `#[tool]` route generates `impl ToolArgs` from the fn ident + doc.
    assert_eq!(<Add as ToolArgs>::NAME, "add");
    assert_eq!(
        <Add as ToolArgs>::DESCRIPTION,
        "Add x and y to the accumulator."
    );
    assert_eq!(<Reset as ToolArgs>::NAME, "reset");
}

#[tokio::test]
async fn dispatches_both_methods() {
    let mut calc = Typed(Calc::default());

    let r = calc
        .call(use_call("Calc__add", serde_json::json!({"x": 2, "y": 3})))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content.to_string(), "5");
    assert_eq!(calc.0.acc, 5);

    let r = calc.call(use_call("reset", serde_json::json!({}))).await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(calc.0.acc, 0);
}

#[tokio::test]
async fn arg_validation_error_is_reported() {
    let mut calc = Typed(Calc::default());
    let r = calc
        .call(use_call(
            "Calc__add",
            serde_json::json!({"x": "nope", "y": 3}),
        ))
        .await;
    assert!(r.is_error);
    assert!(r.content.to_string().contains("Invalid arguments"));
}

#[tokio::test]
async fn on_init_hook_runs() {
    let mut calc = Typed(Calc::default());
    let mut prompt = Prompt::default();
    calc.on_init(&mut prompt).await.unwrap();
    assert!(calc.0.inited);
}

#[tokio::test]
async fn generic_tool_defaults_name_to_ident() {
    assert_eq!(<Holder as Methods>::NAME, "Holder");
    let mut holder = Typed(Holder::default());
    let names: Vec<_> = holder
        .definitions()
        .into_iter()
        .map(|d| d.name.into_owned())
        .collect();
    assert_eq!(names, vec!["Holder__put".to_string()]);

    let r = holder
        .call(use_call("Holder__put", serde_json::json!({"v": "x"})))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(holder.0.items.len(), 1);
}

#[tokio::test]
async fn save_and_load_round_trip() {
    let mut calc = Calc {
        acc: 42,
        inited: false,
    };
    // `Calc` impls both `Tool` and `Methods` (they share these method names),
    // and both are in scope here, so qualify which one we mean.
    let json = Tool::save_json(&mut calc).await;
    let mut other = Calc::default();
    Tool::load_json(&mut other, json).await.unwrap();
    assert_eq!(other.acc, 42);
}
