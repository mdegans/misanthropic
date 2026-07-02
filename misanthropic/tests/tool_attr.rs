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
    torn_down: bool,
}

#[tool(name = "Calc")]
impl Calc {
    /// Add x and y to the accumulator.
    #[method(allowed_callers(code_execution_20260120))]
    async fn add(&mut self, args: Add) -> Result<Content, Content> {
        self.acc += args.x + args.y;
        Ok(self.acc.to_string().into())
    }

    /// Reset the accumulator.
    #[method(defer_loading)]
    async fn reset(&mut self, _args: Reset) -> Result<Content, Content> {
        self.acc = 0;
        Ok("reset".into())
    }

    #[on_init]
    async fn init(
        &mut self,
        _prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.inited = true;
        Ok(())
    }

    #[on_teardown]
    async fn teardown(
        &mut self,
        _prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.torn_down = true;
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
// generics threading (`impl Holder`) and the ident-derived name
// default, the two things the `Notepad` port relies on.
#[derive(Deserialize, JsonSchema)]
struct Put {
    v: String,
}

#[derive(Default)]
struct Holder {
    items: Vec<std::borrow::Cow<'static, str>>,
}

#[tool]
impl Holder {
    /// Put a value.
    #[method]
    async fn put(&mut self, args: Put) -> Result<Content, Content> {
        self.items.push(args.v.into());
        Ok("ok".into())
    }
}

fn use_call(name: &str, input: serde_json::Value) -> Use {
    Use::new(name.to_string(), input).with_id("id")
}

#[test]
fn name_and_namespaced_definitions() {
    assert_eq!(<Calc as Methods>::NAME, "Calc");
    let names: Vec<_> = Typed(Calc::default())
        .definitions()
        .into_iter()
        .map(|d| d.name().to_string())
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

#[test]
// The `assert!`s deliberately pin the value of a compile-time associated const
// (`DEFER_LOADING`) as a regression guard — the point of the test, not a lint to
// flatten.
#[allow(clippy::assertions_on_constants)]
fn method_defer_loading_attribute_flows_through() {
    // `#[method(defer_loading)]` on `reset` defers only that method; `add`
    // (a bare `#[method]`) is left loaded.
    assert!(!<Add as ToolArgs>::DEFER_LOADING);
    assert!(<Reset as ToolArgs>::DEFER_LOADING);

    let defs = Typed(Calc::default()).definitions();
    let defer = |name: &str| {
        defs.iter()
            .find(|d| d.name() == name)
            .unwrap()
            .as_method()
            .unwrap()
            .defer_loading
    };
    assert_eq!(defer("Calc__add"), None);
    assert_eq!(defer("Calc__reset"), Some(true));
}

#[test]
fn method_allowed_callers_attribute_flows_through() {
    use misanthropic::tool::AllowedCaller;

    // `#[method(allowed_callers(code_execution_20260120))]` on `add` opts only
    // that method into programmatic calling; `reset` keeps the default.
    assert_eq!(
        <Add as ToolArgs>::ALLOWED_CALLERS,
        &[AllowedCaller::code_execution_20260120()]
    );
    assert!(<Reset as ToolArgs>::ALLOWED_CALLERS.is_empty());

    let defs = Typed(Calc::default()).definitions();
    let callers = |name: &str| {
        defs.iter()
            .find(|d| d.name() == name)
            .unwrap()
            .as_method()
            .unwrap()
            .allowed_callers
            .clone()
    };
    assert_eq!(
        callers("Calc__add"),
        Some(vec![AllowedCaller::code_execution_20260120()])
    );
    assert_eq!(callers("Calc__reset"), None);
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
async fn on_teardown_hook_runs() {
    // The `#[on_teardown]`-tagged fn is forwarded through the generated
    // `Methods`/`Tool` impls, just like `on_init`.
    let mut calc = Typed(Calc::default());
    let mut prompt = Prompt::default();
    calc.on_teardown(&mut prompt).await.unwrap();
    assert!(calc.0.torn_down);
}

#[tokio::test]
async fn generic_tool_defaults_name_to_ident() {
    assert_eq!(<Holder as Methods>::NAME, "Holder");
    let mut holder = Typed(Holder::default());
    let names: Vec<_> = holder
        .definitions()
        .into_iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(names, vec!["Holder__put".to_string()]);

    let r = holder
        .call(use_call("Holder__put", serde_json::json!({"v": "x"})))
        .await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(holder.0.items.len(), 1);
}

// A `#[tool(flat)]` tool: method names reach the wire bare.
#[derive(Deserialize, JsonSchema)]
struct Ping {}

#[derive(Default)]
struct Sonar {
    pings: usize,
}

#[tool(flat, name = "sonar")]
impl Sonar {
    /// Ping.
    #[method]
    async fn ping(&mut self, _args: Ping) -> Result<Content, Content> {
        self.pings += 1;
        Ok("pong".into())
    }
}

// A second flat tool sharing `Sonar`'s method *name*, to prove the route
// collision assert. Its own `Args` type: the macro impls `ToolArgs` on the
// args struct, so one struct can't serve methods of two tools.
#[derive(Deserialize, JsonSchema)]
struct PingB {}

#[derive(Default)]
struct SonarB;

#[tool(flat, name = "sonar_b")]
impl SonarB {
    /// Ping.
    #[method]
    async fn ping(&mut self, _args: PingB) -> Result<Content, Content> {
        Ok("pong".into())
    }
}

#[test]
// Pins a compile-time associated const, like `DEFER_LOADING` above.
#[allow(clippy::assertions_on_constants)]
fn flat_definitions_are_bare() {
    assert!(<Sonar as Methods>::FLAT);
    assert!(!<Calc as Methods>::FLAT);
    let names: Vec<_> = Typed(Sonar::default())
        .definitions()
        .into_iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(names, vec!["ping".to_string()]);
}

#[tokio::test]
async fn flat_tool_in_flat_toolbox_is_bare_end_to_end() {
    use misanthropic::tool::ToolBox;

    let mut tools = ToolBox::flat().add(Sonar::default());
    let mut prompt = Prompt::default();
    tools.prepare(&mut prompt).await.unwrap();

    let names: Vec<_> = prompt
        .tools
        .as_ref()
        .unwrap()
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(names, vec!["ping".to_string()]);

    // The model emits exactly the installed name; it must route.
    let r = tools.call(use_call("ping", serde_json::json!({}))).await;
    assert!(!r.is_error, "{}", r.content);
    assert_eq!(r.content.to_string(), "pong");
}

#[tokio::test]
async fn flat_tool_in_default_toolbox_keeps_the_box_segment() {
    use misanthropic::tool::ToolBox;

    // Only the tool is flat: the (non-flat) box still adds its segment.
    let mut tools = ToolBox::new().add(Sonar::default());
    let mut prompt = Prompt::default();
    tools.prepare(&mut prompt).await.unwrap();

    let names: Vec<_> = prompt
        .tools
        .as_ref()
        .unwrap()
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(names, vec!["toolbox__ping".to_string()]);

    let r = tools
        .call(use_call("toolbox__ping", serde_json::json!({})))
        .await;
    assert!(!r.is_error, "{}", r.content);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "already claimed")]
fn flat_route_collision_asserts_in_debug() {
    use misanthropic::tool::ToolBox;

    // Two different flat tools exposing the same bare method name.
    let _ = ToolBox::flat().add(Sonar::default()).add(SonarB);
}

#[tokio::test]
async fn save_and_load_round_trip() {
    let mut calc = Calc {
        acc: 42,
        ..Default::default()
    };
    // `Calc` impls both `Tool` and `Methods` (they share these method names),
    // and both are in scope here, so qualify which one we mean.
    let json = Tool::save_json(&mut calc).await;
    let mut other = Calc::default();
    Tool::load_json(&mut other, json).await.unwrap();
    assert_eq!(other.acc, 42);
}
