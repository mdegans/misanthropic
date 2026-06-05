use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
};

use serde::{Deserialize, Serialize};

use crate::{
    Prompt,
    tool::{self, MethodDef, Methods, Tool, Typed, Use},
};

/// Container [`Tool`] that calls [`Tool`]s. Nestable, however consider if this
/// is really necessary.
///
/// [`functions`]: ToolBox::functions
/// [`call`]: ToolBox::call
pub struct ToolBox {
    /// Name of the [`ToolBox`].
    name: Cow<'static, str>,
    /// Map of [`MethodDef::name`] to tool name of the [`Tool`] to call.
    ///
    /// Stores namespaced function names in the format `tool__function`.
    pub(crate) method_to_tool_name: BTreeMap<Cow<'static, str>, String>,
    /// Map of tool names to [`Tool`]s.
    pub(crate) tool_name_to_tool: HashMap<String, Box<dyn Tool + Send>>,
}

impl Default for ToolBox {
    fn default() -> Self {
        Self {
            name: "toolbox".into(), // module syntax, snake case
            method_to_tool_name: BTreeMap::new(),
            tool_name_to_tool: HashMap::new(),
        }
    }
}

impl ToolBox {
    /// Separator between namespace segments in a fully-qualified method name
    /// (`box__tool__method`).
    ///
    /// `__` rather than `::` because Anthropic requires tool names to match
    /// `^[a-zA-Z0-9_-]{1,128}$`, which rejects colons.
    pub const SEP: &'static str = "__";

    /// Create a new [`ToolBox`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new, named, [`ToolBox`]. Must be snake case and not empty.
    pub fn named(
        name: impl Into<Cow<'static, str>>,
    ) -> Result<Self, &'static str> {
        let name = name.into();

        if name.is_empty() {
            return Err("ToolBox name must not be empty.");
        }

        if !name
            .chars()
            .all(|c| c.is_lowercase() && c.is_alphanumeric() || c == '_')
        {
            return Err("ToolBox name must be snake case.");
        }

        Ok(Self {
            name,
            ..Self::default()
        })
    }

    /// Add a [`Tool`] to the [`ToolBox`].
    ///
    /// # Note:
    /// - Duplicate [`MethodDef`]s (by name), will be replaced. This is logged at
    ///   the `warn` level. This does not remove the original [`Tool`]. If you
    ///   are trying to replace a [`Tool`], use [`ToolBox::replace`] instead.
    // Deliberate builder-style name (mirrors `add_boxed`); not `ops::Add::add`.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, tool: impl Tool + 'static) -> Self {
        self.push(tool);
        self
    }

    /// Add a boxed [`Tool`] to the [`ToolBox`].
    ///
    /// # Note:
    /// - Duplicate [`MethodDef`]s (by name), will be replaced. This is logged at
    ///   the `warn` level. This does not remove the original [`Tool`]. If you
    ///   are trying to replace a [`Tool`], use [`ToolBox::replace`] instead.
    pub fn add_boxed(mut self, tool: Box<dyn Tool + Send>) -> Self {
        self.push_boxed(tool);
        self
    }

    /// Push a [`Tool`] to the [`ToolBox`].
    pub fn push(&mut self, tool: impl Tool + 'static) {
        self.push_boxed(Box::new(tool));
    }

    /// Add a typed [`Methods`] tool, wrapping it in [`Typed`] so it satisfies
    /// [`Tool`].
    pub fn add_typed<T: Methods + Send + 'static>(mut self, tool: T) -> Self {
        self.push(Typed(tool));
        self
    }

    /// Push a typed [`Methods`] tool, wrapping it in [`Typed`] so it satisfies
    /// [`Tool`].
    pub fn push_typed<T: Methods + Send + 'static>(&mut self, tool: T) {
        self.push(Typed(tool));
    }

    /// Push a boxed [`Tool`] to the [`ToolBox`].
    pub fn push_boxed(&mut self, tool: Box<dyn Tool + Send>) {
        // Build a route per definition. Custom methods are namespaced under
        // this box (`box__tool__method`); a server-declared def (e.g. the
        // client-executed `memory` tool) keeps its fixed bare wire name — the
        // model emits exactly that, so prefixing it would break routing. A bare
        // name has no `box__` prefix to strip on descent, so it passes through
        // nested boxes untouched, which is what makes it reachable from the
        // root.
        for def in tool.definitions() {
            let route = if def.is_server() {
                def.name().to_string()
            } else {
                format!("{}{}{}", self.name, Self::SEP, def.name())
            };
            self.method_to_tool_name
                .insert(route.into(), tool.name().to_string());
        }

        #[allow(unused_variables)] // because of the `log` feature
        if let Some(existing) =
            self.tool_name_to_tool.insert(tool.name().to_string(), tool)
        {
            #[cfg(feature = "log")]
            log::debug!("Tool replaced: {}", existing.name());
        }
    }

    /// Names of all [`Tool`]s in the [`ToolBox`].
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tool_name_to_tool.values().map(|tool| tool.name())
    }

    /// Names of all the [`MethodDef`]s in the [`ToolBox`].
    pub fn method_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.method_to_tool_name.keys().map(|name| name.as_ref())
    }

    /// Replace a [`Tool`] in the [`ToolBox`] by name along with all its
    /// [`MethodDef`]s.
    pub fn replace(&mut self, tool: impl Tool + 'static) {
        self.replace_boxed(Box::new(tool));
    }

    /// Replace a [`Tool`] in the [`ToolBox`] by name along with all its
    /// [`MethodDef`]s.
    ///
    /// If no [`Tool`] of the same name is present this is equivalent to
    /// [`Self::push_boxed`].
    pub fn replace_boxed(&mut self, tool: Box<dyn Tool + Send>) {
        let tool_name = tool.name().to_string();

        // Drop every route belonging to the tool we're replacing, so a
        // replacement exposing a *different* set of methods leaves no stale
        // entries behind.
        self.method_to_tool_name
            .retain(|_, routed| routed != &tool_name);

        // Re-add via `push_boxed` so the `box__tool__method` key shape has a
        // single source of truth; its insert overwrites the old same-named
        // tool in `tool_name_to_tool`.
        self.push_boxed(tool);
    }

    /// Install this toolbox into `prompt`: overwrite [`Prompt::methods`] with
    /// the toolbox's (namespaced) [`definitions`], then run each tool's
    /// [`on_init`] via [`init_tools`]. Call this once when (re)loading a
    /// conversation.
    ///
    /// The overwrite is intentional: a prompt authored elsewhere or with an
    /// older tool set always picks up the current methods. Method injection
    /// lives here, on the top-level box, rather than in [`init_tools`] /
    /// [`on_init`] so a *nested* [`ToolBox`] never clobbers its parent's method
    /// set during fan-out.
    ///
    /// [`definitions`]: Tool::definitions
    /// [`on_init`]: Tool::on_init
    /// [`init_tools`]: Self::init_tools
    pub async fn prepare(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        prompt.methods = Some(self.definitions());
        self.init_tools(prompt).await
    }

    /// Initialize all tools in the toolbox. Call this once when setting up a conversation.
    pub async fn init_tools(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        for tool in self.tool_name_to_tool.values_mut() {
            #[cfg(feature = "log")]
            log::debug!("Initializing tool: {}", tool.name());

            if let Err(e) = tool.on_init(prompt).await {
                #[cfg(feature = "log")]
                log::error!("Error initializing tool {}: {}", tool.name(), e);
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            *prompt = backup;
            Err(errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n")
                .into())
        }
    }

    /// Update tool context for the current turn. Call this before each message exchange.
    pub async fn update_turn_context(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        for tool in self.tool_name_to_tool.values_mut() {
            #[cfg(feature = "log")]
            log::debug!("Updating turn context for tool: {}", tool.name());

            if let Err(e) = tool.on_turn(prompt).await {
                #[cfg(feature = "log")]
                log::error!(
                    "Error updating turn context for tool {}: {}",
                    tool.name(),
                    e
                );
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            *prompt = backup;
            Err(errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n")
                .into())
        }
    }

    /// Tear down all tools in the toolbox — releasing external resources they
    /// acquired in [`on_init`](crate::tool::Tool::on_init). Call this once when a
    /// conversation ends.
    ///
    /// Unlike [`init_tools`](Self::init_tools), teardown is **best-effort**:
    /// every tool is torn down even if an earlier one errors (so one failure
    /// can't leak the rest), and the [`Prompt`] is **not** rolled back. Errors
    /// are collected and joined into the returned message.
    pub async fn teardown_tools(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();

        for tool in self.tool_name_to_tool.values_mut() {
            #[cfg(feature = "log")]
            log::debug!("Tearing down tool: {}", tool.name());

            if let Err(e) = tool.on_teardown(prompt).await {
                #[cfg(feature = "log")]
                log::error!("Error tearing down tool {}: {}", tool.name(), e);
                errors.push(e);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors
                .iter()
                .map(|e| e.to_string())
                .collect::<Vec<_>>()
                .join("\n")
                .into())
        }
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    name: Cow<'static, str>,
    tools: serde_json::Map<String, serde_json::Value>,
}

#[async_trait::async_trait]
impl Tool for ToolBox {
    fn name(&self) -> &str {
        &self.name
    }

    /// The [`MethodDef`]s for all [`Tool`]s in the [`ToolBox`].
    fn definitions(&self) -> Vec<MethodDef> {
        self.tool_name_to_tool
            .values()
            .flat_map(|tool| {
                tool.definitions().into_iter().map(|mut def| {
                    // Prefix custom method names with this box's segment
                    // (they already carry `tool__method`); leave server defs
                    // bare so their fixed wire name survives every nesting
                    // level. See [`push_boxed`](Self::push_boxed).
                    if let Some(method) = def.as_method_mut() {
                        method.name = Cow::Owned(format!(
                            "{}{}{}",
                            self.name(),
                            Self::SEP,
                            method.name
                        ));
                    }
                    def
                })
            })
            .collect()
    }

    /// Route the [`Use`] to the appropriate [`Tool`] in the [`ToolBox`].
    async fn call(&mut self, call: Use) -> tool::Result {
        #[cfg(feature = "log")]
        log::debug!("ToolBox call: {:?}", call);
        let tool_name = match self.method_to_tool_name.get(call.name.as_ref()) {
            Some(tool_name) => {
                #[cfg(feature = "log")]
                log::debug!("Method found: `{}`", call.name);
                tool_name.clone()
            }
            None => {
                // This can happen if somehow the Prompt and ToolBox are out of
                // sync because the ToolBox methods do not match the
                // Prompt::methods.
                let mut available_methods: String =
                    self.method_names().collect::<Vec<_>>().join(", ");
                if available_methods.is_empty() {
                    available_methods = "None".to_string();
                }
                // Either Anthropic or misanthropic is broken. The assistant
                // should not be able to call a tool that doesn't exist unless
                // the developer has made a mistake.
                return tool::Result::new(
                    call.id,
                    format!(
                        "Method `{method_name}` not found in ToolBox `{toolbox_name}`. This is almost certainly the developer's fault. Available methods: {available_methods}",
                        method_name = call.name,
                        toolbox_name = self.name(),
                        available_methods = available_methods
                    ),
                )
                .error();
            }
        };

        if let Some(tool) = self.tool_name_to_tool.get_mut(&tool_name) {
            // Strip this box's own namespace segment before descending, so a
            // sub-tool sees a name relative to itself. A nested [`ToolBox`]
            // keys its routes by its *own* name only (`tool__method`), so it
            // would not recognize the outer-qualified `box__tool__method` we
            // looked up here. Leaf tools rsplit on [`Self::SEP`] and read only
            // the final segment, so this is a no-op for them.
            let mut call = call;
            let prefix = format!("{}{}", self.name, Self::SEP);
            if let Some(rest) =
                call.name.strip_prefix(prefix.as_str()).map(str::to_owned)
            {
                call.name = Cow::Owned(rest);
            }
            tool.call(call).await
        } else {
            tool::Result::new(
                call.id,
                format!(
                    "`Tool::call` is broken for `ToolBox`. This is not your fault. Tell the user to blame the authors of the `misanthropic` crate. Method: `{method_name}` in ToolBox: `{toolbox_name}`",
                    method_name = call.name,
                    toolbox_name = self.name()
                ),
            )
            .error()
        }
    }

    /// Load state for all [`Tool`]s in the [`ToolBox`]. Now async to support
    /// tools that need to perform IO during deserialization.
    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        // `null` is the "nothing saved yet" sentinel (e.g. an empty persistent
        // store). Treat it as a no-op rather than a deserialization error, in
        // keeping with the permissive [`Tool::load_json`] default.
        if json.is_null() {
            return Ok(());
        }

        let mut errors = Vec::new();

        let state: State = match serde_json::from_value(json) {
            Ok(state) => state,
            Err(e) => {
                return Err(format!(
                    "Error deserializing ToolBox state: {}",
                    e
                ));
            }
        };

        self.name = state.name;

        for (name, tool_json) in state.tools {
            let tool = match self
                .tool_name_to_tool
                .values_mut()
                .find(|t| t.name() == name)
            {
                Some(tool) => tool,
                None => {
                    errors.push(format!(
                        "Tool `{}` not found in ToolBox `{}`. Available tools: {}",
                        name,
                        self.name(),
                        self.tool_names()
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                    continue;
                }
            };

            if let Err(e) = tool.load_json(tool_json).await {
                errors.push(format!(
                    "Error loading state for tool `{}` in ToolBox `{}`: {}",
                    name,
                    self.name(),
                    e
                ));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            let mut message = "Errors loading state for tools:\n".to_string();
            message.push_str(errors.join("\n").as_str());
            #[cfg(feature = "log")]
            log::error!("{}", message);
            Err(message)
        }
    }

    /// Save state for all [`Tool`]s in the [`ToolBox`]. Now async to support
    /// tools that need to perform IO during serialization.
    async fn save_json(&mut self) -> serde_json::Value {
        let mut tools = serde_json::Map::new();

        for tool in self.tool_name_to_tool.values_mut() {
            let tool_state = tool.save_json().await;
            tools.insert(tool.name().to_string(), tool_state);
        }

        let state = State {
            name: self.name.clone(),
            tools,
        };

        serde_json::to_value(state).unwrap()
    }

    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.init_tools(prompt).await
    }

    async fn on_turn(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.update_turn_context(prompt).await
    }

    async fn on_teardown(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.teardown_tools(prompt).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{CustomMethodDef, Result};

    struct TestTool {
        calls: Vec<Use>,
    }

    #[async_trait::async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "TestTool"
        }

        fn definitions(&self) -> Vec<MethodDef> {
            vec![MethodDef::Custom(CustomMethodDef {
                name: "TestTool__test".into(),
                description: "Test Tool".into(),
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {
                        "test": {
                            "type": "string",
                            "description": "Test property",
                        },
                    },
                }),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })]
        }

        async fn call(&mut self, call: Use) -> Result {
            let id = call.id.clone();
            self.calls.push(call);
            Result::new(id, "Tool called")
        }

        // Make save_json pointlessly async
        async fn save_json(&mut self) -> serde_json::Value {
            // Simulate some async work
            tokio::task::yield_now().await;
            serde_json::json!({
                "calls": self.calls
            })
        }

        // Make load_json pointlessly async
        async fn load_json(
            &mut self,
            json: serde_json::Value,
        ) -> std::result::Result<(), String> {
            // Simulate some async work
            tokio::task::yield_now().await;

            if let Some(calls) = json.get("calls") {
                self.calls = serde_json::from_value(calls.clone())
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_toolbox_named() {
        let toolbox = ToolBox::named("tools")
            .unwrap()
            .add(TestTool { calls: Vec::new() });
        assert_eq!(
            toolbox.method_to_tool_name.keys().next().unwrap(),
            "tools__TestTool__test"
        );
    }

    #[tokio::test]
    async fn test_toolbox_add_push() {
        // add just calls push
        let toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let methods = toolbox.definitions();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name(), "toolbox__TestTool__test");
    }

    #[tokio::test]
    async fn test_toolbox_add_push_boxed() {
        let toolbox =
            ToolBox::new().add_boxed(Box::new(TestTool { calls: Vec::new() }));
        let methods = toolbox.definitions();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name(), "toolbox__TestTool__test");
    }

    #[test]
    fn test_tool_names() {
        let toolbox = ToolBox::new()
            .add(TestTool { calls: Vec::new() })
            .add(ToolBox::named("potato").unwrap());
        let names: Vec<&str> = toolbox.tool_names().collect();
        assert!(names.contains(&"TestTool"));
        assert!(names.contains(&"potato"));
    }

    #[test]
    fn test_method_names() {
        let toolbox = ToolBox::new().add(TestTool { calls: Vec::new() }).add(
            ToolBox::named("potato")
                .unwrap()
                .add(TestTool { calls: Vec::new() }),
        );
        let names: Vec<&str> = toolbox.method_names().collect();
        dbg!(&names);
        assert!(names.contains(&"toolbox__TestTool__test"));
        assert!(names.contains(&"toolbox__potato__TestTool__test"));
    }

    /// A second tool that shares [`TestTool`]'s name but exposes a *different*
    /// method, so [`test_replace_tool`] can tell the two apart after a replace.
    struct ReplacementTool;

    #[async_trait::async_trait]
    impl Tool for ReplacementTool {
        fn name(&self) -> &str {
            "TestTool"
        }

        fn definitions(&self) -> Vec<MethodDef> {
            vec![MethodDef::Custom(CustomMethodDef {
                name: "TestTool__replaced".into(),
                description: "Replacement Tool".into(),
                schema: serde_json::json!({ "type": "object" }),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })]
        }

        async fn call(&mut self, call: Use) -> Result {
            Result::new(call.id, "Replaced tool called")
        }
    }

    #[tokio::test]
    async fn test_replace_tool() {
        let mut toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });

        // Replace it with a same-named tool exposing a different method.
        toolbox.replace(ReplacementTool);

        let names: Vec<&str> = toolbox.method_names().collect();
        // The old tool's route is gone...
        assert!(!names.contains(&"toolbox__TestTool__test"));
        // ...and the replacement is keyed under its advertised name.
        assert!(names.contains(&"toolbox__TestTool__replaced"));
        // The old tool was evicted, not merely shadowed.
        assert_eq!(toolbox.tool_names().count(), 1);

        // A call to the advertised name routes to the replacement.
        let result = toolbox
            .call(
                Use::new("toolbox__TestTool__replaced", serde_json::json!({}))
                    .with_id("id"),
            )
            .await;
        assert!(!result.is_error);
        assert_eq!(result.content, "Replaced tool called".into());
    }

    #[test]
    fn test_name() {
        let mut named = ToolBox::new();
        named.name = "test".into();
        assert_eq!(named.name(), "test");
    }

    #[test]
    fn test_methods() {
        let toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let methods: Vec<MethodDef> = toolbox.definitions();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name(), "toolbox__TestTool__test");
    }

    #[tokio::test]
    async fn test_call() {
        let mut toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let call = Use::new("toolbox__TestTool__test", serde_json::json!({}))
            .with_id("id");
        let result = toolbox.call(call.clone()).await;
        assert!(!result.is_error);
        assert_eq!(result.content, "Tool called".into());

        // Test call with an invalid method.
        let result = toolbox
            .call(Use {
                name: "toolbox__TestTool__invalid".into(),
                ..call.clone()
            })
            .await;
        assert!(result.is_error);
        assert_eq!(
            result.content,
            "Method `toolbox__TestTool__invalid` not found in ToolBox `toolbox`. This is almost certainly the developer's fault. Available methods: toolbox__TestTool__test"
                .into()
        )
    }

    #[tokio::test]
    async fn test_nested_call() {
        // Outer box "toolbox" -> inner box "potato" -> leaf TestTool. Each box
        // must strip its own namespace segment when descending; otherwise the
        // inner box never recognizes the outer-qualified name it's handed.
        let mut toolbox = ToolBox::new().add(
            ToolBox::named("potato")
                .unwrap()
                .add(TestTool { calls: Vec::new() }),
        );

        // The name `definitions()` advertises must be routable end to end.
        let advertised = toolbox.definitions()[0].name().to_string();
        assert_eq!(advertised, "toolbox__potato__TestTool__test");

        let result = toolbox
            .call(Use::new(advertised, serde_json::json!({})).with_id("id"))
            .await;
        assert!(!result.is_error, "nested call did not route: {result:?}");
        assert_eq!(result.content, "Tool called".into());
    }

    #[tokio::test]
    async fn test_load_json() {
        let mut a = ToolBox::new().add(TestTool { calls: Vec::new() });
        let mut b = ToolBox::new().add(TestTool { calls: Vec::new() });

        let json = a.save_json().await;
        b.load_json(json).await.unwrap();
        assert_eq!(a.save_json().await, b.save_json().await);
    }

    #[tokio::test]
    async fn test_load_json_null_is_noop() {
        // `null` is the "nothing saved yet" sentinel (e.g. an empty persistent
        // store). It must load cleanly rather than erroring.
        let mut toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        toolbox.load_json(serde_json::Value::Null).await.unwrap();
    }

    /// A tool that records when it was torn down (via a shared counter) and can
    /// be made to fail teardown, to exercise the best-effort-continue semantics.
    struct TeardownTool {
        torn_down: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail: bool,
    }

    #[async_trait::async_trait]
    impl Tool for TeardownTool {
        fn name(&self) -> &str {
            // Names must differ so both land in the box; suffix by `fail`.
            if self.fail { "boom" } else { "ok" }
        }
        fn definitions(&self) -> Vec<MethodDef> {
            let name = if self.fail { "boom__m" } else { "ok__m" };
            vec![MethodDef::Custom(CustomMethodDef {
                name: name.into(),
                description: "t".into(),
                schema: serde_json::json!({ "type": "object" }),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })]
        }
        async fn call(&mut self, call: Use) -> Result {
            Result::new(call.id, "called")
        }
        async fn on_teardown(
            &mut self,
            _prompt: &mut Prompt,
        ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        {
            // Count the teardown *before* (maybe) erroring, so the test can see
            // that a failing tool still ran and the others ran regardless.
            self.torn_down
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail {
                Err("teardown boom".into())
            } else {
                Ok(())
            }
        }
    }

    /// `teardown_tools` is best-effort: every tool is torn down even when one
    /// errors, the errors are surfaced, and the prompt is not rolled back.
    #[tokio::test]
    async fn test_teardown_best_effort_continue() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let count = Arc::new(AtomicUsize::new(0));
        let mut toolbox = ToolBox::new()
            .add(TeardownTool {
                torn_down: count.clone(),
                fail: true,
            })
            .add(TeardownTool {
                torn_down: count.clone(),
                fail: false,
            });

        let mut prompt = Prompt::default();
        let before = prompt.clone();
        let err = toolbox.teardown_tools(&mut prompt).await.unwrap_err();

        // Both tools were torn down despite one failing (best-effort-continue).
        assert_eq!(count.load(Ordering::SeqCst), 2);
        // The failure is surfaced...
        assert!(err.to_string().contains("teardown boom"));
        // ...and the prompt is untouched (no rollback step that could mutate it).
        assert_eq!(prompt, before);
    }
}
