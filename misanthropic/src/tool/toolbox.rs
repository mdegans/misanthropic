use std::{borrow::Cow, collections::BTreeMap};

use serde::{Deserialize, Serialize};

use crate::{
    Prompt,
    tool::{
        self, Mailbox, MethodDef, Methods, Notifications, Tool, Typed, Use,
    },
};

/// Container [`Tool`] that calls [`Tool`]s. Nestable, however consider if this
/// is really necessary.
///
/// Tools are offered in insertion order, which leads the cached prefix: it is
/// deterministic only if tools are registered in a fixed order. Append new
/// tools at the end so the earlier tools' cache still hits.
///
/// [`call`]: ToolBox::call
pub struct ToolBox {
    /// Name of the [`ToolBox`].
    name: Cow<'static, str>,
    /// Map of [`MethodDef::name`] to tool name of the [`Tool`] to call.
    ///
    /// Stores namespaced function names in the format `tool__function`.
    pub(crate) method_to_tool_name: BTreeMap<Cow<'static, str>, String>,
    /// The [`Tool`]s, in insertion order — the order [`Tool::definitions`]
    /// renders them in, so appending one keeps the earlier tools' cache.
    tools: Vec<Box<dyn Tool + Send>>,
    /// Map of tool names to their index in `tools`.
    tool_index: BTreeMap<String, usize>,
    /// This box's outbox — owns the aggregate channel. Each tool gets a
    /// send-only [`derive`](Mailbox::derive)d handle on it; the box's own
    /// receiver is taken by [`Tool::subscribe`]. `None` after
    /// [`teardown_tools`](Self::teardown_tools) drops it; a nested [`ToolBox`]
    /// adopts its parent's (send-only) handle here (see [`ToolBox`]'s
    /// [`Tool::connect`]).
    mailbox: Option<Mailbox>,
    /// The consumer end a driver [`park`](Self::park)ed after a run, handed
    /// out again by the next [`Tool::subscribe`] — the channel outlives the
    /// box's own sender, as the tools hold theirs.
    parked: Option<Notifications>,
    /// Source-path prefix for stamping child mailboxes. `None` at the root (a
    /// source is the bare tool name); `Some("root/child")` once nested, so
    /// sources compose `parent/child/leaf`.
    source_prefix: Option<String>,
    /// When `true`, this box adds no `box__` segment to wire names or routes.
    /// See [`flat`](Self::flat).
    flat: bool,
}

impl Default for ToolBox {
    fn default() -> Self {
        Self {
            name: "toolbox".into(), // module syntax, snake case
            method_to_tool_name: BTreeMap::new(),
            tools: Vec::new(),
            tool_index: BTreeMap::new(),
            mailbox: Some(Mailbox::new("toolbox")),
            parked: None,
            source_prefix: None,
            flat: false,
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

    /// Create a `ToolBox` that adds **no** `box__` segment to wire names or
    /// routes: a [`FLAT`](Methods::FLAT) tool's methods reach the wire
    /// completely bare (`create_post`), a namespaced tool's as `tool__method`.
    /// The box still has a [`name`](Tool::name) (state key, mailbox source) —
    /// flatness only affects method naming. Bare names forfeit the collision
    /// protection namespacing exists for; [`push_boxed`](Self::push_boxed)
    /// debug-asserts that no route is claimed by two different tools.
    pub fn flat() -> Self {
        Self {
            flat: true,
            ..Self::default()
        }
    }

    /// Add a [`Tool`] to the [`ToolBox`].
    ///
    /// # Note:
    /// - A duplicate [`MethodDef`] (by name) overwrites the earlier route, and a
    ///   [`Tool`] whose name already exists replaces the earlier tool in place
    ///   (keeping its position). Stale routes from a differing method set are
    ///   not pruned, so treat tool names as unique.
    // Deliberate builder-style name (mirrors `add_boxed`); not `ops::Add::add`.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, tool: impl Tool + 'static) -> Self {
        self.push(tool);
        self
    }

    /// Add a boxed [`Tool`] to the [`ToolBox`].
    ///
    /// # Note:
    /// - A duplicate [`MethodDef`] (by name) overwrites the earlier route, and a
    ///   [`Tool`] whose name already exists replaces the earlier tool in place
    ///   (keeping its position). Stale routes from a differing method set are
    ///   not pruned, so treat tool names as unique.
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
    pub fn push_boxed(&mut self, mut tool: Box<dyn Tool + Send>) {
        // Build a route per definition. Custom methods are namespaced under
        // this box (`box__tool__method`); a server-declared def (e.g. the
        // client-executed `memory` tool) keeps its fixed bare wire name — the
        // model emits exactly that, so prefixing it would break routing. A bare
        // name has no `box__` prefix to strip on descent, so it passes through
        // nested boxes untouched, which is what makes it reachable from the
        // root. A [`flat`](Self::flat) box adds no segment of its own.
        for def in tool.definitions() {
            let route = if def.is_server() || self.flat {
                def.name().to_string()
            } else {
                format!("{}{}{}", self.name, Self::SEP, def.name())
            };
            let claimed = self
                .method_to_tool_name
                .insert(route.clone().into(), tool.name().to_string());
            // Two *different* tools claiming one route is unreachable code
            // waiting to run — the risk `flat` naming reintroduces. Replacing
            // a same-named tool stays a documented overwrite.
            debug_assert!(
                claimed.as_deref().is_none_or(|t| t == tool.name()),
                "method route `{route}` already claimed by tool \
                 `{}`; renaming or un-flattening one of the tools is required",
                claimed.as_deref().unwrap_or_default(),
            );
        }

        // Hand the tool a send-only handle on this box's channel, stamped with
        // its (namespaced) source, so it can push [`Notification`]s. Skipped once
        // the box has been torn down (no mailbox).
        if let Some(mailbox) = &self.mailbox {
            let source = match &self.source_prefix {
                Some(prefix) => format!("{prefix}/{}", tool.name()),
                None => tool.name().to_string(),
            };
            tool.connect(mailbox.derive(source));
        }

        // A same-named tool is replaced in place, so it keeps its position
        // (and the cached prefix up to it).
        match self.tool_index.get(tool.name()) {
            Some(&index) => {
                #[allow(unused_variables)] // because of the `log` feature
                let existing = std::mem::replace(&mut self.tools[index], tool);
                #[cfg(feature = "log")]
                log::debug!("Tool replaced: {}", existing.name());
            }
            None => {
                self.tool_index
                    .insert(tool.name().to_string(), self.tools.len());
                self.tools.push(tool);
            }
        }
    }

    /// Names of all [`Tool`]s in the [`ToolBox`], in insertion order.
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tools.iter().map(|tool| tool.name())
    }

    /// The [`Tool`] named `name`, if any.
    pub(crate) fn tool_mut(
        &mut self,
        name: &str,
    ) -> Option<&mut (dyn Tool + Send + 'static)> {
        let index = *self.tool_index.get(name)?;
        Some(self.tools[index].as_mut())
    }

    /// Names of all the [`MethodDef`]s in the [`ToolBox`].
    pub fn method_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.method_to_tool_name.keys().map(|name| name.as_ref())
    }

    /// Install this toolbox into `prompt`: overwrite [`Prompt::tools`] with
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
        prompt.tools = Some(self.definitions());
        self.init_tools(prompt).await
    }

    /// Initialize all tools in the toolbox. Call this once when setting up a conversation.
    pub async fn init_tools(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        for tool in self.tools.iter_mut() {
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

        for tool in self.tools.iter_mut() {
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

    /// Give back the stream [`Tool::subscribe`] handed out, so the next
    /// subscriber — a resumed `Chat` — gets it, with whatever
    /// the tools pushed in between still queued. A driver parks it after
    /// [`teardown_tools`](Self::teardown_tools), which drops only the box's
    /// own sender.
    #[cfg(any(feature = "chat", test))]
    pub(crate) fn park(&mut self, notifications: Notifications) {
        self.parked = Some(notifications);
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

        for tool in self.tools.iter_mut() {
            #[cfg(feature = "log")]
            log::debug!("Tearing down tool: {}", tool.name());

            if let Err(e) = tool.on_teardown(prompt).await {
                #[cfg(feature = "log")]
                log::error!("Error tearing down tool {}: {}", tool.name(), e);
                errors.push(e);
            }
        }

        // Drop our own outbox so a `recv()`-driven consumer can see the stream
        // close once the tools (which hold the other senders) also drop theirs.
        self.mailbox = None;

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

    /// The [`MethodDef`]s for all [`Tool`]s in the [`ToolBox`], in insertion
    /// order. Register tools in a fixed order so identical boxes share a
    /// prompt cache, and append new ones so earlier tools' cache still hits.
    fn definitions(&self) -> Vec<MethodDef> {
        self.tools
            .iter()
            .flat_map(|tool| {
                tool.definitions().into_iter().map(|mut def| {
                    // Prefix custom method names with this box's segment
                    // (they already carry `tool__method`); leave server defs
                    // bare so their fixed wire name survives every nesting
                    // level, and add nothing when the box is
                    // [`flat`](Self::flat). See [`push_boxed`](Self::push_boxed).
                    if !self.flat
                        && let Some(method) = def.as_method_mut()
                    {
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
                // Prompt::tools.
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

        // Borrow the fields apart: the name and flatness are read below.
        let index = self.tool_index.get(&tool_name).copied();
        if let Some(tool) = index.map(|index| &mut self.tools[index]) {
            // Strip this box's own namespace segment before descending, so a
            // sub-tool sees a name relative to itself. A nested [`ToolBox`]
            // keys its routes by its *own* name only (`tool__method`), so it
            // would not recognize the outer-qualified `box__tool__method` we
            // looked up here. Leaf tools rsplit on [`Self::SEP`] and read only
            // the final segment, so this is a no-op for them. A
            // [`flat`](Self::flat) box added no segment, so none is stripped —
            // guarded rather than assumed, against a method name that happens
            // to start with `{box}__`.
            let mut call = call;
            let prefix = format!("{}{}", self.name, Self::SEP);
            if !self.flat
                && let Some(rest) =
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
            let tool = match self.tool_mut(&name) {
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

        for tool in self.tools.iter_mut() {
            let tool_state = tool.save_json().await;
            tools.insert(tool.name().to_string(), tool_state);
        }

        let state = State {
            name: self.name.clone(),
            tools,
        };

        serde_json::to_value(state).unwrap()
    }

    fn connect(&mut self, mailbox: Mailbox) {
        // Nested: adopt the parent's (send-only) handle and re-stamp our whole
        // subtree's sources under the path that reaches us. Children were
        // connected to our *own* channel when they were added, so re-connect
        // them onto the parent's now; nested child boxes recurse through this
        // same method. The adopted handle has no receiver, so our own
        // `subscribe` now yields `None` — pushes flow to the parent.
        let prefix = mailbox.source().to_string();
        for tool in self.tools.iter_mut() {
            let source = format!("{prefix}/{}", tool.name());
            tool.connect(mailbox.derive(source));
        }
        self.source_prefix = Some(prefix);
        self.mailbox = Some(mailbox);
    }

    fn subscribe(&mut self) -> Option<Notifications> {
        self.parked
            .take()
            .or_else(|| self.mailbox.as_mut().and_then(Mailbox::subscribe))
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

    /// A one-method stand-in named `.0`, its method described by `.1`.
    struct Named(&'static str, &'static str);

    #[async_trait::async_trait]
    impl Tool for Named {
        fn name(&self) -> &str {
            self.0
        }

        fn definitions(&self) -> Vec<MethodDef> {
            let name = format!("{}__run", self.0);
            let def = CustomMethodDef::with_string_param(
                name, self.1, "what", "What.", true,
            );
            vec![MethodDef::Custom(def)]
        }

        async fn call(&mut self, call: Use) -> Result {
            Result::new(call.id, "ran")
        }
    }

    const NAMES: [&str; 8] = ["h", "c", "f", "a", "g", "b", "e", "d"];

    /// A box holding `names`, added in that order.
    fn boxed<'n>(names: impl IntoIterator<Item = &'n &'static str>) -> ToolBox {
        names
            .into_iter()
            .fold(ToolBox::new(), |b, n| b.add(Named(n, "Run.")))
    }

    /// The wire names of `toolbox`'s definitions, in order.
    fn def_names(toolbox: &ToolBox) -> Vec<String> {
        let defs = toolbox.definitions().into_iter();
        defs.map(|def| def.name().to_string()).collect()
    }

    /// Tools render first in the cached prefix, in the order they were added
    /// — so the same registration order gives the same prefix every time.
    #[test]
    fn test_definitions_follow_insertion_order() {
        let expected = |names: &mut dyn Iterator<Item = &&str>| {
            names
                .map(|n| format!("toolbox__{n}__run"))
                .collect::<Vec<_>>()
        };
        let forward = boxed(&NAMES);
        let reverse = boxed(NAMES.iter().rev());

        assert_eq!(def_names(&forward), expected(&mut NAMES.iter()));
        assert_eq!(def_names(&reverse), expected(&mut NAMES.iter().rev()));
        assert!(forward.tool_names().eq(NAMES));
        assert_eq!(def_names(&boxed(&NAMES)), def_names(&forward));
    }

    /// Appending a tool leaves the earlier definitions a byte-identical
    /// prefix, so a cache breakpoint on them still hits.
    #[test]
    fn test_append_keeps_definitions_prefix() {
        let json = |toolbox: &ToolBox| {
            serde_json::to_string(&toolbox.definitions()).unwrap()
        };
        let before = boxed(&NAMES);
        let after = boxed(&NAMES).add(Named("z", "Run."));

        let (before, after) = (json(&before), json(&after));
        // Drop the closing `]` so the shorter list is a strict prefix.
        let open = &before[..before.len() - 1];
        assert!(after.starts_with(open), "{after}\n!starts_with\n{open}");
        assert!(after.len() > before.len());
    }

    /// Re-adding a same-named tool replaces it where it stood.
    #[test]
    fn test_replace_keeps_position() {
        let toolbox = boxed(&NAMES).add(Named("f", "Run again."));

        assert!(toolbox.tool_names().eq(NAMES));
        let defs = toolbox.definitions();
        assert_eq!(defs.len(), NAMES.len());
        let f = NAMES.iter().position(|n| *n == "f").unwrap();
        let Some(MethodDef::Custom(def)) = defs.get(f) else {
            panic!("expected a custom def at {f}");
        };
        assert_eq!(def.name, "toolbox__f__run");
        assert_eq!(def.description, "Run again.");
    }

    /// A nested box keeps its own insertion order, at the position it was
    /// added in its parent.
    #[test]
    fn test_nested_definitions_order() {
        let inner = ToolBox::named("inner")
            .unwrap()
            .add(Named("y", "Run."))
            .add(Named("x", "Run."));
        let toolbox = ToolBox::new()
            .add(Named("b", "Run."))
            .add(inner)
            .add(Named("a", "Run."));

        assert!(toolbox.tool_names().eq(["b", "inner", "a"]));
        assert_eq!(
            def_names(&toolbox),
            [
                "toolbox__b__run",
                "toolbox__inner__y__run",
                "toolbox__inner__x__run",
                "toolbox__a__run",
            ]
        );
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

    /// A push-only leaf: stores its [`Mailbox`] on connect and emits one
    /// [`Notification`] from `on_init`, to prove nested sources compose.
    #[derive(Default)]
    struct Pusher {
        mailbox: Option<Mailbox>,
    }

    #[async_trait::async_trait]
    impl Tool for Pusher {
        fn name(&self) -> &str {
            "leaf"
        }
        fn definitions(&self) -> Vec<MethodDef> {
            Vec::new()
        }
        async fn call(&mut self, call: Use) -> Result {
            Result::new(call.id, "noop")
        }
        fn connect(&mut self, mailbox: Mailbox) {
            self.mailbox = Some(mailbox);
        }
        async fn on_init(
            &mut self,
            _prompt: &mut Prompt,
        ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>>
        {
            if let Some(mailbox) = &self.mailbox {
                let _ = mailbox.send(
                    crate::prompt::message::Content::text("ping"),
                    vec![crate::prompt::message::Role::User],
                );
            }
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_nested_mailbox_source_composes() {
        // root -> "mid" -> "inner" -> leaf. The leaf's push must reach the
        // root's subscriber, stamped with the full path (the root's own name is
        // implicit and omitted), proving each nesting re-stamps and re-wires the
        // subtree onto the parent's channel.
        let inner = ToolBox::named("inner").unwrap().add(Pusher::default());
        let mid = ToolBox::named("mid").unwrap().add_boxed(Box::new(inner));
        let mut root = ToolBox::new().add_boxed(Box::new(mid));

        let mut notes = root.subscribe().expect("root box has an outbox");
        let mut prompt = Prompt::default();
        root.prepare(&mut prompt).await.unwrap();

        let note = notes.try_recv().expect("leaf push reached the root");
        assert_eq!(&*note.source, "mid/inner/leaf");
        assert!(notes.try_recv().is_err(), "exactly one push");
    }

    /// A parked stream outlives teardown: the tools keep their senders, so
    /// a push after it is waiting for the next subscriber.
    #[tokio::test]
    async fn test_parked_stream_survives_teardown() {
        let mut root = ToolBox::new().add(Pusher::default());
        let mut prompt = Prompt::default();

        let notes = root.subscribe().expect("root box has an outbox");
        root.prepare(&mut prompt).await.unwrap(); // pushes "ping"
        root.teardown_tools(&mut prompt).await.unwrap();
        root.park(notes);
        root.prepare(&mut prompt).await.unwrap(); // pushes again

        let mut notes = root.subscribe().expect("the parked stream");
        assert!(notes.try_recv().is_ok() && notes.try_recv().is_ok());
        assert!(root.subscribe().is_none(), "handed out once");
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
