use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
};

use serde::{Deserialize, Serialize};

use crate::{
    Prompt,
    tool::{self, Method, Tool, Use},
};

/// Container [`Tool`] that calls [`Tool`]s. Nestable, however consider if this
/// is really necessary.
///
/// [`functions`]: ToolBox::functions
/// [`call`]: ToolBox::call
pub struct ToolBox {
    /// Name of the [`ToolBox`].
    name: Cow<'static, str>,
    /// Map of [`Method::name`] to tool name of the [`Tool`] to call.
    ///
    /// Stores namespaced function names in the format `tool::function`.
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
    /// - Duplicate [`Method`]s (by name), will be replaced. This is logged at
    ///   the `warn` level. This does not remove the original [`Tool`]. If you
    ///   are trying to replace a [`Tool`], use [`ToolBox::replace`] instead.
    pub fn add(mut self, tool: impl Tool + Send + 'static) -> Self {
        self.push(tool);
        self
    }

    /// Add a boxed [`Tool`] to the [`ToolBox`].
    ///
    /// # Note:
    /// - Duplicate [`Method`]s (by name), will be replaced. This is logged at
    ///   the `warn` level. This does not remove the original [`Tool`]. If you
    ///   are trying to replace a [`Tool`], use [`ToolBox::replace`] instead.
    pub fn add_boxed(mut self, tool: Box<dyn Tool + Send>) -> Self {
        self.push_boxed(tool);
        self
    }

    /// Push a [`Tool`] to the [`ToolBox`].
    pub fn push(&mut self, tool: impl Tool + Send + 'static) {
        self.push_boxed(Box::new(tool));
    }

    /// Push a boxed [`Tool`] to the [`ToolBox`].
    pub fn push_boxed(&mut self, tool: Box<dyn Tool + Send>) {
        // Append the function names to self.functions.
        for method in tool.methods() {
            self.method_to_tool_name.insert(
                format!("{}::{}", self.name, method.name).into(),
                tool.name().to_string(),
            );
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

    /// Names of all the [`Method`]s in the [`ToolBox`].
    pub fn method_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.method_to_tool_name.keys().map(|name| name.as_ref())
    }

    /// Replace a [`Tool`] in the [`ToolBox`] by name along with all its
    /// [`Method`]s.
    pub fn replace(&mut self, tool: impl Tool + Send + 'static) {
        self.replace_boxed(Box::new(tool));
    }

    /// Replace a [`Tool`] in the [`ToolBox`] by name along with all its
    /// [`Method`]s.
    pub fn replace_boxed(&mut self, tool: Box<dyn Tool + Send>) {
        let self_name = self.name.as_ref();
        let tool_name = tool.name().to_string();
        let function_names = tool.methods().map(|method| {
            format!(
                "{self_name}::{tool}::{method}",
                tool = tool.name(),
                method = method.name
            )
        });

        // Remove the old tool and its functions.
        for name in function_names {
            if let Some(old_tool_name) = self
                .method_to_tool_name
                .insert(name.into(), tool_name.clone())
            {
                self.tool_name_to_tool.remove(&old_tool_name);
            }
        }
    }

    /// Initialize all tools in the toolbox. Call this once when setting up a conversation.
    pub async fn init_tools(
        &mut self,
        prompt: &mut Prompt<'_>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        for (_, tool) in &mut self.tool_name_to_tool {
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
        prompt: &mut Prompt<'_>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        for (_, tool) in &mut self.tool_name_to_tool {
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

    /// The [`Method`]s for all [`Tool`]s in the [`ToolBox`].
    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new(self.tool_name_to_tool.values().flat_map(|tool| {
            tool.methods().map(|mut method| {
                // Append our prefix to the function name, which should already
                // include `tool::function` format for the function name.
                method.name =
                    Cow::Owned(format!("{}::{}", self.name(), method.name));
                method
            })
        }))
    }

    /// Route the [`Use`] to the appropriate [`Tool`] in the [`ToolBox`].
    async fn call<'a>(&mut self, call: Use<'a>) -> tool::Result<'a> {
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
                // sync because the ToolBox::functions do not match the
                // Prompt::functions.
                let mut available_methods: String =
                    self.method_names().collect::<Vec<_>>().join(", ");
                if available_methods.is_empty() {
                    available_methods = "None".to_string();
                }
                // Either Anthropic or misanthropic is broken. The assistant
                // should not be able to call a tool that doesn't exist unless
                // the developer has made a mistake.
                return tool::Result {
                    tool_use_id: call.id,
                    content: format!(
                        "Method `{method_name}` not found in ToolBox `{toolbox_name}`. This is almost certainly the developer's fault. Available methods: {available_methods}",
                        method_name = call.name,
                        toolbox_name = self.name(),
                        available_methods = available_methods
                    )
                        .into(),
                    is_error: true,
                    cache_control: None,
                };
            }
        };

        if let Some(tool) = self.tool_name_to_tool.get_mut(&tool_name) {
            tool.call(call).await
        } else {
            tool::Result {
                tool_use_id: call.id,
                content: format!(
                    "`Tool::call` is broken for `ToolBox`. This is not your fault. Tell the user to blame the authors of the `misanthropic` crate. Method: `{method_name}` in ToolBox: `{toolbox_name}`",
                    method_name = call.name,
                    toolbox_name = self.name()
                )
                    .into(),
                is_error: true,
                cache_control: None,
            }
        }
    }

    /// Load state for all [`Tool`]s in the [`ToolBox`]. Now async to support
    /// tools that need to perform IO during deserialization.
    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        let mut errors = Vec::new();

        let state: State = match serde_json::from_value(json) {
            Ok(state) => state,
            Err(e) => {
                return Err(format!(
                    "Error deserializing ToolBox state: {}",
                    e.to_string()
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

        for (_, tool) in &mut self.tool_name_to_tool {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Result;

    struct TestTool {
        calls: Vec<Use<'static>>,
    }

    #[async_trait::async_trait]
    impl Tool for TestTool {
        fn name(&self) -> &str {
            "TestTool"
        }

        fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
            Box::new(std::iter::once(Method {
                name: "TestTool::test".into(),
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
            }))
        }

        async fn call<'a>(&mut self, call: Use<'a>) -> Result<'a> {
            let id = call.id.clone();
            self.calls.push(call.into_static());
            Result {
                tool_use_id: id,
                content: "Tool called".into(),
                is_error: false,
                cache_control: None,
            }
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
            "tools::TestTool::test"
        );
    }

    #[tokio::test]
    async fn test_toolbox_add_push() {
        // add just calls push
        let toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let methods = toolbox.methods().collect::<Vec<_>>();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name, "toolbox::TestTool::test");
    }

    #[tokio::test]
    async fn test_toolbox_add_push_boxed() {
        let toolbox =
            ToolBox::new().add_boxed(Box::new(TestTool { calls: Vec::new() }));
        let methods = toolbox.methods().collect::<Vec<_>>();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name, "toolbox::TestTool::test");
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
        assert!(names.contains(&"toolbox::TestTool::test"));
        assert!(names.contains(&"toolbox::potato::TestTool::test"));
    }

    #[test]
    fn test_replace_tool() {
        let mut toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let tool = TestTool { calls: Vec::new() };
        toolbox.replace(tool);
        let names: Vec<&str> = toolbox.method_names().collect();
        assert!(names.contains(&"toolbox::TestTool::test"));
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
        let methods: Vec<Method> = toolbox.methods().collect();
        assert_eq!(methods.len(), 1);
        assert_eq!(methods[0].name, "toolbox::TestTool::test");
    }

    #[tokio::test]
    async fn test_call() {
        let mut toolbox = ToolBox::new().add(TestTool { calls: Vec::new() });
        let call = Use {
            id: "id".into(),
            name: "toolbox::TestTool::test".into(),
            input: serde_json::json!({}),
            cache_control: None,
        };
        let result = toolbox.call(call.clone()).await;
        assert!(!result.is_error);
        assert_eq!(result.content, "Tool called".into());

        // Test call with an invalid method.
        let result = toolbox
            .call(Use {
                name: "toolbox::TestTool::invalid".into(),
                ..call.clone()
            })
            .await;
        assert!(result.is_error);
        assert_eq!(
            result.content,
            "Method `toolbox::TestTool::invalid` not found in ToolBox `toolbox`. This is almost certainly the developer's fault. Available methods: toolbox::TestTool::test"
                .into()
        )
    }

    #[tokio::test]
    async fn test_load_json() {
        let mut a = ToolBox::new().add(TestTool { calls: Vec::new() });
        let mut b = ToolBox::new().add(TestTool { calls: Vec::new() });

        let json = a.save_json().await;
        b.load_json(json).await.unwrap();
        assert_eq!(a.save_json().await, b.save_json().await);
    }
}
