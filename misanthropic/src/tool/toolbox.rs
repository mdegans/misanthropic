use std::{
    borrow::Cow,
    collections::{BTreeMap, HashMap},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    tool::{self, Method, Tool, Use},
    Prompt,
};

/// Container [`Tool`] that calls [`Tool`]s. Nestable, however consider if this
/// is really necessary.
///
/// [`functions`]: ToolBox::functions
/// [`call`]: ToolBox::call
pub struct ToolBox {
    /// Unique identifier for the [`ToolBox`].
    id: Uuid,
    /// Name of the [`ToolBox`].
    name: Cow<'static, str>,
    /// Map of [`Method::name`] to index in [`tools`] of the [`Tool`] to call.
    ///
    /// Stores namespaced function names in the format `tool::function`.
    ///
    /// [`tools`]: ToolBox::tools
    pub(crate) method_to_tool_id: BTreeMap<Cow<'static, str>, Uuid>,
    /// Vector of [`Tool`]s to call.
    pub(crate) tool_id_to_tool: HashMap<Uuid, Box<dyn Tool + Send>>,
}

impl Default for ToolBox {
    fn default() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: "toolbox".into(), // module syntax, snake case
            method_to_tool_id: BTreeMap::new(),
            tool_id_to_tool: HashMap::new(),
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
            self.method_to_tool_id.insert(
                format!("{}::{}", self.name, method.name).into(),
                tool.id(),
            );
        }

        if let Some(existing) = self.tool_id_to_tool.insert(tool.id(), tool) {
            #[cfg(feature = "log")]
            log::debug!("Tool replaced: {}", existing.name());
        }
    }

    /// Names of all [`Tool`]s in the [`ToolBox`].
    pub fn tool_names(&self) -> impl Iterator<Item = &str> {
        self.tool_id_to_tool.values().map(|tool| tool.name())
    }

    /// Names of all the [`Method`]s in the [`ToolBox`].
    pub fn method_names(&self) -> impl ExactSizeIterator<Item = &str> {
        self.method_to_tool_id.keys().map(|name| name.as_ref())
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
        let function_names = tool.methods().map(|method| {
            format!(
                "{self_name}::{tool}::{method}",
                tool = tool.name(),
                method = method.name
            )
        });

        // Remove the old tool and its functions.
        for name in function_names {
            if let Some(old_id) =
                self.method_to_tool_id.insert(name.into(), tool.id())
            {
                self.tool_id_to_tool.remove(&old_id);
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct State {
    name: Cow<'static, str>,
    id: Uuid,
    tools: serde_json::Map<String, serde_json::Value>,
}

#[async_trait::async_trait]
impl Tool for ToolBox {
    /// Unique identifier per instance of [`ToolBox`].
    fn id(&self) -> Uuid {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// The [`Method`]s for all [`Tool`]s in the [`ToolBox`].
    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new(self.tool_id_to_tool.values().flat_map(|tool| {
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
        let index = match self.method_to_tool_id.get(call.name.as_ref()) {
            Some(index) => {
                #[cfg(feature = "log")]
                log::debug!("Method found: `{}`", call.name);
                *index
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
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                };
            }
        };

        if let Some(tool) = self.tool_id_to_tool.get_mut(&index) {
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
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
        }
    }

    /// Load state for all [`Tool`]s in the [`ToolBox`].
    fn load_json(
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
        self.id = state.id;

        for (name, tool_json) in state.tools {
            let tool = match self
                .tool_id_to_tool
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

            if let Err(e) = tool.load_json(tool_json) {
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
            log::error!("{}", message);
            Err(message)
        }
    }

    /// Save state for all [`Tool`]s in the [`ToolBox`].
    fn save_json(&self) -> serde_json::Value {
        let state = State {
            name: self.name.clone(),
            id: self.id,
            tools: self
                .tool_id_to_tool
                .iter()
                .map(|(_, tool)| (tool.name().to_string(), tool.save_json()))
                .collect(),
        };

        serde_json::to_value(state).unwrap()
    }

    /// Setup the [`Prompt`] by calling this method on all children, collecting
    /// any errors. If there are any errors, any changes to the prompt are
    /// reverted.
    fn apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut errors = Vec::new();
        let backup = prompt.clone();

        #[allow(unused_variables)]
        for (_, tool) in &self.tool_id_to_tool {
            #[cfg(feature = "log")]
            log::debug!("Setting up `Prompt` for `{}` tool.", tool.name());

            if let Err(e) = tool.apply_to_prompt(prompt) {
                #[cfg(feature = "log")]
                log::error!(
                    "Error setting up `Prompt` for `{name}` tool: {e}",
                    name = tool.name(),
                );

                errors.push(e);
            } else {
                #[cfg(feature = "log")]
                log::debug!(
                    "Sucessful setup of `Prompt` for {name} tool.",
                    name = tool.name()
                );
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Result;

    struct TestTool {
        calls: Vec<Use<'static>>,
    }

    #[async_trait::async_trait]
    impl Tool for TestTool {
        fn id(&self) -> Uuid {
            Uuid::nil()
        }

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
                #[cfg(feature = "prompt-caching")]
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
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }
        }
    }

    #[tokio::test]
    async fn test_toolbox_named() {
        let toolbox = ToolBox::named("tools")
            .unwrap()
            .add(TestTool { calls: Vec::new() });
        assert_eq!(
            toolbox.method_to_tool_id.keys().next().unwrap(),
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
    fn test_id() {
        // ToolBox ids are unique per instance.
        let a = ToolBox::new();
        let b = ToolBox::new();
        assert_ne!(a.id(), b.id());
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
            #[cfg(feature = "prompt-caching")]
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

    #[test]
    fn test_load_json() {
        let a = ToolBox::new().add(TestTool { calls: Vec::new() });
        let mut b = ToolBox::new().add(TestTool { calls: Vec::new() });
        assert_ne!(a.id(), b.id());

        let json = a.save_json();
        b.load_json(json).unwrap();
        assert_eq!(a.id(), b.id());
        assert_eq!(a.save_json(), b.save_json());
    }
}
