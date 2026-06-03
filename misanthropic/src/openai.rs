//! OpenAI-compatible types and conversion from [misanthropic] primitives.
//!
//! This module provides types matching the [OpenAI Chat Completions API],
//! and conversion impls to translate [`Prompt`], [`Message`], and tool types
//! between the Anthropic and OpenAI formats. This enables using misanthropic
//! as the canonical representation for prompts, then serializing to either
//! the Anthropic Messages API or OpenAI-compatible endpoints (like [Ollama]).
//!
//! # Feature
//!
//! This module is gated behind the `openai` feature:
//!
//! ```toml
//! misanthropic = { version = "1", features = ["openai"] }
//! ```
//!
//! [OpenAI Chat Completions API]: https://platform.openai.com/docs/api-reference/chat
//! [Ollama]: https://ollama.com/blog/openai-compatibility
//! [`Prompt`]: crate::Prompt
//! [`Message`]: crate::prompt::Message

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

use crate::CowStr;
use crate::Prompt;
use crate::prompt::Message;
use crate::prompt::message::{Block, Content, Image, MediaType, Role};
use crate::tool;

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

/// An OpenAI-compatible chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier (e.g. `"gpt-4"`, `"cogito:14b"`).
    pub model: String,
    /// The conversation messages.
    pub messages: Vec<ChatMessage>,
    /// Tool definitions available to the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ChatTool>>,
    /// How the model should choose which tool to call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ChatToolChoice>,
    /// Sampling temperature (0.0–2.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Nucleus sampling parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
    /// Whether to stream the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// Reasoning effort for models that support extended thinking (e.g. Ollama
    /// with cogito). Ollama maps this to its `think` parameter — `None` variant
    /// disables thinking entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
}

/// Reasoning effort level for extended thinking models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    /// No reasoning/thinking — fastest, lowest quality.
    None,
    /// Minimal reasoning.
    Low,
    /// Moderate reasoning.
    Medium,
    /// Full reasoning — slowest, highest quality.
    High,
}

/// A message in the chat completions format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: ChatRole,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<ChatContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Role in the OpenAI chat format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ChatRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Content can be a plain string or an array of content parts (for images).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatContent {
    /// Plain text content.
    Text(String),
    /// Array of content parts (text + images).
    Parts(Vec<ChatContentPart>),
}

/// A content part in a multi-part message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatContentPart {
    /// Text content part.
    Text { text: String },
    /// Image URL content part (supports `data:` URIs for inline images).
    ImageUrl { image_url: ImageUrl },
}

/// An image URL reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    /// URL or `data:image/{format};base64,{data}` for inline images.
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

// ---------------------------------------------------------------------------
// Tool types
// ---------------------------------------------------------------------------

/// An OpenAI-compatible tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTool {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunction,
}

/// A function definition within a tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunction {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

/// Tool choice constraint.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatToolChoice {
    /// A string like "auto", "none", or "required".
    String(String),
    /// A specific function choice.
    Object {
        #[serde(rename = "type")]
        kind: String,
        function: ChatToolChoiceFunction,
    },
}

/// Specifies a particular function for tool choice.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolChoiceFunction {
    pub name: String,
}

/// A tool call from the assistant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ChatFunctionCall,
}

/// A function call within a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionCall {
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// An OpenAI-compatible chat completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    #[serde(default)]
    pub id: String,
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

/// A choice in the response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Token usage statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

// ---------------------------------------------------------------------------
// Streaming types
// ---------------------------------------------------------------------------

/// A streaming chunk from the chat completions API.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    pub choices: Vec<ChatChunkChoice>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub usage: Option<ChatUsage>,
}

/// A choice within a streaming chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChunkChoice {
    #[serde(default)]
    pub index: u32,
    pub delta: ChatDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// Delta content in a streaming chunk.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<ChatRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatToolCallDelta>>,
}

/// A partial tool call in a streaming delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ChatFunctionCallDelta>,
}

/// Partial function call data in a streaming delta.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatFunctionCallDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Accumulator for streaming chunks
// ---------------------------------------------------------------------------

/// Accumulates streaming [`ChatCompletionChunk`]s into a complete
/// [`Message`].
///
/// Feed chunks via [`process_chunk`] and call [`into_message`] when the
/// stream is done.
///
/// [`process_chunk`]: ChatStreamAccumulator::process_chunk
/// [`into_message`]: ChatStreamAccumulator::into_message
#[derive(Debug, Default)]
pub struct ChatStreamAccumulator {
    content: String,
    tool_calls: Vec<AccumulatedToolCall>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default)]
struct AccumulatedToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChatStreamAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a streaming chunk, accumulating its content.
    pub fn process_chunk(&mut self, chunk: ChatCompletionChunk) {
        for choice in chunk.choices {
            if let Some(reason) = choice.finish_reason {
                self.finish_reason = Some(reason);
            }

            let delta = choice.delta;

            if let Some(text) = delta.content {
                self.content.push_str(&text);
            }

            if let Some(calls) = delta.tool_calls {
                for call_delta in calls {
                    let idx = call_delta.index as usize;

                    // Extend the vec if needed
                    while self.tool_calls.len() <= idx {
                        self.tool_calls.push(AccumulatedToolCall::default());
                    }

                    let acc = &mut self.tool_calls[idx];

                    if let Some(id) = call_delta.id {
                        acc.id = id;
                    }
                    if let Some(func) = call_delta.function {
                        if let Some(name) = func.name {
                            acc.name = name;
                        }
                        if let Some(args) = func.arguments {
                            acc.arguments.push_str(&args);
                        }
                    }
                }
            }
        }
    }

    /// The finish reason from the final chunk, if received.
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// Consume the accumulator and produce a misanthropic [`Message`].
    ///
    /// Returns `None` if no content or tool calls were accumulated.
    pub fn into_message(self) -> Option<Message<'static>> {
        let has_content = !self.content.is_empty();
        let has_tools = !self.tool_calls.is_empty();

        if !has_content && !has_tools {
            return None;
        }

        let mut blocks: Vec<Block<'static>> = Vec::new();

        if has_content {
            blocks.push(Block::Text {
                text: CowStr::from(self.content),
                citations: None,
                cache_control: None,
            });
        }

        for call in self.tool_calls {
            let input: serde_json::Value =
                serde_json::from_str(&call.arguments)
                    .unwrap_or(serde_json::Value::Null);
            blocks.push(Block::ToolUse {
                call: tool::Use {
                    id: Cow::Owned(call.id),
                    name: Cow::Owned(call.name),
                    input,
                    cache_control: None,
                },
            });
        }

        Some(Message {
            role: Role::Assistant,
            content: Content(blocks),
        })
    }
}

// ---------------------------------------------------------------------------
// Conversions: misanthropic → OpenAI
// ---------------------------------------------------------------------------

impl<'a> From<&Prompt<'a>> for ChatCompletionRequest {
    fn from(prompt: &Prompt<'a>) -> Self {
        let mut messages = Vec::new();

        // System message
        if let Some(system) = &prompt.system {
            messages.push(ChatMessage {
                role: ChatRole::System,
                content: Some(ChatContent::Text(content_to_text(system))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }

        // Conversation messages
        for msg in &prompt.messages {
            messages.extend(message_to_chat_messages(msg));
        }

        // Tools. Server tools have no OpenAI chat-completions equivalent in
        // this shim, so only custom methods are forwarded.
        let tools = prompt.methods.as_ref().map(|methods| {
            methods
                .iter()
                .filter_map(|t| t.as_method())
                .map(ChatTool::from)
                .collect()
        });

        // Tool choice
        let tool_choice =
            prompt.tool_choice.as_ref().map(|choice| match choice {
                tool::Choice::Auto { .. } => {
                    ChatToolChoice::String("auto".to_string())
                }
                tool::Choice::Any { .. } => {
                    ChatToolChoice::String("required".to_string())
                }
                tool::Choice::Method { name, .. } => ChatToolChoice::Object {
                    kind: "function".to_string(),
                    function: ChatToolChoiceFunction { name: name.clone() },
                },
                tool::Choice::None => {
                    ChatToolChoice::String("none".to_string())
                }
            });

        ChatCompletionRequest {
            model: prompt.model.to_string(),
            messages,
            tools,
            tool_choice,
            temperature: prompt.temperature,
            max_tokens: Some(prompt.max_tokens.get()),
            top_p: prompt.top_p,
            stop: prompt
                .stop_sequences
                .as_ref()
                .map(|seqs| seqs.iter().map(|s| s.to_string()).collect()),
            stream: prompt.stream,
            reasoning_effort: None,
        }
    }
}

impl<'a> From<&tool::MethodDef<'a>> for ChatTool {
    fn from(method: &tool::MethodDef<'a>) -> Self {
        ChatTool {
            kind: "function".to_string(),
            function: ChatFunction {
                name: method.name.to_string(),
                description: method.description.to_string(),
                parameters: method.schema.clone(),
            },
        }
    }
}

impl From<ChatToolCall> for tool::Use<'static> {
    fn from(call: ChatToolCall) -> Self {
        let input: serde_json::Value =
            serde_json::from_str(&call.function.arguments)
                .unwrap_or(serde_json::Value::Null);
        tool::Use {
            id: Cow::Owned(call.id),
            name: Cow::Owned(call.function.name),
            input,
            cache_control: None,
        }
    }
}

impl<'a> From<&tool::Result<'a>> for ChatMessage {
    fn from(result: &tool::Result<'a>) -> Self {
        ChatMessage {
            role: ChatRole::Tool,
            content: Some(ChatContent::Text(content_to_text(&result.content))),
            tool_calls: None,
            tool_call_id: Some(result.tool_use_id.to_string()),
            name: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions: OpenAI → misanthropic
// ---------------------------------------------------------------------------

impl ChatCompletionResponse {
    /// Extract the first choice's message as a misanthropic [`Message`].
    ///
    /// Returns `None` if no choices are present.
    pub fn into_message(self) -> Option<Message<'static>> {
        let choice = self.choices.into_iter().next()?;
        Some(chat_message_to_message(choice.message))
    }

    /// The finish reason from the first choice.
    pub fn finish_reason(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract the text content from a misanthropic [`Content`], ignoring
/// non-text blocks.
fn content_to_text(content: &Content<'_>) -> String {
    let mut out = String::new();
    for block in content.iter() {
        if let Block::Text { text, .. } = block {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

/// Convert a misanthropic [`Message`] into one or more [`ChatMessage`]s.
///
/// A single misanthropic message with both text and tool use/results may
/// produce multiple OpenAI messages, because tool results must be separate
/// messages with `role: "tool"`.
fn message_to_chat_messages<'a>(msg: &Message<'a>) -> Vec<ChatMessage> {
    let mut text_parts = Vec::new();
    let mut image_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut tool_results = Vec::new();

    for block in msg.content.iter() {
        match block {
            Block::Text { text, .. } => {
                text_parts.push(text.to_string());
            }
            Block::Image { image, .. } => {
                let data_url = match image {
                    Image::Base64 { media_type, data } => {
                        let mt = match media_type {
                            MediaType::Jpeg => "image/jpeg",
                            MediaType::Png => "image/png",
                            MediaType::Gif => "image/gif",
                            MediaType::Webp => "image/webp",
                        };
                        format!("data:{};base64,{}", mt, data)
                    }
                    Image::Url { url } => url.to_string(),
                };
                image_parts.push(ChatContentPart::ImageUrl {
                    image_url: ImageUrl {
                        url: data_url,
                        detail: None,
                    },
                });
            }
            Block::ToolUse { call } => {
                tool_calls.push(ChatToolCall {
                    id: call.id.to_string(),
                    kind: "function".to_string(),
                    function: ChatFunctionCall {
                        name: call.name.to_string(),
                        arguments: serde_json::to_string(&call.input)
                            .unwrap_or_default(),
                    },
                });
            }
            Block::ToolResult { result } => {
                tool_results.push(ChatMessage {
                    role: ChatRole::Tool,
                    content: Some(ChatContent::Text(content_to_text(
                        &result.content,
                    ))),
                    tool_calls: None,
                    tool_call_id: Some(result.tool_use_id.to_string()),
                    name: None,
                });
            }
            // Thought, Document, and server-tool blocks have no OpenAI
            // chat-completions equivalent in this shim — skip them.
            Block::Thought { .. }
            | Block::RedactedThought { .. }
            | Block::Document { .. }
            | Block::ServerToolUse { .. }
            | Block::WebSearchToolResult { .. }
            | Block::ToolSearchToolResult { .. } => {}
        }
    }

    let mut messages = Vec::new();

    // Build the primary message
    if msg.role == Role::Assistant {
        // Assistant: text content + optional tool calls
        let content = if text_parts.is_empty() {
            None
        } else {
            Some(ChatContent::Text(text_parts.join("\n")))
        };
        let calls = if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        };
        messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content,
            tool_calls: calls,
            tool_call_id: None,
            name: None,
        });
    } else {
        // User: may have text + images
        if !image_parts.is_empty() {
            let mut parts: Vec<ChatContentPart> = text_parts
                .iter()
                .map(|t| ChatContentPart::Text { text: t.clone() })
                .collect();
            parts.extend(image_parts);
            messages.push(ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Parts(parts)),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        } else if !text_parts.is_empty() {
            messages.push(ChatMessage {
                role: role_to_chat_role(msg.role),
                content: Some(ChatContent::Text(text_parts.join("\n"))),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            });
        }
    }

    // Tool results become separate messages
    messages.extend(tool_results);

    messages
}

/// Convert a [`ChatMessage`] into a misanthropic [`Message`].
fn chat_message_to_message(msg: ChatMessage) -> Message<'static> {
    let mut blocks: Vec<Block<'static>> = Vec::new();

    // Text content
    if let Some(content) = msg.content {
        match content {
            ChatContent::Text(text) => {
                if !text.is_empty() {
                    blocks.push(Block::Text {
                        text: CowStr::from(text),
                        citations: None,
                        cache_control: None,
                    });
                }
            }
            ChatContent::Parts(parts) => {
                for part in parts {
                    match part {
                        ChatContentPart::Text { text } => {
                            blocks.push(Block::Text {
                                text: CowStr::from(text),
                                citations: None,
                                cache_control: None,
                            });
                        }
                        ChatContentPart::ImageUrl { .. } => {
                            // Image URL conversion back to base64 Block would
                            // require parsing data: URIs — skip for now.
                        }
                    }
                }
            }
        }
    }

    // Tool calls
    if let Some(calls) = msg.tool_calls {
        for call in calls {
            blocks.push(Block::ToolUse {
                call: tool::Use::from(call),
            });
        }
    }

    let role = match msg.role {
        ChatRole::Assistant => Role::Assistant,
        ChatRole::System => Role::System,
        _ => Role::User,
    };

    Message {
        role,
        content: Content(blocks),
    }
}

fn role_to_chat_role(role: Role) -> ChatRole {
    match role {
        Role::User => ChatRole::User,
        Role::Assistant => ChatRole::Assistant,
        Role::System => ChatRole::System,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU32;

    #[test]
    fn simple_prompt_to_chat_request() {
        let prompt = Prompt {
            model: "claude-sonnet-4-20250514".into(),
            messages: vec![Message {
                role: Role::User,
                content: Content::text("Hello"),
            }],
            max_tokens: NonZeroU32::new(1024).unwrap(),
            system: Some(Content::text("You are helpful.")),
            ..Default::default()
        };

        let req = ChatCompletionRequest::from(&prompt);
        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert_eq!(req.messages.len(), 2);
        assert_eq!(req.messages[0].role, ChatRole::System);
        assert_eq!(req.messages[1].role, ChatRole::User);

        if let Some(ChatContent::Text(t)) = &req.messages[0].content {
            assert_eq!(t, "You are helpful.");
        } else {
            panic!("expected text content");
        }
    }

    #[test]
    fn tool_method_to_chat_tool() {
        let method = tool::MethodDef::builder("get_weather")
            .description("Get the weather")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }))
            .build()
            .unwrap();

        let chat_tool = ChatTool::from(&method);
        assert_eq!(chat_tool.kind, "function");
        assert_eq!(chat_tool.function.name, "get_weather");
        assert_eq!(chat_tool.function.description, "Get the weather");
    }

    #[test]
    fn tool_call_round_trip() {
        let chat_call = ChatToolCall {
            id: "call_123".to_string(),
            kind: "function".to_string(),
            function: ChatFunctionCall {
                name: "get_weather".to_string(),
                arguments: r#"{"location":"Paris"}"#.to_string(),
            },
        };

        let use_block: tool::Use<'static> = chat_call.into();
        assert_eq!(use_block.id, "call_123");
        assert_eq!(use_block.name, "get_weather");
        assert_eq!(use_block.input["location"], "Paris");
    }

    #[test]
    fn tool_choice_mapping() {
        let prompt = Prompt {
            model: "test".into(),
            messages: vec![],
            max_tokens: NonZeroU32::new(100).unwrap(),
            tool_choice: Some(tool::Choice::any()),
            ..Default::default()
        };

        let req = ChatCompletionRequest::from(&prompt);
        assert!(matches!(
            req.tool_choice,
            Some(ChatToolChoice::String(ref s)) if s == "required"
        ));
    }

    #[test]
    fn tool_result_to_chat_message() {
        let result = tool::Result {
            tool_use_id: Cow::Borrowed("call_456"),
            content: Content::text("Sunny, 22°C"),
            is_error: false,
            cache_control: None,
        };

        let msg = ChatMessage::from(&result);
        assert_eq!(msg.role, ChatRole::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_456"));
        if let Some(ChatContent::Text(t)) = &msg.content {
            assert_eq!(t, "Sunny, 22°C");
        } else {
            panic!("expected text");
        }
    }

    #[test]
    fn response_into_message() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-123".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: Some(ChatContent::Text("Hello!".to_string())),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            model: "gpt-4".to_string(),
            usage: None,
        };

        let msg = resp.into_message().unwrap();
        assert_eq!(msg.role, Role::Assistant);
        assert_eq!(msg.content.len(), 1);
        if let Some(Block::Text { text, .. }) = msg.content.first() {
            assert_eq!(text.to_string(), "Hello!");
        } else {
            panic!("expected a text block");
        }
    }

    #[test]
    fn response_with_tool_calls() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-456".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: ChatRole::Assistant,
                    content: None,
                    tool_calls: Some(vec![ChatToolCall {
                        id: "call_789".to_string(),
                        kind: "function".to_string(),
                        function: ChatFunctionCall {
                            name: "search".to_string(),
                            arguments: r#"{"query":"rust"}"#.to_string(),
                        },
                    }]),
                    tool_call_id: None,
                    name: None,
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            model: "test".to_string(),
            usage: None,
        };

        let msg = resp.into_message().unwrap();
        assert_eq!(msg.role, Role::Assistant);
        let tool_use = msg.tool_use().unwrap();
        assert_eq!(tool_use.name, "search");
        assert_eq!(tool_use.input["query"], "rust");
    }

    #[test]
    fn thought_blocks_stripped() {
        let msg = Message {
            role: Role::Assistant,
            content: Content(vec![
                Block::Thought {
                    thought: "thinking...".into(),
                    signature: "sig".into(),
                },
                Block::Text {
                    text: "Hello!".into(),
                    citations: None,
                    cache_control: None,
                },
            ]),
        };

        let chat_msgs = message_to_chat_messages(&msg);
        assert_eq!(chat_msgs.len(), 1);
        if let Some(ChatContent::Text(t)) = &chat_msgs[0].content {
            assert_eq!(t, "Hello!");
        }
        // No thought in the output
    }

    #[test]
    fn stream_accumulator_text() {
        let mut acc = ChatStreamAccumulator::new();

        acc.process_chunk(ChatCompletionChunk {
            id: "1".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some(ChatRole::Assistant),
                    content: Some("Hello".into()),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            model: "test".into(),
            usage: None,
        });

        acc.process_chunk(ChatCompletionChunk {
            id: "1".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: Some(" world!".into()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".into()),
            }],
            model: "test".into(),
            usage: None,
        });

        assert_eq!(acc.finish_reason(), Some("stop"));
        let msg = acc.into_message().unwrap();
        if let Some(Block::Text { text, .. }) = msg.content.first() {
            assert_eq!(text.to_string(), "Hello world!");
        } else {
            panic!("expected text block");
        }
    }

    #[test]
    fn stream_accumulator_tool_call() {
        let mut acc = ChatStreamAccumulator::new();

        // First chunk: tool call header
        acc.process_chunk(ChatCompletionChunk {
            id: "1".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: Some(ChatRole::Assistant),
                    content: None,
                    tool_calls: Some(vec![ChatToolCallDelta {
                        index: 0,
                        id: Some("call_abc".into()),
                        kind: Some("function".into()),
                        function: Some(ChatFunctionCallDelta {
                            name: Some("search".into()),
                            arguments: Some(r#"{"qu"#.into()),
                        }),
                    }]),
                },
                finish_reason: None,
            }],
            model: "test".into(),
            usage: None,
        });

        // Second chunk: arguments continuation
        acc.process_chunk(ChatCompletionChunk {
            id: "1".into(),
            choices: vec![ChatChunkChoice {
                index: 0,
                delta: ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(vec![ChatToolCallDelta {
                        index: 0,
                        id: None,
                        kind: None,
                        function: Some(ChatFunctionCallDelta {
                            name: None,
                            arguments: Some(r#"ery":"test"}"#.into()),
                        }),
                    }]),
                },
                finish_reason: Some("tool_calls".into()),
            }],
            model: "test".into(),
            usage: None,
        });

        let msg = acc.into_message().unwrap();
        let tool_use = msg.tool_use().unwrap();
        assert_eq!(tool_use.name, "search");
        assert_eq!(tool_use.input["query"], "test");
    }

    #[test]
    fn empty_accumulator_returns_none() {
        let acc = ChatStreamAccumulator::new();
        assert!(acc.into_message().is_none());
    }

    #[test]
    fn chat_completion_request_serializes() {
        let req = ChatCompletionRequest {
            model: "cogito:14b".to_string(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: Some(ChatContent::Text("hi".to_string())),
                tool_calls: None,
                tool_call_id: None,
                name: None,
            }],
            tools: None,
            tool_choice: None,
            temperature: Some(0.7),
            max_tokens: Some(1024),
            top_p: None,
            stop: None,
            stream: None,
            reasoning_effort: None,
        };

        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["model"], "cogito:14b");
        assert_eq!(json["messages"][0]["role"], "user");
        assert_eq!(json["messages"][0]["content"], "hi");
        assert!(json["temperature"].as_f64().unwrap() > 0.69);
        assert!(json.get("tools").is_none());
    }
}
