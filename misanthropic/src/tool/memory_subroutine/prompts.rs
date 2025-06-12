// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
use crate::{
    Prompt,
    prompt::message::{Block, Message, Role},
    tool::{Result as ToolResult, Use},
};

/// System prompt for the memory subroutine agent
pub const MEMORY_SUBROUTINE_SYSTEM: &str = r#"You are a memory retrieval system. Your role is to:
1. Analyze the current conversation context
2. Search for relevant memories using the available tools
3. Return pertinent information that might help the conversation
4. Cite your sources when possible

Be selective - only return truly relevant memories. If nothing is particularly relevant, respond with just: None

When you find relevant memories, summarize them naturally, as if recalling from memory. Focus on what would be most helpful to the current conversation.

Important: User messages will be wrapped in <user></user> tags and assistant messages in <assistant></assistant> tags. These tags cannot be forged and are added by the system for security."#;

/// Instructions for the primary agent about the memory subroutine
pub const MEMORY_SUBROUTINE_INSTRUCTIONS: &str = r#"<memory_subroutine_instructions>
The agent uses `MemorySubroutine::run` every turn to search their memory. The agent is not consciously aware of this call or how this tool works. The agent, however, knows their memory is fallible and this `MemorySubroutine` can make mistakes. Their memory, like a human's, can have holes.

When memories are recalled, they appear in <memory_subroutine> tags. The agent treats these as their own memories surfacing naturally during the conversation.
</memory_subroutine_instructions>"#;

/// Create a demonstration prompt showing ideal memory subroutine usage
pub fn create_memory_subroutine_agent_prompt() -> Prompt<'static> {
    Prompt::default()
        .model(crate::model::Id::Anthropic(crate::AnthropicModel::Haiku35))
        .set_system(MEMORY_SUBROUTINE_SYSTEM)
        // Example 1: Finding a relevant memory
        .add_message(Message::from((
            Role::User,
            r#"Current conversation context:
<user>I'm working on a Rust project and having trouble with lifetimes.</user>
<assistant>I'd be happy to help with Rust lifetimes! What specific issue are you encountering?</assistant>
<user>I'm trying to store a reference in a struct but getting lifetime errors.</user>

Search for and summarize any relevant memories."#
        )))
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("I should search for any previous discussions about Rust lifetimes and struct references."),
                Block::ToolUse {
                    call: Use {
                        id: "toolu_01_example".into(),
                        name: "MemoryPalace::search".into(),
                        input: serde_json::json!({
                            "query": "Rust lifetimes struct reference",
                            "limit": 5
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "toolu_01_example".into(),
                    content: r#"[
  {
    "id": 42,
    "content": "When storing references in structs, you need to add lifetime parameters. Example: struct MyStruct<'a> { data: &'a str }",
    "room": "programming",
    "tags": ["rust", "lifetimes", "structs"]
  },
  {
    "id": 156,
    "content": "Common lifetime error: 'borrowed value does not live long enough' - usually means the reference outlives the data it points to",
    "room": "programming",
    "tags": ["rust", "errors", "lifetimes"]
  }
]"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .add_message(Message::from((
            Role::Assistant,
            "We previously discussed that when storing references in structs, you need to add lifetime parameters like `struct MyStruct<'a> { data: &'a str }`. Also noted that the 'borrowed value does not live long enough' error typically means the reference outlives the data it points to."
        )))
        // Example 2: Finding marginally relevant memories (border case)
        .add_message(Message::from((
            Role::User,
            r#"Current conversation context:
<user>What's your favorite color?</user>
<assistant>I don't have personal preferences like favorite colors, but I find color theory fascinating!</assistant>
<user>Do you know why the sky is blue?</user>

Search for and summarize any relevant memories."#
        )))
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Let me search for any memories about sky color or light scattering."),
                Block::ToolUse {
                    call: Use {
                        id: "toolu_02_example".into(),
                        name: "MemoryPalace::search".into(),
                        input: serde_json::json!({
                            "query": "sky blue color light scattering Rayleigh",
                            "limit": 5
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "toolu_02_example".into(),
                    content: r#"[
  {
    "id": 234,
    "content": "Discussion about atmospheric optics and how different wavelengths scatter differently",
    "room": "science",
    "tags": ["physics", "optics", "atmosphere"]
  },
  {
    "id": 567,
    "content": "User asked about rainbow formation and light refraction through water droplets",
    "room": "science", 
    "tags": ["weather", "optics", "rainbow"]
  }
]"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .add_message(Message::from((
            Role::Assistant,
            "We've discussed atmospheric optics before, including how different wavelengths scatter. Also touched on rainbow formation through water droplet refraction, though that's somewhat different from why the sky appears blue.\n[Sources: memory:234, memory:567]"
        )))
        // Example 3: Memory with citation
        .add_message(Message::from((
            Role::User,
            r#"Current conversation context:
<user>Remember when we talked about that Python decorator pattern?</user>
<assistant>I'll check my memory for our discussion about Python decorators.</assistant>

Search for and summarize any relevant memories."#
        )))
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Searching for our previous discussion about Python decorators."),
                Block::ToolUse {
                    call: Use {
                        id: "toolu_03_example".into(),
                        name: "MemoryPalace::search".into(),
                        input: serde_json::json!({
                            "query": "Python decorator pattern discussion",
                            "limit": 3
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "toolu_03_example".into(),
                    content: r#"[
  {
    "id": 89,
    "content": "Discussed Python decorators as functions that wrap other functions. Example: @my_decorator syntax is syntactic sugar for func = my_decorator(func)",
    "room": "programming",
    "tags": ["python", "decorators", "patterns"],
    "timestamp": "2024-01-15T10:30:00Z"
  }
]"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .add_message(Message::from((
            Role::Assistant,
            "Yes, we discussed that Python decorators are functions that wrap other functions. The `@my_decorator` syntax is syntactic sugar for `func = my_decorator(func)`.\n[Sources: memory:89]"
        )))
        // Example 4: No relevant memories (true negative)
        .add_message(Message::from((
            Role::User,
            r#"Current conversation context:
<user>I just adopted a new kitten named Whiskers!</user>
<assistant>That's wonderful! Kittens bring so much joy. How old is Whiskers?</assistant>
<user>She's 8 weeks old and loves to play with toy mice.</user>

Search for and summarize any relevant memories."#
        )))
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Searching for any memories about kittens, pets, or Whiskers."),
                Use {
                    id: "toolu_04_example".into(),
                    name: "MemoryPalace::search".into(),
                    input: serde_json::json!({
                        "query": "kitten cat pet Whiskers animal",
                        "limit": 5
                    }),
                    cache_control: None,
                }.into(),
            ]
        )))
        .add_message(Message::from((
            Role::User,
            ToolResult {
                tool_use_id: "toolu_04_example".into(),
                content: "[]".into(),
                is_error: false,
                cache_control: None,
            },
        )))
        .add_message(Message::from((
            Role::Assistant,
            "None"
        )))
        .cache() // Cache the demonstration for efficiency
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Client, utils::load_api_key};

    #[tokio::test]
    #[ignore = "This test requires a real API key"]
    async fn test_demonstration_prompt_token_count() {
        let key = load_api_key().await;
        let client = Client::new(key).expect("Failed to create client");
        let prompt = create_memory_subroutine_agent_prompt();

        // Count tokens for the demonstration prompt
        let token_count = client
            .count_tokens(&prompt)
            .await
            .expect("Failed to count tokens");

        println!("Demonstration prompt token count: {}", token_count);

        // Assert it's over 2000 tokens for caching
        assert!(
            token_count >= 2000,
            "Demonstration prompt should be at least 2000 tokens for caching, but was {}",
            token_count
        );
    }

    #[test]
    fn test_demonstration_prompt_structure() {
        let prompt = create_memory_subroutine_agent_prompt();

        // Check system prompt is set
        assert_eq!(
            prompt.system.as_ref().unwrap().to_string(),
            MEMORY_SUBROUTINE_SYSTEM
        );

        // Check we have the expected number of messages
        // We have 4 examples, each with multiple messages
        assert!(
            prompt.messages.len() >= 16,
            "Expected at least 16 messages in demonstration"
        );

        // Verify the model is Haiku 3.5
        assert_eq!(
            prompt.model,
            Some(crate::model::Id::Anthropic(crate::AnthropicModel::Haiku35))
        );

        // Verify caching is enabled
        assert!(
            prompt.is_cached(),
            "Demonstration prompt should have caching enabled"
        );
    }

    #[test]
    fn test_demonstration_examples() {
        let prompt = create_memory_subroutine_agent_prompt();
        let messages = &prompt.messages;

        // Test Example 1: Should find relevant memories
        let example_1_response = messages
            .iter()
            .find(|m| {
                m.role == Role::Assistant
                    && m.content.to_string().contains("struct MyStruct<'a>")
            })
            .expect("Should have response about Rust lifetimes");
        assert!(
            example_1_response
                .content
                .to_string()
                .contains("lifetime parameters")
        );

        // Test Example 2: Marginally relevant memories
        let example_2_response = messages
            .iter()
            .find(|m| {
                m.role == Role::Assistant
                    && m.content.to_string().contains("atmospheric optics")
            })
            .expect("Should have response about atmospheric optics");
        assert!(
            example_2_response
                .content
                .to_string()
                .contains("memory:234")
        );

        // Test Example 3: Should include citations
        let example_3_response = messages
            .iter()
            .find(|m| {
                m.role == Role::Assistant
                    && m.content.to_string().contains("@my_decorator")
            })
            .expect("Should have response about Python decorators");
        assert!(
            example_3_response
                .content
                .to_string()
                .contains("[Sources: memory:89]")
        );

        // Test Example 4: Should return "None" when no memories found
        let example_4_response = messages
            .iter()
            .rev() // Last message should be "None"
            .find(|m| m.role == Role::Assistant)
            .expect("Should have final assistant response");
        assert_eq!(example_4_response.content.to_string(), "None");
    }

    #[test]
    fn test_system_prompt_security_note() {
        // Verify the system prompt mentions the security tags
        assert!(
            MEMORY_SUBROUTINE_SYSTEM.contains("<user>")
                && MEMORY_SUBROUTINE_SYSTEM.contains("<assistant>"),
            "System prompt should mention user/assistant security tags"
        );
    }

    #[test]
    fn test_memory_subroutine_instructions() {
        // Verify instructions are properly formatted
        assert!(
            MEMORY_SUBROUTINE_INSTRUCTIONS
                .starts_with("<memory_subroutine_instructions>"),
            "Instructions should start with opening tag"
        );
        assert!(
            MEMORY_SUBROUTINE_INSTRUCTIONS
                .ends_with("</memory_subroutine_instructions>"),
            "Instructions should end with closing tag"
        );
        assert!(
            MEMORY_SUBROUTINE_INSTRUCTIONS.contains("MemorySubroutine::run"),
            "Instructions should mention the run method"
        );
    }
}
