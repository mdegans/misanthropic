//! [`Butler`] tool - An intelligent assistant that manages and queries the Memory Palace.

use super::{MemoryPalace, Method, Tool, Use};
use crate::{Prompt, prompt::message::Block};
use serde_json::json;

const BUTLER_INSTRUCTIONS: &str = r#"<butler_instructions>You have access to a Butler - an intelligent assistant that can help you query and organize your Memory Palace. The Butler uses semantic search and contextual understanding to find relevant information and provide thoughtful responses to your questions.</butler_instructions>"#;

/// A Butler tool that provides intelligent access to the Memory Palace.
#[derive(Debug)]
pub struct Butler<C: surrealdb::Connection> {
    /// The underlying memory palace that the butler manages.
    memory_palace: MemoryPalace<C>,
    /// Butler's own notes and context about conversations.
    context_notes: Vec<String>,
    /// Recent queries for context.
    recent_queries: Vec<String>,
}

impl<C: surrealdb::Connection> Butler<C> {
    const NAME: &'static str = "Butler";

    /// Create a new Butler from an existing Memory Palace.
    pub fn from_memory_palace(memory_palace: MemoryPalace<C>) -> Self {
        Self {
            memory_palace,
            context_notes: Vec::new(),
            recent_queries: Vec::new(),
        }
    }

    /// Add a query to recent queries (keep last 10).
    fn add_recent_query(&mut self, query: &str) {
        self.recent_queries.push(query.to_string());
        if self.recent_queries.len() > 10 {
            self.recent_queries.remove(0);
        }
    }

    /// Generate context for the butler's response.
    fn generate_context(&self) -> String {
        let mut context = String::new();

        if !self.recent_queries.is_empty() {
            context.push_str("Recent queries:\n");
            for (i, query) in
                self.recent_queries.iter().rev().take(3).enumerate()
            {
                context.push_str(&format!("{}. {}\n", i + 1, query));
            }
            context.push('\n');
        }

        if !self.context_notes.is_empty() {
            context.push_str("Butler's context notes:\n");
            for note in &self.context_notes {
                context.push_str(&format!("- {}\n", note));
            }
        }

        context
    }

    /// Intelligently query the memory palace and provide a contextual response.
    async fn ask_butler(&mut self, question: &str) -> Result<String, String> {
        self.add_recent_query(question);

        // Extract key terms from the question for search
        let search_terms = self.extract_search_terms(question);
        #[cfg(feature = "log")]
        log::debug!("Extracted search terms: {:?}", search_terms);

        let mut all_results = Vec::new();
        let mut seen_content = std::collections::HashSet::new();

        // Search for each term
        for term in &search_terms {
            #[cfg(feature = "log")]
            log::trace!("Searching for term: '{}'", term);
            match self.memory_palace.search(term).await {
                Ok(results) => {
                    #[cfg(feature = "log")]
                    log::trace!(
                        "Found {} results for term '{}'",
                        results.len(),
                        term
                    );
                    for (room, id, memory) in results {
                        #[cfg(feature = "log")]
                        log::trace!(
                            "Result - Room: {}, Content: {}",
                            room,
                            memory.content
                        );
                        // Avoid duplicates
                        if seen_content.insert(memory.content.clone()) {
                            all_results.push((room, id, memory));
                        }
                    }
                }
                Err(e) => {
                    #[cfg(feature = "log")]
                    log::error!("Search failed for term '{}': {}", term, e);
                    return Err(format!(
                        "Search failed for term '{}': {}",
                        term, e
                    ));
                }
            }
        }

        #[cfg(feature = "log")]
        log::debug!("Total unique results: {}", all_results.len());

        // Generate a thoughtful response
        if all_results.is_empty() {
            Ok(format!(
                "I searched the Memory Palace for information related to '{}', but couldn't find any relevant memories. \
                 You might want to store some information about this topic first using the Memory Palace store function.\n\n\
                 {}",
                question,
                self.generate_context()
            ))
        } else {
            let mut response = format!(
                "Based on your question '{}', I found {} relevant memories in the Memory Palace:\n\n",
                question,
                all_results.len()
            );

            // Group results by room for better organization
            let mut rooms: std::collections::HashMap<String, Vec<_>> =
                std::collections::HashMap::new();
            for (room, id, memory) in all_results {
                rooms.entry(room).or_default().push((id, memory));
            }

            for (room_name, memories) in rooms {
                response.push_str(&format!("**From {}:**\n", room_name));
                for (id, memory) in memories {
                    response.push_str(&format!(
                        "- {}\n  Tags: {}\n  (ID: {})\n\n",
                        memory.content,
                        memory.tags.join(", "),
                        id
                    ));
                }
            }

            // Add butler context
            let context = self.generate_context();
            if !context.is_empty() {
                response.push_str(&format!("**Context:**\n{}", context));
            }

            Ok(response)
        }
    }

    /// Extract search terms from a natural language question.
    fn extract_search_terms(&self, question: &str) -> Vec<String> {
        // Simple term extraction - in a real implementation you might use NLP
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
            "for", "of", "with", "by", "what", "how", "when", "where", "why",
            "who", "is", "are", "was", "were", "do", "does", "did", "can",
            "could", "should", "would", "will",
        ];

        let terms: Vec<String> = question
            .split_whitespace()
            .filter_map(|word| {
                // Strip punctuation from the word
                let word_clean: String =
                    word.chars().filter(|c| c.is_alphabetic()).collect();
                let word_lower = word_clean.to_lowercase();

                #[cfg(feature = "log")]
                log::trace!(
                    "Processing word: '{}' -> '{}' (lowercase: '{}')",
                    word,
                    word_clean,
                    word_lower
                );

                if word_lower.len() > 2
                    && !stop_words.contains(&word_lower.as_str())
                {
                    #[cfg(feature = "log")]
                    log::trace!("Including word: '{}'", word_lower);
                    Some(word_lower)
                } else {
                    #[cfg(feature = "log")]
                    log::trace!("Excluding word: '{}'", word_lower);
                    None
                }
            })
            .collect();

        #[cfg(feature = "log")]
        log::trace!("Final extracted terms: {:?}", terms);
        terms
    }

    /// Add a note to the butler's context.
    async fn add_context_note(&mut self, note: &str) -> Result<String, String> {
        self.context_notes.push(note.to_string());
        // Keep only the last 20 context notes
        if self.context_notes.len() > 20 {
            self.context_notes.remove(0);
        }
        Ok(format!("Added context note: {}", note))
    }

    /// Get the butler's current context and recent activity.
    fn get_status(&self) -> String {
        let mut status = String::new();

        status.push_str(&format!(
            "Butler Status:\n- Memory Palace: `{:?}`\n- Context notes: {}\n- Recent queries: {}\n\n",
            self.memory_palace.db,
            self.context_notes.len(),
            self.recent_queries.len()
        ));

        status.push_str(&self.generate_context());

        status
    }
}

#[async_trait::async_trait]
impl<C: surrealdb::Connection> Tool for Butler<C> {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("Butler::ask")
                .description("Ask the Butler a question. The Butler will search the Memory Palace and provide a contextual response.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The question to ask the Butler."
                        }
                    },
                    "required": ["question"]
                }))
                .build()
                .unwrap(),

            Method::builder("Butler::add_context")
                .description("Add a context note for the Butler to remember about the current conversation.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "A context note for the Butler to remember."
                        }
                    },
                    "required": ["note"]
                }))
                .build()
                .unwrap(),

            Method::builder("Butler::status")
                .description("Get the Butler's current status and context.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build()
                .unwrap(),
        ].into_iter())
    }

    async fn call<'a>(&mut self, call: Use<'a>) -> super::Result<'a> {
        let method_name = call.name.split("::").last().unwrap_or(&call.name);

        match method_name {
            "ask" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let question =
                    match input.get("question").and_then(|v| v.as_str()) {
                        Some(question) => question,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content:
                                    "Missing required 'question' parameter"
                                        .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                match self.ask_butler(question).await {
                    Ok(response) => super::Result {
                        tool_use_id: call.id,
                        content: response.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Butler error: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }

            "add_context" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let note = match input.get("note").and_then(|v| v.as_str()) {
                    Some(note) => note,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'note' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.add_context_note(note).await {
                    Ok(response) => super::Result {
                        tool_use_id: call.id,
                        content: response.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to add context note: {}", err)
                            .into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }

            "status" => {
                let status = self.get_status();
                super::Result {
                    tool_use_id: call.id,
                    content: status.into(),
                    is_error: false,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                }
            }

            _ => super::Result {
                tool_use_id: call.id,
                content: format!("Unknown method: {}", method_name).into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    async fn save_json(&mut self) -> serde_json::Value {
        let mut data = json!({
            "context_notes": self.context_notes,
            "recent_queries": self.recent_queries,
        });

        // Save the memory palace state
        data["memory_palace"] = self.memory_palace.save_json().await;

        data
    }

    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> Result<(), String> {
        if let Some(context_notes) =
            json.get("context_notes").and_then(|v| v.as_array())
        {
            self.context_notes = context_notes
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        if let Some(recent_queries) =
            json.get("recent_queries").and_then(|v| v.as_array())
        {
            self.recent_queries = recent_queries
                .iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect();
        }

        // Load memory palace state if available
        if let Some(memory_palace_data) = json.get("memory_palace") {
            self.memory_palace
                .load_json(memory_palace_data.clone())
                .await
                .map_err(|e| format!("Failed to load memory palace: {}", e))?;
        }

        Ok(())
    }

    fn apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Add butler instructions to the system prompt
        if let Some(system) = &mut prompt.system {
            for block in system.iter_mut() {
                if let Block::Text { text, .. } = block {
                    if text.contains("<butler_instructions>") {
                        let mut new_text = BUTLER_INSTRUCTIONS.to_string();
                        new_text.push('\n');
                        new_text.push_str("<butler_status>\n");
                        new_text.push_str(&self.get_status());
                        new_text.push_str("</butler_status>");
                        *text = new_text.into();
                        return Ok(());
                    }
                }
            }

            // Not found, append to system prompt
            let mut full_text = BUTLER_INSTRUCTIONS.to_string();
            full_text.push('\n');
            full_text.push_str("<butler_status>\n");
            full_text.push_str(&self.get_status());
            full_text.push_str("</butler_status>");
            system.push(full_text);
        } else {
            // No system prompt, create one
            let mut full_text = BUTLER_INSTRUCTIONS.to_string();
            full_text.push('\n');
            full_text.push_str("<butler_status>\n");
            full_text.push_str(&self.get_status());
            full_text.push_str("</butler_status>");
            prompt.system = Some(full_text.into());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use surrealdb::engine::local::{Db, Mem};
    use surrealdb::opt::Config;

    async fn new_test_db() -> surrealdb::Surreal<Db> {
        let config = Config::default().strict();
        let db = surrealdb::Surreal::new::<Mem>(config).await.unwrap();
        // No need to manually create namespace/database - MemoryPalace will handle it
        db
    }

    #[tokio::test]
    async fn test_butler_ask() {
        let memory_palace =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();
        let mut butler = Butler::from_memory_palace(memory_palace);

        // First, store some information via the memory palace
        butler
            .memory_palace
            .store_memory(
                "Programming",
                "Rust is a systems programming language",
                vec!["rust".to_string(), "programming".to_string()],
            )
            .await
            .unwrap();
        butler
            .memory_palace
            .store_memory(
                "Programming",
                "Python is great for scripting",
                vec!["python".to_string(), "scripting".to_string()],
            )
            .await
            .unwrap();

        // Now ask the butler about Rust
        let response = butler
            .ask_butler("What do you know about Rust?")
            .await
            .unwrap();
        dbg!(&response);
        assert!(response.contains("Rust is a systems programming language"));
        assert!(response.contains("Programming"));
    }

    #[tokio::test]
    async fn test_butler_context() {
        let memory_palace =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();
        let mut butler = Butler::from_memory_palace(memory_palace);

        // Add context note
        butler
            .add_context_note("User is learning Rust programming")
            .await
            .unwrap();

        // Check status includes context
        let status = butler.get_status();
        assert!(status.contains("User is learning Rust programming"));
    }

    #[tokio::test]
    async fn test_butler_save_load() {
        let memory_palace1 =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();
        let mut butler1 = Butler::from_memory_palace(memory_palace1);
        butler1.add_context_note("Test context").await.unwrap();
        butler1.add_recent_query("test query");

        let json = butler1.save_json().await;

        let memory_palace2 =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();
        let mut butler2 = Butler::from_memory_palace(memory_palace2);
        butler2.load_json(json).await.unwrap();

        assert_eq!(butler1.context_notes, butler2.context_notes);
        assert_eq!(butler1.recent_queries, butler2.recent_queries);
    }
}
