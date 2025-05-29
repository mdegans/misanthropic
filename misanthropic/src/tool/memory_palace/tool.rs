use super::MemoryPalace;
use crate::tool::memory_palace::db::ensure_initialized;
use crate::tool::{self, Method, Tool, Use};
use crate::{Prompt, prompt::message::Block};
use serde_json::json;

#[async_trait::async_trait]
impl Tool for MemoryPalace {
    fn name(&self) -> &str {
        Self::NAME
    }

    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Check if instructions are already present to avoid duplication
        if let Some(system) = &mut prompt.system {
            for block in system.iter_mut() {
                if let Block::Text { text, .. } = block {
                    if text.contains("<memory_palace_instructions>") {
                        return Ok(()); // Already initialized
                    }
                }
            }

            // If not found, append the instructions
            system.push(super::MEMORY_PALACE_INSTRUCTIONS);
        }

        // Add memory palace instructions to the system prompt
        prompt.system = Some(super::MEMORY_PALACE_INSTRUCTIONS.into());
        Ok(())
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("MemoryPalace::store")
                .description("Store a new memory in the palace.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "The room where the memory belongs."
                        },
                        "content": {
                            "type": "string",
                            "description": "The content of the memory."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Tags for categorizing the memory."
                        }
                    },
                    "required": ["room", "content"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::search")
                .description("Search for memories by content, room, or tags.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query (text to find in memories)."
                        }
                    },
                    "required": ["query"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::summary")
                .description("Get a summary of recent and important memories.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::connect")
                .description("Connect two rooms in the palace.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "room1": {
                            "type": "string",
                            "description": "The first room to connect."
                        },
                        "room2": {
                            "type": "string",
                            "description": "The second room to connect."
                        }
                    },
                    "required": ["room1", "room2"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::list_rooms")
                .description("List all rooms with their memory counts and connections.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::relate")
                .description("Create or update a relationship between two memories.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id1": {
                            "type": "number",
                            "description": "ID of the first memory."
                        },
                        "memory_id2": {
                            "type": "number",
                            "description": "ID of the second memory."
                        },
                        "relationship_type": {
                            "type": "string",
                            "description": "Type of the relationship."
                        },
                        "strength": {
                            "type": "number",
                            "description": "Strength of the relationship (0.0 to 1.0).",
                            "default": 1.0
                        }
                    },
                    "required": ["memory_id1", "memory_id2", "relationship_type"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::find_related")
                .description("Find memories related to a given memory.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "number",
                            "description": "ID of the memory to find relations for."
                        },
                        "max_depth": {
                            "type": "number",
                            "description": "Maximum depth for relationship traversal.",
                            "default": 2
                        },
                        "min_strength": {
                            "type": "number",
                            "description": "Minimum strength of relationships to consider.",
                            "default": 0.1
                        }
                    },
                    "required": ["memory_id"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::extract_concepts")
                .description("Extract and link concepts from memory content.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "number",
                            "description": "ID of the memory to extract concepts from."
                        },
                        "concepts": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "List of concept names to extract."
                        }
                    },
                    "required": ["memory_id", "concepts"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::find_by_concept")
                .description("Find memories by concept with enhanced scoring.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "concept_name": {
                            "type": "string",
                            "description": "The name of the concept to search for."
                        }
                    },
                    "required": ["concept_name"]
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::graph_stats")
                .description("Get statistics and insights about the memory graph.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
                }))
                .build()
                .unwrap(),
            Method::builder("MemoryPalace::find_bfs")
                .description("Find memories using breadth-first search with decay factor for semantic distance.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "number",
                            "description": "ID of the starting memory for BFS exploration."
                        },
                        "max_distance": {
                            "type": "number",
                            "description": "Maximum distance to explore in the graph.",
                            "default": 3
                        },
                        "decay_factor": {
                            "type": "number",
                            "description": "Decay factor for path strength (0.0 to 1.0).",
                            "default": 0.8
                        },
                        "min_score": {
                            "type": "number",
                            "description": "Minimum path score threshold.",
                            "default": 0.1
                        }
                    },
                    "required": ["memory_id", "max_distance", "decay_factor", "min_score"]
                }))
                .build()
                .unwrap(),
        ].into_iter())
    }

    async fn call<'a>(&mut self, call: Use<'a>) -> tool::Result<'a> {
        let method_name = call.name.split("::").last().unwrap_or(&call.name);

        match method_name {
            "store" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let room = match input.get("room").and_then(|v| v.as_str()) {
                    Some(r) => r,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let content = match input.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'content' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let tags = input.get("tags").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<&str>>()
                });

                let tags_refs = tags.as_ref().map(|v| v.iter().copied()).unwrap_or_else(|| [].iter().copied());

                match self.store_memory(room, content, tags_refs).await {
                    Ok(memory_id) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Memory stored with ID: {} in room '{}'", memory_id, room).into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to store memory: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "search" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let query = match input.get("query").and_then(|v| v.as_str()) {
                    Some(q) => q,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'query' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.search(query).await {
                    Ok(results) => {
                        if results.is_empty() {
                            tool::Result {
                                tool_use_id: call.id,
                                content: format!("No memories found for query: '{}'", query).into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!("Found {} memories for query '{}':\n\n", results.len(), query);
                            for (room_name, memory_id, memory) in results {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nTags: {}\nCreated: {}\n\n",
                                    room_name,
                                    memory_id,
                                    memory.content,
                                    memory.tags.join(", "),
                                    memory.created_at.format("%Y-%m-%d %H:%M:%S")
                                ));
                            }

                            tool::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to search memories: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "summary" => {
                match self.get_context_summary().await {
                    Ok(summary) => tool::Result {
                        tool_use_id: call.id,
                        content: summary.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to get summary: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "connect" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let room1 = match input.get("room1").and_then(|v| v.as_str()) {
                    Some(r) => r,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room1' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let room2 = match input.get("room2").and_then(|v| v.as_str()) {
                    Some(r) => r,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room2' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.connect_rooms(room1, room2).await {
                    Ok(()) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Rooms '{}' and '{}' connected.", room1, room2).into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to connect rooms: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "list_rooms" => {
                match self.list_rooms().await {
                    Ok(rooms) => {
                        let mut response = String::new();
                        for (name, description, memory_count, connections) in rooms {
                            response.push_str(&format!(
                                "Room: {}\nDescription: {}\nMemories: {}\nConnections: {}\n\n",
                                name,
                                description,
                                memory_count,
                                connections.join(", ")
                            ));
                        }

                        tool::Result {
                            tool_use_id: call.id,
                            content: response.into(),
                            is_error: false,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        }
                    }
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to list rooms: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "relate" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id1 = match input.get("memory_id1").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'memory_id1' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id2 = match input.get("memory_id2").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'memory_id2' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let relationship_type = match input.get("relationship_type").and_then(|v| v.as_str()) {
                    Some(r) => r,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'relationship_type' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let strength = input
                    .get("strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);

                match self.relate_memories(memory_id1, memory_id2, relationship_type, strength).await {
                    Ok(msg) => tool::Result {
                        tool_use_id: call.id,
                        content: msg.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to relate memories: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "find_related" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id = match input.get("memory_id").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'memory_id' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let max_depth = input
                    .get("max_depth")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(2);

                let min_strength = input
                    .get("min_strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.1);

                match self.find_related_memories(memory_id, max_depth, min_strength).await {
                    Ok(results) => {
                        if results.is_empty() {
                            tool::Result {
                                tool_use_id: call.id,
                                content: format!("No related memories found for ID '{}'", memory_id).into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!(
                                "Found {} related memories for ID '{}':\n\n",
                                results.len(),
                                memory_id
                            );
                            for (room_name, related_memory_id, memory, relationship_type, strength) in results {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nRelationship: {} (Strength: {:.2})\n\n",
                                    room_name, related_memory_id, memory.content, relationship_type, strength
                                ));
                            }

                            tool::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to find related memories: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "extract_concepts" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id = match input.get("memory_id").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'memory_id' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let concepts = input.get("concepts").and_then(|v| v.as_array()).map(|arr| {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<&str>>()
                });

                let concepts_refs = concepts.as_ref().map(|v| v.iter().copied()).unwrap_or_else(|| [].iter().copied());

                match self.extract_concepts(memory_id, concepts_refs).await {
                    Ok(msg) => tool::Result {
                        tool_use_id: call.id,
                        content: msg.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to extract concepts: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "find_by_concept" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let concept_name = match input.get("concept_name").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'concept_name' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.find_memories_by_concept(concept_name).await {
                    Ok(results) => {
                        if results.is_empty() {
                            tool::Result {
                                tool_use_id: call.id,
                                content: format!("No memories found for concept: '{}'", concept_name).into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!(
                                "Found {} memories for concept '{}':\n\n",
                                results.len(),
                                concept_name
                            );
                            for (room_name, memory_id, memory, confidence) in results {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nConfidence: {:.2}\n\n",
                                    room_name, memory_id, memory.content, confidence
                                ));
                            }

                            tool::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to find memories by concept: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "graph_stats" => {
                match self.get_graph_stats().await {
                    Ok(stats) => tool::Result {
                        tool_use_id: call.id,
                        content: stats.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to get graph stats: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "find_bfs" => {
                let input = match call.input.as_object() {
                    Some(obj) => obj,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id = match input.get("memory_id").and_then(|v| v.as_i64()) {
                    Some(id) => id,
                    None => {
                        return tool::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'memory_id' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let max_distance = input
                    .get("max_distance")
                    .and_then(|v| v.as_u64())
                    .map(|v| v as u32)
                    .unwrap_or(3);

                let decay_factor = input
                    .get("decay_factor")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.8);

                let min_score = input
                    .get("min_score")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.1);

                match self.find_memories_bfs(memory_id, max_distance, decay_factor, min_score).await {
                    Ok(results) => {
                        if results.is_empty() {
                            tool::Result {
                                tool_use_id: call.id,
                                content: format!("No memories found within distance {} from ID '{}'", max_distance, memory_id).into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!(
                                "Found {} memories within distance {} from ID '{}':\n\n",
                                results.len(),
                                max_distance,
                                memory_id
                            );
                            for (room_name, bfs_memory_id, memory, score, distance) in results {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nPath Score: {:.3}\nDistance: {} hops\n\n",
                                    room_name, bfs_memory_id, memory.content, score, distance
                                ));
                            }

                            tool::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(err) => tool::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to find memories via BFS: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            _ => tool::Result {
                tool_use_id: call.id,
                content: format!(
                    "Unknown method '{}'. Available methods: store, search, summary, connect, list_rooms, relate, find_related, find_bfs, extract_concepts, find_by_concept, graph_stats",
                    method_name
                )
                .into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    async fn save_json(&mut self) -> serde_json::Value {
        // Only export the schema name since the actual state lives in the database
        json!({
            "schema_name": self.schema_name
        })
    }

    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> Result<(), String> {
        let data = if let serde_json::Value::Object(obj) = json {
            obj
        } else {
            return Err("Input must be a JSON object".to_string());
        };

        // Only restore the schema name - the database state persists independently
        if let Some(schema_name) =
            data.get("schema_name").and_then(|v| v.as_str())
        {
            self.schema_name = schema_name.to_string();
            // Re-initialize to ensure the schema exists
            ensure_initialized(&self.pool, &self.schema_name)
                .await
                .map_err(|e| e.to_string())?;
        }

        Ok(())
    }
}
