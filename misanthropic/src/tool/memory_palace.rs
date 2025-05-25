//! [`MemoryPalace`] tool for hierarchical knowledge organization using SurrealDB.

use super::{Method, Tool, Use};
use crate::{Prompt, prompt::message::Block};
use serde::{Deserialize, Serialize};
use serde_json::json;
use surrealdb::{RecordId, Surreal};

const MEMORY_PALACE_INSTRUCTIONS: &str = r#"<memory_palace_instructions>You have access to a Memory Palace - a spatial knowledge organization system powered by SurrealDB. Your memories are organized into rooms with semantic relationships, tags, and full-text search capabilities. Use this to store, organize, and retrieve knowledge across conversations.</memory_palace_instructions>"#;

/// A memory item stored in the palace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Memory {
    /// The actual content/knowledge stored.
    pub(crate) content: String,
    /// Room this memory belongs to.
    room: String,
    /// Tags for categorization and search.
    pub(crate) tags: Vec<String>,
    /// When this memory was created (as string for SurrealDB).
    created_at: String,
    /// When this memory was last accessed.
    last_accessed: String,
}

/// A room in the memory palace.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Room {
    /// Name of the room.
    name: String,
    /// Description of what this room contains.
    description: String,
}

/// A connection between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Connection {
    /// Source room ID.
    from: RecordId,
    /// Target room ID.
    to: RecordId,
    /// Optional description of the relationship.
    description: Option<String>,
}

/// Database record with ID for queries.
#[derive(Debug, Deserialize)]
struct Record<T> {
    id: RecordId,
    #[serde(flatten)]
    data: T,
}

/// A Memory Palace tool using SurrealDB for semantic storage.
#[derive(Debug)]
pub struct MemoryPalace<C: surrealdb::Connection> {
    /// SurrealDB connection.
    pub(crate) db: Surreal<C>,
    /// Whether the database has been initialized.
    initialized: bool,
}

impl<C: surrealdb::Connection> MemoryPalace<C> {
    const NAME: &'static str = "MemoryPalace";

    /// Create a new [`MemoryPalace`]` from an existing [`Surreal`] DB.
    /// Initializes the database if it hasn't been done yet.
    pub async fn from_db(db: Surreal<C>) -> Result<Self, String> {
        let mut new = Self {
            db,
            initialized: false,
        };

        // Ensure the database is initialized
        new.ensure_initialized().await?;

        Ok(new)
    }

    /// Initialize the database connection and schema.
    async fn ensure_initialized(&mut self) -> Result<(), String> {
        if self.initialized {
            return Ok(());
        }

        // Create namespace first
        self.db
            .query("DEFINE NAMESPACE IF NOT EXISTS memory_palace;")
            .await
            .map_err(|e| format!("Failed to create namespace: {}", e))?;

        // Use the namespace
        self.db
            .use_ns("memory_palace")
            .await
            .map_err(|e| format!("Failed to use namespace: {}", e))?;

        // Now create the database within the namespace
        self.db
            .query("DEFINE DATABASE IF NOT EXISTS palace;")
            .await
            .map_err(|e| format!("Failed to create database: {}", e))?;

        // Use the database
        self.db
            .use_db("palace")
            .await
            .map_err(|e| format!("Failed to use database: {}", e))?;

        // Enhanced schema with graph relationships and future vector support
        self.db
            .query(
                r#"
                DEFINE TABLE memory SCHEMAFULL;
                DEFINE FIELD content ON TABLE memory TYPE string;
                DEFINE FIELD room ON TABLE memory TYPE string;
                DEFINE FIELD tags ON TABLE memory TYPE array<string>;
                DEFINE FIELD created_at ON TABLE memory TYPE datetime;
                DEFINE FIELD last_accessed ON TABLE memory TYPE datetime;
                DEFINE FIELD access_count ON TABLE memory TYPE int DEFAULT 0;
                DEFINE INDEX room_index ON TABLE memory COLUMNS room;
                DEFINE INDEX tags_index ON TABLE memory COLUMNS tags;
                DEFINE INDEX content_index ON TABLE memory COLUMNS content;

                DEFINE TABLE room SCHEMAFULL;
                DEFINE FIELD name ON TABLE room TYPE string;
                DEFINE FIELD description ON TABLE room TYPE string;
                DEFINE FIELD created_at ON TABLE room TYPE datetime DEFAULT time::now();
                DEFINE INDEX room_name_index ON TABLE room COLUMNS name UNIQUE;

                DEFINE TABLE connection SCHEMAFULL;
                DEFINE FIELD from ON TABLE connection TYPE record<room>;
                DEFINE FIELD to ON TABLE connection TYPE record<room>;
                DEFINE FIELD description ON TABLE connection TYPE option<string>;
                DEFINE FIELD strength ON TABLE connection TYPE int DEFAULT 1;
                DEFINE FIELD created_at ON TABLE connection TYPE datetime DEFAULT time::now();

                -- Graph relationships between memories (relates, references, etc.)
                DEFINE TABLE relates SCHEMAFULL;
                DEFINE FIELD in ON TABLE relates TYPE record<memory>;
                DEFINE FIELD out ON TABLE relates TYPE record<memory>;
                DEFINE FIELD relationship_type ON TABLE relates TYPE string DEFAULT 'related';
                DEFINE FIELD strength ON TABLE relates TYPE float DEFAULT 1.0;
                DEFINE FIELD created_at ON TABLE relates TYPE datetime DEFAULT time::now();

                -- For future: concept extraction and linking
                DEFINE TABLE concept SCHEMAFULL;
                DEFINE FIELD name ON TABLE concept TYPE string;
                DEFINE FIELD description ON TABLE concept TYPE option<string>;
                DEFINE FIELD created_at ON TABLE concept TYPE datetime DEFAULT time::now();
                DEFINE INDEX concept_name_index ON TABLE concept COLUMNS name UNIQUE;

                DEFINE TABLE mentions SCHEMAFULL;
                DEFINE FIELD in ON TABLE mentions TYPE record<memory>;
                DEFINE FIELD out ON TABLE mentions TYPE record<concept>;
                DEFINE FIELD confidence ON TABLE mentions TYPE float DEFAULT 1.0;
                "#,
            )
            .await
            .map_err(|e| format!("Failed to create schema: {}", e))?;

        self.initialized = true;
        Ok(())
    }

    /// Store a memory in a specific room.
    pub(crate) async fn store_memory<S: std::fmt::Display>(
        &mut self,
        room_name: S,
        content: S,
        tags: Vec<String>,
    ) -> Result<String, String> {
        self.ensure_initialized().await?;

        // Ensure room exists
        let room_query = r#"
            SELECT * FROM room WHERE name = $room_name;
        "#;
        let existing_rooms: Vec<Record<Room>> = self
            .db
            .query(room_query)
            .bind(("room_name", room_name.to_string()))
            .await
            .map_err(|e| format!("Failed to query rooms: {}", e))?
            .take(0)
            .map_err(|e| format!("Failed to take rooms result: {}", e))?;

        if existing_rooms.is_empty() {
            // Create the room
            let _room: Option<Record<Room>> = self
                .db
                .create("room")
                .content(Room {
                    name: room_name.to_string(),
                    description: format!("Room for {}", room_name),
                })
                .await
                .map_err(|e| format!("Failed to create room: {}", e))?;
        }

        // Create the memory using SurrealDB's time::now() function
        let create_query = r#"
            CREATE memory SET 
                content = $content,
                room = $room,
                tags = $tags,
                created_at = time::now(),
                last_accessed = time::now();
        "#;

        let results: Vec<Record<Memory>> = self
            .db
            .query(create_query)
            .bind(("content", content.to_string()))
            .bind(("room", room_name.to_string()))
            .bind(("tags", tags))
            .await
            .map_err(|e| format!("Failed to create memory: {}", e))?
            .take(0)
            .map_err(|e| format!("Failed to take create result: {}", e))?;

        match results.first() {
            Some(record) => Ok(record.id.to_string()),
            None => Err("Failed to create memory record".to_string()),
        }
    }

    /// Search for memories using basic text matching and filters.
    pub(crate) async fn search<S: std::fmt::Display>(
        &mut self,
        query: S,
    ) -> Result<Vec<(String, String, Memory)>, String> {
        self.ensure_initialized().await?;

        let query_lower = query.to_string().to_lowercase();
        #[cfg(feature = "log")]
        log::debug!("Memory Palace searching for: '{}'", query_lower);

        // Use basic string matching - search in content, tags, and room names (case-insensitive)
        // Simplify array search - just check if any tag contains the query
        let search_query = r#"
            SELECT * FROM memory 
            WHERE string::lowercase(content) CONTAINS $query
               OR string::lowercase(room) CONTAINS $query
               OR array::any(tags, |$tag| string::lowercase($tag) CONTAINS $query)
            ORDER BY created_at DESC;
        "#;

        let results: Vec<Record<Memory>> = self
            .db
            .query(search_query)
            .bind(("query", query_lower))
            .await
            .map_err(|e| {
                #[cfg(feature = "log")]
                log::error!("Search query failed: {}", e);
                format!("Failed to search memories: {}", e)
            })?
            .take(0)
            .map_err(|e| {
                #[cfg(feature = "log")]
                log::error!("Failed to take search results: {}", e);
                format!("Failed to take search results: {}", e)
            })?;

        #[cfg(feature = "log")]
        log::debug!("Found {} memories", results.len());

        // Update last_accessed for found memories using SurrealDB's time::now()
        for record in &results {
            let update_query = r#"
                UPDATE $id SET last_accessed = time::now();
            "#;

            let _: Vec<Record<Memory>> = self
                .db
                .query(update_query)
                .bind(("id", record.id.clone()))
                .await
                .map_err(|e| format!("Failed to update last_accessed: {}", e))?
                .take(0)
                .map_err(|e| format!("Failed to take update result: {}", e))?;
        }

        // Convert to expected format
        Ok(results
            .into_iter()
            .map(|record| {
                #[cfg(feature = "log")]
                log::trace!(
                    "Memory result - Room: {}, Content: {}, Tags: {:?}",
                    record.data.room,
                    record.data.content,
                    record.data.tags
                );
                (record.data.room.clone(), record.id.to_string(), record.data)
            })
            .collect())
    }

    /// Optimized search for multiple terms using a single query with ranking.
    pub(crate) async fn search_optimized(
        &mut self,
        search_terms: &[String],
    ) -> Result<Vec<(String, String, Memory)>, String> {
        self.ensure_initialized().await?;

        if search_terms.is_empty() {
            return Ok(Vec::new());
        }

        #[cfg(feature = "log")]
        log::debug!(
            "Memory Palace optimized search for terms: {:?}",
            search_terms
        );

        // Build a more sophisticated query with scoring using proper SurrealDB boolean handling
        let mut query_conditions = Vec::new();
        let mut score_expressions = Vec::new();
        let mut bindings = Vec::new();

        for (i, term) in search_terms.iter().enumerate() {
            let term_lower = term.to_lowercase();
            let param_name = format!("term{}", i);

            // Search conditions (OR logic for any match)
            query_conditions.push(format!(
                "(string::lowercase(content) CONTAINS ${param} OR string::lowercase(room) CONTAINS ${param} OR array::any(tags, |$tag| string::lowercase($tag) CONTAINS ${param}))",
                param = param_name
            ));

            // Score expressions using count() to convert boolean to number
            // SurrealDB: count(condition) gives us 1 for true, 0 for false
            score_expressions.push(format!(
                "count(string::lowercase(content) CONTAINS ${param}) * 3 + count(string::lowercase(room) CONTAINS ${param}) * 2 + count(array::any(tags, |$tag| string::lowercase($tag) CONTAINS ${param})) * 1",
                param = param_name
            ));

            bindings.push((param_name, term_lower));
        }

        // Create relevance score by summing all score expressions
        let relevance_score = score_expressions.join(" + ");

        let search_query = format!(
            r#"
            SELECT *, ({relevance_score}) AS relevance_score FROM memory 
            WHERE {conditions}
            ORDER BY relevance_score DESC, created_at DESC;
            "#,
            conditions = query_conditions.join(" OR "),
            relevance_score = relevance_score
        );

        #[cfg(feature = "log")]
        log::trace!("Optimized search query: {}", search_query);

        let mut query = self.db.query(&search_query);
        for (param_name, value) in bindings {
            query = query.bind((param_name, value));
        }

        let results: Vec<Record<Memory>> = query
            .await
            .map_err(|e| {
                #[cfg(feature = "log")]
                log::error!("Optimized search query failed: {}", e);
                format!("Failed to search memories: {}", e)
            })?
            .take(0)
            .map_err(|e| {
                #[cfg(feature = "log")]
                log::error!("Failed to take search results: {}", e);
                format!("Failed to take search results: {}", e)
            })?;

        #[cfg(feature = "log")]
        log::debug!("Found {} memories with optimized search", results.len());

        // Convert to Memory records and update last_accessed
        let mut memory_results = Vec::new();
        for record in results {
            let id = record.id.to_string();

            // Update last_accessed for this memory
            let update_query = r#"
                UPDATE $id SET last_accessed = time::now();
            "#;

            let _: Vec<Record<Memory>> = self
                .db
                .query(update_query)
                .bind(("id", record.id.clone()))
                .await
                .map_err(|e| format!("Failed to update last_accessed: {}", e))?
                .take(0)
                .map_err(|e| format!("Failed to take update result: {}", e))?;

            #[cfg(feature = "log")]
            log::trace!(
                "Memory result - Room: {}, Content: {}, Tags: {:?}",
                record.data.room,
                record.data.content,
                record.data.tags
            );

            memory_results.push((record.data.room.clone(), id, record.data));
        }

        Ok(memory_results)
    }

    /// Connect two rooms in the palace.
    async fn connect_rooms<S: std::fmt::Display>(
        &mut self,
        room1: S,
        room2: S,
    ) -> Result<(), String> {
        self.ensure_initialized().await?;

        // Find both rooms
        let room_query = r#"
            SELECT * FROM room WHERE name = $room_name;
        "#;

        let room1_records: Vec<Record<Room>> = self
            .db
            .query(room_query)
            .bind(("room_name", room1.to_string()))
            .await
            .map_err(|e| format!("Failed to query room1: {}", e))?
            .take(0)
            .map_err(|e| format!("Failed to take room1 result: {}", e))?;

        let room2_records: Vec<Record<Room>> = self
            .db
            .query(room_query)
            .bind(("room_name", room2.to_string()))
            .await
            .map_err(|e| format!("Failed to query room2: {}", e))?
            .take(0)
            .map_err(|e| format!("Failed to take room2 result: {}", e))?;

        if room1_records.is_empty() {
            return Err(format!("Room '{}' does not exist", room1));
        }
        if room2_records.is_empty() {
            return Err(format!("Room '{}' does not exist", room2));
        }

        let room1_id = &room1_records[0].id;
        let room2_id = &room2_records[0].id;

        // Create bidirectional connections
        let _: Option<Record<Connection>> = self
            .db
            .create("connection")
            .content(Connection {
                from: room1_id.clone(),
                to: room2_id.clone(),
                description: None,
            })
            .await
            .map_err(|e| format!("Failed to create connection 1->2: {}", e))?;

        let _: Option<Record<Connection>> = self
            .db
            .create("connection")
            .content(Connection {
                from: room2_id.clone(),
                to: room1_id.clone(),
                description: None,
            })
            .await
            .map_err(|e| format!("Failed to create connection 2->1: {}", e))?;

        Ok(())
    }

    /// List all rooms with their memory counts and connections.
    async fn list_rooms(
        &mut self,
    ) -> Result<Vec<(String, String, usize, Vec<String>)>, String> {
        self.ensure_initialized().await?;

        let room_query = r#"
            SELECT *, 
                   (SELECT count() FROM memory WHERE room = $parent.name)[0] AS memory_count,
                   (SELECT room.name FROM connection WHERE from = $parent.id) AS connected_rooms
            FROM room;
        "#;

        let rooms: Vec<serde_json::Value> = self
            .db
            .query(room_query)
            .await
            .map_err(|e| format!("Failed to query rooms: {}", e))?
            .take(0)
            .map_err(|e| format!("Failed to take rooms result: {}", e))?;

        Ok(rooms
            .into_iter()
            .map(|room| {
                let name = room["name"].as_str().unwrap_or("").to_string();
                let description =
                    room["description"].as_str().unwrap_or("").to_string();
                let memory_count =
                    room["memory_count"].as_u64().unwrap_or(0) as usize;
                let connected_rooms = room["connected_rooms"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                (name, description, memory_count, connected_rooms)
            })
            .collect())
    }
}

// Note: We can't implement Serialize/Deserialize for MemoryPalace due to the SurrealDB connection
// Instead, we'll implement custom save/load that exports/imports the data

#[async_trait::async_trait]
impl<C: surrealdb::Connection> Tool for MemoryPalace<C> {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("MemoryPalace::store")
                .description("Store a memory in a specific room of the palace.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "Name of the room to store the memory in."
                        },
                        "content": {
                            "type": "string",
                            "description": "The knowledge/memory content to store."
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags for categorizing this memory."
                        }
                    },
                    "required": ["room", "content"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::search")
                .description("Search for memories using full-text search, tags, or room names.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query to find relevant memories. Supports full-text search."
                        }
                    },
                    "required": ["query"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::connect")
                .description("Connect two rooms in the palace to show their relationship.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "room1": {
                            "type": "string",
                            "description": "First room to connect."
                        },
                        "room2": {
                            "type": "string",
                            "description": "Second room to connect."
                        }
                    },
                    "required": ["room1", "room2"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::list_rooms")
                .description("List all rooms in the palace with their descriptions, memory counts, and connections.")
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
            "store" => {
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

                let room = match input.get("room").and_then(|v| v.as_str()) {
                    Some(room) => room,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let content =
                    match input.get("content").and_then(|v| v.as_str()) {
                        Some(content) => content,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content: "Missing required 'content' parameter"
                                    .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                let tags = input
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();

                match self
                    .store_memory(room.to_string(), content.to_string(), tags)
                    .await
                {
                    Ok(memory_id) => super::Result {
                        tool_use_id: call.id,
                        content: format!(
                            "Memory stored in room '{}' with ID '{}'",
                            room, memory_id
                        )
                        .into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to store memory: {}", err)
                            .into(),
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let query = match input.get("query").and_then(|v| v.as_str()) {
                    Some(query) => query,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'query' parameter"
                                .into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.search(query).await {
                    Ok(results) => {
                        if results.is_empty() {
                            super::Result {
                                tool_use_id: call.id,
                                content: format!(
                                    "No memories found matching '{}'",
                                    query
                                )
                                .into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!(
                                "Found {} memories matching '{}':\n\n",
                                results.len(),
                                query
                            );
                            for (room_name, memory_id, memory) in results {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nTags: {}\n\n",
                                    room_name,
                                    memory_id,
                                    memory.content,
                                    memory.tags.join(", ")
                                ));
                            }

                            super::Result {
                                tool_use_id: call.id,
                                content: response.into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        }
                    }
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Search failed: {}", err).into(),
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let room1 = match input.get("room1").and_then(|v| v.as_str()) {
                    Some(room) => room,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room1' parameter"
                                .into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let room2 = match input.get("room2").and_then(|v| v.as_str()) {
                    Some(room) => room,
                    None => {
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room2' parameter"
                                .into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self
                    .connect_rooms(room1.to_string(), room2.to_string())
                    .await
                {
                    Ok(()) => super::Result {
                        tool_use_id: call.id,
                        content: format!(
                            "Connected rooms '{}' and '{}'",
                            room1, room2
                        )
                        .into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: err.into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }

            "list_rooms" => match self.list_rooms().await {
                Ok(rooms) => {
                    if rooms.is_empty() {
                        super::Result {
                                tool_use_id: call.id,
                                content: "The memory palace is empty. No rooms have been created yet.".into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                    } else {
                        let mut response = format!(
                            "Memory Palace contains {} rooms:\n\n",
                            rooms.len()
                        );
                        for (name, description, count, connections) in rooms {
                            response.push_str(&format!(
                                    "Room: {}\nDescription: {}\nMemories: {}\nConnections: {}\n\n",
                                    name,
                                    description,
                                    count,
                                    if connections.is_empty() {
                                        "None".to_string()
                                    } else {
                                        connections.join(", ")
                                    }
                                ));
                        }

                        super::Result {
                            tool_use_id: call.id,
                            content: response.into(),
                            is_error: false,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        }
                    }
                }
                Err(err) => super::Result {
                    tool_use_id: call.id,
                    content: format!("Failed to list rooms: {}", err).into(),
                    is_error: true,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                },
            },

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
        if let Err(_) = self.ensure_initialized().await {
            return json!({"error": "Failed to initialize database"});
        }

        // Export all data from the database - use separate queries for SurrealDB 2.x
        let memories_query = "SELECT * FROM memory";
        let rooms_query = "SELECT * FROM room";
        let connections_query = "SELECT * FROM connection";

        let memories = match self.db.query(memories_query).await {
            Ok(mut result) => match result.take::<Vec<serde_json::Value>>(0) {
                Ok(data) => data,
                Err(_) => return json!({"error": "Failed to export memories"}),
            },
            Err(_) => return json!({"error": "Failed to query memories"}),
        };

        let rooms = match self.db.query(rooms_query).await {
            Ok(mut result) => match result.take::<Vec<serde_json::Value>>(0) {
                Ok(data) => data,
                Err(_) => return json!({"error": "Failed to export rooms"}),
            },
            Err(_) => return json!({"error": "Failed to query rooms"}),
        };

        let connections = match self.db.query(connections_query).await {
            Ok(mut result) => match result.take::<Vec<serde_json::Value>>(0) {
                Ok(data) => data,
                Err(_) => {
                    return json!({"error": "Failed to export connections"});
                }
            },
            Err(_) => return json!({"error": "Failed to query connections"}),
        };

        json!({
            "memories": memories,
            "rooms": rooms,
            "connections": connections
        })
    }

    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> Result<(), String> {
        self.ensure_initialized().await?;

        // Clear existing data
        self.db
            .query("DELETE memory; DELETE room; DELETE connection;")
            .await
            .map_err(|e| format!("Failed to clear database: {}", e))?;

        // Import rooms first
        if let Some(rooms) = json.get("rooms").and_then(|v| v.as_array()) {
            for room in rooms {
                if let Ok(room_data) =
                    serde_json::from_value::<Room>(room.clone())
                {
                    let _: Option<Record<Room>> = self
                        .db
                        .create("room")
                        .content(room_data)
                        .await
                        .map_err(|e| format!("Failed to import room: {}", e))?;
                }
            }
        }

        // Import memories
        if let Some(memories) = json.get("memories").and_then(|v| v.as_array())
        {
            for memory in memories {
                if let Ok(memory_data) =
                    serde_json::from_value::<Memory>(memory.clone())
                {
                    let _: Option<Record<Memory>> = self
                        .db
                        .create("memory")
                        .content(memory_data)
                        .await
                        .map_err(|e| {
                            format!("Failed to import memory: {}", e)
                        })?;
                }
            }
        }

        // Import connections
        if let Some(connections) =
            json.get("connections").and_then(|v| v.as_array())
        {
            for connection in connections {
                if let Ok(connection_data) =
                    serde_json::from_value::<Connection>(connection.clone())
                {
                    let _: Option<Record<Connection>> = self
                        .db
                        .create("connection")
                        .content(connection_data)
                        .await
                        .map_err(|e| {
                            format!("Failed to import connection: {}", e)
                        })?;
                }
            }
        }

        Ok(())
    }

    fn apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // For now, just add static instructions since we can't await in this method
        // In a future version, we could add room summaries here
        let palace_context = "<memory_palace>\nMemory Palace is available for storing and retrieving knowledge.\n</memory_palace>";

        if let Some(system) = &mut prompt.system {
            for block in system.iter_mut() {
                if let Block::Text { text, .. } = block {
                    if text.contains("<memory_palace_instructions>") {
                        let mut new_text =
                            MEMORY_PALACE_INSTRUCTIONS.to_string();
                        new_text.push('\n');
                        new_text.push_str(palace_context);
                        *text = new_text.into();
                        return Ok(());
                    }
                }
            }

            // Not found, append to system prompt
            let mut full_text = MEMORY_PALACE_INSTRUCTIONS.to_string();
            full_text.push('\n');
            full_text.push_str(palace_context);
            system.push(full_text);
        } else {
            // No system prompt, create one
            let mut full_text = MEMORY_PALACE_INSTRUCTIONS.to_string();
            full_text.push('\n');
            full_text.push_str(palace_context);
            prompt.system = Some(full_text.into());
        }

        Ok(())
    }
}

#[cfg(all(test, feature = "kv-mem"))]
mod tests {
    use super::*;
    use surrealdb::engine::local::{Db, Mem};
    use surrealdb::opt::Config;

    async fn new_test_db() -> Surreal<Db> {
        let config = Config::default().strict();
        let db = Surreal::new::<Mem>(config).await.unwrap();
        // No need to manually create namespace/database - MemoryPalace will handle it
        db
    }

    #[tokio::test]
    #[allow(unused_variables)] // Used in tests
    async fn test_memory_palace_store_and_search() {
        let mut palace =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();

        // Store some memories
        let id1 = palace
            .store_memory(
                "Rust",
                "async/await is for concurrent programming",
                vec!["rust".to_string(), "async".to_string()],
            )
            .await
            .unwrap();

        let _id2 = palace
            .store_memory(
                "Rust",
                "Traits define shared behavior",
                vec!["rust".to_string(), "traits".to_string()],
            )
            .await
            .unwrap();

        // Search for memories
        let results = palace.search("async").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].2.content,
            "async/await is for concurrent programming"
        );

        let results = palace.search("rust").await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_memory_palace_rooms_and_connections() {
        let mut palace =
            MemoryPalace::from_db(new_test_db().await).await.unwrap();

        // Create rooms with memories
        palace
            .store_memory("Rust", "Systems programming", vec![])
            .await
            .unwrap();
        palace
            .store_memory("WebAssembly", "Compile target", vec![])
            .await
            .unwrap();

        // Connect rooms
        palace.connect_rooms("Rust", "WebAssembly").await.unwrap();

        // List rooms
        let rooms = palace.list_rooms().await.unwrap();
        assert_eq!(rooms.len(), 2);

        // Check that rooms are connected
        let rust_room =
            rooms.iter().find(|(name, _, _, _)| name == "Rust").unwrap();
        assert!(rust_room.3.contains(&"WebAssembly".to_string()));
    }
}
