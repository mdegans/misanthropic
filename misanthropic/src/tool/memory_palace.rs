//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.

use super::{Method, Tool, Use};
use crate::{Prompt, prompt::message::Block};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{FromRow, PgPool, Row};

const MEMORY_PALACE_INSTRUCTIONS: &str = r#"<memory_palace_instructions>You have access to a Memory Palace - a spatial knowledge organization system. Your memories are organized into rooms with semantic relationships, tags, and full-text search capabilities. Use this to store, organize, and retrieve knowledge across conversations.</memory_palace_instructions>"#;

/// A memory item stored in the palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub(crate) struct Memory {
    /// Database record ID.
    pub(crate) id: i64,
    /// The actual content/knowledge stored.
    pub(crate) content: String,
    /// Room this memory belongs to.
    pub(crate) room: String,
    /// Tags for categorization and search (stored as JSONB).
    #[sqlx(json)]
    pub(crate) tags: Vec<String>,
    /// When this memory was created.
    pub(crate) created_at: chrono::DateTime<chrono::Utc>,
    /// When this memory was last updated (managed by database trigger).
    pub(crate) last_updated: chrono::DateTime<chrono::Utc>,
}

/// A room in the memory palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
struct Room {
    /// Database record ID.
    id: i64,
    /// Name of the room.
    name: String,
    /// Description of what this room contains.
    description: String,
    /// When this room was created.
    created_at: chrono::DateTime<chrono::Utc>,
}

/// A connection between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
struct Connection {
    /// Database record ID.
    id: i64,
    /// Source room name.
    from_room: String,
    /// Target room name.
    to_room: String,
    /// Optional description of the relationship.
    description: Option<String>,
    /// Strength of the connection.
    strength: i32,
    /// When this connection was created.
    created_at: chrono::DateTime<chrono::Utc>,
}

/// A Memory Palace tool using PostgreSQL for reliable storage.
#[derive(Debug)]
pub struct MemoryPalace {
    /// PostgreSQL connection pool.
    pub(crate) pool: PgPool,
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool.
    /// Initializes the database schema if it hasn't been done yet.
    pub async fn from_pool(pool: PgPool) -> Result<Self, String> {
        let mut new = Self { pool };

        // Ensure the database is initialized - this is our class invariant
        new.ensure_initialized().await?;

        Ok(new)
    }

    /// Initialize the database schema with proper indexes and triggers.
    async fn ensure_initialized(&mut self) -> Result<(), String> {
        // Create tables with proper indexes and triggers
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS rooms (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                tags JSONB NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS room_connections (
                id BIGSERIAL PRIMARY KEY,
                from_room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                to_room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                description TEXT,
                strength INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_room, to_room)
            );

            CREATE TABLE IF NOT EXISTS memory_relationships (
                id BIGSERIAL PRIMARY KEY,
                from_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relationship_type VARCHAR(100) NOT NULL DEFAULT 'related',
                strength FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_memory_id, to_memory_id)
            );

            CREATE TABLE IF NOT EXISTS concepts (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT, -- Optional description for complex concepts
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS memory_concepts (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                concept_id BIGINT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                confidence FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(memory_id, concept_id)
            );

            -- Trigger to automatically update last_updated timestamp
            CREATE OR REPLACE FUNCTION update_last_updated_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_updated = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql';

            DROP TRIGGER IF EXISTS update_memories_last_updated ON memories;
            CREATE TRIGGER update_memories_last_updated
                BEFORE UPDATE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_last_updated_column();

            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_memories_room ON memories(room);
            CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(to_tsvector('english', content));
            CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING gin(tags);
            CREATE INDEX IF NOT EXISTS idx_room_connections_from ON room_connections(from_room);
            CREATE INDEX IF NOT EXISTS idx_memory_relationships_from ON memory_relationships(from_memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts(memory_id);
            CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts(concept_id);
        "#)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create schema: {}", e))?;

        Ok(())
    }

    /// Store a memory in a specific room.
    pub(crate) async fn store_memory(
        &mut self,
        room_name: &str,
        content: &str,
        tags: impl IntoIterator<Item = &str>,
    ) -> Result<String, String> {
        // Ensure room exists
        sqlx::query(
            r#"
            INSERT INTO rooms (name, description) 
            VALUES ($1, $2) 
            ON CONFLICT (name) DO NOTHING
        "#,
        )
        .bind(room_name)
        .bind(format!("Room for {}", room_name))
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create room: {}", e))?;

        // Convert tags to Vec<String> for JSON serialization
        let tags: Vec<&str> = tags.into_iter().collect();
        let tags_json = serde_json::to_value(&tags)
            .map_err(|e| format!("Failed to serialize tags: {}", e))?;

        let row = sqlx::query(
            r#"
            INSERT INTO memories (content, room, tags) 
            VALUES ($1, $2, $3) 
            RETURNING id
        "#,
        )
        .bind(content)
        .bind(room_name)
        .bind(tags_json)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| format!("Failed to create memory: {}", e))?;

        let memory_id: i64 = row.get("id");
        Ok(memory_id.to_string())
    }

    /// Search for memories using full-text search and filters.
    pub(crate) async fn search(
        &mut self,
        query: &str,
    ) -> Result<Vec<(String, String, Memory)>, String> {
        #[cfg(feature = "log")]
        log::debug!("Memory Palace searching for: '{}'", query);

        let memories: Vec<Memory> = sqlx::query_as(
            r#"
            SELECT id, content, room, tags, created_at, last_updated
            FROM memories 
            WHERE 
                content ILIKE $1 
                OR room ILIKE $1 
                OR tags::text ILIKE $1
            ORDER BY 
                CASE 
                    WHEN content ILIKE $1 THEN 3
                    WHEN room ILIKE $1 THEN 2
                    WHEN tags::text ILIKE $1 THEN 1
                    ELSE 0
                END DESC,
                created_at DESC
        "#,
        )
        .bind(format!("%{}%", query))
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to search memories: {}", e))?;

        let results: Vec<_> = memories
            .into_iter()
            .map(|memory| (memory.room.clone(), memory.id.to_string(), memory))
            .collect();

        #[cfg(feature = "log")]
        log::debug!("Found {} memories", results.len());

        Ok(results)
    }

    /// Connect two rooms in the palace.
    async fn connect_rooms(
        &mut self,
        room1: &str,
        room2: &str,
    ) -> Result<(), String> {
        // Create bidirectional connections
        sqlx::query(
            r#"
            INSERT INTO room_connections (from_room, to_room, strength) 
            VALUES ($1, $2, 1), ($2, $1, 1)
            ON CONFLICT (from_room, to_room) DO NOTHING
        "#,
        )
        .bind(room1)
        .bind(room2)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create connections: {}", e))?;

        Ok(())
    }

    /// List all rooms with their memory counts and connections.
    async fn list_rooms(
        &mut self,
    ) -> Result<Vec<(String, String, usize, Vec<String>)>, String> {
        let rows = sqlx::query(
            r#"
            SELECT 
                r.name,
                r.description,
                COUNT(m.id) as memory_count
            FROM rooms r
            LEFT JOIN memories m ON r.name = m.room
            GROUP BY r.name, r.description
            ORDER BY r.name
        "#,
        )
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to query rooms: {}", e))?;

        let mut results = Vec::new();
        for row in rows {
            let room_name: String = row.get("name");
            let description: String = row.get("description");
            let memory_count: i64 = row.get("memory_count");

            // Get connections for this room
            let connection_rows = sqlx::query(
                r#"
                SELECT to_room FROM room_connections WHERE from_room = $1
            "#,
            )
            .bind(&room_name)
            .fetch_all(&self.pool)
            .await
            .map_err(|e| format!("Failed to query connections: {}", e))?;

            let connections: Vec<String> = connection_rows
                .into_iter()
                .map(|row| row.get("to_room"))
                .collect();

            results.push((
                room_name,
                description,
                memory_count as usize,
                connections,
            ));
        }

        Ok(results)
    }

    /// Create a relationship between two memories.
    pub(crate) async fn relate_memories(
        &mut self,
        memory_id1: i64,
        memory_id2: i64,
        relationship_type: &str,
        strength: f64,
    ) -> Result<String, String> {
        sqlx::query(r#"
            INSERT INTO memory_relationships (from_memory_id, to_memory_id, relationship_type, strength)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (from_memory_id, to_memory_id) 
            DO UPDATE SET relationship_type = $3, strength = $4
        "#)
        .bind(memory_id1)
        .bind(memory_id2)
        .bind(relationship_type)
        .bind(strength)
        .execute(&self.pool)
        .await
        .map_err(|e| format!("Failed to create relationship: {}", e))?;

        Ok(format!(
            "Created {} relationship between {} and {} with strength {}",
            relationship_type, memory_id1, memory_id2, strength
        ))
    }

    /// Find memories related to a given memory through graph traversal.
    /// Uses recursive CTE for multi-depth traversal.
    pub(crate) async fn find_related_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, Memory, String, f64)>, String> {
        // Use recursive CTE for proper graph traversal
        let rows = sqlx::query(
            r#"
            WITH RECURSIVE related_memories(memory_id, relationship_type, strength, depth) AS (
                -- Base case: direct relationships
                SELECT mr.to_memory_id, mr.relationship_type, mr.strength, 1 as depth
                FROM memory_relationships mr
                WHERE mr.from_memory_id = $1 AND mr.strength >= $3
                
                UNION
                
                -- Recursive case: follow relationships up to max_depth
                SELECT mr.to_memory_id, mr.relationship_type, mr.strength, rm.depth + 1
                FROM memory_relationships mr
                JOIN related_memories rm ON mr.from_memory_id = rm.memory_id
                WHERE rm.depth < $2 AND mr.strength >= $3
            )
            SELECT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                   rm.relationship_type, rm.strength
            FROM related_memories rm
            JOIN memories m ON rm.memory_id = m.id
            ORDER BY rm.strength DESC, rm.depth ASC
        "#,
        )
        .bind(memory_id)
        .bind(max_depth as i32)
        .bind(min_strength)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to find related memories: {}", e))?;

        let mut results = Vec::new();
        for row in rows {
            let memory_id: i64 = row.get("id");
            let tags_json: serde_json::Value = row.get("tags");
            let tags: Vec<String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            let relationship_type: String = row.get("relationship_type");
            let strength: f64 = row.get("strength");

            let memory = Memory {
                id: memory_id,
                content: row.get("content"),
                room: row.get("room"),
                tags,
                created_at: row.get("created_at"),
                last_updated: row.get("last_updated"),
            };

            results.push((
                memory.room.clone(),
                memory.id.to_string(),
                memory,
                relationship_type,
                strength,
            ));
        }

        Ok(results)
    }

    /// Extract and create concept nodes from memory content.
    pub(crate) async fn extract_concepts(
        &mut self,
        memory_id: i64,
        concepts: impl IntoIterator<Item = &str>,
    ) -> Result<String, String> {
        let mut created_concepts = Vec::new();

        for concept_name in concepts {
            // Create or get concept
            let concept_row = sqlx::query(
                r#"
                INSERT INTO concepts (name) VALUES ($1)
                ON CONFLICT (name) DO NOTHING
                RETURNING id
            "#,
            )
            .bind(concept_name)
            .fetch_optional(&self.pool)
            .await
            .map_err(|e| format!("Failed to create concept: {}", e))?;

            let concept_id: i64 = if let Some(row) = concept_row {
                row.get("id")
            } else {
                // Concept already exists, get its ID
                sqlx::query("SELECT id FROM concepts WHERE name = $1")
                    .bind(concept_name)
                    .fetch_one(&self.pool)
                    .await
                    .map_err(|e| format!("Failed to get concept ID: {}", e))?
                    .get("id")
            };

            // Link memory to concept
            sqlx::query(
                r#"
                INSERT INTO memory_concepts (memory_id, concept_id, confidence)
                VALUES ($1, $2, 1.0)
                ON CONFLICT (memory_id, concept_id) DO NOTHING
            "#,
            )
            .bind(memory_id)
            .bind(concept_id)
            .execute(&self.pool)
            .await
            .map_err(|e| format!("Failed to link memory to concept: {}", e))?;

            created_concepts.push(concept_name.to_owned());
        }

        Ok(format!(
            "Extracted and linked {} concepts: {}",
            created_concepts.len(),
            created_concepts.join(", ")
        ))
    }

    /// Find memories by concept.
    pub(crate) async fn find_memories_by_concept(
        &mut self,
        concept_name: &str,
    ) -> Result<Vec<(String, String, Memory, f64)>, String> {
        let rows = sqlx::query(
            r#"
            SELECT 
                m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                mc.confidence
            FROM memory_concepts mc
            JOIN memories m ON mc.memory_id = m.id
            JOIN concepts c ON mc.concept_id = c.id
            WHERE c.name = $1
            ORDER BY mc.confidence DESC
        "#,
        )
        .bind(concept_name)
        .fetch_all(&self.pool)
        .await
        .map_err(|e| format!("Failed to find memories by concept: {}", e))?;

        let mut results = Vec::new();
        for row in rows {
            let memory_id: i64 = row.get("id");
            let tags_json: serde_json::Value = row.get("tags");
            let tags: Vec<String> =
                serde_json::from_value(tags_json).unwrap_or_default();
            let confidence: f64 = row.get("confidence");

            let memory = Memory {
                id: memory_id,
                content: row.get("content"),
                room: row.get("room"),
                tags,
                created_at: row.get("created_at"),
                last_updated: row.get("last_updated"),
            };

            results.push((
                memory.room.clone(),
                memory.id.to_string(),
                memory,
                confidence,
            ));
        }

        Ok(results)
    }

    /// Get graph statistics and insights.
    pub(crate) async fn get_graph_stats(&mut self) -> Result<String, String> {
        let stats = sqlx::query(r#"
            SELECT 
                (SELECT COUNT(*) FROM memories) as total_memories,
                (SELECT COUNT(*) FROM rooms) as total_rooms,
                (SELECT COUNT(*) FROM memory_relationships) as total_relationships,
                (SELECT COUNT(*) FROM concepts) as total_concepts,
                (SELECT COUNT(*) FROM memory_concepts) as total_mentions
        "#)
        .fetch_one(&self.pool)
        .await
        .map_err(|e| format!("Failed to get stats: {}", e))?;

        let total_memories: i64 = stats.get("total_memories");
        let total_rooms: i64 = stats.get("total_rooms");
        let total_relationships: i64 = stats.get("total_relationships");
        let total_concepts: i64 = stats.get("total_concepts");
        let total_mentions: i64 = stats.get("total_mentions");

        Ok(format!(
            "Graph Statistics:\n\
            - Total Memories: {}\n\
            - Total Rooms: {}\n\
            - Total Relationships: {}\n\
            - Total Concepts: {}\n\
            - Total Concept Mentions: {}\n\
            - Average Relationships per Memory: {:.2}\n\
            - Average Concepts per Memory: {:.2}",
            total_memories,
            total_rooms,
            total_relationships,
            total_concepts,
            total_mentions,
            if total_memories > 0 {
                total_relationships as f64 / total_memories as f64
            } else {
                0.0
            },
            if total_memories > 0 {
                total_mentions as f64 / total_memories as f64
            } else {
                0.0
            }
        ))
    }
}

#[async_trait::async_trait]
impl Tool for MemoryPalace {
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
                            "description": "Tags for categorizing this memory (can be empty array).",
                            "default": []
                        }
                    },
                    "required": ["room", "content", "tags"]
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

            Method::builder("MemoryPalace::relate")
                .description("Create a relationship between two memories for graph traversal.")
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
                            "description": "Type of relationship (e.g., 'related', 'contradicts', 'builds_on', 'example_of').",
                            "default": "related"
                        },
                        "strength": {
                            "type": "number",
                            "description": "Strength of the relationship (0.0 to 1.0).",
                            "default": 1.0
                        }
                    },
                    "required": ["memory_id1", "memory_id2", "relationship_type", "strength"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::find_related")
                .description("Find memories related to a given memory through graph traversal.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "number",
                            "description": "ID of the memory to find relationships for."
                        },
                        "max_depth": {
                            "type": "number",
                            "description": "Maximum depth for graph traversal.",
                            "default": 2
                        },
                        "min_strength": {
                            "type": "number",
                            "description": "Minimum relationship strength to include.",
                            "default": 0.1
                        }
                    },
                    "required": ["memory_id", "max_depth", "min_strength"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::extract_concepts")
                .description("Extract and link concepts from a memory for semantic organization.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "number",
                            "description": "ID of the memory to extract concepts from."
                        },
                        "concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of concepts to extract and link to this memory."
                        }
                    },
                    "required": ["memory_id", "concepts"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::find_by_concept")
                .description("Find memories that mention a specific concept.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Name of the concept to search for."
                        }
                    },
                    "required": ["concept"]
                }))
                .build()
                .unwrap(),

            Method::builder("MemoryPalace::graph_stats")
                .description("Get statistics about the memory graph structure.")
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

                let tags: Vec<&str> = input
                    .get("tags")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                    .unwrap_or_default();

                match self.store_memory(room, content, tags).await {
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

                match self.connect_rooms(room1, room2).await {
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

            "relate" => {
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

                let memory_id1 =
                    match input.get("memory_id1").and_then(|v| v.as_i64()) {
                        Some(id) => id,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content:
                                    "Missing required 'memory_id1' parameter"
                                        .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                let memory_id2 =
                    match input.get("memory_id2").and_then(|v| v.as_i64()) {
                        Some(id) => id,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content:
                                    "Missing required 'memory_id2' parameter"
                                        .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                let relationship_type = input
                    .get("relationship_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("related");

                let strength = input
                    .get("strength")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);

                match self
                    .relate_memories(
                        memory_id1,
                        memory_id2,
                        relationship_type,
                        strength,
                    )
                    .await
                {
                    Ok(message) => super::Result {
                        tool_use_id: call.id,
                        content: message.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!(
                            "Failed to create relationship: {}",
                            err
                        )
                        .into(),
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id =
                    match input.get("memory_id").and_then(|v| v.as_i64()) {
                        Some(id) => id,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content:
                                    "Missing required 'memory_id' parameter"
                                        .into(),
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

                match self
                    .find_related_memories(memory_id, max_depth, min_strength)
                    .await
                {
                    Ok(results) => {
                        if results.is_empty() {
                            super::Result {
                                tool_use_id: call.id,
                                content: format!(
                                    "No related memories found for ID '{}'",
                                    memory_id
                                )
                                .into(),
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
                            for (
                                room_name,
                                related_memory_id,
                                memory,
                                rel_type,
                                strength,
                            ) in results
                            {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nRelationship: {} (strength: {})\n\n",
                                    room_name,
                                    related_memory_id,
                                    memory.content,
                                    rel_type,
                                    strength
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
                        content: format!(
                            "Failed to find related memories: {}",
                            err
                        )
                        .into(),
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let memory_id =
                    match input.get("memory_id").and_then(|v| v.as_i64()) {
                        Some(id) => id,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content:
                                    "Missing required 'memory_id' parameter"
                                        .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                let concepts: Vec<&str> = input
                    .get("concepts")
                    .and_then(|v| v.as_array())
                    .map(|arr| arr.iter().filter_map(|v| v.as_str()).collect())
                    .unwrap_or_default();

                match self.extract_concepts(memory_id, concepts).await {
                    Ok(message) => super::Result {
                        tool_use_id: call.id,
                        content: message.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to extract concepts: {}", err)
                            .into(),
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Input must be an object".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                let concept =
                    match input.get("concept").and_then(|v| v.as_str()) {
                        Some(concept) => concept,
                        None => {
                            return super::Result {
                                tool_use_id: call.id,
                                content: "Missing required 'concept' parameter"
                                    .into(),
                                is_error: true,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            };
                        }
                    };

                match self.find_memories_by_concept(concept).await {
                    Ok(results) => {
                        if results.is_empty() {
                            super::Result {
                                tool_use_id: call.id,
                                content: format!(
                                    "No memories found for concept '{}'",
                                    concept
                                )
                                .into(),
                                is_error: false,
                                #[cfg(feature = "prompt-caching")]
                                cache_control: None,
                            }
                        } else {
                            let mut response = format!(
                                "Found {} memories for concept '{}':\n\n",
                                results.len(),
                                concept
                            );
                            for (room_name, memory_id, memory, confidence) in
                                results
                            {
                                response.push_str(&format!(
                                    "Room: {}\nID: {}\nContent: {}\nConfidence: {}\n\n",
                                    room_name,
                                    memory_id,
                                    memory.content,
                                    confidence
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
                        content: format!(
                            "Failed to find memories by concept: {}",
                            err
                        )
                        .into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }

            "graph_stats" => match self.get_graph_stats().await {
                Ok(stats) => super::Result {
                    tool_use_id: call.id,
                    content: stats.into(),
                    is_error: false,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                },
                Err(err) => super::Result {
                    tool_use_id: call.id,
                    content: format!("Failed to get graph stats: {}", err)
                        .into(),
                    is_error: true,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                },
            },

            _ => super::Result {
                tool_use_id: call.id,
                content: format!(
                    "Method '{}' not implemented yet",
                    method_name
                )
                .into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            },
        }
    }

    /// Save the current state of the palace to JSON.
    async fn save_json(&mut self) -> serde_json::Value {
        if let Err(_) = self.ensure_initialized().await {
            return json!({"error": "Failed to initialize database"});
        }

        // Export all data as JSON
        let memories_result = sqlx::query("SELECT * FROM memories ORDER BY id")
            .fetch_all(&self.pool)
            .await;

        let rooms_result = sqlx::query("SELECT * FROM rooms ORDER BY id")
            .fetch_all(&self.pool)
            .await;

        let connections_result =
            sqlx::query("SELECT * FROM room_connections ORDER BY id")
                .fetch_all(&self.pool)
                .await;

        match (memories_result, rooms_result, connections_result) {
            (Ok(memory_rows), Ok(room_rows), Ok(connection_rows)) => {
                let memories: Vec<serde_json::Value> = memory_rows
                    .into_iter()
                    .map(|row| {
                        let tags_json: serde_json::Value = row.get("tags");
                        json!({
                            "id": row.get::<i64, _>("id"),
                            "content": row.get::<String, _>("content"),
                            "room": row.get::<String, _>("room"),
                            "tags": tags_json,
                            "created_at": row.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
                            "last_accessed": row.get::<chrono::DateTime<chrono::Utc>, _>("last_accessed").to_rfc3339(),
                        })
                    })
                    .collect();

                let rooms: Vec<serde_json::Value> = room_rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "id": row.get::<i64, _>("id"),
                            "name": row.get::<String, _>("name"),
                            "description": row.get::<String, _>("description"),
                            "created_at": row.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
                        })
                    })
                    .collect();

                let connections: Vec<serde_json::Value> = connection_rows
                    .into_iter()
                    .map(|row| {
                        json!({
                            "id": row.get::<i64, _>("id"),
                            "from_room": row.get::<String, _>("from_room"),
                            "to_room": row.get::<String, _>("to_room"),
                            "description": row.get::<Option<String>, _>("description"),
                            "strength": row.get::<i32, _>("strength"),
                            "created_at": row.get::<chrono::DateTime<chrono::Utc>, _>("created_at").to_rfc3339(),
                        })
                    })
                    .collect();

                json!({
                    "memories": memories,
                    "rooms": rooms,
                    "connections": connections,
                })
            }
            _ => json!({"error": "Failed to export data"}),
        }
    }

    /// Load the palace state from JSON.
    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> Result<(), String> {
        self.ensure_initialized().await?;

        // Clear existing data
        sqlx::query("TRUNCATE memories, rooms, room_connections RESTART IDENTITY CASCADE")
            .execute(&self.pool)
            .await
            .map_err(|e| format!("Failed to clear database: {}", e))?;

        // Import data
        if let Some(rooms) = json.get("rooms").and_then(|v| v.as_array()) {
            for room in rooms {
                if let (Some(name), Some(description)) = (
                    room.get("name").and_then(|v| v.as_str()),
                    room.get("description").and_then(|v| v.as_str()),
                ) {
                    sqlx::query(
                        "INSERT INTO rooms (name, description) VALUES ($1, $2)",
                    )
                    .bind(name)
                    .bind(description)
                    .execute(&self.pool)
                    .await
                    .map_err(|e| format!("Failed to import room: {}", e))?;
                }
            }
        }

        if let Some(memories) = json.get("memories").and_then(|v| v.as_array())
        {
            for memory in memories {
                if let (Some(content), Some(room), Some(tags)) = (
                    memory.get("content").and_then(|v| v.as_str()),
                    memory.get("room").and_then(|v| v.as_str()),
                    memory.get("tags"),
                ) {
                    sqlx::query("INSERT INTO memories (content, room, tags) VALUES ($1, $2, $3)")
                        .bind(content)
                        .bind(room)
                        .bind(tags)
                        .execute(&self.pool)
                        .await
                        .map_err(|e| format!("Failed to import memory: {}", e))?;
                }
            }
        }

        if let Some(connections) =
            json.get("connections").and_then(|v| v.as_array())
        {
            for connection in connections {
                if let (Some(from_room), Some(to_room)) = (
                    connection.get("from_room").and_then(|v| v.as_str()),
                    connection.get("to_room").and_then(|v| v.as_str()),
                ) {
                    let description = connection.get("description");
                    let strength = connection
                        .get("strength")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(1) as i32;

                    sqlx::query("INSERT INTO room_connections (from_room, to_room, description, strength) VALUES ($1, $2, $3, $4)")
                        .bind(from_room)
                        .bind(to_room)
                        .bind(description)
                        .bind(strength)
                        .execute(&self.pool)
                        .await
                        .map_err(|e| format!("Failed to import connection: {}", e))?;
                }
            }
        }

        Ok(())
    }

    /// Apply the memory palace context to a prompt.
    fn apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let palace_context = "<memory_palace>\nMemory Palace is available for storing and retrieving knowledge.\n</memory_palace>";

        if prompt.system.is_none() {
            let full_text = MEMORY_PALACE_INSTRUCTIONS.to_string();
            prompt.system = Some(full_text.into());
        } else {
            let system_content = prompt.system.as_mut().unwrap();

            match system_content {
                crate::prompt::message::Content::SinglePart(text) => {
                    if text.contains("</memory_palace>") {
                        let new_text = text.replace(
                            "</memory_palace>",
                            &format!("\n{}", palace_context),
                        );
                        *text = new_text.into();
                    } else {
                        let existing_text = text.clone();
                        *system_content = vec![
                            Block::Text {
                                text: existing_text,
                                cache_control: None,
                            },
                            Block::Text {
                                text: palace_context.into(),
                                cache_control: None,
                            },
                        ]
                        .into();
                    }
                }
                crate::prompt::message::Content::MultiPart(blocks) => {
                    let mut found = false;
                    for block in blocks.iter_mut() {
                        if let Block::Text { text, .. } = block {
                            if text.contains("</memory_palace>") {
                                let new_text = text.replace(
                                    "</memory_palace>",
                                    &format!("\n{}", palace_context),
                                );
                                *text = new_text.into();
                                found = true;
                                break;
                            }
                        }
                    }

                    if !found {
                        blocks.push(Block::Text {
                            text: palace_context.into(),
                            cache_control: None,
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    // use super::*;

    // Tests would need to be updated to use a test PostgreSQL instance
    // or use sqlx::test with migrations
}
