//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.

use super::{Method, Tool, Use};
use crate::{Prompt, prompt::message::Block};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::{FromRow, PgPool, Row};

const MEMORY_PALACE_INSTRUCTIONS: &str = r#"<memory_palace_instructions>You have access to a Memory Palace - a spatial knowledge organization system that helps you store, organize, and retrieve knowledge across conversations.

## Key Concepts:
- **Rooms**: Organize memories by topic (e.g., "science", "cooking", "personal_facts")
- **Memories**: Individual pieces of knowledge with content, tags, and timestamps
- **Relationships**: Connect related memories for graph traversal and discovery
- **Concepts**: Extract and link semantic concepts for advanced querying

## Best Practices:
- Use descriptive room names that group related knowledge
- Add relevant tags to make memories searchable
- Create relationships between related memories to build knowledge graphs
- Extract key concepts to enable semantic search

Start with `MemoryPalace::store` to save important information, then use `MemoryPalace::search` to find it later.</memory_palace_instructions>"#;

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
    /// The schema name to use for all operations.
    schema_name: String,
}

impl MemoryPalace {
    const NAME: &'static str = "MemoryPalace";

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool.
    /// Uses the default 'public' schema.
    pub async fn from_pool(pool: PgPool) -> Result<Self, String> {
        Self::from_pool_with_schema(pool, "public".to_string()).await
    }

    /// Create a new [`MemoryPalace`] from an existing PostgreSQL pool with a specific schema.
    /// Initializes the database schema if it hasn't been done yet.
    pub async fn from_pool_with_schema(
        pool: PgPool,
        schema_name: String,
    ) -> Result<Self, String> {
        let mut new = Self { pool, schema_name };

        // Ensure the database is initialized - this is our class invariant
        new.ensure_initialized().await?;

        Ok(new)
    }

    /// Initialize the database schema with proper indexes and triggers.
    async fn ensure_initialized(&mut self) -> Result<(), String> {
        // Execute schema creation statements in a transaction to ensure atomicity
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        // Set search path for this transaction
        sqlx::query(&format!("SET search_path TO {}", self.schema_name))
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("Failed to set search path: {}", e))?;

        // Create tables first
        let table_statements = [
            r#"CREATE TABLE IF NOT EXISTS rooms (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#,
            r#"CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                room VARCHAR(255) NOT NULL,
                tags JSONB NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#,
            r#"CREATE TABLE IF NOT EXISTS room_connections (
                id BIGSERIAL PRIMARY KEY,
                from_room VARCHAR(255) NOT NULL,
                to_room VARCHAR(255) NOT NULL,
                description TEXT,
                strength INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_room, to_room)
            )"#,
            r#"CREATE TABLE IF NOT EXISTS memory_relationships (
                id BIGSERIAL PRIMARY KEY,
                from_memory_id BIGINT NOT NULL,
                to_memory_id BIGINT NOT NULL,
                relationship_type VARCHAR(100) NOT NULL DEFAULT 'related',
                strength FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_memory_id, to_memory_id)
            )"#,
            r#"CREATE TABLE IF NOT EXISTS concepts (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#,
            r#"CREATE TABLE IF NOT EXISTS memory_concepts (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL,
                concept_id BIGINT NOT NULL,
                confidence FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(memory_id, concept_id)
            )"#,
        ];

        for statement in table_statements {
            sqlx::query(statement)
                .execute(&mut *tx)
                .await
                .map_err(|e| format!("Failed to create table: {}", e))?;
        }

        // Add foreign key constraints after tables are created
        let constraint_statements = [
            r#"ALTER TABLE memories DROP CONSTRAINT IF EXISTS memories_room_fkey"#,
            r#"ALTER TABLE memories ADD CONSTRAINT memories_room_fkey 
               FOREIGN KEY (room) REFERENCES rooms(name) ON DELETE CASCADE"#,
            r#"ALTER TABLE room_connections DROP CONSTRAINT IF EXISTS room_connections_from_room_fkey"#,
            r#"ALTER TABLE room_connections ADD CONSTRAINT room_connections_from_room_fkey 
               FOREIGN KEY (from_room) REFERENCES rooms(name) ON DELETE CASCADE"#,
            r#"ALTER TABLE room_connections DROP CONSTRAINT IF EXISTS room_connections_to_room_fkey"#,
            r#"ALTER TABLE room_connections ADD CONSTRAINT room_connections_to_room_fkey 
               FOREIGN KEY (to_room) REFERENCES rooms(name) ON DELETE CASCADE"#,
            r#"ALTER TABLE memory_relationships DROP CONSTRAINT IF EXISTS memory_relationships_from_memory_id_fkey"#,
            r#"ALTER TABLE memory_relationships ADD CONSTRAINT memory_relationships_from_memory_id_fkey 
               FOREIGN KEY (from_memory_id) REFERENCES memories(id) ON DELETE CASCADE"#,
            r#"ALTER TABLE memory_relationships DROP CONSTRAINT IF EXISTS memory_relationships_to_memory_id_fkey"#,
            r#"ALTER TABLE memory_relationships ADD CONSTRAINT memory_relationships_to_memory_id_fkey 
               FOREIGN KEY (to_memory_id) REFERENCES memories(id) ON DELETE CASCADE"#,
            r#"ALTER TABLE memory_concepts DROP CONSTRAINT IF EXISTS memory_concepts_memory_id_fkey"#,
            r#"ALTER TABLE memory_concepts ADD CONSTRAINT memory_concepts_memory_id_fkey 
               FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE"#,
            r#"ALTER TABLE memory_concepts DROP CONSTRAINT IF EXISTS memory_concepts_concept_id_fkey"#,
            r#"ALTER TABLE memory_concepts ADD CONSTRAINT memory_concepts_concept_id_fkey 
               FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE"#,
        ];

        for statement in constraint_statements {
            sqlx::query(statement)
                .execute(&mut *tx)
                .await
                .map_err(|e| format!("Failed to add constraint: {}", e))?;
        }

        // Create functions and triggers
        let function_statements = [
            r#"CREATE OR REPLACE FUNCTION update_last_updated_column()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.last_updated = NOW();
                RETURN NEW;
            END;
            $$ language 'plpgsql'"#,
            r#"DROP TRIGGER IF EXISTS update_memories_last_updated ON memories"#,
            r#"CREATE TRIGGER update_memories_last_updated
                BEFORE UPDATE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_last_updated_column()"#,
        ];

        for statement in function_statements {
            sqlx::query(statement)
                .execute(&mut *tx)
                .await
                .map_err(|e| {
                    format!("Failed to create function/trigger: {}", e)
                })?;
        }

        // Create indexes
        let index_statements = [
            r#"CREATE INDEX IF NOT EXISTS idx_memories_room ON memories(room)"#,
            r#"CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(to_tsvector('english', content))"#,
            r#"CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING gin(tags)"#,
            r#"CREATE INDEX IF NOT EXISTS idx_room_connections_from ON room_connections(from_room)"#,
            r#"CREATE INDEX IF NOT EXISTS idx_memory_relationships_from ON memory_relationships(from_memory_id)"#,
            r#"CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts(memory_id)"#,
            r#"CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts(concept_id)"#,
        ];

        for statement in index_statements {
            sqlx::query(statement)
                .execute(&mut *tx)
                .await
                .map_err(|e| format!("Failed to create index: {}", e))?;
        }

        // Commit the transaction
        tx.commit()
            .await
            .map_err(|e| format!("Failed to commit schema creation: {}", e))?;

        Ok(())
    }

    /// Execute a query with proper schema context
    async fn execute_with_schema<'q, F, R>(
        &self,
        operation: F,
    ) -> Result<R, String>
    where
        F: for<'c> FnOnce(
            &'c mut sqlx::Transaction<'_, sqlx::Postgres>,
        ) -> std::pin::Pin<
            Box<
                dyn std::future::Future<Output = Result<R, sqlx::Error>>
                    + Send
                    + 'c,
            >,
        >,
    {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| format!("Failed to begin transaction: {}", e))?;

        // Set search path for this transaction
        sqlx::query(&format!("SET search_path TO {}", self.schema_name))
            .execute(&mut *tx)
            .await
            .map_err(|e| format!("Failed to set search path: {}", e))?;

        let result = operation(&mut tx)
            .await
            .map_err(|e| format!("Database operation failed: {}", e))?;

        tx.commit()
            .await
            .map_err(|e| format!("Failed to commit transaction: {}", e))?;

        Ok(result)
    }

    /// Store a memory in a specific room.
    pub(crate) async fn store_memory(
        &mut self,
        room_name: impl Into<String>,
        content: impl Into<String>,
        tags: impl IntoIterator<Item = &str>,
    ) -> Result<i64, String> {
        let tags: Vec<String> =
            tags.into_iter().map(|s| s.to_string()).collect();
        let tags_json = serde_json::to_value(&tags)
            .map_err(|e| format!("Failed to serialize tags: {}", e))?;
        let room_name = room_name.into();
        let content = content.into();

        self.execute_with_schema(|tx| {
            Box::pin(async move {
                // Ensure room exists
                sqlx::query(
                    r#"
                    INSERT INTO rooms (name, description) 
                    VALUES ($1, $2) 
                    ON CONFLICT (name) DO NOTHING
                "#,
                )
                .bind(&room_name)
                .bind(format!("Room for {}", room_name))
                .execute(&mut **tx)
                .await?;

                let row = sqlx::query(
                    r#"
                    INSERT INTO memories (content, room, tags) 
                    VALUES ($1, $2, $3) 
                    RETURNING id
                "#,
                )
                .bind(&content)
                .bind(&room_name)
                .bind(&tags_json)
                .fetch_one(&mut **tx)
                .await?;

                Ok(row.get::<i64, _>("id"))
            })
        })
        .await
    }

    /// Search for memories using full-text search and filters.
    pub(crate) async fn search(
        &mut self,
        query: &str,
    ) -> Result<Vec<(String, String, Memory)>, String> {
        #[cfg(feature = "log")]
        log::debug!("Memory Palace searching for: '{}'", query);

        let query_pattern = format!("%{}%", query);

        self.execute_with_schema(|tx| {
            Box::pin(async move {
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
                .bind(&query_pattern)
                .fetch_all(&mut **tx)
                .await?;

                let results: Vec<_> = memories
                    .into_iter()
                    .map(|memory| {
                        (memory.room.clone(), memory.id.to_string(), memory)
                    })
                    .collect();

                Ok(results)
            })
        })
        .await
    }

    /// Connect two rooms in the palace.
    async fn connect_rooms(
        &mut self,
        room1: impl Into<String>,
        room2: impl Into<String>,
    ) -> Result<(), String> {
        let room1 = room1.into();
        let room2 = room2.into();
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                sqlx::query(
                    r#"
                    INSERT INTO room_connections (from_room, to_room, strength) 
                    VALUES ($1, $2, 1), ($2, $1, 1)
                    ON CONFLICT (from_room, to_room) DO NOTHING
                "#,
                )
                .bind(room1)
                .bind(room2)
                .execute(&mut **tx)
                .await?;

                Ok(())
            })
        })
        .await
    }

    /// List all rooms with their memory counts and connections.
    async fn list_rooms(
        &mut self,
    ) -> Result<Vec<(String, String, usize, Vec<String>)>, String> {
        self.execute_with_schema(|tx| {
            Box::pin(async move {
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
                .fetch_all(&mut **tx)
                .await?;

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
                    .fetch_all(&mut **tx)
                    .await?;

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
            })
        }).await
    }

    /// Create a relationship between two memories.
    pub(crate) async fn relate_memories(
        &mut self,
        memory_id1: i64,
        memory_id2: i64,
        relationship_type: impl Into<String>,
        strength: f64,
    ) -> Result<String, String> {
        let relationship_type = relationship_type.into();
        let okmsg = format!(
            "Created {} relationship between {} and {} with strength {}",
            relationship_type, memory_id1, memory_id2, strength
        );
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                sqlx::query(r#"
                    INSERT INTO memory_relationships (from_memory_id, to_memory_id, relationship_type, strength)
                    VALUES ($1, $2, $3, $4)
                    ON CONFLICT (from_memory_id, to_memory_id) 
                    DO UPDATE SET relationship_type = $3, strength = $4
                "#)
                .bind(memory_id1)
                .bind(memory_id2)
                .bind(&relationship_type)
                .bind(strength)
                .execute(&mut **tx)
                .await?;

                Ok(())
            })
        }).await?;

        Ok(okmsg)
    }

    /// Find memories related to a given memory through graph traversal.
    pub(crate) async fn find_related_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, Memory, String, f64)>, String> {
        self.execute_with_schema(|tx| {
            Box::pin(async move {
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
                .fetch_all(&mut **tx)
                .await?;

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
            })
        }).await
    }

    /// Extract and create concept nodes from memory content.
    pub(crate) async fn extract_concepts(
        &mut self,
        memory_id: i64,
        concepts: impl IntoIterator<Item = &str>,
    ) -> Result<String, String> {
        let concept_names: Vec<String> =
            concepts.into_iter().map(|s| s.to_string()).collect();

        let created_concepts = self.execute_with_schema(|tx| {
            Box::pin(async move {
                let mut created = Vec::new();

                for concept_name in &concept_names {
                    // Create or get concept
                    let concept_row = sqlx::query(
                        r#"
                        INSERT INTO concepts (name) VALUES ($1)
                        ON CONFLICT (name) DO NOTHING
                        RETURNING id
                    "#,
                    )
                    .bind(concept_name)
                    .fetch_optional(&mut **tx)
                    .await?;

                    let concept_id: i64 = if let Some(row) = concept_row {
                        row.get("id")
                    } else {
                        // Concept already exists, get its ID
                        sqlx::query("SELECT id FROM concepts WHERE name = $1")
                            .bind(concept_name)
                            .fetch_one(&mut **tx)
                            .await?
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
                    .execute(&mut **tx)
                    .await?;

                    created.push(concept_name.clone());
                }

                Ok(created)
            })
        }).await?;

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
        let concept_name = concept_name.to_string();

        self.execute_with_schema(|tx| {
            Box::pin(async move {
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
                .bind(&concept_name)
                .fetch_all(&mut **tx)
                .await?;

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
            })
        }).await
    }

    /// Get graph statistics and insights.
    pub(crate) async fn get_graph_stats(&mut self) -> Result<String, String> {
        let stats = self.execute_with_schema(|tx| {
            Box::pin(async move {
                sqlx::query(r#"
                    SELECT 
                        (SELECT COUNT(*) FROM memories) as total_memories,
                        (SELECT COUNT(*) FROM rooms) as total_rooms,
                        (SELECT COUNT(*) FROM memory_relationships) as total_relationships,
                        (SELECT COUNT(*) FROM concepts) as total_concepts,
                        (SELECT COUNT(*) FROM memory_concepts) as total_mentions
                "#)
                .fetch_one(&mut **tx)
                .await
            })
        }).await?;

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

    /// Get a summary of recent and important memories for prompt context.
    async fn get_context_summary(&mut self) -> Result<String, String> {
        let (recent_memories, top_relationships) = self.execute_with_schema(|tx| {
            Box::pin(async move {
                // Get recent memories
                let recent_memories = sqlx::query(
                    r#"
                    SELECT content, room, tags, created_at
                    FROM memories 
                    ORDER BY created_at DESC 
                    LIMIT 5
                "#,
                )
                .fetch_all(&mut **tx)
                .await?;

                // Get top relationships by strength
                let top_relationships = sqlx::query(
                    r#"
                    SELECT m1.content as from_content, m2.content as to_content, 
                           mr.relationship_type, mr.strength
                    FROM memory_relationships mr
                    JOIN memories m1 ON mr.from_memory_id = m1.id
                    JOIN memories m2 ON mr.to_memory_id = m2.id
                    ORDER BY mr.strength DESC
                    LIMIT 3
                "#,
                )
                .fetch_all(&mut **tx)
                .await?;

                Ok((recent_memories, top_relationships))
            })
        }).await?;

        let mut context = String::new();

        if !recent_memories.is_empty() {
            context.push_str("Recent memories:\n");
            for row in recent_memories {
                let content: String = row.get("content");
                let room: String = row.get("room");
                let tags_json: serde_json::Value = row.get("tags");
                let tags: Vec<String> =
                    serde_json::from_value(tags_json).unwrap_or_default();

                context.push_str(&format!(
                    "- [{}] {}",
                    room,
                    if content.len() > 50 {
                        format!("{}...", &content[..50])
                    } else {
                        content
                    }
                ));
                if !tags.is_empty() {
                    context.push_str(&format!(" ({})", tags.join(", ")));
                }
                context.push('\n');
            }
        }

        if !top_relationships.is_empty() {
            context.push_str("\nKey relationships:\n");
            for row in top_relationships {
                let from_content: String = row.get("from_content");
                let to_content: String = row.get("to_content");
                let rel_type: String = row.get("relationship_type");
                let strength: f64 = row.get("strength");

                context.push_str(&format!(
                    "- {} --[{}]({:.1})--> {}\n",
                    if from_content.len() > 30 {
                        format!("{}...", &from_content[..30])
                    } else {
                        from_content
                    },
                    rel_type,
                    strength,
                    if to_content.len() > 30 {
                        format!("{}...", &to_content[..30])
                    } else {
                        to_content
                    }
                ));
            }
        }

        if context.is_empty() {
            context = "Memory palace is empty - ready to store new knowledge!"
                .to_string();
        }

        Ok(context)
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

            Method::builder("MemoryPalace::summary")
                .description("Get a summary of recent memories and key relationships for context.")
                .schema(json!({
                    "type": "object",
                    "properties": {},
                    "required": []
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
                    Some(room) => room.to_string(),
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
                        Some(content) => content.to_string(),
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

                match self.store_memory(room.clone(), content, tags).await {
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
            "summary" => {
                match self.get_context_summary().await {
                    Ok(summary) => super::Result {
                        tool_use_id: call.id,
                        content: summary.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Failed to get summary: {}", err)
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
                                    room_name, memory_id, memory.content, memory.tags.join(", ")
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
                    Some(room) => room.to_string(),
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
                    Some(room) => room.to_string(),
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

                match self.connect_rooms(room1.clone(), room2.clone()).await {
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
                        relationship_type.to_string(),
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
                                    room_name, related_memory_id, memory.content, rel_type, strength
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
                                    room_name, memory_id, memory.content, confidence
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
                    "Unknown method '{}'. Available methods: store, search, summary, connect, list_rooms, relate, find_related, extract_concepts, find_by_concept, graph_stats",
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
        // Export all data as JSON using execute_with_schema
        let export_result = self
            .execute_with_schema(|tx| {
                Box::pin(async move {
                    let memories_result =
                        sqlx::query("SELECT * FROM memories ORDER BY id")
                            .fetch_all(&mut **tx)
                            .await;

                    let rooms_result =
                        sqlx::query("SELECT * FROM rooms ORDER BY id")
                            .fetch_all(&mut **tx)
                            .await;

                    let connections_result = sqlx::query(
                        "SELECT * FROM room_connections ORDER BY id",
                    )
                    .fetch_all(&mut **tx)
                    .await;

                    Ok((memories_result, rooms_result, connections_result))
                })
            })
            .await;

        match export_result {
            Ok((Ok(memory_rows), Ok(room_rows), Ok(connection_rows))) => {
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
                            "last_updated": row.get::<chrono::DateTime<chrono::Utc>, _>("last_updated").to_rfc3339(),
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
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                // Clear existing data
                sqlx::query("TRUNCATE memories, rooms, room_connections RESTART IDENTITY CASCADE")
                    .execute(&mut **tx)
                    .await?;

                // Import data
                if let Some(rooms) = json.get("rooms").and_then(|v| v.as_array()) {
                    for room in rooms {
                        if let (Some(name), Some(description)) = (
                            room.get("name").and_then(|v| v.as_str()),
                            room.get("description").and_then(|v| v.as_str()),
                        ) {
                            sqlx::query("INSERT INTO rooms (name, description) VALUES ($1, $2)")
                                .bind(name)
                                .bind(description)
                                .execute(&mut **tx)
                                .await?;
                        }
                    }
                }

                if let Some(memories) = json.get("memories").and_then(|v| v.as_array()) {
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
                                .execute(&mut **tx)
                                .await?;
                        }
                    }
                }

                if let Some(connections) = json.get("connections").and_then(|v| v.as_array()) {
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
                                .execute(&mut **tx)
                                .await?;
                        }
                    }
                }

                Ok(())
            })
        }).await
    }

    /// Initialize the Memory Palace instructions in the prompt.
    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Add the static instructions once
        if let Some(system) = &mut prompt.system {
            self.update_or_add_instructions(system).await?;
        } else {
            prompt.system = Some(MEMORY_PALACE_INSTRUCTIONS.into());
        }
        Ok(())
    }
}

impl MemoryPalace {
    /// Update or add memory palace instructions to the system content
    async fn update_or_add_instructions<'a>(
        &self,
        system: &mut crate::prompt::message::Content<'a>,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        match system {
            crate::prompt::message::Content::SinglePart(text) => {
                if text.contains("<memory_palace_instructions>") {
                    // Replace existing instructions
                    let parts: Vec<&str> =
                        text.split("<memory_palace_instructions>").collect();
                    if parts.len() >= 2 {
                        let before = parts[0];
                        let after_parts: Vec<&str> = parts[1]
                            .split("</memory_palace_instructions>")
                            .collect();
                        let after = if after_parts.len() > 1 {
                            after_parts[1]
                        } else {
                            ""
                        };

                        let new_text = format!(
                            "{}{}{}",
                            before, MEMORY_PALACE_INSTRUCTIONS, after
                        );
                        *text = new_text.into();
                    }
                } else {
                    // Add instructions to existing content
                    let existing_text = text.clone();
                    *system = vec![
                        Block::Text {
                            text: existing_text,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        },
                        Block::Text {
                            text: MEMORY_PALACE_INSTRUCTIONS.into(),
                            #[cfg(feature = "prompt-caching")]
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
                        if text.contains("<memory_palace_instructions>") {
                            let parts: Vec<&str> = text
                                .split("<memory_palace_instructions>")
                                .collect();
                            if parts.len() >= 2 {
                                let before = parts[0];
                                let after_parts: Vec<&str> = parts[1]
                                    .split("</memory_palace_instructions>")
                                    .collect();
                                let after = if after_parts.len() > 1 {
                                    after_parts[1]
                                } else {
                                    ""
                                };

                                let new_text = format!(
                                    "{}{}{}",
                                    before, MEMORY_PALACE_INSTRUCTIONS, after
                                );
                                *text = new_text.into();
                                found = true;
                                break;
                            }
                        }
                    }
                }

                if !found {
                    blocks.push(Block::Text {
                        text: MEMORY_PALACE_INSTRUCTIONS.into(),
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::PgPool;

    async fn create_test_palace(test_id: &str) -> MemoryPalace {
        let database_url =
            std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
                "postgresql://postgres@localhost:5432/misanthropic_test"
                    .to_string()
            });

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Create a unique schema for this test to avoid conflicts
        let schema_name = format!("test_{}", test_id.replace("-", "_"));

        // Drop and recreate schema in a transaction
        let mut tx = pool.begin().await.expect("Failed to begin transaction");

        sqlx::query(&format!("DROP SCHEMA IF EXISTS {} CASCADE", schema_name))
            .execute(&mut *tx)
            .await
            .expect("Failed to drop test schema");

        sqlx::query(&format!("CREATE SCHEMA {}", schema_name))
            .execute(&mut *tx)
            .await
            .expect("Failed to create test schema");

        tx.commit().await.expect("Failed to commit schema setup");

        // Create the MemoryPalace and initialize it with the schema
        let mut palace = MemoryPalace {
            pool,
            schema_name: schema_name.clone(),
        };

        palace
            .ensure_initialized()
            .await
            .expect("Failed to initialize test palace");

        // Set search_path for subsequent operations
        sqlx::query(&format!("SET search_path TO {}", schema_name))
            .execute(&palace.pool)
            .await
            .expect("Failed to set search path for operations");

        palace
    }

    #[tokio::test]
    async fn test_store_and_search_memory() {
        let mut palace = create_test_palace("store_and_search_memory").await;

        // Store a memory
        let memory_id = palace
            .store_memory(
                "library",
                "The capital of France is Paris",
                ["geography", "facts"],
            )
            .await
            .expect("Failed to store memory");

        // Search for the memory
        let results = palace
            .search("France")
            .await
            .expect("Failed to search memories");

        assert_eq!(results.len(), 1);
        let (room, id, memory) = &results[0];
        assert_eq!(room, "library");
        assert_eq!(id, &memory_id.to_string());
        assert_eq!(memory.content, "The capital of France is Paris");
        assert!(memory.tags.contains(&"geography".to_string()));
        assert!(memory.tags.contains(&"facts".to_string()));
    }

    #[tokio::test]
    async fn test_search_empty_results() {
        let mut palace = create_test_palace("search_empty_results").await;

        let results = palace
            .search("nonexistent")
            .await
            .expect("Failed to search memories");

        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_by_room() {
        let mut palace = create_test_palace("search_by_room").await;

        palace
            .store_memory("kitchen", "Recipe for pasta", ["cooking"])
            .await
            .expect("Failed to store memory");

        let results = palace
            .search("kitchen")
            .await
            .expect("Failed to search memories");

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "kitchen");
    }

    #[tokio::test]
    async fn test_search_by_tags() {
        let mut palace = create_test_palace("search_by_tags").await;

        palace
            .store_memory(
                "study",
                "Python is a programming language",
                ["programming", "python"],
            )
            .await
            .expect("Failed to store memory");

        let results = palace
            .search("programming")
            .await
            .expect("Failed to search memories");

        assert_eq!(results.len(), 1);
        assert!(results[0].2.tags.contains(&"programming".to_string()));
    }

    #[tokio::test]
    async fn test_list_rooms() {
        let mut palace = create_test_palace("list_rooms").await;

        // Initially no rooms
        let rooms = palace.list_rooms().await.expect("Failed to list rooms");
        assert!(rooms.is_empty());

        // Add some memories to create rooms
        palace
            .store_memory("library", "Book about history", ["history"])
            .await
            .expect("Failed to store memory");

        palace
            .store_memory("kitchen", "Recipe for cookies", ["cooking"])
            .await
            .expect("Failed to store memory");

        palace
            .store_memory("library", "Another book", ["literature"])
            .await
            .expect("Failed to store memory");

        let rooms = palace.list_rooms().await.expect("Failed to list rooms");
        assert_eq!(rooms.len(), 2);

        // Find library room
        let library_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "library")
            .unwrap();
        assert_eq!(library_room.2, 2); // 2 memories in library

        // Find kitchen room
        let kitchen_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "kitchen")
            .unwrap();
        assert_eq!(kitchen_room.2, 1); // 1 memory in kitchen
    }

    #[tokio::test]
    async fn test_connect_rooms() {
        let mut palace = create_test_palace("connect_rooms").await;

        // Create rooms by storing memories
        palace
            .store_memory("library", "A book", [])
            .await
            .expect("Failed to store memory");

        palace
            .store_memory("study", "Study notes", [])
            .await
            .expect("Failed to store memory");

        // Connect the rooms
        palace
            .connect_rooms("library", "study")
            .await
            .expect("Failed to connect rooms");

        // Check connections
        let rooms = palace.list_rooms().await.expect("Failed to list rooms");
        let library_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "library")
            .unwrap();
        assert!(library_room.3.contains(&"study".to_string()));

        let study_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "study")
            .unwrap();
        assert!(study_room.3.contains(&"library".to_string()));
    }

    #[tokio::test]
    async fn test_memory_relationships() {
        let mut palace = create_test_palace("memory_relationships").await;

        // Store two memories
        let memory_id1 = palace
            .store_memory("science", "E = mc²", ["physics", "einstein"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory(
                "science",
                "Theory of relativity",
                ["physics", "einstein"],
            )
            .await
            .expect("Failed to store memory");

        // Create a relationship
        let result = palace
            .relate_memories(memory_id1, memory_id2, "related_to", 0.9)
            .await
            .expect("Failed to create relationship");

        assert!(result.contains("related_to"));
        assert!(result.contains(&memory_id1.to_string()));
        assert!(result.contains(&memory_id2.to_string()));

        // Find related memories
        let related = palace
            .find_related_memories(memory_id1, 2, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 1);
        assert_eq!(related[0].2.id, memory_id2);
        assert_eq!(related[0].3, "related_to");
        assert_eq!(related[0].4, 0.9);
    }

    #[tokio::test]
    async fn test_concepts() {
        let mut palace = create_test_palace("concepts").await;

        // Store a memory
        let memory_id = palace
            .store_memory(
                "science",
                "Photosynthesis converts light to energy",
                ["biology"],
            )
            .await
            .expect("Failed to store memory");

        // Extract concepts
        let result = palace
            .extract_concepts(
                memory_id,
                ["photosynthesis", "energy", "biology"],
            )
            .await
            .expect("Failed to extract concepts");

        assert!(result.contains("3 concepts"));
        assert!(result.contains("photosynthesis"));
        assert!(result.contains("energy"));
        assert!(result.contains("biology"));

        // Find memories by concept
        let memories = palace
            .find_memories_by_concept("photosynthesis")
            .await
            .expect("Failed to find memories by concept");

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].2.id, memory_id);
        assert_eq!(memories[0].3, 1.0); // confidence
    }

    #[tokio::test]
    async fn test_graph_stats() {
        let mut palace = create_test_palace("graph_stats").await;

        // Initially empty
        let stats =
            palace.get_graph_stats().await.expect("Failed to get stats");

        assert!(stats.contains("Total Memories: 0"));
        assert!(stats.contains("Total Rooms: 0"));

        // Add some data
        let memory_id1 = palace
            .store_memory("room1", "Memory 1", ["tag1"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("room2", "Memory 2", ["tag2"])
            .await
            .expect("Failed to store memory");

        palace
            .relate_memories(memory_id1, memory_id2, "related", 1.0)
            .await
            .expect("Failed to relate memories");

        palace
            .extract_concepts(memory_id1, ["concept1"])
            .await
            .expect("Failed to extract concepts");

        let stats =
            palace.get_graph_stats().await.expect("Failed to get stats");

        assert!(stats.contains("Total Memories: 2"));
        assert!(stats.contains("Total Rooms: 2"));
        assert!(stats.contains("Total Relationships: 1"));
        assert!(stats.contains("Total Concepts: 1"));
    }

    #[tokio::test]
    async fn test_tool_interface() {
        let palace = create_test_palace("tool_interface").await;

        // Test name
        assert_eq!(palace.name(), "MemoryPalace");

        // Test methods
        let methods: Vec<_> = palace.methods().collect();
        assert!(!methods.is_empty());

        let method_names: Vec<_> =
            methods.iter().map(|m| m.name.as_ref()).collect();
        assert!(method_names.contains(&"MemoryPalace::store"));
        assert!(method_names.contains(&"MemoryPalace::search"));
        assert!(method_names.contains(&"MemoryPalace::list_rooms"));
    }

    #[tokio::test]
    async fn test_tool_call_store() {
        let mut palace = create_test_palace("tool_call_store").await;

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::store".into(),
            input: json!({
                "room": "test_room",
                "content": "test content",
                "tags": ["test_tag"]
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Memory stored"));
        assert!(result.content.to_string().contains("test_room"));
    }

    #[tokio::test]
    async fn test_tool_call_search() {
        let mut palace = create_test_palace("tool_call_search").await;

        // First store something to search for
        palace
            .store_memory(
                "library",
                "A fascinating book about AI",
                ["technology", "AI"],
            )
            .await
            .expect("Failed to store memory");

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::search".into(),
            input: json!({
            "query": "AI"
                   }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Found 1 memories"));
        assert!(result.content.to_string().contains("fascinating book"));
    }

    #[tokio::test]
    async fn test_tool_call_invalid_method() {
        let mut palace = create_test_palace("tool_call_invalid_method").await;

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::invalid_method".into(),
            input: json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Unknown method"));
        assert!(result.content.to_string().contains("Available methods"));
    }

    #[tokio::test]
    async fn test_tool_call_missing_parameters() {
        let mut palace =
            create_test_palace("tool_call_missing_parameters").await;

        // Test store without required parameters
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::store".into(),
            input: json!({
                "room": "test_room"
                // missing content and tags
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Missing required"));
    }

    #[tokio::test]
    async fn test_save_load_json() {
        let mut palace1 = create_test_palace("save_load_json").await;

        // Add some data
        palace1
            .store_memory("room1", "content1", ["tag1"])
            .await
            .expect("Failed to store memory");

        palace1
            .store_memory("room2", "content2", ["tag2"])
            .await
            .expect("Failed to store memory");

        // Save state
        let json = palace1.save_json().await;
        assert!(json.is_object());

        // Create new palace and load state (use same schema)
        let mut palace2 = create_test_palace("save_load_json").await;
        palace2.load_json(json).await.expect("Failed to load state");

        // Verify data was loaded
        let results =
            palace2.search("content1").await.expect("Failed to search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2.content, "content1");
    }

    #[tokio::test]
    async fn test_apply_to_prompt() {
        let mut palace = create_test_palace("apply_to_prompt").await;
        let mut prompt = Prompt::default();

        palace.on_init(&mut prompt).await.unwrap();
        palace.on_turn(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert!(system_content.contains("Memory Palace"));
        assert!(system_content.contains("MemoryPalace::store"));
        assert!(system_content.contains("MemoryPalace::search"));
        assert!(system_content.contains("Rooms")); // Changed from "rooms" to "Rooms"
        assert!(system_content.contains("Relationships")); // Changed from "relationships" to "Relationships"
    }

    #[tokio::test]
    async fn test_enhanced_prompt_application() {
        let mut palace =
            create_test_palace("enhanced_prompt_application").await;
        let mut prompt = Prompt::default();

        palace.on_init(&mut prompt).await.unwrap();
        palace.on_turn(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert!(system_content.contains("Memory Palace"));
        assert!(system_content.contains("MemoryPalace::store"));
        assert!(system_content.contains("MemoryPalace::search"));
        assert!(system_content.contains("Rooms")); // Changed from "rooms" to "Rooms"
        assert!(system_content.contains("Relationships")); // Changed from "relationships" to "Relationships"
    }
}
