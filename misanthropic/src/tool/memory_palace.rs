//! [`MemoryPalace`] tool for hierarchical knowledge organization using PostgreSQL.

use crate::{prompt::message::Block, Prompt};

use super::{Method, Tool, Use};
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
- On your first turn with a user call `MemoryPalace::summary` to get a context summary of recent and important memories.
- Do not call `MemoryPalace::summary` in the middle of a conversation since any alterations to the palace will already be in context.
- Use descriptive room names that group related knowledge
- Add relevant tags to make memories searchable
- Create relationships between related memories to build knowledge graphs

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

/// Helper struct for room listing with memory count
#[derive(Debug, Clone, FromRow)]
struct RoomWithCount {
    name: String,
    description: String,
    memory_count: i64,
}

/// Helper struct for connection listing
#[derive(Debug, Clone, FromRow)]
struct RoomConnection {
    to_room: String,
}

/// Helper struct for memory relationships
#[derive(Debug, Clone, FromRow)]
struct RelatedMemory {
    id: i64,
    content: String,
    room: String,
    tags: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    last_updated: chrono::DateTime<chrono::Utc>,
    relationship_type: String,
    strength: f64,
}

/// Helper struct for concept-based memory search
#[derive(Debug, Clone, FromRow)]
struct ConceptMemory {
    id: i64,
    content: String,
    room: String,
    tags: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    last_updated: chrono::DateTime<chrono::Utc>,
    confidence: f64,
}

/// Helper struct for graph statistics
#[derive(Debug, Clone, FromRow)]
struct GraphStats {
    total_memories: i64,
    total_rooms: i64,
    total_relationships: i64,
    total_concepts: i64,
    total_mentions: i64,
}

/// Helper struct for recent memories summary
#[derive(Debug, Clone, FromRow)]
struct RecentMemory {
    content: String,
    room: String,
    tags: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
}

/// Helper struct for top relationships summary
#[derive(Debug, Clone, FromRow)]
struct TopRelationship {
    from_content: String,
    to_content: String,
    relationship_type: String,
    strength: f64,
}

/// Helper struct for BFS memory discovery
#[derive(Debug, Clone, FromRow)]
struct BfsMemory {
    id: i64,
    content: String,
    room: String,
    tags: serde_json::Value,
    created_at: chrono::DateTime<chrono::Utc>,
    last_updated: chrono::DateTime<chrono::Utc>,
    distance: i32,
    path_strength: f64,
}

/// Helper struct for blended search results
#[derive(Debug, Clone)]
struct ScoredMemory {
    memory: Memory,
    room: String,
    relevance_score: f64,
    recency_score: f64,
    relationship_score: f64,
    final_score: f64,
}

/// A Memory Palace tool using PostgreSQL for reliable storage.
#[derive(Debug)]
pub struct MemoryPalace {
    /// PostgreSQL connection pool.
    pub(crate) pool: PgPool,
    /// The schema name to use for all operations.
    pub(crate) schema_name: String,
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
    pub(crate) async fn ensure_initialized(&mut self) -> Result<(), String> {
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

    /// Search for memories using blended scoring that combines relevance, recency, and relationships.
    pub(crate) async fn search(
        &mut self,
        query: &str,
    ) -> Result<Vec<(String, String, Memory)>, String> {
        #[cfg(feature = "log")]
        log::debug!("Memory Palace searching for: '{}'", query);

        let query_pattern = format!("%{}%", query);

        self.execute_with_schema(|tx| {
            Box::pin(async move {
                // Get relevance-based results (top 10)
                let relevance_memories: Vec<Memory> = sqlx::query_as(
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
                        END DESC
                    LIMIT 10
                "#,
                )
                .bind(&query_pattern)
                .fetch_all(&mut **tx)
                .await?;

                // Get recency-based results (top 10 most recently updated)
                let recent_memories: Vec<Memory> = sqlx::query_as(
                    r#"
                    SELECT id, content, room, tags, created_at, last_updated
                    FROM memories 
                    WHERE 
                        content ILIKE $1 
                        OR room ILIKE $1 
                        OR tags::text ILIKE $1
                    ORDER BY last_updated DESC
                    LIMIT 10
                "#,
                )
                .bind(&query_pattern)
                .fetch_all(&mut **tx)
                .await?;

                // Get relationship-based results (memories connected to relevant ones)
                let relationship_memories: Vec<Memory> = sqlx::query_as(
                    r#"
                    WITH relevant_memories AS (
                        SELECT id
                        FROM memories 
                        WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
                        LIMIT 5
                    ),
                    related_via_relationships AS (
                        SELECT DISTINCT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                               mr.strength
                        FROM memory_relationships mr
                        JOIN memories m ON mr.to_memory_id = m.id
                        JOIN relevant_memories rm ON mr.from_memory_id = rm.id
                        WHERE mr.strength >= 0.3
                        ORDER BY mr.strength DESC
                        LIMIT 10
                    )
                    SELECT id, content, room, tags, created_at, last_updated
                    FROM related_via_relationships
                "#,
                )
                .bind(&query_pattern)
                .fetch_all(&mut **tx)
                .await?;

                // Combine and score all memories
                let mut scored_memories = std::collections::HashMap::new();
                let now = chrono::Utc::now();

                // Score relevance memories
                for (i, memory) in relevance_memories.into_iter().enumerate() {
                    let relevance_score = (10 - i) as f64 / 10.0; // 1.0 to 0.1
                    let recency_score = calculate_recency_score(&memory.last_updated, &now);
                    
                    scored_memories.insert(memory.id, ScoredMemory {
                        room: memory.room.clone(),
                        memory,
                        relevance_score,
                        recency_score,
                        relationship_score: 0.0,
                        final_score: 0.0, // Will calculate after
                    });
                }

                // Boost recency scores for recent memories
                for (i, memory) in recent_memories.into_iter().enumerate() {
                    let recency_boost = (10 - i) as f64 / 10.0;
                    let recency_score = calculate_recency_score(&memory.last_updated, &now);
                    
                    scored_memories.entry(memory.id)
                        .and_modify(|sm| sm.recency_score = f64::max(sm.recency_score, recency_score + recency_boost * 0.3))
                        .or_insert_with(|| ScoredMemory {
                            room: memory.room.clone(),
                            memory,
                            relevance_score: 0.0,
                            recency_score: recency_score + recency_boost * 0.3,
                            relationship_score: 0.0,
                            final_score: 0.0,
                        });
                }

                // Add relationship scores
                for (i, memory) in relationship_memories.into_iter().enumerate() {
                    let relationship_score = (10 - i) as f64 / 10.0;
                    
                    scored_memories.entry(memory.id)
                        .and_modify(|sm| sm.relationship_score = f64::max(sm.relationship_score, relationship_score))
                        .or_insert_with(|| ScoredMemory {
                            room: memory.room.clone(),
                            recency_score: calculate_recency_score(&memory.last_updated, &now),
                            memory,
                            relevance_score: 0.0,
                            relationship_score,
                            final_score: 0.0,
                        });
                }

                // Calculate final scores with weighted combination
                let mut final_memories: Vec<_> = scored_memories.into_values().map(|mut sm| {
                    // Weighted combination: 50% relevance, 30% recency, 20% relationships
                    sm.final_score = sm.relevance_score * 0.5 + sm.recency_score * 0.3 + sm.relationship_score * 0.2;
                    sm
                }).collect();

                // Sort by final score and take top results
                final_memories.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
                final_memories.truncate(10);

                let results: Vec<_> = final_memories
                    .into_iter()
                    .map(|sm| (sm.room, sm.memory.id.to_string(), sm.memory))
                    .collect();

                Ok(results)
            })
        })
        .await
    }

    /// Find memories using BFS with decay factor for distance.
    pub(crate) async fn find_memories_bfs(
        &mut self,
        start_memory_id: i64,
        max_distance: u32,
        decay_factor: f64,
        min_score: f64,
    ) -> Result<Vec<(String, String, Memory, f64, i32)>, String> {
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                let rows: Vec<BfsMemory> = sqlx::query_as(
                    r#"
                    WITH RECURSIVE memory_bfs(memory_id, distance, path_strength, visited) AS (
                        -- Base case: starting memory
                        SELECT $1::BIGINT, 0, 1.0::FLOAT, ARRAY[$1::BIGINT]
                        
                        UNION
                        
                        -- Recursive case: explore neighbors
                        SELECT 
                            mr.to_memory_id,
                            mb.distance + 1,
                            mb.path_strength * mr.strength * $3::FLOAT, -- Apply decay factor
                            mb.visited || mr.to_memory_id
                        FROM memory_bfs mb
                        JOIN memory_relationships mr ON mb.memory_id = mr.from_memory_id
                        WHERE 
                            mb.distance < $2::INT
                            AND mr.to_memory_id != ALL(mb.visited) -- Avoid cycles
                            AND mb.path_strength * mr.strength * $3::FLOAT >= $4::FLOAT -- Min score threshold
                    )
                    SELECT DISTINCT 
                        m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                        mb.distance, mb.path_strength
                    FROM memory_bfs mb
                    JOIN memories m ON mb.memory_id = m.id
                    WHERE mb.memory_id != $1 -- Exclude starting memory
                    ORDER BY mb.path_strength DESC, mb.distance ASC
                "#,
                )
                .bind(start_memory_id)
                .bind(max_distance as i32)
                .bind(decay_factor)
                .bind(min_score)
                .fetch_all(&mut **tx)
                .await?;

                let mut results = Vec::new();
                for row in rows {
                    let tags: Vec<String> =
                        serde_json::from_value(row.tags).unwrap_or_default();

                    let memory = Memory {
                        id: row.id,
                        content: row.content,
                        room: row.room.clone(),
                        tags,
                        created_at: row.created_at,
                        last_updated: row.last_updated,
                    };

                    results.push((
                        memory.room.clone(),
                        memory.id.to_string(),
                        memory,
                        row.path_strength,
                        row.distance,
                    ));
                }

                Ok(results)
            })
        }).await
    }

    /// Connect two rooms in the palace.
    pub(crate) async fn connect_rooms(
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
    pub(crate) async fn list_rooms(
        &mut self,
    ) -> Result<Vec<(String, String, usize, Vec<String>)>, String> {
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                let rooms: Vec<RoomWithCount> = sqlx::query_as(
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
                for room in rooms {
                    // Get connections for this room
                    let connections: Vec<RoomConnection> = sqlx::query_as(
                        r#"
                        SELECT to_room FROM room_connections WHERE from_room = $1
                    "#,
                    )
                    .bind(&room.name)
                    .fetch_all(&mut **tx)
                    .await?;

                    let connection_names: Vec<String> = connections
                        .into_iter()
                        .map(|conn| conn.to_room)
                        .collect();

                    results.push((
                        room.name,
                        room.description,
                        room.memory_count as usize,
                        connection_names,
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

    /// Find memories related to a given memory through graph traversal with enhanced scoring.
    pub(crate) async fn find_related_memories(
        &mut self,
        memory_id: i64,
        max_depth: u32,
        min_strength: f64,
    ) -> Result<Vec<(String, String, Memory, String, f64)>, String> {
        self.execute_with_schema(|tx| {
            Box::pin(async move {
                let rows: Vec<RelatedMemory> = sqlx::query_as(
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
                    ORDER BY 
                        -- Primary: Relationship strength
                        rm.strength DESC, 
                        -- Secondary: Recency of updates
                        m.last_updated DESC,
                        -- Tertiary: Graph depth (closer relationships first)
                        rm.depth ASC
                "#,
                )
                .bind(memory_id)
                .bind(max_depth as i32)
                .bind(min_strength)
                .fetch_all(&mut **tx)
                .await?;

                let mut results = Vec::new();
                for row in rows {
                    let tags: Vec<String> =
                        serde_json::from_value(row.tags).unwrap_or_default();

                    let memory = Memory {
                        id: row.id,
                        content: row.content,
                        room: row.room.clone(),
                        tags,
                        created_at: row.created_at,
                        last_updated: row.last_updated,
                    };

                    results.push((
                        memory.room.clone(),
                        memory.id.to_string(),
                        memory,
                        row.relationship_type,
                        row.strength,
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

    /// Find memories by concept with enhanced relevance scoring.
    pub(crate) async fn find_memories_by_concept(
        &mut self,
        concept_name: &str,
    ) -> Result<Vec<(String, String, Memory, f64)>, String> {
        let concept_name = concept_name.to_string();

        self.execute_with_schema(|tx| {
            Box::pin(async move {
                let rows: Vec<ConceptMemory> = sqlx::query_as(
                    r#"
                    SELECT 
                        m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                        mc.confidence
                    FROM memory_concepts mc
                    JOIN memories m ON mc.memory_id = m.id
                    JOIN concepts c ON mc.concept_id = c.id
                    WHERE c.name = $1
                    ORDER BY 
                        -- Primary: Concept confidence
                        mc.confidence DESC,
                        -- Secondary: Recency of updates
                        m.last_updated DESC,
                        -- Tertiary: Creation time
                        m.created_at DESC
                "#,
                )
                .bind(&concept_name)
                .fetch_all(&mut **tx)
                .await?;

                let mut results = Vec::new();
                for row in rows {
                    let tags: Vec<String> =
                        serde_json::from_value(row.tags).unwrap_or_default();

                    let memory = Memory {
                        id: row.id,
                        content: row.content,
                        room: row.room.clone(),
                        tags,
                        created_at: row.created_at,
                        last_updated: row.last_updated,
                    };

                    results.push((
                        memory.room.clone(),
                        memory.id.to_string(),
                        memory,
                        row.confidence,
                    ));
                }

                Ok(results)
            })
        }).await
    }

    /// Get graph statistics and insights.
    pub(crate) async fn get_graph_stats(&mut self) -> Result<String, String> {
        let stats: GraphStats = self.execute_with_schema(|tx| {
            Box::pin(async move {
                sqlx::query_as(r#"
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

        Ok(format!(
            "Graph Statistics:\n\
            - Total Memories: {}\n\
            - Total Rooms: {}\n\
            - Total Relationships: {}\n\
            - Total Concepts: {}\n\
            - Total Concept Mentions: {}\n\
            - Average Relationships per Memory: {:.2}\n\
            - Average Concepts per Memory: {:.2}",
            stats.total_memories,
            stats.total_rooms,
            stats.total_relationships,
            stats.total_concepts,
            stats.total_mentions,
            if stats.total_memories > 0 {
                stats.total_relationships as f64 / stats.total_memories as f64
            } else {
                0.0
            },
            if stats.total_memories > 0 {
                stats.total_mentions as f64 / stats.total_memories as f64
            } else {
                0.0
            }
        ))
    }

    /// Get a summary of recent and important memories for prompt context.
    async fn get_context_summary(&mut self) -> Result<String, String> {
        let (recent_memories, top_relationships) = self.execute_with_schema(|tx| {
            Box::pin(async move {
                // Get recent memories based on last_updated (more relevant for agents)
                let recent_memories: Vec<RecentMemory> = sqlx::query_as(
                    r#"
                    SELECT content, room, tags, created_at
                    FROM memories 
                    ORDER BY last_updated DESC, created_at DESC 
                    LIMIT 5
                "#,
                )
                .fetch_all(&mut **tx)
                .await?;

                // Get top relationships by strength, but also consider recency
                let top_relationships: Vec<TopRelationship> = sqlx::query_as(
                    r#"
                    SELECT m1.content as from_content, m2.content as to_content, 
                           mr.relationship_type, mr.strength
                    FROM memory_relationships mr
                    JOIN memories m1 ON mr.from_memory_id = m1.id
                    JOIN memories m2 ON mr.to_memory_id = m2.id
                    ORDER BY 
                        mr.strength DESC,
                        GREATEST(m1.last_updated, m2.last_updated) DESC
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
            for memory in recent_memories {
                let tags: Vec<String> =
                    serde_json::from_value(memory.tags).unwrap_or_default();

                // Format the date to show how recent the memory is
                let now = chrono::Utc::now();
                let duration = now.signed_duration_since(memory.created_at);
                let time_desc = if duration.num_days() > 0 {
                    format!("{} days ago", duration.num_days())
                } else if duration.num_hours() > 0 {
                    format!("{} hours ago", duration.num_hours())
                } else if duration.num_minutes() > 0 {
                    format!("{} minutes ago", duration.num_minutes())
                } else {
                    "just now".to_string()
                };

                context.push_str(&format!(
                    "- [{}] {} ({})",
                    memory.room,
                    if memory.content.len() > 50 {
                        format!("{}...", &memory.content[..50])
                    } else {
                        memory.content
                    },
                    time_desc
                ));
                if !tags.is_empty() {
                    context.push_str(&format!(" [{}]", tags.join(", ")));
                }
                context.push('\n');
            }
        }

        if !top_relationships.is_empty() {
            context.push_str("\nKey relationships:\n");
            for rel in top_relationships {
                context.push_str(&format!(
                    "- {} --[{}]({:.1})--> {}\n",
                    if rel.from_content.len() > 30 {
                        format!("{}...", &rel.from_content[..30])
                    } else {
                        rel.from_content
                    },
                    rel.relationship_type,
                    rel.strength,
                    if rel.to_content.len() > 30 {
                        format!("{}...", &rel.to_content[..30])
                    } else {
                        rel.to_content
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

/// Calculate a recency score based on how recently a memory was updated.
/// Returns a score between 0.0 and 1.0, with 1.0 being most recent.
/// Uses exponential decay with configurable half-life.
fn calculate_recency_score(
    last_updated: &chrono::DateTime<chrono::Utc>,
    now: &chrono::DateTime<chrono::Utc>,
) -> f64 {
    let duration = now.signed_duration_since(*last_updated);
    let hours_ago = duration.num_minutes() as f64 / 60.0;
    
    // Use a 24-hour half-life for recency scoring
    // This means memories from 24 hours ago get score 0.5
    // Memories from 48 hours ago get score 0.25, etc.
    let half_life_hours = 24.0;
    
    // Exponential decay: score = 2^(-hours_ago / half_life)
    let score = 2_f64.powf(-hours_ago / half_life_hours);
    
    // Clamp to reasonable bounds
    score.max(0.001).min(1.0)
}

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
            system.push(MEMORY_PALACE_INSTRUCTIONS);
        }

        // Add memory palace instructions to the system prompt
        prompt.system = Some(MEMORY_PALACE_INSTRUCTIONS.into());
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
                    Some(r) => r,
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

                let content = match input.get("content").and_then(|v| v.as_str()) {
                    Some(c) => c,
                    None => {
                        return super::Result {
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
                    Ok(memory_id) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Memory stored with ID: {} in room '{}'", memory_id, room).into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
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
                    Some(q) => q,
                    None => {
                        return super::Result {
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
                            super::Result {
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
                        content: format!("Failed to search memories: {}", err).into(),
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
                    Some(r) => r,
                    None => {
                        return super::Result {
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
                        return super::Result {
                            tool_use_id: call.id,
                            content: "Missing required 'room2' parameter".into(),
                            is_error: true,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        };
                    }
                };

                match self.connect_rooms(room1, room2).await {
                    Ok(()) => super::Result {
                        tool_use_id: call.id,
                        content: format!("Rooms '{}' and '{}' connected.", room1, room2).into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
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

                        super::Result {
                            tool_use_id: call.id,
                            content: response.into(),
                            is_error: false,
                            #[cfg(feature = "prompt-caching")]
                            cache_control: None,
                        }
                    }
                    Err(err) => super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                    Ok(msg) => super::Result {
                        tool_use_id: call.id,
                        content: msg.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                            super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                    Ok(msg) => super::Result {
                        tool_use_id: call.id,
                        content: msg.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                            super::Result {
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
                        content: format!("Failed to find memories by concept: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            "graph_stats" => {
                match self.get_graph_stats().await {
                    Ok(stats) => super::Result {
                        tool_use_id: call.id,
                        content: stats.into(),
                        is_error: false,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                    Err(err) => super::Result {
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
                        return super::Result {
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
                        return super::Result {
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
                            super::Result {
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
                        content: format!("Failed to find memories via BFS: {}", err).into(),
                        is_error: true,
                        #[cfg(feature = "prompt-caching")]
                        cache_control: None,
                    },
                }
            }
            _ => super::Result {
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

    async fn load_json(&mut self, json: serde_json::Value) -> Result<(), String> {
        let data = if let serde_json::Value::Object(obj) = json {
            obj
        } else {
            return Err("Input must be a JSON object".to_string());
        };
        
        // Only restore the schema name - the database state persists independently
        if let Some(schema_name) = data.get("schema_name").and_then(|v| v.as_str()) {
            self.schema_name = schema_name.to_string();
            // Re-initialize to ensure the schema exists
            self.ensure_initialized().await?;
        }

        Ok(())
    }
}