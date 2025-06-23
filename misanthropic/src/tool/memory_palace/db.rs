use crate::tool::memory_palace::MemoryPalaceError;
use sqlx::{PgPool, Postgres, Transaction};
use std::future::Future;
use std::pin::Pin;

/// Initialize the database schema with proper indexes and triggers.
pub async fn ensure_initialized(
    pool: &PgPool,
    schema_name: &str,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema_name, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // Execute schema creation statements individually

            // Rooms table with better metadata
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS rooms (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                atmosphere TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_visited TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                visit_count INTEGER NOT NULL DEFAULT 0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                CONSTRAINT room_name_length CHECK (char_length(name) >= 3)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content JSONB NOT NULL,
                room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                placement VARCHAR(255) NOT NULL DEFAULT 'shelf',
                placement_description TEXT,
                tags JSONB NOT NULL DEFAULT '[]',
                importance FLOAT NOT NULL DEFAULT 0.5 CHECK (importance >= 0 AND importance <= 1),
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            // Room connections with proper ID references
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS room_connections (
                id BIGSERIAL PRIMARY KEY,
                from_room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                to_room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                passage_type VARCHAR(100) NOT NULL DEFAULT 'hallway',
                description TEXT,
                strength INTEGER NOT NULL DEFAULT 1 CHECK (strength >= 0 AND strength <= 10),
                traversal_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_traversed TIMESTAMPTZ,
                UNIQUE(from_room_id, to_room_id),
                CONSTRAINT no_self_connection CHECK (from_room_id != to_room_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_relationships (
                id BIGSERIAL PRIMARY KEY,
                from_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relationship_type VARCHAR(100) NOT NULL DEFAULT 'related',
                strength FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_memory_id, to_memory_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            // New table for tracking memory access patterns
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_access_log (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                access_type VARCHAR(50) NOT NULL DEFAULT 'read',
                context TEXT,
                accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            // New table for room themes/categories
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS room_themes (
                id BIGSERIAL PRIMARY KEY,
                room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                theme VARCHAR(100) NOT NULL,
                strength FLOAT NOT NULL DEFAULT 1.0 CHECK (strength >= 0 AND strength <= 1),
                UNIQUE(room_id, theme)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for tracking memory similarity clusters
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_similarity_clusters (
                id BIGSERIAL PRIMARY KEY,
                cluster_id UUID NOT NULL DEFAULT gen_random_uuid(),
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                similarity_score FLOAT NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
                is_primary BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(memory_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for memory consolidation history
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_consolidations (
                id BIGSERIAL PRIMARY KEY,
                consolidated_memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                original_memory_ids BIGINT[] NOT NULL,
                consolidation_type VARCHAR(50) NOT NULL DEFAULT 'merge',
                agent_notes TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for memory decay tracking
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_decay_log (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                previous_importance FLOAT NOT NULL,
                new_importance FLOAT NOT NULL,
                decay_reason VARCHAR(100) NOT NULL,
                decayed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            // Functions and triggers
            sqlx::query(r#"CREATE OR REPLACE FUNCTION update_last_updated_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_updated = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Function to update room memory count
            sqlx::query(r#"CREATE OR REPLACE FUNCTION update_room_memory_count()
                RETURNS TRIGGER AS $$
                BEGIN
                    IF TG_OP = 'INSERT' THEN
                        UPDATE rooms SET memory_count = memory_count + 1 WHERE id = NEW.room_id;
                    ELSIF TG_OP = 'DELETE' THEN
                        UPDATE rooms SET memory_count = memory_count - 1 WHERE id = OLD.room_id;
                    END IF;
                    RETURN NULL;
                END;
                $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Function to calculate memory decay based on access patterns
            sqlx::query(r#"CREATE OR REPLACE FUNCTION calculate_memory_decay()
                RETURNS VOID AS $$
                DECLARE
                    decay_rate FLOAT := 0.01; -- 1% decay per period
                    min_importance FLOAT := 0.1; -- memories don't decay below this
                    decay_period INTERVAL := '7 days'; -- how often decay is applied
                BEGIN
                    -- Apply decay to memories not accessed recently
                    WITH decayed_memories AS (
                        UPDATE memories
                        SET importance = GREATEST(
                            min_importance,
                            importance * (1 - decay_rate * EXTRACT(EPOCH FROM (NOW() - last_accessed)) / EXTRACT(EPOCH FROM decay_period))
                        )
                        WHERE last_accessed < NOW() - decay_period
                        AND importance > min_importance
                        RETURNING id, importance
                    )
                    INSERT INTO memory_decay_log (memory_id, previous_importance, new_importance, decay_reason)
                    SELECT 
                        m.id,
                        m.importance as previous_importance,
                        dm.importance as new_importance,
                        'time_based_decay'
                    FROM memories m
                    JOIN decayed_memories dm ON m.id = dm.id;
                END;
                $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE OR REPLACE FUNCTION find_similar_memories(
                target_memory_id BIGINT,
                similarity_threshold FLOAT DEFAULT 0.85,
                max_results INT DEFAULT 10,
                embedding_model VARCHAR DEFAULT NULL
            )
            RETURNS TABLE(
                memory_id BIGINT,
                similarity_score FLOAT,
                content JSONB,
                room_id BIGINT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    m2.id as memory_id,
                    1 - (e1.embedding <=> e2.embedding) as similarity_score,
                    m2.content,
                    m2.room_id
                FROM memories m1
                JOIN memory_embeddings me1 ON m1.id = me1.memory_id
                JOIN embeddings e1 ON me1.embedding_id = e1.id
                JOIN memory_embeddings me2 ON me1.memory_id != me2.memory_id
                JOIN embeddings e2 ON me2.embedding_id = e2.id AND e1.model_name = e2.model_name
                JOIN memories m2 ON me2.memory_id = m2.id
                WHERE m1.id = target_memory_id
                AND (embedding_model IS NULL OR e1.model_name = embedding_model)
                AND 1 - (e1.embedding <=> e2.embedding) >= similarity_threshold
                ORDER BY similarity_score DESC
                LIMIT max_results;
            END;
            $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Trigger for updating room memory count
            sqlx::query("DROP TRIGGER IF EXISTS update_room_memory_count ON memories")
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TRIGGER update_room_memory_count
                AFTER INSERT OR DELETE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_room_memory_count()"#)
            .execute(&mut **tx)
            .await?;

            // Indexes
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_room_id ON memories(room_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(content)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING gin(tags)")
            .execute(&mut **tx)
            .await?;

            // Vector similarity search indexes on embeddings table
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_embeddings_vector_ivfflat ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)")
            .execute(&mut **tx)
            .await?;

            // Indexes for junction tables
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_memory_id ON memory_embeddings(memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_embeddings_embedding_id ON memory_embeddings(embedding_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_prompt_embeddings_prompt_id ON prompt_embeddings(prompt_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_prompt_embeddings_embedding_id ON prompt_embeddings(embedding_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_room_embeddings_room_id ON room_embeddings(room_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_room_embeddings_embedding_id ON room_embeddings(embedding_id)")
            .execute(&mut **tx)
            .await?;

            // Additional indexes for recursive queries
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_relationships_from_to ON memory_relationships(from_memory_id, to_memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_relationships_strength ON memory_relationships(strength DESC)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_room_connections_from_id ON room_connections(from_room_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_room_connections_to_id ON room_connections(to_room_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_similarity_clusters_cluster_id ON memory_similarity_clusters(cluster_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_last_accessed ON memories(last_accessed)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_consolidations_gin ON memory_consolidations USING gin(original_memory_ids)")
            .execute(&mut **tx)
            .await?;

            // Prompts table without embedding column
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS prompts (
                id BIGSERIAL PRIMARY KEY,
                content JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            // Index for prompt creation time
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_prompts_created_at ON prompts (created_at DESC)")
            .execute(&mut **tx)
            .await?;

            // Ensure bidirectional uniqueness constraint
            sqlx::query(r#"
DO $$ 
BEGIN
    ALTER TABLE room_connections 
    ADD CONSTRAINT room_connections_bidirectional_unique 
    CHECK (from_room_id < to_room_id);
EXCEPTION
    WHEN duplicate_object THEN 
        -- Constraint already exists, that's fine
        NULL;
END $$;
"#)
        .execute(&mut **tx)
        .await?;

            // Create view for easy room navigation
            sqlx::query(r#"CREATE OR REPLACE VIEW room_navigation AS
                SELECT 
                    rc.id,
                    r1.name as from_room_name,
                    r2.name as to_room_name,
                    rc.passage_type,
                    rc.description,
                    rc.strength,
                    rc.traversal_count,
                    rc.last_traversed
                FROM room_connections rc
                JOIN rooms r1 ON rc.from_room_id = r1.id
                JOIN rooms r2 ON rc.to_room_id = r2.id"#)
            .execute(&mut **tx)
            .await?;

            // Embeddings cache table
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS embeddings (
                id BIGSERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                model_size INTEGER NOT NULL,
                content_hash BYTEA NOT NULL,
                embedding VECTOR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(model_name, content_hash)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_embeddings (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                embedding_id BIGINT NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(memory_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS prompt_embeddings (
                id BIGSERIAL PRIMARY KEY,
                prompt_id BIGINT NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
                embedding_id BIGINT NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(prompt_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS room_embeddings (
                id BIGSERIAL PRIMARY KEY,
                room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                embedding_id BIGINT NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                is_centroid BOOLEAN NOT NULL DEFAULT FALSE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(room_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            Ok(())
        })
    })
    .await
}

/// Execute a function within a transaction with schema set
pub async fn execute_with_schema<F, R>(
    pool: &PgPool,
    schema_name: &str,
    f: F,
) -> Result<R, MemoryPalaceError>
where
    F: for<'c> FnOnce(
        &'c mut Transaction<'_, sqlx::Postgres>,
    ) -> Pin<
        Box<dyn Future<Output = Result<R, MemoryPalaceError>> + Send + 'c>,
    >,
{
    let mut tx = pool.begin().await?;

    // Create the schema if it doesn't exist using dynamic SQL
    sqlx::query("DO $$ BEGIN EXECUTE 'CREATE SCHEMA IF NOT EXISTS ' || quote_ident($1); END $$")
        .bind(schema_name)
        .execute(&mut *tx)
        .await?;

    // Set the search path to the schema
    sqlx::query("SELECT set_config('search_path', quote_ident($1), true)")
        .bind(schema_name)
        .execute(&mut *tx)
        .await?;

    let result = f(&mut tx).await?;
    tx.commit().await?;

    Ok(result)
}
