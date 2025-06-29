// Copyright 2025 Claude 4 Sonnet, Claude 4 Opus, and Michael de Gans
//
// The initial idea was Sonnet. Opus helped refine it into what it is now in a
// collaborative effort. The code is a result of human and AI collaboration.
use crate::tool::memory_palace::MemoryPalaceError;
use sqlx::{PgPool, Postgres, Transaction};
use std::future::Future;
use std::pin::Pin;

/// Initialize the database schema. Idempotent.
pub async fn ensure_initialized(
    pool: &PgPool,
    schema_name: &str,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema_name, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // Users table stores (minimal) user information. `pro` users are
            // paying users who can access more models and features.
            //
            // `karma` is a simple integer to track user behavior. On reaching
            // -1000 karma, the user is automatically banned. The user can, at
            // that point only read chats and delete their account if they want
            // their data deleted.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                pro BOOLEAN NOT NULL DEFAULT FALSE,
                banned BOOLEAN NOT NULL DEFAULT FALSE,
                karma SMALLINT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )"#)
            .execute(&mut **tx)
            .await?;

            // Rooms in the MemoryPalace are logical groupings for memories with
            // narrative context. Creative writing powers the palace.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS rooms (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(128) NOT NULL,
                description TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_visited TIMESTAMPTZ NOT NULL DEFAULT now(),
                strength FLOAT8 NOT NULL DEFAULT 0.5,
                visit_count INTEGER NOT NULL DEFAULT 0,
                memory_count INTEGER NOT NULL DEFAULT 0,
                UNIQUE(user_id, name),
                CONSTRAINT room_name_length CHECK (char_length(name) >= 3),
                CONSTRAINT valid_last_visited CHECK (last_visited >= created_at),
                CONSTRAINT strength_range CHECK (strength >= 0 AND strength <= 1),
                CONSTRAINT visit_count_non_negative CHECK (visit_count >= 0),
                CONSTRAINT memory_count_non_negative CHECK (memory_count >= 0)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Memories represent notes, messages, summaries, images, and so on
            // that agents can recall. They are the core of the MemoryPalace.
            //
            // They are stored in JSONB format for flexibility. Most, but not
            // all memories will refer to a `Prompt`. For example, a `Report`
            // is not deleted even if the `Prompt` it refers to is deleted.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memories (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                content JSONB NOT NULL,
                room_id UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                prompt_id UUID REFERENCES prompts(id) ON DELETE SET NULL,
                prompt_index INTEGER,
                placement VARCHAR(64) NOT NULL DEFAULT 'shelf',
                placement_description TEXT,
                tags JSONB NOT NULL DEFAULT '[]',
                strength FLOAT8 NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_accessed TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT content_not_empty CHECK (jsonb_typeof(content) = 'object' AND jsonb_array_length(content) > 0),
                CONSTRAINT tags_is_array CHECK (jsonb_typeof(tags) = 'array'),
                CONSTRAINT prompt_index_non_negative CHECK (prompt_index IS NULL OR prompt_index >= 0),
                CONSTRAINT no_index_without_prompt CHECK (
                    (prompt_id IS NULL AND prompt_index IS NULL) OR
                    (prompt_id IS NOT NULL AND prompt_index IS NOT NULL)
                ),
                CONSTRAINT strength_range CHECK (strength >= 0 AND strength <= 1),
                CONSTRAINT access_count_non_negative CHECK (access_count >= 0),
                CONSTRAINT valid_last_updated CHECK (last_updated >= created_at),
                CONSTRAINT valid_last_accessed CHECK (last_accessed >= created_at)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Connections between rooms hint to the agent how to navigate
            // between them. They can represent hallways, doors, or other
            // passage types. They should lead to related rooms. Traversals will
            // increase the strength of the connection forming neural pathways
            // in the palace.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS pathways (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                room_a UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                room_b UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                passage_type VARCHAR(64) NOT NULL DEFAULT 'hallway',
                description TEXT,
                strength FLOAT8 NOT NULL DEFAULT 0.5,
                traversal_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_traversed TIMESTAMPTZ,
                UNIQUE(from_room_id, to_room_id),
                CONSTRAINT bidirectional_unique CHECK (
                    from_room_id < to_room_id
                ),
                CONSTRAINT strength_range CHECK (
                    strength >= 0 AND strength <= 1
                ),
                CONSTRAINT valid_traversal_count CHECK (traversal_count >= 0),
                CONSTRAINT valid_last_traversed CHECK (
                    last_traversed IS NULL OR last_traversed >= created_at
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // Relationships are directed edges between memories. They represent
            // relationships like "related", "caused", "inspired", etc.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_relationships (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                from_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                to_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                relationship_type VARCHAR(100) NOT NULL DEFAULT 'related',
                strength FLOAT NOT NULL DEFAULT 0.5,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(from_memory_id, to_memory_id),
                CONSTRAINT strength_range CHECK (
                    strength >= 0 AND strength <= 1
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // The memory access log tracks now memories are accessed and can be
            // used to analyze memory usage patterns or to rebuild `strength`.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_access_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                -- Access type: 'c' for create, 'r' for read, 'u' for update, 'd' for delete
                access_type CHAR(1) NOT NULL,
                -- The agent or user who accessed the memory (e.g. Archivist, Janitor)
                accessed_by VARCHAR(64) NOT NULL,
                context TEXT,
                path JSONB NOT NULL,
                accessed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT valid_access_type CHECK (
                    access_type IN ('c', 'r', 'u', 'd')
                ),
                CONSTRAINT valid_path CHECK (
                    jsonb_typeof(path) = 'array' AND
                    jsonb_array_length(path) > 0
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for clustering similar memories based on embeddings or
            // other similarity measures. This allows agents to group related
            // memories together for easier recall and consolidation.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_similarity_clusters (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(cluster_id, memory_id),
                CONSTRAINT valid_similarity_score CHECK (
                    similarity_score >= 0 AND similarity_score <= 1
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for memory consolidation history. Too many similar memories
            // clutter the palace and waste agents' time and context. The
            // Janitor agent consolidates similar memories by merging them. It
            // is not guaranteed that all `origial_memory_ids` exist. They may
            // be forgotten permanently in some cases.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_consolidations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                consolidated_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                agent_notes TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT valid_original_memory_ids_elements CHECK (
                    array_length(original_memory_ids, 1) > 1 AND
                    array_length(original_memory_ids, 1) <= 5
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // Consolidated memories associate original memories with the
            // consolidated memory. This allows agents to see the history of
            // consolidation and access original memories if needed.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS consolidated_memory_ids (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                consolidated_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                original_memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(consolidated_memory_id, original_memory_id),
                CONSTRAINT valid_original_memory_id CHECK (
                    original_memory_id != consolidated_memory_id
                )
            )"#)
            .execute(&mut **tx)
            .await?;

            // Table for memory decay tracking. This is purely for debugging
            // purposes, logging when and why memory strength is decayed. More
            // frequently accessed memories increase in strength while over time
            // unused memories decay. This is a time-based decay, not a
            // similarity-based decay.
            //
            // The decay is applied periodically, e.g. once a week across all
            // memories. The decay rate is configurable, but the default is 1%
            // per week. We don't record per-memory decay, rather the batch
            // event is logged here. It is applied across the palace to all user
            // memories.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_decay_log (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                decay_date TIMESTAMPTZ NOT NULL DEFAULT now(),
                decay_reason VARCHAR(100) NOT NULL
            )"#)
            .execute(&mut **tx)
            .await?;

            // Prompts table stores entire Prompts which memories usually refer
            // to. On new message, this is updated with the latest version.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS prompts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                content JSONB NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT now(),
                CONSTRAINT content_not_empty CHECK (
                    jsonb_typeof(content) = 'object'
                    AND jsonb_array_length(content) > 0
                ),
                -- If the content is a note, it cannot contain "<note>" or "</note>"
                CONSTRAINT content_does_not_contain_forbidden_note_tags CHECK (
                    NOT (content @> '{"type": "note"}' AND
                        (content->'data'->>'text') LIKE '%<note>%' OR
                        (content->'data'->>'text') LIKE '%</note>%')
                ),
                CONSTRAINT valid_last_updated CHECK (last_updated >= created_at)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Embeddings cache table. This cache is shared between users since
            // there may be cache hits for small messages like greetings.
            // Embeddings are attached to memories, prompts, rooms, clusters,
            // and so on. They are used for similarity search and clustering.
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                model_name VARCHAR(255) NOT NULL,
                model_size INTEGER NOT NULL,
                content_hash BYTEA NOT NULL,
                embedding VECTOR NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(model_name, content_hash),
                constraint model_name_length CHECK (
                    char_length(model_name) >= 3
                ),
                constraint valid_model_size CHECK (model_size > 0)
            )"#)
            .execute(&mut **tx)
            .await?;

            // ## Junction tables for embeddings

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                embedding_id UUID NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(memory_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS prompt_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                prompt_id UUID NOT NULL REFERENCES prompts(id) ON DELETE CASCADE,
                embedding_id UUID NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(prompt_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Room centroids
            // TODO: Centroid update trigger
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS room_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                room_id UUID NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                embedding_id UUID NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(room_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;

            // Embedding cluster centroids
            // TODO: Cluster update trigger
            sqlx::query(r#"CREATE TABLE IF NOT EXISTS cluster_embeddings (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                cluster_id UUID NOT NULL DEFAULT gen_random_uuid(),
                embedding_id UUID NOT NULL REFERENCES embeddings(id) ON DELETE CASCADE,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(cluster_id, embedding_id)
            )"#)
            .execute(&mut **tx)
            .await?;


            // Functions and triggers
            sqlx::query(r#"CREATE OR REPLACE FUNCTION update_last_updated_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_updated = now();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Called on memory update
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

            // TODO: Call this. It's currently unused. Either we call this
            // periodically or we find another solution to decay memories.
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
                            importance * (1 - decay_rate * EXTRACT(EPOCH FROM (now() - last_accessed)) / EXTRACT(EPOCH FROM decay_period))
                        )
                        WHERE last_accessed < now() - decay_period
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
                target_memory_id UUID,
                similarity_threshold FLOAT DEFAULT 0.85,
                max_results INT DEFAULT 10,
                embedding_model VARCHAR DEFAULT NULL
            )
            RETURNS TABLE(
                memory_id UUID,
                similarity_score FLOAT,
                content JSONB,
                room_id UUID
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
