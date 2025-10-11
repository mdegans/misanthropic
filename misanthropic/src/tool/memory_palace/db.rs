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
                strength FLOAT8 NOT NULL DEFAULT 0.5,
                traversal_count INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_traversed TIMESTAMPTZ,
                UNIQUE(room_a, room_b),
                CONSTRAINT bidirectional_unique CHECK (
                    room_a < room_b
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
                cluster_id UUID NOT NULL DEFAULT gen_random_uuid(),
                memory_id UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                similarity_score FLOAT NOT NULL DEFAULT 0.5,
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
                original_memory_ids UUID[] NOT NULL,
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
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                memories_affected INTEGER NOT NULL DEFAULT 0,
                decay_rate FLOAT NOT NULL DEFAULT 0.01,
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
                        SET strength = GREATEST(
                            min_importance,
                            strength * (1 - decay_rate * EXTRACT(EPOCH FROM (now() - last_accessed)) / EXTRACT(EPOCH FROM decay_period))
                        )
                        WHERE last_accessed < now() - decay_period
                        AND strength > min_importance
                        RETURNING id, strength
                    )
                    INSERT INTO memory_decay_log (user_id, memories_affected, decay_rate, decay_reason)
                    SELECT 
                        m.user_id,
                        COUNT(dm.id),
                        decay_rate,
                        'time_based_decay'
                    FROM memories m
                    JOIN decayed_memories dm ON m.id = dm.id
                    GROUP BY m.user_id;
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

            // Function to calculate centroid embedding for a room
            sqlx::query(r#"CREATE OR REPLACE FUNCTION calculate_room_centroid(
                room_id_param UUID,
                model_name_param VARCHAR
            )
            RETURNS UUID AS $$
            DECLARE
                new_embedding_id UUID;
                centroid_vector VECTOR;
            BEGIN
                -- Calculate the centroid of all memory embeddings in the room
                SELECT AVG(e.embedding)::VECTOR INTO centroid_vector
                FROM memories m
                JOIN memory_embeddings me ON m.id = me.memory_id
                JOIN embeddings e ON me.embedding_id = e.id
                WHERE m.room_id = room_id_param
                AND (e.model_name = model_name_param);
                
                IF centroid_vector IS NULL THEN
                    RETURN NULL;
                END IF;
                
                -- Create a new embedding for the centroid
                INSERT INTO embeddings (model_name, model_size, content_hash, embedding)
                VALUES (
                    model_name_param,
                    array_length(centroid_vector::float[], 1),
                    sha256(room_id_param::text::bytea),
                    centroid_vector
                )
                ON CONFLICT (model_name, content_hash) 
                DO UPDATE SET embedding = EXCLUDED.embedding
                RETURNING id INTO new_embedding_id;
                
                -- Update room_embeddings
                INSERT INTO room_embeddings (user_id, room_id, embedding_id)
                SELECT m.user_id, room_id_param, new_embedding_id
                FROM memories m
                WHERE m.room_id = room_id_param
                LIMIT 1
                ON CONFLICT (room_id, embedding_id) DO NOTHING;
                
                RETURN new_embedding_id;
            END;
            $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Function to calculate centroid embedding for a cluster
            sqlx::query(r#"CREATE OR REPLACE FUNCTION calculate_cluster_centroid(
                cluster_id_param UUID,
                user_id_param UUID,
                model_name_param VARCHAR
            )
            RETURNS UUID AS $$
            DECLARE
                new_embedding_id UUID;
                centroid_vector VECTOR;
            BEGIN
                -- Calculate the centroid of all memory embeddings in the cluster
                SELECT AVG(e.embedding)::VECTOR INTO centroid_vector
                FROM memory_similarity_clusters msc
                JOIN memory_embeddings me ON msc.memory_id = me.memory_id
                JOIN embeddings e ON me.embedding_id = e.id
                WHERE msc.cluster_id = cluster_id_param
                AND msc.user_id = user_id_param
                AND (e.model_name = model_name_param);

                IF centroid_vector IS NULL THEN
                    RETURN NULL;
                END IF;

                -- Create a new embedding for the centroid
                INSERT INTO embeddings (model_name, model_size, content_hash, embedding)
                VALUES (
                    model_name_param,
                    array_length(centroid_vector::float[], 1),
                    sha256(cluster_id_param::text::bytea),
                    centroid_vector
                )
                ON CONFLICT (model_name, content_hash) 
                DO UPDATE SET embedding = EXCLUDED.embedding
                RETURNING id INTO new_embedding_id;

                -- Update cluster_embeddings
                INSERT INTO cluster_embeddings (user_id, cluster_id, embedding_id)
                VALUES (user_id_param, cluster_id_param, new_embedding_id)
                ON CONFLICT (cluster_id, embedding_id) DO NOTHING;

                RETURN new_embedding_id;
            END;
            $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Function to update room centroid after memory changes
            sqlx::query(r#"CREATE OR REPLACE FUNCTION update_room_centroid()
            RETURNS TRIGGER AS $$
            BEGIN
                -- For INSERT or UPDATE, use NEW.room_id
                IF TG_OP IN ('INSERT', 'UPDATE') THEN
                    PERFORM calculate_room_centroid(NEW.room_id);
                END IF;
                
                -- For UPDATE with room change, update old room too
                IF TG_OP = 'UPDATE' AND OLD.room_id != NEW.room_id THEN
                    PERFORM calculate_room_centroid(OLD.room_id);
                END IF;
                
                -- For DELETE, use OLD.room_id
                IF TG_OP = 'DELETE' THEN
                    PERFORM calculate_room_centroid(OLD.room_id);
                END IF;
                
                RETURN NULL;
            END;
            $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            // Function to update cluster centroids when membership changes
            sqlx::query(r#"CREATE OR REPLACE FUNCTION update_cluster_centroid()
            RETURNS TRIGGER AS $$
            BEGIN
                IF TG_OP IN ('INSERT', 'UPDATE') THEN
                    PERFORM calculate_cluster_centroid(NEW.cluster_id, NEW.user_id);
                END IF;
                
                IF TG_OP = 'UPDATE' AND OLD.cluster_id != NEW.cluster_id THEN
                    PERFORM calculate_cluster_centroid(OLD.cluster_id, OLD.user_id);
                END IF;
                
                IF TG_OP = 'DELETE' THEN
                    PERFORM calculate_cluster_centroid(OLD.cluster_id, OLD.user_id);
                END IF;
                
                RETURN NULL;
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

            // Trigger for updating room centroids when memories change
            sqlx::query("DROP TRIGGER IF EXISTS update_room_centroid_on_memory_change ON memories")
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TRIGGER update_room_centroid_on_memory_change
                AFTER INSERT OR UPDATE OR DELETE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_room_centroid()"#)
            .execute(&mut **tx)
            .await?;

            // Trigger for updating room centroids when memory embeddings change
            sqlx::query("DROP TRIGGER IF EXISTS update_room_centroid_on_embedding_change ON memory_embeddings")
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TRIGGER update_room_centroid_on_embedding_change
                AFTER INSERT OR UPDATE OR DELETE ON memory_embeddings
                FOR EACH ROW
                EXECUTE FUNCTION update_room_centroid()"#)
            .execute(&mut **tx)
            .await?;

            // Trigger for updating cluster centroids
            sqlx::query("DROP TRIGGER IF EXISTS update_cluster_centroid_on_membership ON memory_similarity_clusters")
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TRIGGER update_cluster_centroid_on_membership
                AFTER INSERT OR UPDATE OR DELETE ON memory_similarity_clusters
                FOR EACH ROW
                EXECUTE FUNCTION update_cluster_centroid()"#)
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

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_pathways_room_a ON pathways(room_a)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_pathways_room_b ON pathways(room_b)")
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

            // Create view for easy room navigation
            sqlx::query(r#"CREATE OR REPLACE VIEW room_navigation AS
                SELECT 
                    p.id,
                    r1.name as room_a_name,
                    r2.name as room_b_name,
                    p.passage_type,
                    p.description,
                    p.strength,
                    p.traversal_count,
                    p.last_traversed
                FROM pathways p
                JOIN rooms r1 ON p.room_a = r1.id
                JOIN rooms r2 ON p.room_b = r2.id"#)
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
