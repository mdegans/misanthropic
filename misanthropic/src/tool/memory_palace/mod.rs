use crate::tool::memory_palace::MemoryPalaceError;
use sqlx::{PgPool, Postgres, Transaction};
use std::future::Future;
use std::pin::Pin;
use uuid::Uuid;

/// Initialize the database schema with proper indexes and triggers.
pub async fn ensure_initialized(
    pool: &PgPool,
    schema_name: &str,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema_name, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // Execute schema creation statements individually using sqlx::query!

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS rooms (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                atmosphere TEXT,
                centroid_embedding VECTOR(1536) NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                placement VARCHAR(255) NOT NULL default 'shelf',
                placement_description TEXT,
                tags JSONB NOT NULL DEFAULT '[]',
                embedding VECTOR(1536) NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS room_connections (
                id BIGSERIAL PRIMARY KEY,
                from_room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                to_room_id BIGINT NOT NULL REFERENCES rooms(id) ON DELETE CASCADE,
                passage_type VARCHAR(100) NOT NULL DEFAULT 'hallway',
                description TEXT,
                strength INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_room_id, to_room_id)
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

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS concepts (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TABLE IF NOT EXISTS memory_concepts (
                id BIGSERIAL PRIMARY KEY,
                memory_id BIGINT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                concept_id BIGINT NOT NULL REFERENCES concepts(id) ON DELETE CASCADE,
                confidence FLOAT NOT NULL DEFAULT 1.0,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(memory_id, concept_id)
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

            sqlx::query("DROP TRIGGER IF EXISTS update_memories_last_updated ON memories")
            .execute(&mut **tx)
            .await?;

            sqlx::query(r#"CREATE TRIGGER update_memories_last_updated
                BEFORE UPDATE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_last_updated_column()"#)
            .execute(&mut **tx)
            .await?;

            // Indexes
            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_room ON memories(room)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(to_tsvector('english', content))")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING gin(tags)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_room_connections_from ON room_connections(from_room_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_relationships_from ON memory_relationships(from_memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts(memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query("CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts(concept_id)")
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

impl MemoryPalace {
    /// Find memories similar to a given memory
    pub async fn find_similar_memories(
        &self,
        memory_id: i64,
        similarity_threshold: Option<f32>,
        max_results: Option<i32>,
    ) -> Result<Vec<SimilarMemory>, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            let similarity_threshold = similarity_threshold.unwrap_or(0.85);
            let max_results = max_results.unwrap_or(10);

            Box::pin(async move {
                #[derive(sqlx::FromRow)]
                struct SimilarMemoryRow {
                    memory_id: i64,
                    similarity_score: f32,
                    content: String,
                    room_id: i64,
                }

                let rows: Vec<SimilarMemoryRow> = sqlx::query_as(
                    "SELECT * FROM find_similar_memories($1, $2, $3)",
                )
                .bind(memory_id)
                .bind(similarity_threshold)
                .bind(max_results)
                .fetch_all(&mut **tx)
                .await?;

                let similar_memories = rows
                    .into_iter()
                    .map(|row| SimilarMemory {
                        memory_id: row.memory_id,
                        similarity_score: row.similarity_score,
                        content: row.content,
                        room_id: row.room_id,
                    })
                    .collect();

                Ok(similar_memories)
            })
        })
        .await
    }

    /// Mark memories as part of a similarity cluster for review
    pub async fn create_similarity_cluster(
        &self,
        memory_ids: Vec<i64>,
        similarity_scores: Vec<f32>,
    ) -> Result<Uuid, MemoryPalaceError> {
        if memory_ids.len() != similarity_scores.len() {
            return Err(MemoryPalaceError::InvalidInput(
                "Memory IDs and similarity scores must have the same length"
                    .to_string(),
            ));
        }

        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                let cluster_id = Uuid::new_v4();

                // Mark the first memory as primary
                for (idx, (memory_id, score)) in memory_ids.iter().zip(similarity_scores.iter()).enumerate() {
                    sqlx::query(
                        r#"INSERT INTO memory_similarity_clusters (cluster_id, memory_id, similarity_score, is_primary)
                           VALUES ($1, $2, $3, $4)
                           ON CONFLICT (memory_id) DO UPDATE
                           SET cluster_id = EXCLUDED.cluster_id,
                               similarity_score = EXCLUDED.similarity_score,
                               is_primary = EXCLUDED.is_primary"#
                    )
                    .bind(cluster_id)
                    .bind(memory_id)
                    .bind(score)
                    .bind(idx == 0)
                    .execute(&mut **tx)
                    .await?;
                }

                Ok(cluster_id)
            })
        })
        .await
    }

    /// Consolidate multiple memories into a single memory
    pub async fn consolidate_memories(
        &self,
        original_memory_ids: Vec<i64>,
        consolidated_content: &str,
        room_id: i64,
        agent_notes: Option<&str>,
        consolidation_type: Option<&str>,
    ) -> Result<i64, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            let consolidated_content = consolidated_content.to_string();
            let agent_notes = agent_notes.map(|s| s.to_string());
            let consolidation_type = consolidation_type.unwrap_or("merge").to_string();
            let original_memory_ids = original_memory_ids.clone();

            Box::pin(async move {
                // Calculate average importance from original memories
                let avg_importance: f32 = sqlx::query_scalar(
                    "SELECT AVG(importance)::FLOAT FROM memories WHERE id = ANY($1)"
                )
                .bind(&original_memory_ids)
                .fetch_one(&mut **tx)
                .await?;

                // Merge tags from all original memories
                let merged_tags: serde_json::Value = sqlx::query_scalar(
                    "SELECT jsonb_agg(DISTINCT tag) FROM memories, jsonb_array_elements(tags) as tag WHERE id = ANY($1)"
                )
                .bind(&original_memory_ids)
                .fetch_one(&mut **tx)
                .await?;

                // Create the consolidated memory
                let consolidated_id: i64 = sqlx::query_scalar(
                    r#"INSERT INTO memories (content, room_id, tags, importance)
                       VALUES ($1, $2, $3, $4)
                       RETURNING id"#
                )
                .bind(&consolidated_content)
                .bind(room_id)
                .bind(&merged_tags)
                .bind(avg_importance * 1.1) // Slight boost for consolidated memories
                .fetch_one(&mut **tx)
                .await?;

                // Log the consolidation
                sqlx::query(
                    r#"INSERT INTO memory_consolidations (consolidated_memory_id, original_memory_ids, consolidation_type, agent_notes)
                       VALUES ($1, $2, $3, $4)"#
                )
                .bind(consolidated_id)
                .bind(&original_memory_ids)
                .bind(&consolidation_type)
                .bind(&agent_notes)
                .execute(&mut **tx)
                .await?;

                // Archive or delete original memories based on consolidation type
                if consolidation_type == "merge" {
                    // Soft delete by setting importance to 0
                    sqlx::query(
                        "UPDATE memories SET importance = 0 WHERE id = ANY($1)"
                    )
                    .bind(&original_memory_ids)
                    .execute(&mut **tx)
                    .await?;
                }

                Ok(consolidated_id)
            })
        })
        .await
    }

    /// Apply decay to memories based on access patterns
    pub async fn apply_memory_decay(&self) -> Result<usize, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            Box::pin(async move {
                sqlx::query("SELECT calculate_memory_decay()")
                    .execute(&mut **tx)
                    .await?;

                let decayed_count: i64 = sqlx::query_scalar(
                    "SELECT COUNT(*) FROM memory_decay_log WHERE decayed_at > NOW() - INTERVAL '1 minute'"
                )
                .fetch_one(&mut **tx)
                .await?;

                Ok(decayed_count as usize)
            })
        })
        .await
    }

    /// Get memories that are candidates for consolidation
    pub async fn get_consolidation_candidates(
        &self,
        min_cluster_size: Option<i32>,
        similarity_threshold: Option<f32>,
    ) -> Result<Vec<MemoryCluster>, MemoryPalaceError> {
        execute_with_schema(&self.pool, &self.schema_name, |tx| {
            let min_cluster_size = min_cluster_size.unwrap_or(3);
            let similarity_threshold = similarity_threshold.unwrap_or(0.85);

            Box::pin(async move {
                #[derive(sqlx::FromRow)]
                struct ClusterRow {
                    cluster_id: Uuid,
                    memory_ids: Vec<i64>,
                    avg_similarity: f32,
                    room_names: Vec<String>,
                }

                let rows: Vec<ClusterRow> = sqlx::query_as(
                    r#"WITH cluster_stats AS (
                        SELECT 
                            msc.cluster_id,
                            array_agg(msc.memory_id) as memory_ids,
                            AVG(msc.similarity_score)::FLOAT as avg_similarity,
                            array_agg(DISTINCT r.name) as room_names
                        FROM memory_similarity_clusters msc
                        JOIN memories m ON msc.memory_id = m.id
                        JOIN rooms r ON m.room_id = r.id
                        WHERE msc.similarity_score >= $1
                        GROUP BY msc.cluster_id
                        HAVING COUNT(*) >= $2
                    )
                    SELECT * FROM cluster_stats
                    ORDER BY avg_similarity DESC"#,
                )
                .bind(similarity_threshold)
                .bind(min_cluster_size)
                .fetch_all(&mut **tx)
                .await?;

                let clusters = rows
                    .into_iter()
                    .map(|row| MemoryCluster {
                        cluster_id: row.cluster_id,
                        memory_ids: row.memory_ids,
                        avg_similarity: row.avg_similarity,
                        room_names: row.room_names,
                    })
                    .collect();

                Ok(clusters)
            })
        })
        .await
    }
}

// Add these structs to your types
#[derive(Debug, Clone)]
pub struct SimilarMemory {
    pub memory_id: i64,
    pub similarity_score: f32,
    pub content: String,
    pub room_id: i64,
}

#[derive(Debug, Clone)]
pub struct MemoryCluster {
    pub cluster_id: Uuid,
    pub memory_ids: Vec<i64>,
    pub avg_similarity: f32,
    pub room_names: Vec<String>,
}
