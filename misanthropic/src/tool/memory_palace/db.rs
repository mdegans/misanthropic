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
            // Execute schema creation statements individually using sqlx::query!

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS rooms (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS memories (
                id BIGSERIAL PRIMARY KEY,
                content TEXT NOT NULL,
                room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                tags JSONB NOT NULL DEFAULT '[]',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS room_connections (
                id BIGSERIAL PRIMARY KEY,
                from_room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                to_room VARCHAR(255) NOT NULL REFERENCES rooms(name) ON DELETE CASCADE,
                description TEXT,
                strength INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(from_room, to_room)
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS memory_relationships (
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

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS concepts (
                id BIGSERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                description TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query!(r#"CREATE TABLE IF NOT EXISTS memory_concepts (
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
            sqlx::query!(r#"CREATE OR REPLACE FUNCTION update_last_updated_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.last_updated = NOW();
                    RETURN NEW;
                END;
                $$ language 'plpgsql'"#)
            .execute(&mut **tx)
            .await?;

            sqlx::query!("DROP TRIGGER IF EXISTS update_memories_last_updated ON memories")
            .execute(&mut **tx)
            .await?;

            sqlx::query!(r#"CREATE TRIGGER update_memories_last_updated
                BEFORE UPDATE ON memories
                FOR EACH ROW
                EXECUTE FUNCTION update_last_updated_column()"#)
            .execute(&mut **tx)
            .await?;

            // Indexes
            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memories_room ON memories(room)")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memories_content_gin ON memories USING gin(to_tsvector('english', content))")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memories_tags_gin ON memories USING gin(tags)")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_room_connections_from ON room_connections(from_room)")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memory_relationships_from ON memory_relationships(from_memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts(memory_id)")
            .execute(&mut **tx)
            .await?;

            sqlx::query!("CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts(concept_id)")
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
        Box<dyn Future<Output = Result<R, sqlx::Error>> + Send + 'c>,
    >,
{
    let mut tx = pool.begin().await?;

    // Set search_path for this transaction
    sqlx::query(&format!("SET search_path TO {}", schema_name))
        .execute(&mut *tx)
        .await?;

    let result = f(&mut tx).await?;
    tx.commit().await?;

    Ok(result)
}
