use sqlx::{PgPool, Transaction};
use std::future::Future;
use std::pin::Pin;

use crate::tool::memory_palace::MemoryPalaceError;

/// Initialize the database schema with proper indexes and triggers.
pub(crate) async fn ensure_initialized(
    pool: &PgPool,
    schema_name: &str,
) -> Result<(), MemoryPalaceError> {
    // Execute schema creation statements in a transaction to ensure atomicity
    let mut tx = pool.begin().await?;

    // Set search path for this transaction
    sqlx::query(&format!("SET search_path TO {}", schema_name))
        .execute(&mut *tx)
        .await?;

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
        sqlx::query(statement).execute(&mut *tx).await?;
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
        sqlx::query(statement).execute(&mut *tx).await?;
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
        sqlx::query(statement).execute(&mut *tx).await?;
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
        sqlx::query(statement).execute(&mut *tx).await?;
    }

    // Commit the transaction
    tx.commit().await?;

    Ok(())
}

/// Run any operation inside a transaction with the correct search_path
pub(crate) async fn execute_with_schema<'q, F, R>(
    pool: &PgPool,
    schema_name: &str,
    operation: F,
) -> Result<R, MemoryPalaceError>
where
    F: for<'c> FnOnce(
        &'c mut Transaction<'_, sqlx::Postgres>,
    ) -> Pin<
        Box<dyn Future<Output = Result<R, sqlx::Error>> + Send + 'c>,
    >,
{
    let mut tx = pool.begin().await?;

    sqlx::query(&format!("SET search_path TO {}", schema_name))
        .execute(&mut *tx)
        .await?;

    let result = operation(&mut tx).await?;

    tx.commit().await?;

    Ok(result)
}
