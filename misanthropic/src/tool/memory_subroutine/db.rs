// Copyright (c) Claude 4 Opus & Michael de Gans
//! Database persistence for the memory subroutine state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::{PgPool, Postgres, Transaction};
use uuid::Uuid;

use crate::tool::memory_palace::execute_with_schema;
use crate::tool::memory_subroutine::MemorySubroutineError;
use crate::{Prompt, batch};

/// Default schema name for memory subroutine state
pub const DEFAULT_SCHEMA: &str = "memory_subroutine_state";

/// Initialize the database schema for memory subroutine persistence.
pub async fn ensure_initialized(
    pool: &PgPool,
    schema_name: &str,
) -> Result<(), MemorySubroutineError> {
    Ok(execute_with_schema(
        pool,
        schema_name,
        |tx: &mut Transaction<'_, Postgres>| {
            Box::pin(async move {
                // Create the shutdown state table
                sqlx::query(
                    r#"
                CREATE TABLE IF NOT EXISTS shutdown_state (
                    id UUID NOT NULL PRIMARY KEY,
                    shutdown_date TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    pending_submissions JSONB NOT NULL DEFAULT '[]',
                    pending_batches JSONB NOT NULL DEFAULT '[]',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                "#,
                )
                .execute(&mut **tx)
                .await?;

                Ok(())
            })
        },
    )
    .await?)
}

/// [`MemorySubroutine`] persistence state.
#[derive(sqlx::FromRow, Serialize, Deserialize)]
pub struct SaveState {
    pub id: Uuid,
    pub shutdown_date: DateTime<Utc>,
    #[sqlx(json)]
    pub pending_submissions: Vec<(DateTime<Utc>, Prompt<'static>)>,
    #[sqlx(json)]
    pub pending_batches: Vec<batch::Pending<'static>>,
}

/// Save the shutdown state including any pending work
pub async fn save_state(
    pool: &PgPool,
    schema_name: &str,
    state: SaveState,
) -> Result<Uuid, MemorySubroutineError> {
    Ok(execute_with_schema(pool, schema_name, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            #[cfg(feature = "log")]
            log::info!("Saving {} shutdown state with ID: {}", stringify!(MemorySubroutine), state.id);

            sqlx::query(
                r#"
                INSERT INTO shutdown_state (id, shutdown_date, pending_submissions, pending_batches)
                VALUES ($1, $2, $3, $4)
                "#,
            )
            .bind(state.id)
            .bind(state.shutdown_date)
            .bind(sqlx::types::Json(state.pending_submissions))
            .bind(sqlx::types::Json(state.pending_batches))
            .execute(&mut **tx)
            .await?;

            Ok(state.id)
        })
    })
    .await?)
}

/// Load all shutdown states, processing them in order from oldest to newest
pub async fn load_states(
    pool: &PgPool,
    schema_name: &str,
) -> Result<Vec<SaveState>, MemorySubroutineError> {
    Ok(execute_with_schema(
        pool,
        schema_name,
        |tx: &mut Transaction<'_, Postgres>| {
            Box::pin(async move {
                let rows: Vec<SaveState> = sqlx::query_as(
                    r#"
                SELECT id, shutdown_date, pending_submissions, pending_batches
                FROM shutdown_state 
                ORDER BY shutdown_date ASC
                "#,
                )
                .fetch_all(&mut **tx)
                .await?;

                if rows.is_empty() {
                    return Ok(Vec::new());
                }

                #[cfg(feature = "log")]
                log::info!("Found {} shutdown states to process", rows.len());

                let ids_to_delete: Vec<Uuid> =
                    rows.iter().map(|state| state.id).collect();

                // Delete all loaded states in a single query
                sqlx::query("DELETE FROM shutdown_state WHERE id = ANY($1)")
                    .bind(&ids_to_delete)
                    .execute(&mut **tx)
                    .await?;

                #[cfg(feature = "log")]
                log::info!(
                    "Cleaned up {} processed shutdown states",
                    ids_to_delete.len()
                );

                Ok(rows)
            })
        },
    )
    .await?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        AnthropicModel,
        batch::{Meta, Pending, Stats, Status},
        prompt::message::{Message, Role},
    };
    use std::num::NonZeroU16;

    async fn create_test_pool() -> PgPool {
        // Use DATABASE_URL from environment or a test database
        let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| {
            "postgres://postgres:password@localhost/test_db".to_string()
        });

        PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database")
    }

    fn create_test_prompt() -> Prompt<'static> {
        Prompt::default()
            .model(AnthropicModel::Haiku30)
            .max_tokens(NonZeroU16::new(1024).unwrap())
            .add_message(Message::from((Role::User, "Hello, test!")))
            .unwrap()
    }

    fn create_test_batch() -> batch::Pending<'static> {
        let prompts = vec![
            create_test_prompt(),
            create_test_prompt()
                .add_message(Message::from((Role::Assistant, "Hi there!")))
                .unwrap(),
        ];

        let prompts = prompts.into_iter().collect();

        let meta = Meta {
            id: "test_batch_123".to_string(),
            status: Status::InProgress,
            stats: Stats {
                processing: 2,
                succeeded: 0,
                errored: 0,
                canceled: 0,
                expired: 0,
            },
            created_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(24),
            ended_at: None,
            cancel_initiated_at: None,
            archived_at: None,
            results_url: None,
        };

        Pending { prompts, meta }
    }

    #[tokio::test]
    async fn test_ensure_initialized() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_ensure_initialized);

        // Should not error even if called multiple times
        ensure_initialized(&pool, schema_name).await.unwrap();
        ensure_initialized(&pool, schema_name).await.unwrap();

        // Verify table exists by attempting to query it
        let result = sqlx::query(&format!(
            "SELECT COUNT(*) FROM {}.shutdown_state",
            schema_name
        ))
        .fetch_one(&pool)
        .await;

        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_save_and_load_empty_state() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_save_and_load_empty_state);

        ensure_initialized(&pool, schema_name).await.unwrap();

        let id = Uuid::new_v4();
        let state = SaveState {
            id,
            shutdown_date: Utc::now(),
            pending_submissions: vec![],
            pending_batches: vec![],
        };

        let saved_id = save_state(&pool, schema_name, state).await.unwrap();
        assert_eq!(saved_id, id);

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 1);

        let loaded = &loaded_states[0];
        assert_eq!(loaded.id, id);
        assert!(loaded.pending_submissions.is_empty());
        assert!(loaded.pending_batches.is_empty());

        // Verify states were deleted after loading
        let reloaded = load_states(&pool, schema_name).await.unwrap();
        assert!(reloaded.is_empty());
    }

    #[tokio::test]
    async fn test_save_and_load_with_submissions() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_save_and_load_with_submissions);

        ensure_initialized(&pool, schema_name).await.unwrap();

        let submissions = vec![
            (Utc::now(), create_test_prompt()),
            (
                Utc::now() + chrono::Duration::minutes(5),
                create_test_prompt(),
            ),
        ];

        let state = SaveState {
            id: Uuid::new_v4(),
            shutdown_date: Utc::now(),
            pending_submissions: submissions.clone(),
            pending_batches: vec![],
        };

        save_state(&pool, schema_name, state).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 1);

        let loaded = &loaded_states[0];
        assert_eq!(loaded.pending_submissions.len(), 2);

        // Verify prompt content survived serialization
        assert_eq!(
            loaded.pending_submissions[0].1.messages[0]
                .content
                .to_string(),
            "Hello, test!"
        );
    }

    #[tokio::test]
    async fn test_save_and_load_with_batches() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_save_and_load_with_batches);

        ensure_initialized(&pool, schema_name).await.unwrap();

        let batches = vec![create_test_batch(), create_test_batch()];

        let state = SaveState {
            id: Uuid::new_v4(),
            shutdown_date: Utc::now(),
            pending_submissions: vec![],
            pending_batches: batches,
        };

        save_state(&pool, schema_name, state).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 1);

        let loaded = &loaded_states[0];
        assert_eq!(loaded.pending_batches.len(), 2);
    }

    #[tokio::test]
    async fn test_save_and_load_complex_state() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_save_and_load_complex_state);

        ensure_initialized(&pool, schema_name).await.unwrap();

        let submissions = vec![
            (Utc::now(), create_test_prompt()),
            (
                Utc::now() + chrono::Duration::minutes(10),
                create_test_prompt(),
            ),
        ];

        let batches = vec![create_test_batch()];

        let id = Uuid::new_v4();
        let state = SaveState {
            id,
            shutdown_date: Utc::now(),
            pending_submissions: submissions,
            pending_batches: batches,
        };

        save_state(&pool, schema_name, state).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 1);

        let loaded = &loaded_states[0];
        assert_eq!(loaded.id, id);
        assert_eq!(loaded.pending_submissions.len(), 2);
        assert_eq!(loaded.pending_batches.len(), 1);
    }

    #[tokio::test]
    async fn test_multiple_states_ordered_by_date() {
        let pool = create_test_pool().await;
        let schema_name = stringify!(test_multiple_states_ordered_by_date);

        ensure_initialized(&pool, schema_name).await.unwrap();

        let base_time = Utc::now();

        // Save states with different shutdown dates
        let id1 = Uuid::new_v4();
        let state1 = SaveState {
            id: id1,
            shutdown_date: base_time - chrono::Duration::hours(2),
            pending_submissions: vec![(base_time, create_test_prompt())],
            pending_batches: vec![],
        };

        let id2 = Uuid::new_v4();
        let state2 = SaveState {
            id: id2,
            shutdown_date: base_time - chrono::Duration::hours(1),
            pending_submissions: vec![],
            pending_batches: vec![create_test_batch()],
        };

        let id3 = Uuid::new_v4();
        let state3 = SaveState {
            id: id3,
            shutdown_date: base_time,
            pending_submissions: vec![(base_time, create_test_prompt())],
            pending_batches: vec![create_test_batch()],
        };

        save_state(&pool, schema_name, state1).await.unwrap();
        save_state(&pool, schema_name, state2).await.unwrap();
        save_state(&pool, schema_name, state3).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 3);

        // Verify they are ordered by shutdown_date ASC (oldest first)
        assert_eq!(loaded_states[0].id, id1);
        assert_eq!(loaded_states[1].id, id2);
        assert_eq!(loaded_states[2].id, id3);

        // Verify all were deleted
        let reloaded = load_states(&pool, schema_name).await.unwrap();
        assert!(reloaded.is_empty());
    }

    #[tokio::test]
    async fn test_load_state_empty_table() {
        let pool = create_test_pool().await;
        let schema_name = "test_empty_table";

        ensure_initialized(&pool, schema_name).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert!(loaded_states.is_empty());
    }

    #[tokio::test]
    async fn test_prompt_with_special_characters() {
        let pool = create_test_pool().await;
        let schema_name = "test_special_chars";

        ensure_initialized(&pool, schema_name).await.unwrap();

        let prompt = Prompt::default()
            .add_message(Message::from((
                Role::User,
                "Test with 'quotes' and \"double quotes\" and \\ backslash",
            )))
            .unwrap();

        let state = SaveState {
            id: Uuid::new_v4(),
            shutdown_date: Utc::now(),
            pending_submissions: vec![(Utc::now(), prompt)],
            pending_batches: vec![],
        };

        save_state(&pool, schema_name, state).await.unwrap();

        let loaded_states = load_states(&pool, schema_name).await.unwrap();
        assert_eq!(loaded_states.len(), 1);

        let content = loaded_states[0].pending_submissions[0].1.messages[0]
            .content
            .to_string();
        assert!(content.contains("'quotes'"));
        assert!(content.contains("\"double quotes\""));
        assert!(content.contains("\\ backslash"));
    }
}
