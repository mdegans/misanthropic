use crate::tool::memory_palace::{
    MemoryPalaceError, PgPool, Postgres, Transaction, db::execute_with_schema,
    models::*, queries::*,
};
use sqlx::Row;

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

pub async fn store(
    pool: &PgPool,
    schema: &str,
    room: String,
    content: String,
    tags_json: serde_json::Value,
) -> Result<i64, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // Ensure room exists
            sqlx::query(INSERT_ROOM)
                .bind(&room)
                .bind(format!("Room for {}", room))
                .execute(&mut **tx)
                .await?;

            let row = sqlx::query(INSERT_MEMORY_RETURNING_ID)
                .bind(&content)
                .bind(&room)
                .bind(&tags_json)
                .fetch_one(&mut **tx)
                .await?;

            Ok(row.get("id"))
        })
    })
    .await
}

pub async fn search(
    pool: &PgPool,
    schema: &str,
    query: &str,
) -> Result<Vec<(String, String, Memory)>, MemoryPalaceError> {
    let pattern = format!("%{}%", query.trim());
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // relevance
            let rel = sqlx::query_as::<_, Memory>(SEARCH_RELEVANCE)
                .bind(&pattern)
                .fetch_all(&mut **tx)
                .await?;
            // recency
            let rec = sqlx::query_as::<_, Memory>(SEARCH_RECENCY)
                .bind(&pattern)
                .fetch_all(&mut **tx)
                .await?;
            // relationships
            let rels = sqlx::query_as::<_, Memory>(SEARCH_RELATIONSHIPS)
                .bind(&pattern)
                .fetch_all(&mut **tx)
                .await?;

            // Combine and score all memories
            let mut scored_memories = std::collections::HashMap::new();
            let now = chrono::Utc::now();

            // Score relevance memories
            for (i, memory) in rel.into_iter().enumerate() {
                let relevance_score = (10 - i) as f64 / 10.0; // 1.0 to 0.1
                let recency_score =
                    calculate_recency_score(&memory.last_updated, &now);

                scored_memories.insert(
                    memory.id,
                    ScoredMemory {
                        room: memory.room.clone(),
                        memory,
                        relevance_score,
                        recency_score,
                        relationship_score: 0.0,
                        final_score: 0.0, // Will calculate after
                    },
                );
            }

            // Boost recency scores for recent memories
            for (i, memory) in rec.into_iter().enumerate() {
                let recency_boost = (10 - i) as f64 / 10.0;
                let recency_score =
                    calculate_recency_score(&memory.last_updated, &now);

                scored_memories
                    .entry(memory.id)
                    .and_modify(|sm| {
                        sm.recency_score = f64::max(
                            sm.recency_score,
                            recency_score + recency_boost * 0.3,
                        )
                    })
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
            for (i, memory) in rels.into_iter().enumerate() {
                let relationship_score = (10 - i) as f64 / 10.0;

                scored_memories
                    .entry(memory.id)
                    .and_modify(|sm| {
                        sm.relationship_score =
                            f64::max(sm.relationship_score, relationship_score)
                    })
                    .or_insert_with(|| ScoredMemory {
                        room: memory.room.clone(),
                        recency_score: calculate_recency_score(
                            &memory.last_updated,
                            &now,
                        ),
                        memory,
                        relevance_score: 0.0,
                        relationship_score,
                        final_score: 0.0,
                    });
            }

            // Calculate final scores with weighted combination
            let mut final_memories: Vec<_> = scored_memories
                .into_values()
                .map(|mut sm| {
                    // Weighted combination: 50% relevance, 30% recency, 20% relationships
                    sm.final_score = sm.relevance_score * 0.5
                        + sm.recency_score * 0.3
                        + sm.relationship_score * 0.2;
                    sm
                })
                .collect();

            // Sort by final score and take top results
            final_memories.sort_by(|a, b| {
                b.final_score
                    .partial_cmp(&a.final_score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
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

pub async fn find_memories_bfs(
    pool: &PgPool,
    schema: &str,
    start_memory_id: i64,
    max_distance: u32,
    decay_factor: f64,
    min_score: f64,
) -> Result<Vec<(String, String, Memory, f64, i32)>, MemoryPalaceError> {
    execute_with_schema(&pool, &schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let rows: Vec<BfsMemory> = sqlx::query_as(FIND_MEMORIES_BFS)
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
    })
    .await
}

pub async fn connect_rooms(
    pool: &PgPool,
    schema: &str,
    room1: String,
    room2: String,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
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

pub async fn list_rooms(
    pool: &PgPool,
    schema: &str,
) -> Result<Vec<(String, String, usize, Vec<String>)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
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

                let connection_names: Vec<String> =
                    connections.into_iter().map(|conn| conn.to_room).collect();

                results.push((
                    room.name,
                    room.description,
                    room.memory_count as usize,
                    connection_names,
                ));
            }

            Ok(results)
        })
    })
    .await
}

pub async fn relate_memories(
    pool: &PgPool,
    schema: &str,
    memory_id1: i64,
    memory_id2: i64,
    relationship_type: String,
    strength: f64,
) -> Result<String, MemoryPalaceError> {
    execute_with_schema(
        pool,
        schema,
        |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let okmsg = format!(
                "Created {} relationship between {} and {} with strength {}",
                relationship_type, memory_id1, memory_id2, strength
            );

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

            Ok(okmsg)
        })
    }).await
}

pub async fn find_related_memories(
    pool: &PgPool,
    schema: &str,
    memory_id: i64,
    max_depth: u32,
    min_strength: f64,
) -> Result<Vec<(String, String, Memory, String, f64)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let rows: Vec<RelatedMemory> =
                sqlx::query_as(FIND_RELATED_MEMORIES)
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
    })
    .await
}

pub async fn extract_concepts(
    pool: &PgPool,
    schema: &str,
    memory_id: i64,
    concepts: Vec<String>,
) -> Result<String, MemoryPalaceError> {
    let created_concepts = execute_with_schema(
        pool,
        schema,
        |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let mut created = Vec::new();

            for concept in &concepts {
                // Create or get concept
                let concept_row = sqlx::query(
                    r#"
                    INSERT INTO concepts (name) VALUES ($1)
                    ON CONFLICT (name) DO NOTHING
                    RETURNING id
                "#,
                )
                .bind(concept)
                .fetch_optional(&mut **tx)
                .await?;

                let concept_id: i64 = if let Some(row) = concept_row {
                    row.get("id")
                } else {
                    // Concept already exists, get its ID
                    sqlx::query("SELECT id FROM concepts WHERE name = $1")
                        .bind(concept)
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

                created.push(concept.clone());
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

pub async fn find_memories_by_concept(
    pool: &PgPool,
    schema: &str,
    concept: String,
) -> Result<Vec<(String, String, Memory, f64)>, MemoryPalaceError> {
    execute_with_schema(
        pool,
        schema,
        |tx: &mut Transaction<Postgres>| {
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
            .bind(&concept)
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

pub async fn get_graph_stats(
    pool: &PgPool,
    schema: &str,
) -> Result<String, MemoryPalaceError> {
    let stats: GraphStats = execute_with_schema(
        pool,
        schema,
        |tx: &mut Transaction<Postgres>| {
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

pub async fn get_context_summary(
    pool: &PgPool,
    schema: &str,
) -> Result<String, MemoryPalaceError> {
    let (recent_memories, top_relationships) =
        execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
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
        })
        .await?;

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
