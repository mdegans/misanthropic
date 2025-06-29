use std::collections::BTreeSet;

use chrono::{DateTime, Utc};

// Copyright 2025 Claude 4 Opus, Claude 4 Sonnet, and Michael de Gans
use crate::{tool::{embedding::TextEmbedding, memory_palace::{
    db::execute_with_schema, models::*, MemoryPalaceError, PgPool, Postgres, Transaction
}}, Prompt};

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
    placement: String,
    keywords: Vec<String>,
) -> Result<i64, MemoryPalaceError> {
    let keywords = serde_json::to_value(&keywords)?;

    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // Ensure room exists
            sqlx::query(
                "INSERT INTO rooms (name, description) VALUES ($1, $2) ON CONFLICT (name) DO NOTHING",
            )
            .bind(&room)
            .bind(format!("Room for {}", room))
            .execute(&mut **tx)
            .await?;

            #[derive(sqlx::FromRow)]
            struct IdRow {
                id: i64,
            }

            let row: IdRow = sqlx::query_as(
                "INSERT INTO memories (content, placement, room, tags) VALUES ($1, $2, $3, $4) RETURNING id",
            )
            .bind(content)
            .bind(placement)
            .bind(room)
            .bind(keywords)
            .fetch_one(&mut **tx)
            .await?;

            Ok(row.id)
        })
    })
    .await
}

pub async fn connect_rooms(
    pool: &PgPool,
    schema: &str,
    room1: String,
    room2: String,
    passage_type: String,
    description: Option<String>,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            // Ensure consistent ordering: smaller name first
            let (from_room, to_room) = if room1 < room2 {
                (room1, room2)
            } else {
                (room2, room1)
            };
            
            // Insert or update the connection
            sqlx::query(
                r#"
                INSERT INTO room_connections (from_room, to_room, passage_type, description, strength) 
                VALUES ($1, $2, $3, $4, 1)
                ON CONFLICT (from_room, to_room) 
                DO UPDATE SET 
                    strength = room_connections.strength + 1,
                    passage_type = EXCLUDED.passage_type,
                    description = COALESCE(EXCLUDED.description, room_connections.description)
                "#,
            )
            .bind(&from_room)
            .bind(&to_room)
            .bind(&passage_type)
            .bind(&description)
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

            sqlx::query(
                r#"
                INSERT INTO memory_relationships (from_memory_id, to_memory_id, relationship_type, strength)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (from_memory_id, to_memory_id) 
                DO UPDATE SET relationship_type = $3, strength = $4
                "#,
            )
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
                        
pub async fn find_resonating_memories(
    pool: &PgPool,
    schema: &str,
    memory_id: i64,
    max_depth: u32,
    min_strength: f64,
) -> Result<Vec<(String, String, Memory, String, f64)>, MemoryPalaceError>
{
    // Early return for invalid inputs
    if max_depth == 0 {
        return Ok(vec![]);
    }
    
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // First, get the source memory to validate it exists
            let source: Memory = sqlx::query_as(
                "SELECT * FROM memories WHERE id = $1"
            )
            .bind(memory_id)
            .fetch_optional(&mut **tx)
            .await?
            .ok_or_else(|| MemoryPalaceError::MemoryNotFound(MemoryId(memory_id)))?;
            
            // Use the recursive CTE with proper depth limiting
            #[derive(sqlx::FromRow)]
            struct RelatedMemoryRow {
                room_name: String,
                memory_id: i64,
                #[sqlx(json)]
                content: MemoryContent,
                room_id: i64,
                placement: String,
                placement_description: Option<String>,
                #[sqlx(json)]
                tags: Vec<String>,
                embedding: Option<Vec<f32>>,
                importance: f32,
                access_count: i32,
                created_at: DateTime<Utc>,
                last_updated: DateTime<Utc>,
                last_accessed: DateTime<Utc>,
                relationship_type: String,
                strength: f64,
                depth: i32,
            }
            
            let rows: Vec<RelatedMemoryRow> = sqlx::query_as(
                r#"
                WITH RECURSIVE related_memories AS (
                    -- Base case: direct relationships
                    SELECT 
                        m.*, 
                        r.name as room_name,
                        mr.relationship_type,
                        mr.strength,
                        1 as depth,
                        ARRAY[m.id] as visited
                    FROM memory_relationships mr
                    JOIN memories m ON (
                        CASE 
                            WHEN mr.from_memory_id = $1 THEN mr.to_memory_id 
                            ELSE mr.from_memory_id 
                        END = m.id
                    )
                    JOIN rooms r ON m.room_id = r.id
                    WHERE (mr.from_memory_id = $1 OR mr.to_memory_id = $1)
                        AND mr.strength >= $3
                    
                    UNION ALL
                    
                    -- Recursive case: indirect relationships
                    SELECT 
                        m.*, 
                        r.name as room_name,
                        mr.relationship_type,
                        rm.strength * mr.strength as strength,
                        rm.depth + 1 as depth,
                        rm.visited || m.id as visited
                    FROM related_memories rm
                    JOIN memory_relationships mr ON (
                        (mr.from_memory_id = rm.id OR mr.to_memory_id = rm.id)
                        AND mr.from_memory_id != ALL(rm.visited)
                        AND mr.to_memory_id != ALL(rm.visited)
                    )
                    JOIN memories m ON (
                        CASE 
                            WHEN mr.from_memory_id = rm.id THEN mr.to_memory_id 
                            ELSE mr.from_memory_id 
                        END = m.id
                    )
                    JOIN rooms r ON m.room_id = r.id
                    WHERE rm.depth < $2
                        AND rm.strength * mr.strength >= $3
                        AND NOT (m.id = ANY(rm.visited))
                )
                SELECT DISTINCT ON (id) 
                    room_name, id, content, room_id, placement, placement_description,
                    tags, embedding, importance, access_count, created_at, last_updated,
                    last_accessed, relationship_type, strength, depth
                FROM related_memories
                ORDER BY id, strength DESC, depth ASC
                LIMIT 50  -- Hard limit to prevent runaway queries
                "#,
            )
            .bind(memory_id)
            .bind(max_depth as i32)
            .bind(min_strength)
            .fetch_all(&mut **tx)
            .await?;
            
            // Convert to expected format
            let results: Vec<(String, String, Memory, String, f64)> = rows
                .into_iter()
                .map(|row| {
                    let memory_row = Memory {
                        id: MemoryId(row.memory_id),
                        content: row.content,
                        room_id: RoomId(row.room_id),
                        placement: row.placement,
                        placement_description: row.placement_description,
                        tags: row.tags,
                        importance: row.importance,
                        access_count: row.access_count,
                        created_at: row.created_at,
                        last_updated: row.last_updated,
                        last_accessed: row.last_accessed,
                    };
                    
                    (
                        row.room_name,
                        memory_row.id.0.to_string(),
                        memory_row,
                        row.relationship_type,
                        row.strength,
                    )
                })
                .collect();
            
            Ok(results)
        })
    })
    .await
}

/// Get rooms within N hops of current room with safety limits
pub async fn get_rooms_within_radius(
    pool: &PgPool,
    schema: &str,
    start_room: &str,
    radius: u32,
) -> Result<Vec<RoomWithJourney>, MemoryPalaceError> {
    // Validate radius
    if radius == 0 {
        return Ok(vec![]);
    }
    
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // First verify the starting room exists
            let start_room_id: Option<RoomId> = sqlx::query_scalar(
                "SELECT id FROM rooms WHERE name = $1"
            )
            .bind(start_room)
            .fetch_optional(&mut **tx)
            .await?;
            
            let start_room_id = start_room_id
                .ok_or_else(|| MemoryPalaceError::RoomNotFound(start_room.to_string()))?;
            
            #[derive(sqlx::FromRow)]
            struct RoomDistanceRow {
                id: i64,
                name: String,
                description: String,
                atmosphere: Option<String>,
                created_at: DateTime<Utc>,
                last_visited: DateTime<Utc>,
                visit_count: i32,
                memory_count: i32,
                distance: i32,
                path: Vec<i64>,
            }
            
            let rows: BTreeSet<RoomDistanceRow> = sqlx::query_as(
                r#"
                WITH RECURSIVE room_graph AS (
                    -- Start room
                    SELECT 
                        r.*,
                        0 as distance,
                        ARRAY[r.id] as path
                    FROM rooms r
                    WHERE r.id = $1
                    
                    UNION ALL
                    
                    -- Connected rooms
                    SELECT 
                        r.*,
                        rg.distance + 1 as distance,
                        rg.path || r.id as path
                    FROM room_graph rg
                    JOIN room_connections rc ON (
                        (rc.from_room_id = rg.id AND rc.to_room_id = r.id) OR
                        (rc.to_room_id = rg.id AND rc.from_room_id = r.id)
                    )
                    JOIN rooms r ON r.id = CASE 
                        WHEN rc.from_room_id = rg.id THEN rc.to_room_id
                        ELSE rc.from_room_id
                    END
                    WHERE rg.distance < $2
                        AND NOT (r.id = ANY(rg.path))  -- Prevent cycles
                )
                SELECT DISTINCT ON (id) *
                FROM room_graph
                WHERE id != $1
                ORDER BY id, distance ASC
                LIMIT 100  -- Hard limit
                "#
            )
            .bind(start_room_id)
            .bind(radius as i32)
            .fetch_all(&mut **tx)
            .await?;
            
            let results: Vec<RoomWithJourney> = rows
                .into_iter()
                .map(|row| RoomWithJourney {
                    room: Room {
                        id: RoomId(row.id),
                        name: row.name,
                        description: row.description,
                        atmosphere: row.atmosphere,
                        centroid: row.centroid_embedding,
                        created_at: row.created_at,
                        last_visited: row.last_visited,
                        visit_count: row.visit_count,
                        memory_count: row.memory_count,
                    },
                    distance: row.distance as u32,
                    path: row.path.into_iter().map(RoomId).collect(),
                })
                .collect();
            
            Ok(results)
        })
    })
    .await
}

pub async fn get_context_summary(
    pool: &PgPool,
    schema: &str,
) -> Result<String, MemoryPalaceError> {
    let (recent_memories, top_relationships) =
        execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
            Box::pin(async move {
                // Get recent memories based on last_updated (more relevant for agents)
                let recent_memories: Vec<Memory> = sqlx::query_as(
                    r#"
                SELECT *
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
