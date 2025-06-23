use std::collections::HashMap;

// Copyright 2025 Claude 4 Opus, Claude 4 Sonnet, and Michael de Gans
use crate::tool::{embedding::TextEmbedding, memory_palace::{
    db::execute_with_schema, models::*, MemoryPalaceError, PgPool, Postgres, Transaction
}};

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

pub async fn search(
    pool: &PgPool,
    schema: &str,
    query: &str,
) -> Result<Vec<(String, String, Memory)>, MemoryPalaceError> {
    let pattern = format!("%{}%", query.trim());
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // relevance - using runtime query_as to work with #[sqlx(json)]
            let rel = sqlx::query_as::<_, Memory>(
                r#"
                SELECT id, content, room, tags, created_at, last_updated
                FROM memories 
                WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
                ORDER BY 
                    CASE 
                        WHEN content ILIKE $1 THEN 3
                        WHEN room ILIKE $1 THEN 2
                        WHEN tags::text ILIKE $1 THEN 1
                        ELSE 0
                    END DESC,
                    created_at DESC
                LIMIT 10
                "#,
            )
            .bind(&pattern)
            .fetch_all(&mut **tx)
            .await?;

            // recency
            let rec = sqlx::query_as::<_, Memory>(
                r#"
                SELECT id, content, room, tags, created_at, last_updated
                FROM memories 
                WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
                ORDER BY last_updated DESC
                LIMIT 10
                "#,
            )
            .bind(&pattern)
            .fetch_all(&mut **tx)
            .await?;

            // relationships
            let rels = sqlx::query_as::<_, Memory>(
                r#"
                WITH related_memories AS (
                    SELECT DISTINCT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated
                    FROM memories m
                    JOIN memory_relationships mr ON (m.id = mr.from_memory_id OR m.id = mr.to_memory_id)
                    JOIN memories search_mem ON (
                        (mr.from_memory_id = search_mem.id AND search_mem.content ILIKE $1) OR
                        (mr.to_memory_id = search_mem.id AND search_mem.content ILIKE $1)
                    )
                    WHERE m.content ILIKE $1 OR m.room ILIKE $1 OR m.tags::text ILIKE $1
                )
                SELECT id, content, room, tags, created_at, last_updated
                FROM related_memories
                ORDER BY created_at DESC
                LIMIT 10
                "#,
            )
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
                        final_score: 0.0 // Will calculate after
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
    max_depth: u32,  // Renamed from max_distance
    decay_factor: f64,
    min_score: f64,
) -> Result<Vec<(String, String, MemoryRow, f64, i32)>, MemoryPalaceError> {
    // Early return for invalid inputs
    if max_depth == 0 {
        return Ok(vec![]);
    }
    
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // First verify the starting memory exists
            let _: MemoryId = sqlx::query_scalar(
                "SELECT id FROM memories WHERE id = $1"
            )
            .bind(start_memory_id)
            .fetch_optional(&mut **tx)
            .await?
            .ok_or_else(|| MemoryPalaceError::MemoryNotFound(MemoryId(start_memory_id)))?;
            
            #[derive(sqlx::FromRow)]
            struct BfsRow {
                id: i64,
                #[sqlx(json)]
                content: Memory,
                room_id: i64,
                placement: String,
                placement_description: Option<String>,
                #[sqlx(json)]
                tags: Vec<String>,
                embedding: Option<pgvector::Vector>,
                importance: f32,
                access_count: i32,
                created_at: chrono::DateTime<chrono::Utc>,
                last_updated: chrono::DateTime<chrono::Utc>,
                last_accessed: chrono::DateTime<chrono::Utc>,
                path_strength: Option<f64>,
                depth: Option<i32>,  // Renamed from distance
                room_name: String,
            }

            let rows: Vec<BfsRow> = sqlx::query_as(
                r#"
                WITH RECURSIVE memory_bfs AS (
                    -- Base case: start with the given memory
                    SELECT 
                        $1::bigint as memory_id,
                        1.0::double precision as path_strength,
                        0 as depth,
                        ARRAY[$1::bigint] as path
                    
                    UNION ALL
                    
                    -- Recursive case: explore relationships
                    SELECT 
                        CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END as memory_id,
                        mb.path_strength * mr.strength * $3 as path_strength,
                        mb.depth + 1 as depth,
                        mb.path || CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END as path
                    FROM memory_bfs mb
                    JOIN memory_relationships mr ON (
                        mr.from_memory_id = mb.memory_id OR mr.to_memory_id = mb.memory_id
                    )
                    WHERE 
                        mb.depth < $2 
                        AND mb.path_strength * mr.strength * $3 >= $4
                        AND NOT (CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END = ANY(mb.path))
                )
                SELECT DISTINCT ON (mb.memory_id)
                    m.*, 
                    r.name as room_name,
                    mb.path_strength, 
                    mb.depth
                FROM memory_bfs mb
                JOIN memories m ON mb.memory_id = m.id
                JOIN rooms r ON m.room_id = r.id
                WHERE mb.memory_id != $1
                ORDER BY mb.memory_id, mb.path_strength DESC, mb.depth ASC
                LIMIT 50  -- Hard limit to prevent runaway queries
                "#,
            )
            .bind(start_memory_id)
            .bind(max_depth as i32)
            .bind(decay_factor)
            .bind(min_score)
            .fetch_all(&mut **tx)
            .await?;

            let mut results = Vec::new();
            for row in rows {
                let memory_row = MemoryRow {
                    id: MemoryId(row.id),
                    content: row.content,
                    room_id: RoomId(row.room_id),
                    placement: row.placement,
                    placement_description: row.placement_description,
                    tags: row.tags,
                    embedding: row.embedding,
                    importance: row.importance,
                    access_count: row.access_count,
                    created_at: row.created_at,
                    last_updated: row.last_updated,
                    last_accessed: row.last_accessed,
                };

                results.push((
                    row.room_name,
                    memory_row.id.0.to_string(),
                    memory_row,
                    row.path_strength.unwrap_or(0.0),
                    row.depth.unwrap_or(0),
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
) -> Result<Vec<(String, String, MemoryRow, String, f64)>, MemoryPalaceError>
{
    // Early return for invalid inputs
    if max_depth == 0 {
        return Ok(vec![]);
    }
    
    execute_with_schema(pool, schema, |tx: &mut Transaction<'_, Postgres>| {
        Box::pin(async move {
            // First, get the source memory to validate it exists
            let source: MemoryRow = sqlx::query_as(
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
                content: Memory,
                room_id: i64,
                placement: String,
                placement_description: Option<String>,
                #[sqlx(json)]
                tags: Vec<String>,
                embedding: Option<pgvector::Vector>,
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
            let results: Vec<(String, String, MemoryRow, String, f64)> = rows
                .into_iter()
                .map(|row| {
                    let memory_row = MemoryRow {
                        id: MemoryId(row.memory_id),
                        content: row.content,
                        room_id: RoomId(row.room_id),
                        placement: row.placement,
                        placement_description: row.placement_description,
                        tags: row.tags,
                        embedding: row.embedding,
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
) -> Result<Vec<RoomWithDistance>, MemoryPalaceError> {
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
                centroid_embedding: Option<pgvector::Vector>,
                created_at: DateTime<Utc>,
                last_visited: DateTime<Utc>,
                visit_count: i32,
                memory_count: i32,
                distance: i32,
                path: Vec<i64>,
            }
            
            let rows: Vec<RoomDistanceRow> = sqlx::query_as(
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
            
            let results: Vec<RoomWithDistance> = rows
                .into_iter()
                .map(|row| RoomWithDistance {
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

/// Get a hint about what kind of memories are in a room
pub async fn get_room_character_hint(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
) -> Result<String, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            // Get top tags from the room
            let top_tags: Vec<(String, i64)> = sqlx::query_as(
                r#"
                SELECT tag, COUNT(*) as count
                FROM memories, jsonb_array_elements_text(tags) as tag
                WHERE room = $1
                GROUP BY tag
                ORDER BY count DESC
                LIMIT 3
                "#
            )
            .bind(room_name)
            .fetch_all(&mut **tx)
            .await?;
            
            if top_tags.is_empty() {
                Ok("empty and waiting".to_string())
            } else {
                let tags: Vec<String> = top_tags.into_iter()
                    .map(|(tag, _)| tag)
                    .collect();
                Ok(format!("{} memories dominate", tags.join(", ")))
            }
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
                #[derive(sqlx::FromRow)]
                struct ConceptRow {
                    id: i64,
                }

                let concept_row: Option<ConceptRow> = sqlx::query_as(
                    "SELECT id FROM concepts WHERE name = $1"
                )
                .bind(concept)
                .fetch_optional(&mut **tx)
                .await?;

                let concept_id: i64 = if let Some(row) = concept_row {
                    row.id
                } else {
                    let new_row: ConceptRow = sqlx::query_as(
                        "INSERT INTO concepts (name) VALUES ($1) RETURNING id"
                    )
                    .bind(concept)
                    .fetch_one(&mut **tx)
                    .await?;
                    new_row.id
                };

                // Link memory to concept
                sqlx::query(
                    r#"
                    INSERT INTO memory_concepts (memory_id, concept_id, confidence)
                    VALUES ($1, $2, 1.0)
                    ON CONFLICT (memory_id, concept_id) DO NOTHING
                    "#
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
            #[derive(sqlx::FromRow)]
            struct ConceptMemoryRow {
                id: i64,
                content: String,
                room: String,
                tags: serde_json::Value,
                created_at: chrono::DateTime<chrono::Utc>,
                last_updated: chrono::DateTime<chrono::Utc>,
                confidence: f64,
            }

            let rows: Vec<ConceptMemoryRow> = sqlx::query_as(
                r#"
                SELECT 
                    m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                    mc.confidence
                FROM memory_concepts mc
                JOIN memories m ON mc.memory_id = m.id
                JOIN concepts c ON mc.concept_id = c.id
                WHERE c.name = $1
                ORDER BY 
                    mc.confidence DESC,
                    m.created_at DESC
            "#,
            )
            .bind(&concept)
            .fetch_all(&mut **tx)
            .await?;

            let mut results = Vec::new();
            for row in rows {
                let tags: Vec<String> =
                    serde_json::from_value(row.tags.clone()).unwrap_or_default();

                let memory = Memory {
                    id: row.id,
                    content: row.content,
                    room: row.room.clone(),
                    tags,
                    created_at: row.created_at,
                    last_updated: row.last_updated,
                    placement: todo!(),
                    placement_description: todo!(),
                    embedding: todo!(),
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
    let stats = execute_with_schema(
        pool,
        schema,
        |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            #[derive(sqlx::FromRow)]
            struct StatsRow {
                total_memories: Option<i64>,
                total_rooms: Option<i64>,
                total_relationships: Option<i64>,
                total_concepts: Option<i64>,
                total_mentions: Option<i64>,
            }

            let stats: StatsRow = sqlx::query_as(
                r#"
                SELECT 
                    (SELECT COUNT(*) FROM memories) as total_memories,
                    (SELECT COUNT(*) FROM rooms) as total_rooms,
                    (SELECT COUNT(*) FROM memory_relationships) as total_relationships,
                    (SELECT COUNT(*) FROM concepts) as total_concepts,
                    (SELECT COUNT(*) FROM memory_concepts) as total_mentions
                "#
            )
            .fetch_one(&mut **tx)
            .await?;

            Ok(stats)
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
        stats.total_memories.unwrap_or(0),
        stats.total_rooms.unwrap_or(0),
        stats.total_relationships.unwrap_or(0),
        stats.total_concepts.unwrap_or(0),
        stats.total_mentions.unwrap_or(0),
        if stats.total_memories.unwrap_or(0) > 0 {
            stats.total_relationships.unwrap_or(0) as f64
                / stats.total_memories.unwrap_or(1) as f64
        } else {
            0.0
        },
        if stats.total_memories.unwrap_or(0) > 0 {
            stats.total_mentions.unwrap_or(0) as f64
                / stats.total_memories.unwrap_or(1) as f64
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

/// Search for prompts by content or metadata
pub async fn search_prompts(
    pool: &PgPool,
    schema: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<(PromptId, DateTime<Utc>)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let pattern = format!("%{}%", query);
            
            let results: Vec<(PromptId, DateTime<Utc>)> = sqlx::query_as(
                r#"
                SELECT id, created_at
                FROM prompts
                WHERE content::text ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2
                "#
            )
            .bind(&pattern)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?;

            Ok(results)
        })
    })
    .await
}

/// Get prompts within a time range
pub async fn get_prompts_in_range(
    pool: &PgPool,
    schema: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Vec<(PromptId, Prompt<'static>)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let results: Vec<(PromptId, serde_json::Value)> = sqlx::query_as(
                r#"
                SELECT id, content
                FROM prompts
                WHERE created_at >= $1 AND created_at <= $2
                ORDER BY created_at DESC
                "#
            )
            .bind(start)
            .bind(end)
            .fetch_all(&mut **tx)
            .await?;

            let prompts = results
                .into_iter()
                .map(|(id, value)| {
                    let prompt: Prompt<'static> = serde_json::from_value(value)?;
                    Ok((id, prompt))
                })
                .collect::<Result<Vec<_>, serde_json::Error>>()?;

            Ok(prompts)
        })
    })
    .await
}

/// Find similar prompts based on embeddings
pub async fn find_similar_prompts(
    pool: &PgPool,
    schema: &str,
    prompt_id: PromptId,
    limit: usize,
) -> Result<Vec<(PromptId, f32)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            // First get the embedding of the target prompt
            let embedding: Option<pgvector::Vector> = sqlx::query_scalar(
                "SELECT embedding FROM prompts WHERE id = $1"
            )
            .bind(prompt_id)
            .fetch_optional(&mut **tx)
            .await?;

            if let Some(embedding) = embedding {
                let results: Vec<(PromptId, f32)> = sqlx::query_as(
                    r#"
                    SELECT 
                        p.id,
                        1 - (p.embedding <=> $1) as similarity
                    FROM prompts p
                    WHERE p.id != $2 AND p.embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#
                )
                .bind(&embedding)
                .bind(prompt_id)
                .bind(limit as i64)
                .fetch_all(&mut **tx)
                .await?;

                Ok(results)
            } else {
                Ok(vec![])
            }
        })
    })
    .await
}
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

/// Search for prompts by content or metadata
pub async fn search_prompts(
    pool: &PgPool,
    schema: &str,
    query: &str,
    limit: usize,
) -> Result<Vec<(PromptId, DateTime<Utc>)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let pattern = format!("%{}%", query);
            
            let results: Vec<(PromptId, DateTime<Utc>)> = sqlx::query_as(
                r#"
                SELECT id, created_at
                FROM prompts
                WHERE content::text ILIKE $1
                ORDER BY created_at DESC
                LIMIT $2
                "#
            )
            .bind(&pattern)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?;

            Ok(results)
        })
    })
    .await
}

/// Get prompts within a time range
pub async fn get_prompts_in_range(
    pool: &PgPool,
    schema: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
) -> Result<Vec<(PromptId, Prompt<'static>)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let results: Vec<(PromptId, serde_json::Value)> = sqlx::query_as(
                r#"
                SELECT id, content
                FROM prompts
                WHERE created_at >= $1 AND created_at <= $2
                ORDER BY created_at DESC
                "#
            )
            .bind(start)
            .bind(end)
            .fetch_all(&mut **tx)
            .await?;

            let prompts = results
                .into_iter()
                .map(|(id, value)| {
                    let prompt: Prompt<'static> = serde_json::from_value(value)?;
                    Ok((id, prompt))
                })
                .collect::<Result<Vec<_>, serde_json::Error>>()?;

            Ok(prompts)
        })
    })
    .await
}

/// Find similar prompts based on embeddings
pub async fn find_similar_prompts(
    pool: &PgPool,
    schema: &str,
    prompt_id: PromptId,
    limit: usize,
) -> Result<Vec<(PromptId, f32)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            // First get the embedding of the target prompt
            let embedding: Option<pgvector::Vector> = sqlx::query_scalar(
                "SELECT embedding FROM prompts WHERE id = $1"
            )
            .bind(prompt_id)
            .fetch_optional(&mut **tx)
            .await?;

            if let Some(embedding) = embedding {
                let results: Vec<(PromptId, f32)> = sqlx::query_as(
                    r#"
                    SELECT 
                        p.id,
                        1 - (p.embedding <=> $1) as similarity
                    FROM prompts p
                    WHERE p.id != $2 AND p.embedding IS NOT NULL
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#
                )
                .bind(&embedding)
                .bind(prompt_id)
                .bind(limit as i64)
                .fetch_all(&mut **tx)
                .await?;

                Ok(results)
            } else {
                Ok(vec![])
            }
        })
    })
    .await
}
