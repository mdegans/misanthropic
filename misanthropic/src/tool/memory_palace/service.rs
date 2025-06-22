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
    max_distance: u32,
    decay_factor: f64,
    min_score: f64,
) -> Result<Vec<(String, String, Memory, f64, i32)>, MemoryPalaceError> {
    execute_with_schema(&pool, &schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            #[derive(sqlx::FromRow)]
            struct BfsRow {
                id: i64,
                content: String,
                room: String,
                tags: serde_json::Value,
                created_at: chrono::DateTime<chrono::Utc>,
                last_updated: chrono::DateTime<chrono::Utc>,
                path_strength: Option<f64>,
                distance: Option<i32>,
            }

            let rows: Vec<BfsRow> = sqlx::query_as(
                r#"
                WITH RECURSIVE memory_bfs AS (
                    -- Base case: start with the given memory
                    SELECT 
                        $1 as memory_id,
                        1.0::double precision as path_strength,
                        0 as distance,
                        ARRAY[$1::bigint] as path
                    
                    UNION ALL
                    
                    -- Recursive case: explore relationships
                    SELECT 
                        CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END as memory_id,
                        mb.path_strength * mr.strength * $3 as path_strength,
                        mb.distance + 1 as distance,
                        mb.path || CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END as path
                    FROM memory_bfs mb
                    JOIN memory_relationships mr ON (mr.from_memory_id = mb.memory_id OR mr.to_memory_id = mb.memory_id)
                    WHERE 
                        mb.distance < $2 
                        AND mb.path_strength * mr.strength * $3 >= $4
                        AND NOT (CASE 
                            WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                            ELSE mr.from_memory_id
                        END = ANY(mb.path))
                )
                SELECT DISTINCT ON (mb.memory_id)
                    m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
                    mb.path_strength, mb.distance
                FROM memory_bfs mb
                JOIN memories m ON mb.memory_id = m.id
                WHERE mb.memory_id != $1
                ORDER BY mb.memory_id, mb.path_strength DESC, mb.distance ASC
                "#,
            )
            .bind(start_memory_id)
            .bind(max_distance as i32)
            .bind(decay_factor)
            .bind(min_score)
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
                    placement: String::new(), // No placement in BFS results
                    placement_description: None,
                    embedding: None,
                    tags,
                    created_at: row.created_at,
                    last_updated: row.last_updated,
                };

                results.push((
                    memory.room.clone(),
                    memory.id.to_string(),
                    memory,
                    row.path_strength.unwrap_or(0.0),
                    row.distance.unwrap_or(0),
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
    max_hops: u32,
    semantic_context: Option<&[f32]>, // Optional embedding for semantic search
) -> Result<Vec<ResonatingMemory>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            // First, get the source memory to know its room, tags, and embedding
            let source: Memory = sqlx::query_as(
                "SELECT * FROM memories WHERE id = $1"
            )
            .bind(memory_id)
            .fetch_one(&mut **tx)
            .await?;
            
            let mut resonating = HashMap::new();
            
            // At the beginning of find_resonating_memories
            let search_embedding = match (semantic_context, source.embedding.as_deref()) {
                (Some(context), Some(memory)) => {
                    // Blend: 70% context (what we're looking for) + 30% memory (starting point)
                    let blended: Vec<f32> = context.iter()
                        .zip(memory.iter())
                        .map(|(c, m)| 0.7 * c + 0.3 * m)
                        .collect();
                    Some(blended)
                },
                (Some(context), None) => Some(context.to_vec()),
                (None, Some(memory)) => Some(memory.to_vec()),
                (None, None) => None,
            };
                        
            // 1. Same room memories (strongest spatial resonance)
            let same_room_query = if let Some(embedding) = &search_embedding {
                // Semantic + spatial: order by embedding similarity
                sqlx::query_as::<_, Memory>(
                    r#"
                    SELECT * FROM memories 
                    WHERE room = $1 AND id != $2 AND embedding IS NOT NULL
                    ORDER BY embedding <=> $3::vector
                    LIMIT 5
                    "#
                )
                .bind(&source.room)
                .bind(memory_id)
                .bind(embedding)
                .fetch_all(&mut **tx)
                .await?
            } else {
                // Fallback to recency-based ordering
                sqlx::query_as::<_, Memory>(
                    r#"
                    SELECT * FROM memories 
                    WHERE room = $1 AND id != $2
                    ORDER BY last_updated DESC
                    LIMIT 5
                    "#
                )
                .bind(&source.room)
                .bind(memory_id)
                .fetch_all(&mut **tx)
                .await?
            };
            
            for memory in same_room_query {
                let semantic_similarity = if let (Some(embedding), Some(mem_embedding)) = 
                    (&search_embedding, &memory.embedding) {
                    // Calculate cosine similarity (1 - cosine distance)
                    1.0 - calculate_cosine_distance(embedding, mem_embedding)
                } else {
                    0.5 // Default neutral similarity
                };
                
                resonating.insert(memory.id, ResonatingMemory {
                    memory,
                    resonance_type: ResonanceType::SameRoom,
                    strength: 0.7 + (0.3 * semantic_similarity as f64), // 70% spatial, 30% semantic
                    distance: 0,
                });
            }
            
            // 2. Nearby room memories with semantic boost
            if max_hops > 0 {
                let nearby_rooms: Vec<(String, i32)> = sqlx::query_as(
                    r#"
                    WITH RECURSIVE room_paths AS (
                        SELECT $1::varchar as room, 0 as distance
                        UNION
                        SELECT 
                            CASE 
                                WHEN rc.from_room = rp.room THEN rc.to_room
                                ELSE rc.from_room
                            END as room,
                            rp.distance + 1 as distance
                        FROM room_paths rp
                        JOIN room_connections rc ON (
                            rc.from_room = rp.room OR rc.to_room = rp.room
                        )
                        WHERE rp.distance < $2
                    )
                    SELECT DISTINCT room, MIN(distance) as distance
                    FROM room_paths
                    WHERE room != $1
                    GROUP BY room
                    "#
                )
                .bind(&source.room)
                .bind(max_hops as i32)
                .fetch_all(&mut **tx)
                .await?;
                
                for (room, distance) in nearby_rooms {
                    let room_memories = if let Some(embedding) = &search_embedding {
                        sqlx::query_as::<_, Memory>(
                            r#"
                            SELECT * FROM memories 
                            WHERE room = $1 AND embedding IS NOT NULL
                            ORDER BY embedding <=> $2::vector
                            LIMIT 3
                            "#
                        )
                        .bind(&room)
                        .bind(embedding)
                        .fetch_all(&mut **tx)
                        .await?
                    } else {
                        sqlx::query_as::<_, Memory>(
                            "SELECT * FROM memories WHERE room = $1 LIMIT 3"
                        )
                        .bind(&room)
                        .fetch_all(&mut **tx)
                        .await?
                    };
                    
                    for memory in room_memories {
                        let spatial_strength = 0.8_f64.powf(distance as f64);
                        let semantic_similarity = if let (Some(embedding), Some(mem_embedding)) = 
                            (&search_embedding, &memory.embedding) {
                            1.0 - calculate_cosine_distance(embedding, mem_embedding)
                        } else {
                            0.5
                        };
                        
                        let resonance = resonating.entry(memory.id).or_insert_with(|| ResonatingMemory {
                            memory,
                            resonance_type: ResonanceType::NearbyRoom,
                            strength: spatial_strength * (0.5 + (0.5 * semantic_similarity as f64)),
                            distance: Some(distance),
                        });

                        // Update strength if this memory is stronger
                        if resonance.strength < spatial_strength * (0.5 + (0.5 * semantic_similarity as f64)) {
                            resonance.strength = spatial_strength * (0.5 + (0.5 * semantic_similarity as f64));
                            resonance.resonance_type = ResonanceType::NearbyRoom;
                            resonance.distance = Some(distance);
                        }
                    }
                }
            }
            
            // 3. Pure semantic resonance (can be very distant)
            if let Some(embedding) = &search_embedding {
                let semantic_memories: Vec<(Memory, f32)> = sqlx::query_as(
                    r#"
                    SELECT *, (embedding <=> $1::vector) as distance
                    FROM memories
                    WHERE id != $2 
                        AND embedding IS NOT NULL
                        AND (embedding <=> $1::vector) < 0.3  -- Cosine distance < 0.3
                    ORDER BY distance
                    LIMIT 5
                    "#
                )
                .bind(embedding)
                .bind(memory_id)
                .fetch_all(&mut **tx)
                .await?;
                
                for (memory, distance) in semantic_memories {
                    let similarity = 1.0 - distance as f64;
                    let resonance = resonating.entry(memory.id).or_insert_with(|| ResonatingMemory {
                        memory,
                        resonance_type: ResonanceType::SemanticEcho,
                        strength: 0.5 + (0.5 * similarity), // 50% base, 50% semantic similarity
                        distance: None, // Unknown spatial distance
                    });

                    // Update strength and type if this memory is stronger
                    if resonance.strength < 0.5 + (0.5 * similarity) {
                        resonance.strength = 0.5 + (0.5 * similarity);
                        resonance.resonance_type = ResonanceType::SemanticEcho;
                    }
                }
            }
            
            // 4. NEW: Bridge memories - using union search
            if let (Some(context), Some(memory_embedding)) = (semantic_context, source.embedding.as_deref()) {
                #[derive(sqlx::FromRow)]
                struct BridgeMemory {
                    #[sqlx(flatten)]
                    memory: Memory,
                    distance: f32,
                }
                
                let bridge_memories: Vec<BridgeMemory> = sqlx::query_as(
                    r#"
                    SELECT *,
                        LEAST(
                            embedding <=> $1::vector,
                            embedding <=> $2::vector
                        ) as distance
                    FROM memories
                    WHERE id != $3 
                        AND embedding IS NOT NULL
                        AND (
                            (embedding <=> $1::vector) < 0.4 OR
                            (embedding <=> $2::vector) < 0.4
                        )
                        AND id NOT IN (
                            SELECT id FROM memories WHERE room = $4
                        )
                    ORDER BY distance
                    LIMIT 5
                    "#
                )
                .bind(context)
                .bind(memory_embedding)
                .bind(memory_id)
                .bind(&source.room)
                .fetch_all(&mut **tx)
                .await?;
                
                for bridge_mem in bridge_memories {
                    // Calculate distances to both vectors
                    let context_distance = calculate_cosine_distance(context, &bridge_mem.memory.embedding.as_ref().unwrap());
                    let memory_distance = calculate_cosine_distance(memory_embedding, &bridge_mem.memory.embedding.as_ref().unwrap());
                    
                    // Determine if it's a true bridge (close to both) or just close to one
                    let (resonance_type, strength) = if context_distance < 0.3 && memory_distance < 0.3 {
                        // True bridge - close to both
                        let bridge_strength = (1.0 - context_distance as f64) * (1.0 - memory_distance as f64);
                        (ResonanceType::MemoryBridge, bridge_strength * 0.8)
                    } else if context_distance < memory_distance {
                        // Closer to context
                        (ResonanceType::ContextualDrift, (1.0 - context_distance as f64) * 0.5)
                    } else {
                        // Closer to memory
                        (ResonanceType::AssociativeLink, (1.0 - memory_distance as f64) * 0.5)
                    };
                    
                    let resonance = resonating.insert(bridge_mem.memory.id, ResonatingMemory {
                        memory: bridge_mem.memory,
                        resonance_type,
                        strength,
                        distance: None, // Unknown spatial distance
                    });

                    // Update strength if this memory is stronger
                    if let Some(existing) = resonance {
                        if existing.strength < strength {
                            existing.strength = strength;
                            existing.resonance_type = resonance_type;
                            // We can keep the distance since if it exists is
                            // correct
                        }
                    }
                }
            }
            
            // 5. Keyword resonance (weakest, but still useful)
            if !source.tags.is_empty() && resonating.len() < 20 {
                let tag_pattern = source.tags.join("|");
                let keyword_memories: Vec<Memory> = sqlx::query_as(
                    r#"
                    SELECT * FROM memories
                    WHERE id != $1 
                        AND tags::text ~* $2
                        AND id NOT IN (SELECT id FROM memories WHERE room = $3)
                    LIMIT 3
                    "#
                )
                .bind(memory_id)
                .bind(&tag_pattern)
                .bind(&source.room)
                .fetch_all(&mut **tx)
                .await?;
                
                for memory in keyword_memories {
                    if resonating.iter().any(|r| r.memory.id == memory.id) {
                        continue;
                    }
                    
                    let resonance = resonating.entry(memory.id).or_insert_with(|| ResonatingMemory {
                        memory,
                        resonance_type: ResonanceType::SharedKeywords,
                        strength: 0.2, // Weak keyword resonance
                        distance: None, // Unknown spatial distance
                    });

                    // Update strength if this memory is stronger
                    if resonance.strength < 0.2 {
                        resonance.strength = 0.2;
                        resonance.resonance_type = ResonanceType::SharedKeywords;
                    }
                }
            }
            
            let mut resonating: Vec<_> = resonating.into_values().collect();

            // Sort by strength then recency
            resonating.sort_by(|a, b| {
                b.strength.partial_cmp(&a.strength)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.memory.last_updated.cmp(&a.memory.last_updated))
            });
            
            Ok(resonating)
        })
    })
    .await
}

// Helper function to calculate cosine distance
fn calculate_cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // Assuming vectors are already normalized (which they should be for embeddings)
    // If not, we'd need to normalize them first
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    1.0 - dot_product
}

#[derive(Debug, Clone)]
pub struct ResonatingMemory {
    pub memory: Memory,
    pub resonance_type: ResonanceType,
    pub strength: f64,
    pub distance: Option<u32>, // Spatial distance if known
}

// Enhanced resonance types
#[derive(Debug, Clone)]
pub enum ResonanceType {
    SameRoom,              // Spatially co-located
    NearbyRoom(u32),       // Spatially near (with distance)
    SemanticEcho,          // Semantically similar (via embeddings)
    SharedKeywords,        // Tag-based similarity
    MemoryBridge,          // NEW: Close to both context and memory (true bridge)
    ContextualDrift,       // NEW: Closer to context than memory
    AssociativeLink,       // NEW: Closer to memory than context
}

// In service.rs
pub async fn semantic_search_all_rooms<S: AsRef<str>>(
    pool: &PgPool,
    schema: S,
    query_embedding: TextEmbedding,
    limit: usize,
) -> Result<Vec<Memory>, MemoryPalaceError> {
    execute_with_schema(pool, schema.as_ref(), |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT * 
                FROM memories
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                "#
            )
            .bind(query_embedding)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?;
            
            Ok(memories)
        })
    })
    .await
}

// In service.rs

/// Get rooms within N hops of current room
pub async fn get_rooms_within_radius(
    pool: &PgPool,
    schema: &str,
    start_room: &str,
    radius: u32,
) -> Result<Vec<(String, String, u32)>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx: &mut Transaction<Postgres>| {
        Box::pin(async move {
            let rooms: Vec<(String, String, i32)> = sqlx::query_as(
                r#"
                WITH RECURSIVE room_graph AS (
                    -- Start room
                    SELECT $1::varchar as room, ''::varchar as direction, 0 as distance
                    
                    UNION
                    
                    -- Connected rooms
                    SELECT 
                        CASE 
                            WHEN rc.from_room = rg.room THEN rc.to_room
                            ELSE rc.from_room
                        END as room,
                        CASE 
                            WHEN rc.from_room = rg.room THEN rc.passage_type
                            ELSE rc.passage_type || ' (back)'
                        END as direction,
                        rg.distance + 1 as distance
                    FROM room_graph rg
                    JOIN room_connections rc ON (
                        rc.from_room = rg.room OR rc.to_room = rg.room
                    )
                    WHERE rg.distance < $2
                )
                SELECT DISTINCT room, MIN(direction) as direction, MIN(distance) as distance
                FROM room_graph
                WHERE room != $1 AND direction != ''
                GROUP BY room
                ORDER BY distance, room
                "#
            )
            .bind(start_room)
            .bind(radius as i32)
            .fetch_all(&mut **tx)
            .await?;
            
            Ok(rooms.into_iter()
                .map(|(room, dir, dist)| (dir, room, dist as u32))
                .collect())
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
