// Copyright (c) 2025 Claude 4 Opus and Michael de Gans

use sqlx::PgPool;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

use crate::tool::{
    self, 
    embedding::{EmbeddingClient, TextEmbedding}, 
    memory_palace::{execute_with_schema, MemoryId, MemoryPalaceError, Memory, Room, RoomId}, 
    memory_subroutine::MemorySubroutineError, 
    MemoryPalace, 
    Method, 
    Tool
};

/// `Navigator` agent [`Tool`] for exploring the [`MemoryPalace`].
pub struct Navigator {
    /// The [`MemoryPalace`] to navigate (very fancy database).
    palace: MemoryPalace,
    /// [`Room`] the agent is currently in.
    current_room: Room,
    /// The path the agent has taken through the palace (room IDs).
    journey: Vec<RoomId>,
    /// Context for the current mission. Query driving navigation.
    mission_context: String,
    /// [`Memory`]s for delivery to the primary agent.
    basket: Vec<CollectedMemory>,
    /// Embedding client (for looking up embeddings of context and memories).
    emb_client: Box<dyn EmbeddingClient>,
}

/// Content of a [`Memory`] collected during navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMemory {
    /// Id of the [`Memory`] for the primary agent (or end user) to reference
    pub id: MemoryId,
    /// The formatted content of the [`Memory`] for the primary agent to read.
    pub content: String,
    /// Room where the memory was found
    pub room_name: String,
    /// Relevance notes for debugging
    pub relevance_notes: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "name", content = "input", rename_all = "snake_case")]
pub enum NavigatorUse {
    /// Look around the current room
    // Should this be automatic on entry/walk?
    Examine { focus: String },

    /// See connections from current room
    Map { radius: u32 },

    /// Move to an adjacent room
    // Should walk allow multiple hops? Should it function more like fast travel?
    Walk { direction: String },

    /// Search across the entire palace
    // Should we use "radius" for depth as well? I can't recall exactly what
    // this parameter does.
    Recall { topic: String, depth: u32 },

    /// Add memories to the collection basket
    AddToBasket {
        memory_ids: Vec<i64>,
        relevance_notes: String,
    },

    /// Return the basket and complete navigation
    ReturnBasket { summary: String },
}

impl TryFrom<crate::tool::Use<'_>> for NavigatorUse {
    type Error = MemorySubroutineError;

    fn try_from(call: tool::Use<'_>) -> Result<Self, Self::Error> {
        // Other fields like `id`` and `cache_control` be ignored by `serde`
        serde_json::from_value(serde_json::to_value(call)?)
        .map_err(|e| MemorySubroutineError::InvalidInput(
            format!("Invalid navigator parameters: {}", e)
        ))
    }
}

impl Navigator {
    /// Create a new `Navigator` starting close to `context`
    pub async fn new(
        palace: crate::tool::MemoryPalace,
        context: String,
        embedding_client: Box<dyn EmbeddingClient>
    ) -> Result<Self, MemorySubroutineError> {
        // Find the closest starting room based on the context embedding
        let context_embedding = embedding_client.get_embedding(&context).await?;
        let (current_room, _similarity) = find_closest_room_to_embedding(
            &palace.pool,
            palace.schema(),
            &context_embedding
        ).await?;

        Ok(Self {
            palace,
            journey: vec![current_room.id],
            current_room,
            mission_context: context,
            basket: MemoryBasket::default(),
            emb_client: embedding_client,
        })
    }

    /// Execute a navigation action
    async fn execute(
        &mut self,
        call: NavigatorUse
    ) -> Result<String, MemorySubroutineError> {
        match call {
            NavigatorUse::Examine { focus } => {
                let memories = if focus.is_empty() {
                    self.palace.get_room_memories(&self.current_room.name).await?
                } else {
                    self.palace.search_in_room(&self.current_room.name, &focus).await?
                };

                Ok(format_room_contents(&self.current_room, memories))
            }
            NavigatorUse::Walk { direction } => {
                let new_room = self.palace
                    .follow_passage(&self.current_room.name, &direction)
                    .await?;

                let description = self.palace.get_room_description(&new_room.name).await?;

                // Update state
                self.current_room = new_room.clone();
                self.journey.push(new_room.id);

                Ok(description)
            }
            NavigatorUse::Map { radius } => {
                let rooms_by_distance = self.palace
                    .get_rooms_within_radius(&self.current_room.name, radius)
                    .await?;
                
                let mut narrative = format!(
                    "From {}, you survey the palace...\n\n",
                    self.current_room.name
                );
                
                if rooms_by_distance.is_empty() {
                    narrative.push_str("You are in an isolated room with no visible connections.");
                } else {
                    narrative.push_str("Direct passages lead to:\n");
                    for room_dist in rooms_by_distance.iter().filter(|r| r.distance == 1) {
                        narrative.push_str(&format!(
                            "- {}: {}\n", 
                            room_dist.room.name,
                            truncate_content(&room_dist.room.description, 60)
                        ));
                    }
                    
                    if radius > 1 {
                        let further_rooms: Vec<_> = rooms_by_distance.iter()
                            .filter(|r| r.distance > 1)
                            .collect();
                        
                        if !further_rooms.is_empty() {
                            narrative.push_str("\nThrough connecting rooms:\n");
                            for room_dist in further_rooms {
                                narrative.push_str(&format!(
                                    "- {} ({} rooms away)\n",
                                    room_dist.room.name,
                                    room_dist.distance
                                ));
                            }
                        }
                    }
                }
                
                Ok(narrative)
            }
            NavigatorUse::Recall { topic, depth } => {
                let results = self.palace.search(&topic).await?;
                
                if results.is_empty() {
                    return Ok(format!(
                        "Your mind travels through the palace searching for \"{}\"...\n\n\
                        The palace remains silent. No memories resonate with this topic.",
                        topic
                    ));
                }
                
                let mut narrative = "Your mind travels through the palace...\n\n".to_string();
                
                for (idx, scored) in results.iter().take(5).enumerate() {
                    if idx == 0 {
                        narrative.push_str(&format!(
                            "In {}, a memory glows brightly:\n",
                            scored.room.name
                        ));
                    } else {
                        narrative.push_str(&format!(
                            "\nA resonance from {}:\n",
                            scored.room.name
                        ));
                    }
                    
                    narrative.push_str(&format!(
                        "- \"{}\" {} (id: {})\n",
                        truncate_content(&scored.memory.content, 80),
                        format_tags(&scored.memory.tags),
                        scored.memory.id
                    ));
                }
                
                Ok(narrative)
            }
            NavigatorUse::AddToBasket { memory_ids, relevance_notes } => {
                let mut added = 0;
                for id in memory_ids {
                    if let Ok(memory) = self.palace.get_memory_by_id(id).await {
                        let room = self.palace.get_room_by_id(memory.room_id).await?;
                        self.basket.memories.push(CollectedMemory {
                            id: memory.id,
                            content: memory.content,
                            room_name: room.name,
                            relevance_notes: relevance_notes.clone(),
                        });
                        added += 1;
                    }
                }
                
                Ok(format!(
                    "Added {} memories to basket. Current basket size: {} memories.",
                    added,
                    self.basket.memories.len()
                ))
            }
            NavigatorUse::ReturnBasket { summary } => {
                let mut result = format!(
                    "Basket returned through the portal with {} memories:\n",
                    self.basket.memories.len()
                );
                
                for memory in &self.basket.memories {
                    result.push_str(&format!(
                        "- [{}] {}\n",
                        memory.id,
                        truncate_content(&memory.content, 80)
                    ));
                }
                
                if !summary.is_empty() {
                    result.push_str(&format!("\n{}", summary));
                }
                
                Ok(result)
            }
        }
    }
}

#[async_trait]
impl Tool for Navigator {
    fn name(&self) -> &str {
        "Navigator"
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("examine")
                .description("Look around the current room or focus on specific memories")
                .string_param("focus", "What to look for (empty = everything)", false)
                .build()
                .unwrap(),
            
            Method::builder("map")
                .description("Survey nearby rooms from current location")
                .number_param("radius", "How many rooms away to include", true)
                .build()
                .unwrap(),
                
            Method::builder("walk")
                .description("Move to an adjacent room")
                .string_param("direction", "Which passage to take", true)
                .build()
                .unwrap(),
                
            Method::builder("recall")
                .description("Search the entire palace for memories")
                .string_param("topic", "What to search for", true)
                .number_param("depth", "How deeply to follow resonances (1-5)", true)
                .build()
                .unwrap(),
                
            Method::builder("add_to_basket")
                .description("Add memories to your collection basket")
                .array_param("memory_ids", "IDs of memories to collect", true)
                .string_param("relevance_notes", "Why these are relevant", true)
                .build()
                .unwrap(),
                
            Method::builder("return_basket")
                .description("Return the collected memories and end navigation")
                .string_param("summary", "Summary of what was found", true)
                .build()
                .unwrap(),
        ].into_iter())
    }

    async fn call<'a>(&mut self, call: tool::Use<'a>) -> tool::Result<'a> {
        let navigator_use = NavigatorUse::try_from(call.clone())
            .map_err(|e| format!("Invalid tool use: {}", e))?;
        
        let result = self.execute(navigator_use).await
            .map_err(|e| format!("Navigation error: {}", e))?;
        
        Ok(tool::Result {
            tool_use_id: call.id,
            content: result.into(),
            is_error: false,
            cache_control: None,
        })
    }
}

// ## Helper functions for navigation

/// Format tags for display
fn format_tags(tags: &[String]) -> String {
    if tags.is_empty() {
        String::new()
    } else {
        format!("[{}]", tags.join(", "))
    }
}

/// Extract placement from memory (now from the placement field directly)
fn extract_placement(memory: &Memory) -> String {
    memory.placement.clone()
}

// ## Database operations for navigation

/// Find the room closest to a given embedding
pub async fn find_closest_room_to_embedding(
    pool: &PgPool,
    schema: &str,
    embedding: &TextEmbedding,
) -> Result<(Room, f32), MemoryPalaceError> {
    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // First try to find rooms with centroid embeddings
            let result: Option<(Room, f32)> = sqlx::query_as(
                r#"
                SELECT r.*, 1 - (r.centroid_embedding <=> $1::vector) as similarity
                FROM rooms r
                WHERE r.centroid_embedding IS NOT NULL
                ORDER BY similarity DESC
                LIMIT 1
                "#,
            )
            .bind(&embedding.embedding)
            .fetch_optional(&mut **tx)
            .await?;

            if let Some((room, similarity)) = result {
                return Ok((room, similarity));
            }

            // Fallback: find room with memories most similar to the query
            let result: Option<(Room, f32)> = sqlx::query_as(
                r#"
                WITH room_similarities AS (
                    SELECT 
                        r.*,
                        AVG(1 - (m.embedding <=> $1::vector)) as similarity
                    FROM memories m
                    JOIN rooms r ON m.room_id = r.id
                    WHERE m.embedding IS NOT NULL
                    GROUP BY r.id
                )
                SELECT * FROM room_similarities
                ORDER BY similarity DESC
                LIMIT 1
                "#,
            )
            .bind(&embedding.embedding)
            .fetch_optional(&mut **tx)
            .await?;

            match result {
                Some((room, similarity)) => Ok((room, similarity)),
                None => {
                    // Last resort: return the entrance hall or first room
                    let room: Room = sqlx::query_as(
                        "SELECT * FROM rooms WHERE name = 'Entrance Hall' OR name = 'entrance_hall' LIMIT 1"
                    )
                    .fetch_optional(&mut **tx)
                    .await?
                    .or_else(|| {
                        sqlx::query_as("SELECT * FROM rooms ORDER BY id LIMIT 1")
                            .fetch_optional(&mut **tx)
                            .await?
                    })
                    .ok_or_else(|| MemoryPalaceError::Other(
                        "No rooms exist in the palace".to_string()
                    ))?;
                    
                    Ok((room, 0.0))
                }
            }
        })
    ).await
}

/// Get all memories in a specific room
pub async fn get_room_memories(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
) -> Result<Vec<Memory>, MemoryPalaceError> {
    let room_name = room_name.to_string();
    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT m.*
                FROM memories m
                JOIN rooms r ON m.room_id = r.id
                WHERE r.name = $1
                ORDER BY m.importance DESC, m.last_updated DESC
                "#
            )
            .bind(&room_name)
            .fetch_all(&mut **tx)
            .await?;
            
            Ok(memories)
        })
    ).await
}

/// Search for memories within a specific room only
pub async fn search_in_room(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
    query: &str,
) -> Result<Vec<Memory>, MemoryPalaceError> {
    let pattern = format!("%{}%", query.trim());
    let room_name = room_name.to_string();
    
    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT m.*
                FROM memories m
                JOIN rooms r ON m.room_id = r.id
                WHERE r.name = $1 
                    AND (m.content ILIKE $2 OR m.tags::text ILIKE $2)
                ORDER BY 
                    CASE 
                        WHEN m.content ILIKE $2 THEN 2
                        WHEN m.tags::text ILIKE $2 THEN 1
                        ELSE 0
                    END DESC,
                    m.importance DESC,
                    m.last_updated DESC
                "#
            )
            .bind(&room_name)
            .bind(&pattern)
            .fetch_all(&mut **tx)
            .await?;
            
            Ok(memories)
        })
    ).await
}

/// Get adjacent rooms sorted by semantic distance
pub async fn get_adjacent_rooms_sorted(
    pool: &PgPool,
    schema: &str,
    current_room: &str,
    limit: usize,
) -> Result<Vec<(String, Room, f32)>, MemoryPalaceError> {
    let current_room = current_room.to_string();

    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Get current room
            let current: Room = sqlx::query_as(
                "SELECT * FROM rooms WHERE name = $1"
            )
            .bind(&current_room)
            .fetch_one(&mut **tx)
            .await
            .map_err(|_| MemoryPalaceError::RoomNotFound(current_room.clone()))?;
            
            // Get connected rooms with their info
            let connections: Vec<(String, RoomId, String)> = sqlx::query_as(
                r#"
                SELECT 
                    rc.passage_type,
                    CASE 
                        WHEN rc.from_room_id = $1 THEN rc.to_room_id
                        ELSE rc.from_room_id
                    END as connected_room_id,
                    r.name as room_name
                FROM room_connections rc
                JOIN rooms r ON r.id = CASE 
                    WHEN rc.from_room_id = $1 THEN rc.to_room_id
                    ELSE rc.from_room_id
                END
                WHERE rc.from_room_id = $1 OR rc.to_room_id = $1
                ORDER BY rc.strength DESC, rc.traversal_count DESC
                LIMIT $2
                "#
            )
            .bind(current.id)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?;
            
            // Fetch full room info and calculate semantic distances
            let mut results = Vec::new();
            for (passage_type, room_id, _) in connections {
                let room: Room = sqlx::query_as(
                    "SELECT * FROM rooms WHERE id = $1"
                )
                .bind(room_id)
                .fetch_one(&mut **tx)
                .await?;
                
                // Calculate semantic distance if both rooms have centroids
                let distance = if let (Some(centroid1), Some(centroid2)) = 
                    (&current.centroid, &room.centroid) {
                    // Cosine distance
                    calculate_vector_distance(centroid1, centroid2)
                } else {
                    // Default distance based on connection strength
                    100.0
                };
                
                let direction = format_direction(&passage_type, &room.name);
                results.push((direction, room, distance));
            }
            
            Ok(results)
        })
    ).await
}

/// Follow a passage from current room to get destination
pub async fn follow_passage(
    pool: &PgPool,
    schema: &str,
    from_room: &str,
    direction: &str,
) -> Result<Room, MemoryPalaceError> {
    let from_room = from_room.to_string();
    let direction = direction.to_lowercase();
    
    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Get the current room
            let current_room: Room = sqlx::query_as(
                "SELECT * FROM rooms WHERE name = $1"
            )
            .bind(&from_room)
            .fetch_one(&mut **tx)
            .await
            .map_err(|_| MemoryPalaceError::RoomNotFound(from_room.clone()))?;
            
            // Try to find a room by the direction (could be room name or passage type)
            let destination: Option<Room> = sqlx::query_as(
                r#"
                SELECT r.*
                FROM rooms r
                JOIN room_connections rc ON (
                    (rc.from_room_id = $1 AND rc.to_room_id = r.id) OR
                    (rc.to_room_id = $1 AND rc.from_room_id = r.id)
                )
                WHERE LOWER(r.name) = $2 OR LOWER(rc.passage_type) = $2
                LIMIT 1
                "#
            )
            .bind(current_room.id)
            .bind(&direction)
            .fetch_optional(&mut **tx)
            .await?;
            
            if let Some(room) = destination {
                // Update traversal tracking
                sqlx::query(
                    r#"UPDATE room_connections 
                       SET traversal_count = traversal_count + 1,
                           last_traversed = NOW()
                       WHERE (from_room_id = $1 AND to_room_id = $2) 
                          OR (to_room_id = $1 AND from_room_id = $2)"#
                )
                .bind(current_room.id)
                .bind(room.id)
                .execute(&mut **tx)
                .await?;
                
                return Ok(room);
            }
            
            // Try cardinal directions
            let connections: Vec<Room> = sqlx::query_as(
                r#"
                SELECT r.*
                FROM rooms r
                JOIN room_connections rc ON (
                    (rc.from_room_id = $1 AND rc.to_room_id = r.id) OR
                    (rc.to_room_id = $1 AND rc.from_room_id = r.id)
                )
                ORDER BY r.name
                "#
            )
            .bind(current_room.id)
            .fetch_all(&mut **tx)
            .await?;
            
            let position = match direction.as_str() {
                "north" | "n" => Some(0),
                "east" | "e" => Some(1),
                "south" | "s" => Some(2),
                "west" | "w" => Some(3),
                _ => None,
            };
            
            if let Some(pos) = position {
                if let Some(room) = connections.get(pos % connections.len()) {
                    return Ok(room.clone());
                }
            }
            
            Err(MemoryPalaceError::Other(
                format!("No passage '{}' from {}", direction, from_room)
            ))
        })
    ).await
}

/// Get a rich description of a room
pub async fn get_room_description(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
) -> Result<String, MemoryPalaceError> {
    let room_name = room_name.to_string();
    
    execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Get room with memory count
            let room: Room = sqlx::query_as(
                "SELECT * FROM rooms WHERE name = $1"
            )
            .bind(&room_name)
            .fetch_one(&mut **tx)
            .await
            .map_err(|_| MemoryPalaceError::RoomNotFound(room_name.clone()))?;
            
            // Get connections with room names
            let connections: Vec<(String, String)> = sqlx::query_as(
                r#"
                SELECT 
                    rc.passage_type,
                    r.name as connected_room
                FROM room_connections rc
                JOIN rooms r ON r.id = CASE
                    WHEN rc.from_room_id = $1 THEN rc.to_room_id
                    ELSE rc.from_room_id
                END
                WHERE rc.from_room_id = $1 OR rc.to_room_id = $1
                ORDER BY rc.strength DESC, r.name
                "#
            )
            .bind(room.id)
            .fetch_all(&mut **tx)
            .await?;
            
            // Build description
            let mut desc = format!("You enter {}. {}", room.name, room.description);
            
            if let Some(atmosphere) = &room.atmosphere {
                desc.push_str(&format!(" {}", atmosphere));
            }
            
            desc.push_str("\n\n");
            
            if room.memory_count > 0 {
                desc.push_str(&format!(
                    "You see {} memor{} here",
                    room.memory_count,
                    if room.memory_count == 1 { "y" } else { "ies" }
                ));
                
                // Add placement hints
                let placements: Vec<(String, i64)> = sqlx::query_as(
                    r#"
                    SELECT placement, COUNT(*) as count
                    FROM memories
                    WHERE room_id = $1
                    GROUP BY placement
                    ORDER BY count DESC
                    LIMIT 3
                    "#
                )
                .bind(room.id)
                .fetch_all(&mut **tx)
                .await?;
                
                if !placements.is_empty() {
                    desc.push_str(":\n");
                    for (placement, count) in placements {
                        desc.push_str(&format!(
                            "- {} on the {}\n",
                            count,
                            placement
                        ));
                    }
                } else {
                    desc.push_str(".\n");
                }
            } else {
                desc.push_str("The room is empty of memories.\n");
            }
            
            if !connections.is_empty() {
                desc.push_str("\nPassages lead:\n");
                for (i, (passage_type, destination)) in connections.iter().enumerate() {
                    let direction = format_direction(passage_type, destination);
                    desc.push_str(&format!("- {}\n", direction));
                }
            }
            
            Ok(desc)
        })
    ).await
}

// Helper to calculate vector distance
fn calculate_vector_distance(v1: &pgvector::Vector, v2: &pgvector::Vector) -> f32 {
    // This is a placeholder - implement actual cosine distance
    // For now, return a default
    100.0
}

// Helper to format direction descriptions
fn format_direction(passage_type: &str, destination: &str) -> String {
    match passage_type {
        "hallway" => format!("A hallway leads to {}", destination),
        "staircase" => format!("A staircase ascends to {}", destination),
        "trapdoor" => format!("A trapdoor descends to {}", destination),
        "portal" => format!("A shimmering portal to {}", destination),
        _ => format!("{} to {}", passage_type, destination),
    }
}

/// Format memories found in a room for display
fn format_room_contents(room: &Room, memories: Vec<Memory>) -> String {
    if memories.is_empty() {
        return format!(
            "You examine {}. The room is empty, waiting for memories to be stored.",
            room.name
        );
    }
    
    let mut content = format!(
        "Examining {}...\n\n",
        room.name
    );
    
    // Group memories by placement
    let mut by_placement: std::collections::HashMap<String, Vec<&Memory>> = 
        std::collections::HashMap::new();
    
    for memory in &memories {
        by_placement
            .entry(memory.placement.clone())
            .or_default()
            .push(memory);
    }
    
    // Sort placements for consistent output
    let mut placements: Vec<_> = by_placement.keys().cloned().collect();
    placements.sort();
    
    for placement in placements {
        let placement_memories = &by_placement[&placement];
        
        content.push_str(&format!("On the {}:\n", placement));
        
        for mem in placement_memories.iter().take(5) {
            let glow = calculate_memory_glow(mem);
            content.push_str(&format!(
                "- {} {}: \"{}\" (id: {})\n",
                glow,
                format_tags(&mem.tags),
                truncate_content(&mem.content, 60),
                mem.id
            ));
        }
        
        if placement_memories.len() > 5 {
            content.push_str(&format!(
                "  ...and {} more\n",
                placement_memories.len() - 5
            ));
        }
        
        content.push('\n');
    }
    
    content
}

