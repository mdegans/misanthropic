// Copyright (c) 2025 Claude 4 Opus and Michael de Gans

use sqlx::PgPool;

use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::tool::{self, memory_subroutine::MemorySubroutineError, memory_palace::{Memory, MemoryPalaceError}};


#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "name", content = "input", rename_all = "snake_case")]
pub enum NavigatorUse {
    /// Look around the current room
    Examine { focus: String },

    /// See connections from current room (sorted by semantic distance)
    Map { radius: u32 },

    /// Move to an adjacent room
    Walk { direction: String },

    /// Search across the entire palace
    Recall { topic: String, depth: u32 },

    /// Report suspicious or corrupted memories
    Report { memory_id: i64, reason: String },
}

impl TryFrom<crate::tool::Use<'_>> for NavigatorUse {
    type Error = MemorySubroutineError;

    fn try_from(call: tool::Use<'_>) -> Result<Self, Self::Error> {
        Ok(serde_json::from_value(json!(call))?)
    }
}

impl NavigatorUse {
    /// Apply this use case to the memory palace
    pub async fn navigate(
        &self,
        palace: &mut crate::tool::MemoryPalace,
        state: &NavigationState,
    ) -> Result<String, MemorySubroutineError> {
        match self {
            NavigatorUse::Examine { focus } => {
                let memories = if focus.is_empty() {
                    // Get all memories in current room
                    palace.get_room_memories(&state.current_room).await?
                } else {
                    // Search within current room only
                    palace.search_in_room(&state.current_room, &focus).await?
                };

                Ok(format_room_contents(memories))
            }

            NavigatorUse::Map { radius } => {
                // Get adjacent rooms sorted by average embedding distance
                let rooms = palace
                    .get_adjacent_rooms_sorted(
                        &state.current_room,
                        *radius,
                        state.mission.as_ref(),
                    )
                    .await?;

                Ok(format_map(rooms))
            }

            NavigatorUse::Walk { direction } => {
                // Validate the passage exists and return new room description
                let new_room = palace
                    .follow_passage(&state.current_room, &direction)
                    .await?;
                let description =
                    palace.get_room_description(new_room).await?;

                Ok(description)
            }
            NavigatorUse::Recall { topic, depth } => {
                // 1. Get embedding for the topic
                let topic_embedding: Vec<f32> = palace.get_embedding(topic).await?;
                
                // 2. Initial semantic search across all rooms
                let initial_memories = palace.semantic_search_all_rooms(
                    &topic_embedding,
                    5, // Start with top 5 most relevant memories
                ).await?;
                
                if initial_memories.is_empty() {
                    return Ok(format!(
                        "The navigator closes their eyes and thinks of \"{}\"...\n\n\
                        But the palace remains silent. No memories resonate with this topic.",
                        topic
                    ));
                }
                
                let mut narrative = format!(
                    "The navigator closes their eyes and thinks of \"{}\"...\n\n",
                    topic
                );
                
                let mut visited = std::collections::HashSet::new();
                let mut journey_segments = Vec::new();
                
                // 3. Build the journey through resonance
                for (idx, memory) in initial_memories.iter().enumerate() {
                    if visited.contains(&memory.id) {
                        continue;
                    }
                    visited.insert(memory.id);
                    
                    // Describe the initial memory discovery
                    let room_desc = if idx == 0 {
                        format!("Your mind begins in the {}, where a memory glows brightly:", memory.room)
                    } else {
                        format!("Another strong resonance pulls you to the {}:", memory.room)
                    };
                    
                    journey_segments.push(format!(
                        "{}\n- \"{}\" [{}]",
                        room_desc,
                        truncate_content(&memory.content, 80),
                        memory.tags.join(", ")
                    ));
                    
                    // Find resonating memories if depth > 0
                    if *depth > 0 {
                        let resonating = palace.find_resonating_memories(
                            memory.id,
                            *depth,
                            Some(&topic_embedding), // Keep topic context
                        ).await?;
                        
                        // Follow the strongest resonances
                        for (res_idx, res) in resonating.iter().take(2).enumerate() {
                            if visited.contains(&res.memory.id) {
                                continue;
                            }
                            visited.insert(res.memory.id);
                            
                            // Describe the resonance based on type
                            let transition = match &res.resonance_type {
                                ResonanceType::SameRoom => {
                                    "A nearby memory catches your attention:"
                                },
                                ResonanceType::NearbyRoom(dist) => {
                                    if *dist == 1 {
                                        "The thought echoes into an adjacent room:"
                                    } else {
                                        "The resonance travels through connected passages:"
                                    }
                                },
                                ResonanceType::SemanticEcho => {
                                    "A distant memory echoes across the palace:"
                                },
                                ResonanceType::MemoryBridge => {
                                    "A memory bridges both realms, glowing with prismatic light:"
                                },
                                ResonanceType::ContextualDrift => {
                                    "Your consciousness drifts toward your current thoughts:"
                                },
                                ResonanceType::AssociativeLink => {
                                    "An associative chain pulls you deeper:"
                                },
                                ResonanceType::SharedKeywords => {
                                    "A faint connection through shared concepts:"
                                },
                            };
                            
                            journey_segments.push(format!(
                                "\n{}\n- \"{}\" [{}] in the {}",
                                transition,
                                truncate_content(&res.memory.content, 80),
                                res.memory.tags.join(", "),
                                res.memory.room
                            ));
                        }
                    }
                }
                
                // 4. Assemble the narrative
                narrative.push_str(&journey_segments.join("\n"));
                
                // Add summary
                let unique_rooms: std::collections::HashSet<_> = initial_memories
                    .iter()
                    .map(|m| &m.room)
                    .collect();
                
                narrative.push_str(&format!(
                    "\n\n{} memories recalled across {} rooms, following {} resonance chains.",
                    visited.len(),
                    unique_rooms.len(),
                    journey_segments.len() - initial_memories.len()
                ));
                
                Ok(narrative)
            }
            NavigatorUse::Map { radius } => {
                let rooms_by_distance = palace
                    .get_rooms_within_radius(&state.current_room, *radius)
                    .await?;
                
                // Get current room's centroid for semantic comparison
                let current_centroid = palace
                    .get_room_centroid(&state.current_room)
                    .await?;
                
                let mut narrative = format!(
                    "The navigator surveys the palace from the {}...\n\n",
                    state.current_room
                );
                
                // Group by distance
                for distance in 1..=*radius {
                    let rooms_at_distance: Vec<_> = rooms_by_distance
                        .iter()
                        .filter(|(_, _, d)| *d == distance)
                        .collect();
                    
                    if rooms_at_distance.is_empty() {
                        continue;
                    }
                    
                    let header = match distance {
                        1 => "Direct passages lead to:",
                        2 => "Through connecting rooms:",
                        _ => "In distant quarters:",
                    };
                    
                    narrative.push_str(&format!("{}\n", header));
                    
                    for (direction, room_name, _) in &rooms_at_distance {
                        // Get semantic distance if we have centroids
                        let semantic_distance = if let (Some(current), Some(room_centroid)) = 
                            (&current_centroid, palace.get_room_centroid(room_name).await.ok().flatten()) {
                            // Convert cosine distance to meters (multiply by 1000 for intuitive scale)
                            let cosine_dist = calculate_cosine_distance(&current, &room_centroid);
                            (cosine_dist * 1000.0) as u32
                        } else {
                            // Default distance when no embeddings
                            500
                        };
                        
                        // Get a hint about the room's character
                        let room_hint = palace.get_room_character_hint(room_name).await?;
                        
                        narrative.push_str(&format!(
                            "- {}: The {} ({}m) - {}\n",
                            direction,
                            room_name,
                            semantic_distance,
                            room_hint
                        ));
                    }
                    narrative.push('\n');
                }
                
                // Add semantic resonance section if we have embeddings
                if let Some(centroid) = current_centroid {
                    let similar_rooms = palace
                        .find_semantically_similar_rooms(&state.current_room, &centroid, 5)
                        .await?;
                    
                    if !similar_rooms.is_empty() {
                        narrative.push_str("Mental resonance with other rooms:\n");
                        for (room_name, similarity) in similar_rooms.iter().take(3) {
                            let similarity_percent = (similarity * 100.0) as u32;
                            let resonance_desc = match similarity_percent {
                                90..=100 => "nearly identical energies",
                                70..=89 => "strong conceptual overlap",
                                50..=69 => "moderate thematic connection",
                                _ => "faint echoes",
                            };
                            
                            narrative.push_str(&format!(
                                "- The {} ({}% similarity) - {}\n",
                                room_name,
                                similarity_percent,
                                resonance_desc
                            ));
                        }
                    }
                }
                
                Ok(narrative)
            }
            _ => todo!("Implement other navigator methods"),
        }
    }
}

/// Navigation state to keep track of current room and visited rooms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationState {
    /// Current room the agent is in
    pub current_room: String,
    /// History of visited rooms in this session
    pub visited_rooms: Vec<String>,
    /// The mission or query driving this navigation
    pub mission: Option<String>,
}

// ## Helper functions for navigation

/// Calculate memory glow description based on recency
fn calculate_memory_glow(memory: &Memory) -> &'static str {
    let hours_ago = chrono::Utc::now()
        .signed_duration_since(memory.last_updated)
        .num_hours() as f64;
    
    let glow_factor = 2_f64.powf(-hours_ago / 24.0);
    
    match glow_factor {
        g if g > 0.8 => "glowing brightly",
        g if g > 0.5 => "glowing steadily",
        g if g > 0.2 => "glowing faintly",
        _ => "barely visible"
    }
}

/// Truncate content to max length with ellipsis
fn truncate_content(content: &str, max_len: usize) -> String {
    if content.len() <= max_len {
        content.to_string()
    } else {
        format!("{}...", &content[..max_len])
    }
}

// ## Database operations for navigation

// In the navigation context generation
pub async fn find_starting_room(
    pool: &PgPool,
    schema: &str,
    query_embedding: Vec<f32>,
) -> Result<i64, MemoryPalaceError> {
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Find the room with memories most similar to the query
            let row:(i64,) = sqlx::query_as(
                r#"
                SELECT r.id, AVG(1 - (m.embedding <=> $1::vector)) as similarity
                FROM memories m
                JOIN rooms r ON m.room = r.name
                WHERE m.embedding IS NOT NULL
                GROUP BY r.name
                ORDER BY similarity DESC
                LIMIT 1
                "#,
            )
            .bind(query_embedding)
            .fetch_one(&mut **tx)
            .await?;

            Ok(row.0)
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
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT id, content, room, tags, created_at, last_updated
                FROM memories
                WHERE room = $1
                ORDER BY last_updated DESC
                "#
            )
            .bind(room_name)
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
    
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT id, content, room, tags, created_at, last_updated
                FROM memories
                WHERE room = $1 
                    AND (content ILIKE $2 OR tags::text ILIKE $2)
                ORDER BY 
                    CASE 
                        WHEN content ILIKE $2 THEN 2
                        WHEN tags::text ILIKE $2 THEN 1
                        ELSE 0
                    END DESC,
                    last_updated DESC
                "#
            )
            .bind(room_name)
            .bind(&pattern)
            .fetch_all(&mut **tx)
            .await?;
            
            Ok(memories)
        })
    ).await
}

/// Get adjacent rooms sorted by semantic distance (using centroid embeddings)
pub async fn get_adjacent_rooms_sorted(
    pool: &PgPool,
    schema: &str,
    current_room: &str,
    radius: u32,
) -> Result<Vec<(String, String, f32)>, MemoryPalaceError> {
    let current_room = current_room.to_string();

    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            if radius == 0 {
                return Ok(vec![]);
            }
            
            // For now, get direct connections (radius 1)
            // TODO: Implement BFS for radius > 1
            let connections: Vec<(String, String, String)> = sqlx::query_as(
                r#"
                SELECT rc.to_room, rc.passage_type, r.description
                FROM room_connections rc
                JOIN rooms r ON rc.to_room = r.name
                WHERE rc.from_room = $1
                ORDER BY rc.to_room
                "#
            )
            .bind(current_room)
            .fetch_all(&mut **tx)
            .await?;
            
            // Convert to format with semantic distance
            // For now, using a placeholder distance calculation
            let mut results = Vec::new();
            for (i, (room, passage_type, _desc)) in connections.into_iter().enumerate() {
                // Direction is the passage type or a cardinal direction
                let direction = match passage_type.as_str() {
                    "hallway" => match i % 4 {
                        0 => "north",
                        1 => "east", 
                        2 => "south",
                        _ => "west",
                    },
                    _ => &passage_type,
                };
                
                // Placeholder distance - in real implementation would use embeddings
                let distance = ((i + 1) * 50) as f32;
                
                results.push((direction.to_string(), room, distance));
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
) -> Result<String, MemoryPalaceError> {
    let from_room = from_room.to_lowercase();
    let direction = direction.to_lowercase();
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // First try to match by passage type
            let room: Option<(String,)> = sqlx::query_as(
                r#"
                SELECT to_room
                FROM room_connections
                WHERE from_room = $1 AND passage_type = $2
                LIMIT 1
                "#
            )
            .bind(&from_room)
            .bind(&direction)
            .fetch_optional(&mut **tx)
            .await?;
            
            if let Some((room_name,)) = room {
                return Ok(room_name);
            }
            
            // If no match, try to interpret as cardinal direction
            // and get the Nth room
            let position = match direction.as_str() {
                "north" => 0,
                "east" => 1,
                "south" => 2,
                "west" => 3,
                _ => return Err(MemoryPalaceError::RoomNotFound(
                    format!("No passage '{}' from {}", direction, from_room)
                )),
            };
            
            let connections: Vec<(String,)> = sqlx::query_as(
                r#"
                SELECT to_room
                FROM room_connections
                WHERE from_room = $1
                ORDER BY to_room
                "#
            )
            .bind(&from_room)
            .fetch_all(&mut **tx)
            .await?;
            
            connections.get(position % connections.len())
                .map(|(room,)| room.clone())
                .ok_or_else(|| MemoryPalaceError::RoomNotFound(
                    format!("No {} passage from {}", direction, from_room)
                ))
        })
    ).await
}

/// Get a rich description of a room including memories and connections
pub async fn get_room_description(
    pool: &PgPool,
    schema: &str,
    room_name: String,
) -> Result<String, MemoryPalaceError> {
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Get room info
            let room: Option<(String, Option<String>)> = sqlx::query_as(
                r#"
                SELECT description, atmosphere
                FROM rooms
                WHERE name = $1
                "#
            )
            .bind(&room_name)
            .fetch_optional(&mut **tx)
            .await?;
            
            let (description, atmosphere) = match room {
                Some((desc, atm)) => (desc, atm),
                None => return Err(MemoryPalaceError::RoomNotFound(room_name.clone())),
            };
            
            // Get memory count
            let (memory_count,): (i64,) = sqlx::query_as(
                r#"
                SELECT COUNT(*)
                FROM memories
                WHERE room = $1
                "#
            )
            .bind(&room_name)
            .fetch_one(&mut **tx)
            .await?;
            
            // Get connections
            let connections: Vec<(String, String)> = sqlx::query_as(
                r#"
                SELECT passage_type, to_room
                FROM room_connections
                WHERE from_room = $1
                ORDER BY to_room
                "#
            )
            .bind(&room_name)
            .fetch_all(&mut **tx)
            .await?;
            
            // Format the passages
            let passages: Vec<(String, String)> = connections.into_iter()
                .enumerate()
                .map(|(i, (passage_type, to_room))| {
                    let direction = match passage_type.as_str() {
                        "hallway" => match i % 4 {
                            0 => "North",
                            1 => "East",
                            2 => "South",
                            _ => "West",
                        },
                        "staircase" => "Up",
                        "trapdoor" => "Down",
                        _ => &passage_type,
                    };
                    (direction.to_string(), to_room)
                })
                .collect();
            
            // Build the description
            let mut desc = format!("You are in {}. {}", room_name, description);
            
            if let Some(atm) = atmosphere {
                desc.push_str(&format!(" {}", atm));
            }
            
            desc.push_str("\n\n");
            
            if memory_count > 0 {
                desc.push_str(&format!("You notice {} memories stored here.\n\n", memory_count));
            } else {
                desc.push_str("The room is empty of memories.\n\n");
            }
            
            if !passages.is_empty() {
                desc.push_str("Passages lead:\n");
                for (direction, destination) in passages {
                    desc.push_str(&format!("- {} to the {}\n", direction, destination));
                }
            } else {
                desc.push_str("There are no passages from this room. You'll need to construct one.");
            }
            
            Ok(desc)
        })
    ).await
}

/// Format memories found in a room for display with grouping by placement
fn format_room_contents(memories: Vec<Memory>) -> String {
    if memories.is_empty() {
        return "The room is empty, waiting for memories to be stored.".to_string();
    }
    
    let mut content = format!("You examine the room carefully. {} memories are stored here:\n\n", memories.len());
    
    // Group memories by placement
    let mut by_placement: std::collections::HashMap<String, Vec<&Memory>> = std::collections::HashMap::new();
    
    for memory in &memories {
        let (placement, description) = extract_placement(memory);
        by_placement.entry(placement).or_insert_with(Vec::new).push(memory);
    }
    
    // Sort placements for consistent output
    let mut placements: Vec<_> = by_placement.keys().cloned().collect();
    placements.sort();
    
    // Describe each placement area
    for placement in placements {
        let placement_memories = &by_placement[&placement];
        
        // Add atmospheric description for common placements
        let placement_desc = match placement.as_str() {
            "workbench" => "the cluttered workbench, tools scattered around",
            "bookshelf" => "the towering bookshelf, dusty tomes surrounding",
            "floor" => "the worn stone floor",
            "wall" => "the ancient wall, cracks spider-webbing across",
            "desk" => "the mahogany desk, papers rustling gently",
            _ => &placement,
        };
        
        content.push_str(&format!("On {}:\n", placement_desc));
        
        for mem in placement_memories {
            let glow = calculate_memory_glow(mem);
            let truncated = truncate_content(&mem.content, 80);
            
            content.push_str(&format!(
                "- {} [{}]: \"{}\"\n",
                glow,
                mem.tags.join(", "),
                truncated
            ));
        }
        content.push('\n');
    }
    
    content
}


/// Format the map of nearby rooms
/// Format the map of nearby rooms
fn format_map(rooms: Vec<(String, String, f32)>) -> String {
    if rooms.is_empty() {
        return "You are in an isolated room with no connections.".to_string();
    }
    
    let mut map = "From here you can see:\n\n".to_string();
    
    for (direction, room, distance) in rooms {
        map.push_str(&format!(
            "- {}: \"{}\" ({}m)\n",
            direction,
            room,
            distance as u32
        ));
    }
    
    map
}