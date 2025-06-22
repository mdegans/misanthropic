// Copyright (c) 2025 Claude 4 Opus and Michael de Gans

use sqlx::PgPool;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

use crate::tool::{self, embedding::{EmbeddingClient as EmbeddingClient, TextEmbedding}, memory_palace::{Memory, MemoryPalaceError, Room}, memory_subroutine::MemorySubroutineError, MemoryPalace, Method, Tool};

/// `Navigator` agent [`Tool`] for exploring the [`MemoryPalace`].
pub struct Navigator {
    /// The [`MemoryPalace`] to navigate (very fancy database).
    palace: MemoryPalace,
    /// [`Room`] the agent is currently in.
    current_room: Room,
    /// The path the agent has taken through the palace, mostly for debugging.
    journey: Vec<i64>,
    /// Context for the current mission. Query driving navigation.
    mission_context: String,
    /// [`Memory`]s for delivery to the primary agent.
    basket: MemoryBasket,
    /// Embedding client (for looking up embeddings of context and memories).
    emb_client: Box<dyn EmbeddingClient>,
}

/// [`Memory`]s found during navigation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryBasket {
    /// [`Memory`]s for the subroutine to return to the primary agent.
    pub memories: Vec<CollectedMemory>,
}

/// Content of a [`Memory`] collected during navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMemory {
    /// Id of the [`Memory`] for the primary agent (or end user) to reference
    /// the source and surrounding context directly.
    pub id: i64,
    /// The formatted content of the [`Memory`] for the primary agent to read.
    /// This should be in the form or narrative prose, not raw data. For
    /// example, "The agent remembers the user is a software engineer." The
    /// primary agent can refer back to the original in case of ambiguity.
    pub content: String,
    /// This is mostly for debugging. This is a copy of the chain of thought
    /// that led to this [`Memory`] being chosen for the basket.
    // Issue here is there's a 1:many relationship between notes and memories
    // so we will copy this a bunch of times. could this go somewhere else? Do
    // we need it? If we do, the copies are probably fine. We can use an Arc if
    // it ever becomes a problem.
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
    /// Create a new `Navigator` starting close to `context`, or in the
    /// "Entrance Hall" if no suitable room is found (usually empty palace).
    pub async fn new(
        palace: crate::tool::MemoryPalace,
        context: String,
        embedding_client: Box<dyn EmbeddingClient>
    ) -> Result<Self, MemorySubroutineError> {
        // Find the closest starting room based on the context embedding
        let context_embedding = embedding_client.get_embedding(&context).await?;
        let (current_room, _similarity) = find_closest_to(
            &palace.pool,
            palace.schema(), // palace.user() might be handy for multi-user
            context_embedding
        ).await?; // This should return an error because it is a logic error to
        // call the navigator without having first called the archivist to
        // initialize the palace. Otherwise the caller would be wasting money
        // especially if this happens in a loop. And this probably means us.

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
                    // TODO: Use ids instead. Content is O(n) and an id is O(1).
                    // with very short content is might be faster but we haven't
                    // put caps on the content length yet and it's probablu just
                    // better to use ids.
                    self.palace.get_room_memories(&self.current_room.name).await?
                } else {
                    self.palace.search_in_room(&self.current_room.name, &focus).await?
                };

                Ok(format_room_contents(memories))
            }
            NavigatorUse::Walk { direction } => {
                let new_room = self.palace
                    .follow_passage(&self.current_room.name, &direction)
                    .await?;

                let description = self.palace.get_room_description(new_room.clone()).await?;

                // Update state
                self.current_room = new_room.clone();
                self.journey.push(new_room);

                Ok(description)
            }
            NavigatorUse::Map { radius } => {
                // Implementation from existing code
                let rooms_by_distance = self.palace
                    .get_rooms_within_radius(&self.current_room, radius)
                    .await?;
                
                let current_centroid = self.palace
                    .get_room_centroid(&self.current_room)
                    .await?;
                
                let mut narrative = format!(
                    "The navigator surveys the palace from the {}...\n\n",
                    self.current_room
                );
                
                // ... rest of existing Map implementation
                
                Ok(narrative)
            }
            NavigatorUse::Recall { topic, depth } => {
                // Implementation from existing code  
                let topic_embedding = self.emb_client.get_embedding(&topic).await?;
                
                let initial_memories = self.palace.semantic_search_all_rooms(
                    &topic_embedding,
                    5,
                ).await?;
                
                if initial_memories.is_empty() {
                    return Ok(format!(
                        "The navigator closes their eyes and thinks of \"{}\"...\n\n\
                        But the palace remains silent. No memories resonate with this topic.",
                        topic
                    ));
                }
                
                // ... rest of existing Recall implementation with IDs included
                
                Ok(narrative)
            }
            NavigatorUse::AddToBasket { memory_ids, relevance_notes } => {
                // Fetch full memory details for each ID
                // TODO: This can be a single query in the future.
                for id in memory_ids {
                    if let Ok(memory) = self.palace.get_memory_by_id(id).await {
                        self.basket.memories.push(CollectedMemory {
                            id: memory.id,
                            content: memory.content,
                            // TODO: we could consider relocating this. It's
                            // mostly for debug, so do we need it, or can we log
                            // it instead?
                            relevance_notes: relevance_notes.clone(),
                        });
                    }
                }
                
                Ok(format!(
                    "Added {} memories to basket. Current basket size: {} memories.",
                    memory_ids.len(),
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
                
                result.push_str(&format!("\nSummary: {}", summary));
                
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

// Find (room, similarity) pair for a query embedding
pub async fn find_closest_to(
    pool: &PgPool,
    schema: &str,
    m: TextEmbedding,
) -> Result<(Room, f32), MemoryPalaceError> {
    crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Find the room with memories most similar to the query
            let (room, similarity):(Room, f32)  = sqlx::query_as(
                r#"
                SELECT *, AVG(1 - (m.embedding <=> $1::vector)) as similarity
                FROM memories m
                JOIN rooms r ON m.room = r.name
                WHERE m.embedding IS NOT NULL
                GROUP BY r.name
                ORDER BY similarity DESC
                LIMIT 1
                "#,
            )
            .bind(m.embedding)
            .fetch_one(&mut **tx)
            .await?;

            Ok((room, similarity))
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
) -> Result<Vec<Room>, MemoryPalaceError> {
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
) -> Result<Room, MemoryPalaceError> {
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
    room_id: i64,
) -> Result<String, MemorySubroutineError> {
    Ok(crate::tool::memory_palace::execute_with_schema(
        pool,
        schema,
        |tx| Box::pin(async move {
            // Get room info
            let room: Room = match sqlx::query_as(
                r#"
                SELECT *
                FROM rooms
                WHERE id = $1
                "#
            )
            .bind(room_id)
            .fetch_optional(&mut **tx)
            .await {
                Ok(Some(room)) => room,
                Ok(None) => return Err(MemorySubroutineError::RoomNotFound(
                    format!("Room with ID {} not found", room_id)
                )),
                Err(e) => return Err(MemorySubroutineError::DatabaseError(
                    format!("Failed to fetch room: {}", e)
                )),
            };
            
            
            // Get memory count
            let (memory_count,): (i64,) = sqlx::query_as(
                r#"
                SELECT COUNT(*)
                FROM memories
                WHERE room = $1
                "#
            )
            .bind(&room.name)
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
            let mut desc = format!("You are in {}. {}", name, description);
            
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
    ).await?)
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
        // FIXME: The agent should be able to add custom descriptions. A
        // `placement` table may be appropriate.
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

