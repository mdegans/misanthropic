// Copyright 2025 Claude 4 Opus and Michael de Gans
/// [`Tool`] implementation.
///
/// [`Tool`]: crate::tool::Tool
mod tool;

use serde::{Deserialize, Serialize};

use crate::prompt::message::{Message, Role};
use crate::tool::memory_palace::{MemoryContent, MemoryId, RoomId};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "name", content = "input", rename_all = "snake_case")]
pub enum ArchivistUse {
    // Navigation tools (shared with Navigator)
    Map {
        radius: u32,
    },
    Walk {
        direction: String,
    },
    Examine {
        focus: String,
    },
    Recall {
        topic: String,
        depth: u32,
    },

    // Archivist-specific tools
    Store {
        content: String,
        placement: String,
        keywords: Vec<String>,
    },
    CreateRoom {
        name: String,
        description: String,
        atmosphere: String,
    },
    Connect {
        room1: String,
        room2: String,
        passage_type: String,
        description: Option<String>,
    },
}

impl ArchivistUse {
    pub async fn execute(
        &self,
        palace: &mut MemoryPalace,
        state: &ArchivistState,
    ) -> Result<String, MemorySubroutineError> {
        match self {
            // Shared navigation tools delegate to palace methods
            ArchivistUse::Map { radius } => {
                // Same as Navigator::Map
            }
            ArchivistUse::Walk { direction } => {
                // Same as Navigator::Walk
            }
            // Archivist-specific tools
            ArchivistUse::Store {
                content,
                placement,
                keywords,
            } => {
                // Determine what type of memory this is from context
                let memory =
                    if let Some(pending) = state.pending_memories.first() {
                        match &pending.memory_type {
                            PendingMemoryType::Exchange(messages) => {
                                MemoryContent::Pair {
                                    messages: messages.clone(),
                                    summary: Some(content.clone()),
                                }
                            }
                            PendingMemoryType::Note => MemoryContent::Note {
                                content: content.clone(),
                                tags: keywords.clone(),
                            },
                            PendingMemoryType::Insight {
                                source_ids,
                                confidence,
                            } => MemoryContent::Insight {
                                content: content.clone(),
                                source_memories: source_ids.clone(),
                                confidence: *confidence,
                            },
                        }
                    } else {
                        // Default to note if no pending memory
                        MemoryContent::Note {
                            content: content.clone(),
                            tags: keywords.clone(),
                        }
                    };

                let memory_id = palace
                    .store_memory(
                        &state.current_room.name,
                        memory,
                        placement,
                        None,
                        keywords.clone(),
                        None, // embedding will be added by store_memory
                    )
                    .await?;

                Ok(format!(
                    "Memory successfully placed on the {}. It glows with fresh importance. (id: {})",
                    placement, memory_id
                ))
            }

            ArchivistUse::CreateRoom {
                name,
                description,
                atmosphere,
            } => {
                let room_id = palace
                    .create_room(name, description, Some(atmosphere))
                    .await?;

                // Connect to current room
                palace
                    .connect_rooms(
                        &state.current_room.name,
                        name,
                        Some("passage"),
                        None,
                        None,
                    )
                    .await?;

                Ok(format!(
                    "The {} materializes (id: {}), connected to {} by a new passage.",
                    name, room_id, state.current_room.name
                ))
            }

            ArchivistUse::Connect {
                room1,
                room2,
                passage_type,
                description,
            } => {
                palace
                    .connect_rooms(
                        room1,
                        room2,
                        Some(passage_type),
                        description.as_deref(),
                        None,
                    )
                    .await?;
                Ok(format!(
                    "A {} shimmers into existence, bridging {} and {}.",
                    passage_type, room1, room2
                ))
            }
            _ => todo!("Other shared tools like Examine, Recall, etc."),
        }
    }
}

/// Types of memories the Archivist might store
#[derive(Debug, Clone)]
pub enum PendingMemoryType {
    Exchange(Vec<Message<'static>>),
    Note,
    Insight {
        source_ids: Vec<MemoryId>,
        confidence: f32,
    },
}

/// State for the Archivist agent
#[derive(Debug, Clone)]
pub struct ArchivistState {
    /// Current room the archivist is in
    pub current_room: Room,
    /// Memories to be stored
    pub pending_memories: Vec<PendingMemory>,
}

#[derive(Debug, Clone)]
pub struct PendingMemory {
    pub memory_type: PendingMemoryType,
    pub suggested_room: Option<String>,
    pub suggested_tags: Vec<String>,
}
