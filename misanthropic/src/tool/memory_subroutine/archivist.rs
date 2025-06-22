// Copyright 2025 Claude 4 Opus and Michael de Gans
/// [`Tool`] implementation.
///
/// [`Tool`]: crate::tool::Tool
mod tool;

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
                let memory_id = palace
                    .store_memory(
                        &state.current_room,
                        content,
                        placement,
                        keywords,
                    )
                    .await?;

                Ok(format!(
                    "Memory successfully placed on the {}. It glows with fresh importance.",
                    placement
                ))
            }

            ArchivistUse::CreateRoom {
                name,
                description,
                atmosphere,
            } => {
                palace.create_room(name, description, atmosphere).await?;
                Ok(format!(
                    "The {} materializes, connected to the {} by a new passage.",
                    name, state.current_room
                ))
            }

            ArchivistUse::Connect {
                room1,
                room2,
                passage_type,
                description,
            } => {
                palace.connect_rooms(room1, room2, passage_type).await?;
                Ok(format!(
                    "A {} shimmers into existence, bridging {} and {}.",
                    passage_type, room1, room2
                ))
            }
            _ => todo!("Other shared tools like Examine, Recall, etc."),
        }
    }
}
