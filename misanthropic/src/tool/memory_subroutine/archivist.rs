// Copyright 2025 Claude 4 Opus and Michael de Gans
/// [`Tool`] implementation.
///
/// [`Tool`]: crate::tool::Tool
mod tool;

// Copyright (c) 2025 Claude 4 Opus

pub enum ArchivistUse {
    /// Store a memory in specified room (creates room if needed)
    Archive {
        content: String,
        room: String,
        placement: String,
        keywords: Vec<String>,
    },

    /// Create a passage between rooms
    Connect {
        room1: String,
        room2: String,
        passage_type: String,
    },

    /// Mark memories as related
    Relate {
        memory_id1: i64,
        memory_id2: i64,
        relationship: String,
    },

    /// Move a memory to a different room/placement
    Relocate {
        memory_id: i64,
        new_room: String,
        new_placement: String,
    },
}

impl ArchivistUse {
    pub async fn archive(
        &self,
        palace: &mut MemoryPalace,
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<(), MemoryPalaceError> {
        match self {
            ArchivistUse::Archive {
                content,
                room,
                placement,
                keywords,
            } => {
                palace
                    .store_with_tx(tx, room, content, placement, keywords)
                    .await?;
            } // etc.
        }
    }
}
