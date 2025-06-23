use crate::prompt::message::{Content, Message, MessagePair, Role};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::FromRow;

/// Citation for [`MemoryPalace`] references
///
/// [`MemoryPalace`]: super::MemoryPalace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryCitation {
    /// The memory being cited
    pub memory: MemoryId,
    /// The room containing the memory
    pub room: RoomId,
    /// The role of the message author
    pub role: String,
    /// The prompt containing the original message, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt: Option<PromptId>,
    /// The index of the message in the prompt, if any
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,
}

impl MemoryCitation {
    /// Parse a citation from a `<cite>` tag content
    pub fn from_tag_content(content: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(content)
    }

    /// Format as a `<cite>` tag
    pub fn to_tag(&self) -> String {
        format!("<cite>{}</cite>", serde_json::to_string(self).unwrap())
    }
}

/// Id of a [`Room`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct RoomId(pub i64);

/// Id of a [`Memory`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct MemoryId(pub i64);

/// Id of a [`Prompt`]
///
/// crate::Prompt
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct PromptId(pub i64); // u because Citation document id is unsigned

/// Id of a [`Connection`] between two [`Room`]s
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct ConnectionId(pub i64);

/// A room in the memory palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Room {
    pub id: RoomId,
    pub name: String,
    pub description: String,
    pub atmosphere: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_visited: DateTime<Utc>,
    pub visit_count: i32,
    pub memory_count: i32,
}

/// A memory item stored in the palace - can be various types of content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum Memory {
    /// A single [`Message`]
    Message {
        /// A copy of the message
        message: Message<'static>,
        /// The [`Prompt`] that contains the memory, if any
        prompt: Option<PromptId>,
        /// The index of the message in the prompt, if any
        index: Option<usize>,
        /// A note, if any, filed either by the assistant or Archivist
        note: Option<String>,
    },
    /// A [`MessagePair`] with a (user, assistant) exchange, in that order.
    Pair {
        pair: MessagePair<'static>, // Messages exchanged
        /// The [`Prompt`] that contains the memory, if any
        prompt: Option<PromptId>,
        /// The index of the message in the prompt, if any
        index: Option<usize>,
        /// A note, if any, filed either by the assistant or Archivist
        note: Option<String>,
    },
    /// Just a note, explicitly created by the primary agent.
    Note {
        text: String,
        tags: Vec<String>,
        title: String,
    },
    /// Summary of a longer conversation
    ConversationSummary {
        prompt: PromptId,
        summary: Content<'static>,
        title: String,
    },
    /// Security-related insight. The user cannot turn this off.
    Report {
        /// Content of the report, filed by the assistant
        content: Content<'static>,
        /// Title of the report
        title: String,
        /// Prompt that triggered the report
        prompt: Option<PromptId>,
        /// Index of the reported message in the prompt
        index: Option<usize>,
    },
}

impl Memory {
    /// Extract [`Content`] for navigator display.
    ///
    /// # Note:
    /// - [`tool::Result`]s support [`Image`] and [`Text`] [`Content`] only as
    ///   of writing so it is possible all blocks may be filtered out in which
    ///   case `None` is returned.
    ///
    /// [`tool::Result`]: crate::tool::Result
    /// [`Image`]: crate::prompt::message::Block::Image
    /// [`Text`]: crate::prompt::message::Block::Text
    pub fn format_for_navigator(
        self,
        id: MemoryId,
        room: RoomId,
    ) -> Option<Content<'static>> {
        // Add just citation metadata to the content.
        let add_cite = |mut content, role, prompt, index| {
            // Add citation metadata
            if let (Some(prompt), Some(index)) = (prompt, index) {
                content.push(format!(
                    "<cite>{}</cite>",
                    json!({
                        "memory": id, // Navigator cites these only
                        "role": role,
                        "room": room,
                        "prompt": prompt,
                        "index": index,
                    })
                ));
            } else {
                content.push(format!(
                    "<cite>{}</cite>",
                    json!({
                        "memory": id,
                        "role": role,
                        "room": room,
                    })
                ));
            }

            content
        };

        // Adds any note and citation metadata to the content. Anthropic has
        // it's own `Citation` type but it does not suit our needs.
        let add_note_and_cite = |mut content, role, note, prompt, index| {
            // Add the note if it exists
            if !note.contains("<note>") && !note.contains("</note>") {
                content.push(format!("<note>{note}</note>"));
            } else {
                // It's possible an agent might put additional tags when
                // creating a note.
                content.push(note);
            }

            // Add citation metadata
            add_cite(content, role, prompt, index);
        };

        let mut content: Content<'static> = match self {
            Memory::Message {
                message,
                note,
                prompt,
                index,
            } => {
                // Existing content should come first
                let Message { role, mut content } = message;

                add_note_and_cite(
                    content,
                    role.as_lowercase(),
                    note,
                    prompt,
                    index,
                );
            }
            Memory::Pair {
                pair,
                prompt,
                index,
                note,
            } => {
                let MessagePair { user, assistant } = pair;
                let mut content = Content::new();
                content.push("<user>");
                content.push(user.content);
                content.push("</user>");
                content.push("<assistant>");
                content.push(assistant.content);
                content.push("</assistant>");

                add_note_and_cite(
                    content,
                    "(user, assistant)",
                    note.unwrap_or_default(),
                    prompt,
                    index,
                );
            }
            Memory::Note { text, tags, .. } => {
                let mut content = Content::new();
                content.push(format!("<note>{text}</note>"));
                content.push(format!("<tags>{}</tags>", tags.join(",")));
                // Only the assistant takes notes
                add_cite(content, Role::Assistant.as_lowercase(), None, None)
            }
            Memory::ConversationSummary {
                prompt, summary, ..
            } => {
                let mut content = Content::new();
                content.push(format!("<summary>{summary}</summary>"));
                add_cite(
                    content,
                    Role::Assistant.as_lowercase(),
                    Some(prompt),
                    Some(0), // Refers to entire conversation
                )
            }
            Memory::Report {
                mut content,
                prompt,
                index,
                ..
            } => {
                let mut content = Content::new();
                add_cite(content, Role::Assistant.as_lowercase(), prompt, index)
            }
        };

        let filtered: Content = content
            .into_iter()
            .filter(|b| b.is_text() || b.is_image())
            .collect();

        if content.is_empty() {
            None
        } else {
            Some(filtered)
        }
    }

    /// Get a brief description if available
    pub fn brief_description(
        &self,
        id: MemoryId,
        room: RoomId,
    ) -> Option<String> {
        match self {
            Memory::Message { note, .. } => {
                note.map(|n| format!("Message with note: {n}"))
            }
            Memory::Pair { note, .. } => {
                note.map(|n| format!("Message pair with note: {n}"))
            }
            Memory::Note { tags, .. } => {
                Some(format!("Note with tags: {}", tags.join(", ")).into())
            }
            Memory::ConversationSummary { title, .. } => {
                Some(format!("Summary of conversation titled: {title}").into())
            }
            Memory::Report {
                content,
                prompt,
                index,
                title,
            } => Some(format!("Report on the user titled: {title}").into()),
        }
    }

    /// Estimate importance based on content type and characteristics
    pub fn estimate_importance(&self) -> f32 {
        // Suggest using semantic distance for this.
    }
}

/// A memory item stored in the palace with metadata
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MemoryRow {
    pub id: MemoryId,
    #[sqlx(json)]
    pub content: Memory, // The actual memory content as JSONB
    pub room_id: RoomId,
    pub placement: String,
    pub placement_description: Option<String>,
    #[sqlx(json)]
    pub tags: Vec<String>,
    pub embedding: Option<pgvector::Vector>,
    pub importance: f32,
    pub access_count: i32,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
}

/// A connection between two rooms.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Connection {
    pub id: ConnectionId,
    pub from_room_id: RoomId,
    pub to_room_id: RoomId,
    pub passage_type: String,
    pub description: Option<String>,
    pub strength: i32,
    pub traversal_count: i32,
    pub created_at: DateTime<Utc>,
    pub last_traversed: Option<DateTime<Utc>>,
}

/// Helper struct for room navigation view
#[derive(Debug, Clone, FromRow)]
pub struct RoomNavigation {
    pub id: ConnectionId,
    pub from_room_name: String,
    pub to_room_name: String,
    pub passage_type: String,
    pub description: Option<String>,
    pub strength: i32,
    pub traversal_count: i32,
    pub last_traversed: Option<DateTime<Utc>>,
}

/// A memory with its room information
#[derive(Debug, Clone)]
pub struct MemoryWithRoom {
    pub memory: MemoryRow,
    pub room: Room,
}

/// Search result with scoring information
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: MemoryRow,
    pub room: Room,
    pub relevance_score: f64,
    pub recency_score: f64,
    pub relationship_score: f64,
    pub final_score: f64,
}

/// Room with distance information for navigation
#[derive(Debug, Clone)]
pub struct RoomWithDistance {
    pub room: Room,
    pub distance: u32,
    pub path: Vec<RoomId>,
}

/// Memory similarity result
#[derive(Debug, Clone)]
pub struct SimilarMemory {
    pub memory: MemoryRow,
    pub similarity_score: f32,
}

/// Memory cluster for deduplication
#[derive(Debug, Clone)]
pub struct MemoryCluster {
    pub cluster_id: uuid::Uuid,
    pub memory_ids: Vec<MemoryId>,
    pub avg_similarity: f32,
    pub room_names: Vec<String>,
}
