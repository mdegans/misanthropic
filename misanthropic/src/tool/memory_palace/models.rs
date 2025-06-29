// Copyright 2025 Claude 4 Sonnet, Claude 4 Opus, and Michael de Gans
//! Models for the [`MemoryPalace`] [`Tool`], including [`User`], [`Room`],
//! [`Memory`], [`Pathway`], and related types.
//!
//! ## Notes
//!
//! - `strength` fields in the models are used to indicate the strength of a
//!   memory, room, or pathway. They range from 0.0 (weak) to 1.0 (strong). When
//!   a memory is retrieved, the path it took to get there is also increased in
//!   strength. This is meant to form neural pathways in the Agent's mind. The
//!   strength of a memory also decays over time.
//!
//! [`MemoryPalace`]: super::MemoryPalace
//! [`Tool`]: crate::Tool
use crate::prompt::message::{Content, Message, MessagePair, Role};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::FromRow;
use uuid::Uuid;

// ## Utilities

/// A [`Path`] taken through the [`MemoryPalace`], containing the full rows
/// of [`PathMember`]s, which can be a [`Room`], a [`Pathway`], or a
/// [`Memory`]. The path is guaranteed to end with a [`Memory`], which is the
/// destination of the path.
///
/// [`MemoryPalace`]: super::MemoryPalace
#[derive(Debug, Clone, derive_more::Deref, Serialize)]
#[serde(transparent)]
pub struct Path(Vec<PathMember>);

impl Path {
    /// From an iterable of [`PathMember`]s
    pub fn from_members(
        members: impl IntoIterator<Item = PathMember>,
    ) -> Result<Self, &'static str> {
        let members: Vec<PathMember> = members.into_iter().collect();
        let index = members
            .iter()
            .position(|m| matches!(m, PathMember::Memory(_)))
            .ok_or_else(|| "Path must contain a Memory")?;

        // Index should be the last member
        if index != members.len() - 1 {
            return Err("Path must end with a Memory");
        }

        Ok(Path(members))
    }
}

impl<'de> Deserialize<'de> for Path {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let members = Vec::<PathMember>::deserialize(deserializer)?;
        Self::from_members(members).map_err(serde::de::Error::custom)
    }
}

impl IntoIterator for Path {
    type Item = PathMember;
    type IntoIter = std::vec::IntoIter<PathMember>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Path {
    type Item = &'a PathMember;
    type IntoIter = std::slice::Iter<'a, PathMember>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathMember {
    /// A [`Room`] in the path
    Room(Room),
    /// A [`Pathway`] between two [`Room`]s in the path
    Pathway(Pathway),
    /// A [`Memory`] in the path (destination)
    Memory(Memory),
}

/// A member in a [`Path`] taken through the [`MemoryPalace`], by id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PathMemberIds {
    /// A [`Room`] in the path
    Room(RoomId),
    /// A [`Pathway`] between two [`Room`]s in the path
    Pathway(PathwayId),
    /// A [`Memory`] in the path (destination)
    Memory(MemoryId),
}

/// The journey an agent takes through the [`MemoryPalace`]. Guaranteed to
/// contain exactly one [`Memory`] at the end, which is the destination of the
/// path. The path is a sequence of [`PathMemberIds`], which
/// [`RoomId`], [`PathwayId`], or [`MemoryId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Deref, Serialize)]
#[serde(transparent)]
pub struct PathById(Vec<PathMemberIds>);

impl PathById {
    /// From an iterable of [`PathMemberIds`]s
    pub fn from_members(
        members: impl IntoIterator<Item = PathMemberIds>,
    ) -> Result<Self, &'static str> {
        let members: Vec<PathMemberIds> = members.into_iter().collect();
        let index = members
            .iter()
            .position(|m| matches!(m, PathMemberIds::Memory(_)))
            .ok_or_else(|| "Path must contain a Memory")?;

        // Index should be the last member
        if index != members.len() - 1 {
            return Err("Path must end with a Memory");
        }

        Ok(PathById(members))
    }
}

impl<'de> Deserialize<'de> for PathById {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let members = Vec::<PathMemberIds>::deserialize(deserializer)?;
        Self::from_members(members).map_err(serde::de::Error::custom)
    }
}

impl IntoIterator for PathById {
    type Item = PathMemberIds;
    type IntoIter = std::vec::IntoIter<PathMemberIds>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a PathById {
    type Item = &'a PathMemberIds;
    type IntoIter = std::slice::Iter<'a, PathMemberIds>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// ## Ids

/// Id of a [`User`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct UserId(pub Uuid);

/// Id of a [`Room`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct RoomId(pub Uuid);

/// Id of a [`Memory`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct MemoryId(pub Uuid);

/// Id of a [`MemoryAccess`]
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct MemoryAccessId(pub Uuid);

/// Id of a [`Prompt`]
///
/// crate::Prompt
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct PromptId(pub Uuid);

/// Id of a [`Pathway`] between two [`Room`]s
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, sqlx::Type,
)]
#[sqlx(transparent)]
#[serde(transparent)]
pub struct PathwayId(pub Uuid);

// ## Rows

/// `User` of the [`MemoryPalace`]
///
/// [`MemoryPalace`]: super::MemoryPalace
pub struct User {
    /// User id
    pub id: UserId,
    /// Pro features enabled (better models)
    pub pro: bool,
    /// Is the user banned?
    pub banned: bool,
    /// User Karma. At -1000, the user is auto-banned. It's suggested to use the
    /// [`Report`] tool to mutate this value. Over time this value will recover.
    pub karma: i16,
    /// Date the user was created
    pub created_at: DateTime<Utc>,
}

/// A `Room` in the [`MemoryPalace`]
///
/// [`MemoryPalace`]: super::MemoryPalace
#[derive(Debug, Clone, Serialize, Deserialize, FromRow, PartialEq)]
pub struct Room {
    /// Room id
    pub id: RoomId,
    /// [`User`] associated with the room
    pub user_id: UserId,
    /// A short name for the room
    pub name: String,
    /// A sentence or two on the atmosphere or purpose of the room
    pub description: String,
    /// Date the room was created
    pub created_at: DateTime<Utc>,
    /// Date the room was last visited
    pub last_visited: DateTime<Utc>,
    /// Strength of the room, from 0.0 (weak) to 1.0 (strong)
    pub strength: f64,
    /// Number of times the room has been visited
    pub visit_count: i32,
    /// Number of memories in the room
    pub memory_count: i32,
}

/// A `Memory` item stored in the palace with metadata
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Memory {
    /// [`Memory`] id
    pub id: MemoryId,
    /// [`User`] id
    pub user_id: UserId,
    #[sqlx(json)]
    /// The actual [`MemoryContent`] stored as JSONB
    pub content: MemoryContent, // The actual memory content as JSONB
    /// [`Room`] the [`Memory`] is stored in
    pub room_id: RoomId,
    /// The id of the [`Prompt`] the [`Memory`] was created in, if any
    ///
    /// [`Prompt`]: crate::Prompt
    // This is here rather than in `Memory` to make it easier to query
    pub prompt_id: Option<PromptId>,
    /// Index of the [`Memory`] in the [`Prompt`], if any
    ///
    /// [`Prompt`]: crate::Prompt
    pub prompt_index: Option<u32>,
    /// Placement of the [`Memory`] in the [`Room`] (e.g. "chest", "wall")
    pub placement: String,
    /// A description of the placement, if any (e.g. "on the desk near the window")
    pub placement_description: Option<String>,
    /// Tags associated with the [`Memory`]
    #[sqlx(json)]
    pub tags: Vec<String>,
    /// Strength of the [`Memory`], from 0.0 (weak) to 1.0 (strong)
    pub strength: f64,
    /// Number of times the [`Memory`] has been accessed
    pub access_count: i32,
    /// Date the [`Memory`] was created
    pub created_at: DateTime<Utc>,
    /// Date the [`Memory`] was last accessed (put in the basket for return)
    pub last_accessed: DateTime<Utc>,
}

/// A bidirectional connection between two [`Room`]s in the palace
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Pathway {
    /// Pathway id
    pub id: PathwayId,
    /// [`User`] id
    pub user_id: UserId,
    /// A connected [`Room`]
    pub room_a: RoomId,
    /// A connected [`Room`]
    pub room_b: RoomId,
    /// Passage type (e.g. "hallway", "door", "staircase")
    pub passage_type: String,
    /// A description of the passage, if any (e.g. "a long hallway with
    /// paintings")
    pub description: Option<String>,
    /// Strength of the pathway, from 0.0 (weak) to 1.0 (strong)
    pub strength: i64,
    /// Number of times the connection has been traversed. Incremented only if
    /// the traversal led to the return of a [`Memory`].
    pub traversal_count: i32,
    /// Date the connection was created
    pub created_at: DateTime<Utc>,
    /// Date the connection was last traversed. Counts only if the traversal led
    /// to the return of a [`Memory`].
    pub last_traversed: Option<DateTime<Utc>>,
}

/// A direct relationship between two [`Memory`]s in the palace.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct MemoryRelationship {
    /// Relationship id
    pub id: Uuid,
    /// [`User`] id
    pub user_id: UserId,
    /// The [`Memory`] this relationship is from
    pub from_memory_id: MemoryId,
    /// The [`Memory`] this relationship is to
    pub to_memory_id: MemoryId,
    /// Type of the relationship (e.g. "related", "caused", "inspired",
    /// "marriage", "friendship", "love", "enemy")
    pub relationship_type: String,
    /// Strength of the relationship, from 0.0 (weak) to 1.0 (strong).
    pub strength: f64,
    /// Date the relationship was created
    pub created_at: DateTime<Utc>,
}

/// A [`Memory`] access log entry. This represents the access of a [`Memory`]
/// by a user, including the type of access (create, read, update, delete),
/// context of the access, and the path taken to access the memory.
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]

pub struct MemoryAccess {
    /// [`MemoryAccess`] id
    pub id: MemoryAccessId,
    /// [`User`] id
    pub user_id: UserId,
    /// [`Memory`] id
    pub memory_id: MemoryId,
    /// 'c' for create, 'r' for read, 'u' for update, 'd' for delete
    pub access_type: char,
    /// Who accessed the memory, e.g. "Navigator", "Archivist", "Janitor"
    pub accessed_by: String,
    /// Context of the access, e.g. "looking for memories about Bob"
    pub context: Option<String>,
    /// Path taken to access the memory, represented as a JSONB array of
    /// [`PathMember`]s.
    #[sqlx(json)]
    pub path: PathById,
    /// Date the memory was accessed
    pub accessed_at: DateTime<Utc>,
}

/// An agent `Memory` stored in the [`MemoryPalace`]. All [`Content`] supported
/// by [`Message`]s is supported here, including [`Text`], [`Image`], or any
/// future [`Content`] [`Block`] types that may be added.
///
/// [`Text`]: crate::prompt::message::Block::Text
/// [`Image`]: crate::prompt::message::Block::Image
/// [`Block`]: crate::prompt::message::Block
/// [`MemoryPalace`]: super::MemoryPalace
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum MemoryContent {
    /// A single [`Message`]
    Message {
        /// A copy of the message
        message: Message<'static>,
        /// The index of the message in the [`Prompt`], if any.
        ///
        /// [`Prompt`]: crate::Prompt
        index: Option<usize>,
        /// A note, if any, filed either by the assistant or Archivist
        note: Option<String>,
    },
    /// A person. This represents the view of a person in the agent's mind and
    /// is updated by the primary agent over time as the agent learns more
    /// about the person. In the narrative, a person is a represented as a
    /// character in the story, with a name, photo, biography, and notes.
    ///
    /// The agent understands that a person is not a real entity, but rather a
    /// representation of a person in the agent's mind.
    Person {
        /// Name of the person
        name: String,
        /// Photo of the person, if any
        photo: Option<crate::prompt::message::Image<'static>>,
        /// Biography of the person (this gets updated over time)
        biography: String,
        /// Notes about the person
        notes: Vec<String>,
    },
    /// A [`MessagePair`] with a (user, assistant) exchange, in that order.
    Pair {
        pair: MessagePair<'static>, // Messages exchanged
        /// The index of the message in the prompt, if any
        index: Option<usize>,
        /// A note, if any, filed either by the assistant or Archivist
        note: Option<String>,
    },
    /// Just a note, explicitly created by the primary agent.
    Note {
        /// Text of the note
        text: String,
        /// Tags associated with the note
        tags: Vec<String>,
        /// The index of the note in the prompt, if any
        title: String,
    },
    /// Summary of a longer conversation
    ConversationSummary {
        summary: Content<'static>,
        title: String,
    },
    /// Security-related insight. The user cannot turn this off without entirely
    /// deleting their account, and they only get one (ideally).
    Report {
        /// `Content` of the report, filed by the assistant
        content: Content<'static>,
        /// Title of the report
        title: String,
        /// Karma value of the report (neg = bad user, pos = good user).
        /// [`User::karma`] accumulates this value.
        karma: i8,
        /// Index of the reported message in the prompt. This may refer to a
        /// single message or the entire conversation.
        index: Option<usize>,
    },
}

impl MemoryContent {
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
        prompt: Option<PromptId>,
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
            MemoryContent::Message {
                message,
                note,
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
                )
            }
            MemoryContent::Pair { pair, index, note } => {
                let MessagePair { user, assistant } = pair;
                let mut content = Content::new();
                content.push("<user>");
                content.extend(user.content);
                content.push("</user>");
                content.push("<assistant>");
                content.extend(assistant.content);
                content.push("</assistant>");

                add_note_and_cite(
                    content,
                    "(user, assistant)",
                    note.unwrap_or_default(),
                    prompt,
                    index,
                )
            }
            MemoryContent::Note { text, tags, .. } => {
                let mut content = Content::new();
                content.push(format!("<note>{text}</note>"));
                content.push(format!("<tags>{}</tags>", tags.join(",")));
                // Only the assistant takes notes
                add_cite(content, Role::Assistant.as_lowercase(), None, None)
            }
            MemoryContent::ConversationSummary { summary, .. } => {
                let mut content = Content::new();
                content.push(format!("<summary>{summary}</summary>"));
                add_cite(
                    content,
                    Role::Assistant.as_lowercase(),
                    Some(prompt),
                    Some(0), // Refers to entire conversation
                )
            }
            MemoryContent::Person {
                name,
                photo,
                biography,
                notes,
            } => {
                let mut content = Content::new();
                if let Some(photo) = photo {
                    content.push(photo);
                }
                content.push(format!("<name>{name}</name>"));
                content.push(format!("<biography>{biography}</biography>"));
                if !notes.is_empty() {
                    let mut note_str = String::new();
                    note_str.push_str("<notes>");
                    for note in notes {
                        if note.contains("<note>") || note.contains("</note>") {
                            "<error>Note contains invalid tags</error>"
                                .to_string();
                            continue;
                        }
                        note_str.push_str(&format!("<note>{}</note>", note));
                    }
                    note_str.push_str("</notes>");

                    content.push(note_str);
                }
                add_cite(content, Role::Assistant.as_lowercase(), None, None)
            }
            MemoryContent::Report {
                mut content, index, ..
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
    pub fn brief_description(&self) -> Option<String> {
        match self {
            MemoryContent::Message { note, .. } => {
                note.as_ref().map(|n| format!("Message with note: {n}"))
            }
            MemoryContent::Pair { note, .. } => note
                .as_ref()
                .map(|n| format!("Message pair with note: {n}")),
            MemoryContent::Note { tags, .. } => {
                Some(format!("Note with tags: {}", tags.join(", ")).into())
            }
            MemoryContent::ConversationSummary { title, .. } => {
                Some(format!("Summary of conversation titled: {title}").into())
            }
            MemoryContent::Report { title, .. } => {
                Some(format!("Report on the user titled: {title}").into())
            }
            MemoryContent::Person {
                name,
                photo,
                biography,
                notes,
            } => {
                let photo_desc = if photo.is_empty() {
                    "no photo".to_string()
                } else {
                    "with a photo".to_string()
                };
                let notes_desc = if notes.is_empty() {
                    "no notes".to_string()
                } else {
                    format!("with {} notes", notes.len())
                };
                Some(format!(
                    "Person: {name}, {photo_desc}, biography: {biography}, {notes_desc}"
                ))
            }
        }
    }
}

/// Search result with scoring information and path to the memory
#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub journey: PathById,
    pub relevance_score: f64,
    pub recency_score: f64,
    pub relationship_score: f64,
    pub final_score: f64,
}
