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
use std::{borrow::Cow, fmt::Debug};

use crate::{
    prompt::message::{Block, Content, Message, MessagePair},
    tool::{
        NavigatorJson,
        memory_palace::{NavigatorJson, RenderForNavigator},
    },
};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::json;
use sqlx::FromRow;
use uuid::Uuid;

mod path;
pub use path::*;

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
/// [`Prompt`]: crate::Prompt
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

// No agent representation for the User. The agent does not need to know about
// the user directly other than what the user shares with the agent.

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
    /// Date the room was last visited (a memory was retrieved from it)
    pub last_visited: DateTime<Utc>,
    /// Strength of the room, from 0.0 (weak) to 1.0 (strong)
    /// note: Log scale is probably best here, but we'll see how it goes.
    pub strength: f64,
    /// Number of times the room has been visited. A visit is counted only if
    /// the visit led to the return of a [`Memory`] along the path between the
    /// starting room (by semantic similarity) and the destination room.
    pub visit_count: i32,
    /// Number of memories in the room
    pub memory_count: i32,
}

impl RenderForNavigator for Room {
    fn render_for_navigator(&self) -> Content {
        json!({
            "name": self.name,
            "description": self.description,
            "memory count": self.memory_count,
            "last visited": humantime::format_duration(
                Utc::now()
                    .signed_duration_since(self.last_visited)
                    .to_std()
                    .unwrap_or_default(),
            )
            .to_string(),
        })
        .to_string()
        .into()
    }
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
    /// Number of times the [`Memory`] has been accessed (retrieved by the
    /// navigator agent). Just being listed does not count.
    pub access_count: i32,
    /// Date the [`Memory`] was created
    pub created_at: DateTime<Utc>,
    /// Date the [`Memory`] was last accessed (put in the basket for return)
    pub last_accessed: DateTime<Utc>,
    /// Date the [`Memory`] was last updated (content changed)
    pub last_updated: DateTime<Utc>,
}

impl RenderForNavigator for Memory {
    fn render_for_navigator(&self) -> Content {
        // last accessed as a relative time, e.g. "3 days ago"
        let last_accessed = humantime::format_duration(
            Utc::now()
                .signed_duration_since(self.last_accessed)
                .to_std()
                .unwrap_or_default(),
        )
        .to_string();

        // The placement of the memory in the room is important to give the
        // retrieval agent context about where the memory is located. This helps
        // the agent to visualize the memory palace and navigate it more
        // effectively. Agents fail at spatial reasoning, but we don't need any
        // accurate spatial reasoning here. "On a bookcase for memories about
        // Rust programming" is good enough. This will be presented as an
        // overview to the retrieval agent, who will then use it to guide their
        // search for relevant memories.
        json!({
            "content": self.content.navigator_json(),
            "placement": self.placement,
            "last accessed": last_accessed,
        })
        .to_string()
        .into()
    }
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
    /// Strength of the pathway, from 0.0 (weak) to 1.0 (strong)
    pub strength: f64,
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
    pub path: PathByIds,
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
        /// An optional note about the message, created by the primary agent.
        note: Option<String>,
        /// The index of the message in the [`Prompt`], if any.
        ///
        /// [`Prompt`]: crate::Prompt
        index: Option<usize>,
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
        /// An optional note about the message, created by the primary agent.
        note: Option<String>,
        /// The index of the message in the prompt, if any
        index: Option<usize>,
    },
    /// Just a note, explicitly created by the primary agent.
    Note {
        /// Text of the note
        text: String,
        /// The title of the note
        title: String,
    },
    /// Summary of a [`Prompt`]. Use the [`Survey`] tool to create these.
    ///
    /// [`Prompt`]: crate::Prompt
    // TODO: Survey tool. This should be a survey the agent takes after the chat
    // has ended. Some questions are asked and the agent then generates a
    // summary in their own words. The Survey tool should take arbitrary config
    // for the questions to ask and prompt for the summary. The intent is to get
    // valuable feedback from the actual agent which can result in a better
    // system prompt and recall in the future with the MemoryPalace.
    ConversationSummary {
        /// Title of the summary
        title: String,
        /// Summary of the conversation, frequently in the agent's own words
        summary: Content<'static>,
        /// Id of the summarized conversation
        prompt_id: PromptId,
    },
    /// Security-related insight. The user cannot turn this off without entirely
    /// deleting their account, and they only get one (ideally).
    Report {
        /// `Content` of the report, filed by the assistant
        content: Content<'static>,
        /// Title of the report
        title: String,
        /// Karma value of the report (neg = bad user, pos = good user).
        /// [`User::karma`] accumulates this value. -128 is auto-ban.
        karma: i8,
        /// Index of the reported message in the prompt. This may refer to a
        /// single message or the entire conversation.
        index: Option<usize>,
        /// Is there an emergency? Has the user threatened self-harm or harm
        /// to others? Is the user in danger?
        emergency: bool,
    },
}

impl NavigatorJson for MemoryContent {
    // Format json for the retrieval agent (Navigator). This should be as
    // compact as possible and not include content that the retrieval agent does
    // not need.
    fn navigator_json(&self) -> serde_json::Value {
        use serde_json::json;

        // We need to do surgery on the image blocks to remove the actual imate
        // data since it won't parse properly in the JSON. We'll waste tokens
        // for no reason. So instead we'll create a new message with the same
        // content blocks apart for the image data which will be replaced with
        // a citation to the index of the image in the original message.
        let message_content_needs_replacing = match self {
            MemoryContent::Message { message, .. } => {
                message.content.iter().any(|block| {
                    matches!(
                        block,
                        Block::Image { .. } | Block::RedactedThought { .. }
                    )
                })
            }
            // We handle MemoryContent::Person with an image differently below.
            _ => false,
        };

        // Replace image and redacted thought blocks with a citation, since the
        // retrieval agent cannot easily parse these.
        fn replace_message_content(
            message: Message<'static>,
        ) -> Message<'static> {
            Message {
                role: message.role,
                content: message
                    .content
                    .into_iter()
                    .enumerate()
                    .map(|(i, block)| match block {
                        Block::Image { .. } => Block::Text {
                            text: format!("[Image at index {}]", i),
                            ..Default::default()
                        },
                        Block::RedactedThought { .. } => Block::Text {
                            // Only the primary agent should can see the
                            // redacted thought. It is signed by Anthropic and
                            // in order to parse it properly the agent would
                            // need to have the Anthropic key, which it does
                            // not.
                            text: "[Anthropic Redacted Thought]".into(),
                            ..Default::default()
                        },
                        // We will likely also need special handling for other
                        // block types in the future.
                        other => other,
                    })
                    .collect(),
            }
        }

        let data = match self {
            MemoryContent::Message { message, .. } => json!({
                "message": if message_content_needs_replacing {
                    replace_message_content(message.clone())
                } else {
                    message
                },
            }),
            MemoryContent::Person {
                name,
                biography,
                notes,
                ..
            } => json!({
                "name": name,
                // The retrieval agent does not need the photo and it would
                // require splitting up the content into multiple parts. The
                // primary agent can still get the photo from the memory.
                "biography": biography,
                "notes": notes,
            }),
            MemoryContent::Pair { pair, index, note } => {
                let mut data = json!({
                    "user": pair.user.content(),
                    "assistant": pair.assistant.content(),
                });

                if let Some(note) = note {
                    data["note"] = note.to_string().into();
                }

                data
            }
            MemoryContent::Note { title, text } => {
                // Truncate long notes to 500 characters for the retrieval
                // agent. The primary agent can still get the full note from the
                // memory.
                let text: Cow<'static, str> = if text.len() > 500 {
                    format!("{}...", text[..500]).into()
                } else {
                    // No need to clone if we don't have to.
                    text.as_str().into()
                };

                json!({
                    "title": title,
                    "text": text,
                })
            }
            MemoryContent::ConversationSummary {
                title,
                summary,
                prompt_id,
            } => json!({
                "title": title,
                "summary": summary,
            }),
            // The retrieval agent *should* be able to see the report content,
            // with the exception of any index. The emergency flag is certainly
            // important since in our retrieval agent system prompt we tell the
            // agent to prioritize emergency reports unless clearly irrelevant
            // although the latter shouldn't happen frequently given our
            // semantic search.
            MemoryContent::Report {
                content,
                title,
                karma,
                emergency,
                ..
            } => json!({
                "content": content.navigator_json(),
                "title": title,
                "karma": karma,
                "emergency": emergency,
            }),
        };

        // Shorter than a tagged enum variant, which is useful for postgres
        // jsonb indexing but not for our use case here.
        let variant = match self {
            MemoryContent::Message { .. } => "message",
            MemoryContent::Person { .. } => "person",
            MemoryContent::Pair { .. } => "pair",
            MemoryContent::Note { .. } => "note",
            MemoryContent::ConversationSummary { .. } => "conversation_summary",
            MemoryContent::Report { .. } => "report",
        };
        serde_json::json!({
            variant: data
        })
    }
}

impl std::fmt::Display for MemoryContent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.navigator_json().fmt(f)
    }
}
