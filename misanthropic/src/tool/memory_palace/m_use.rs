use crate::tool;
use serde::{Deserialize, Serialize};

use super::{MemoryPalace, MemoryPalaceError};

/// [`tool::Use`] for the [`MemoryPalace`] tool - a spatial navigation interface.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "name", content = "input")]
pub enum Use {
    /// Inscribe a new memory onto an object in the current room.
    #[serde(rename = "MemoryPalace::inscribe")]
    Inscribe {
        /// The memory to inscribe.
        content: String,
        /// Where to place it (bookshelf, painting, statue, floor, etc.)
        placement: String,
        /// Keywords for finding this memory later.
        keywords: Vec<String>,
    },

    /// Examine the current room or a specific object/area.
    #[serde(rename = "MemoryPalace::examine")]
    Examine {
        /// What to look for (empty string = describe everything).
        focus: String,
    },

    /// Walk through a doorway to an adjacent room.
    #[serde(rename = "MemoryPalace::walk")]
    Walk {
        /// Which doorway to take (north, south, east, west, or specific passage name).
        direction: String,
    },

    /// Construct a new passage connecting rooms.
    #[serde(rename = "MemoryPalace::construct")]
    Construct {
        /// Type of passage (hallway, secret door, bridge, staircase, portal).
        passage_type: String,
        /// The room to connect to.
        destination: String,
    },

    /// View a map of nearby rooms.
    #[serde(rename = "MemoryPalace::map")]
    Map {
        /// How many rooms away to include (1 = adjacent only).
        radius: u32,
    },

    /// Recall memories on a topic from across the entire palace.
    #[serde(rename = "MemoryPalace::recall")]
    Recall {
        /// The topic or memory fragment to search for.
        topic: String,
        /// How deeply to search connecting memories (1-5).
        depth: u32,
    },
}

impl TryFrom<tool::Use<'_>> for Use {
    type Error = MemoryPalaceError;

    fn try_from(call: tool::Use<'_>) -> Result<Self, Self::Error> {
        // Check if this is a MemoryPalace call
        if !call.name.starts_with("MemoryPalace::") {
            return Err(MemoryPalaceError::InvalidInput(format!(
                "Not a MemoryPalace call: {}",
                call.name
            )));
        }

        // Deserialize directly using the tagged enum
        serde_json::from_value(serde_json::json!({
            "name": call.name,
            "input": call.input,
        }))
        .map_err(|e| {
            MemoryPalaceError::InvalidInput(format!(
                "Invalid parameters: {}",
                e
            ))
        })
    }
}

// Helper struct to maintain navigation state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationState {
    /// Current room the agent is in
    pub current_room: String,
    /// History of visited rooms in this session
    pub visited_rooms: Vec<String>,
    /// The mission or query driving this navigation
    pub mission: Option<String>,
}

impl Default for NavigationState {
    fn default() -> Self {
        Self {
            current_room: "Entrance Hall".to_string(),
            visited_rooms: vec!["Entrance Hall".to_string()],
            mission: None,
        }
    }
}

/// Format a room description with memories
pub fn format_room_description(
    room_name: &str,
    room_desc: &str,
    memory_count: usize,
    passages: &[(String, String)], // (direction, destination)
) -> String {
    let mut desc = format!("You are in the {}. {}\n\n", room_name, room_desc);

    if memory_count > 0 {
        desc.push_str(&format!(
            "You notice {} memories stored here.\n\n",
            memory_count
        ));
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

    desc
}

/// Format memory objects as they appear in rooms
pub fn format_memory_object(
    content: &str,
    placement: &str,
    keywords: &[String],
    recency_glow: f64, // 0.0 to 1.0
) -> String {
    let glow_desc = match recency_glow {
        g if g > 0.8 => "glowing brightly",
        g if g > 0.5 => "glowing steadily",
        g if g > 0.2 => "glowing faintly",
        _ => "barely visible",
    };

    format!(
        "On the {}, {} [{}]: \"{}\"",
        placement,
        glow_desc,
        keywords.join(", "),
        content
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_use_serialization() {
        let inscribe = Use::Inscribe {
            content: "The user prefers Playwright over Selenium".to_string(),
            placement: "workbench".to_string(),
            keywords: vec![
                "playwright".to_string(),
                "selenium".to_string(),
                "preference".to_string(),
            ],
        };

        let json = serde_json::to_value(&inscribe).unwrap();
        assert_eq!(json["name"], "MemoryPalace::inscribe");
        assert_eq!(
            json["input"]["content"],
            "The user prefers Playwright over Selenium"
        );

        // Test round-trip
        let deserialized: Use = serde_json::from_value(json).unwrap();
        matches!(deserialized, Use::Inscribe { .. });
    }

    #[test]
    fn test_navigation_state() {
        let mut state = NavigationState::default();
        assert_eq!(state.current_room, "Entrance Hall");

        // Simulate navigation
        state.current_room = "Programming Workshop".to_string();
        state.visited_rooms.push("Programming Workshop".to_string());

        assert_eq!(state.visited_rooms.len(), 2);
    }
}
