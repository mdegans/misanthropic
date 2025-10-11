use serde::{Deserialize, Serialize};

use crate::tool::{
    NavigatorJson,
    memory_palace::{Memory, MemoryId, Pathway, PathwayId, Room, RoomId},
};

/// A [`Path`] taken through the [`MemoryPalace`], containing the full rows of
/// [`PathMember`]s, which can be a [`Room`], a [`Pathway`], or a [`Memory`].
/// The path is guaranteed to end with a [`Memory`], which is the destination of
/// the path.
///
/// [`MemoryPalace`]: super::MemoryPalace
#[derive(Debug, Clone, derive_more::Deref, Serialize)]
#[serde(transparent)]
pub struct Path(Vec<PathMember>);

impl Path {
    /// From an iterable of [`PathMember`]s. Can fail if the path does not end
    /// with a [`Memory`].
    pub fn from_members(
        members: impl IntoIterator<Item = PathMember>,
    ) -> Result<Self, &'static str> {
        let members: Vec<PathMember> = members.into_iter().collect();

        // The last member must be a memory
        if !matches!(members.last(), Some(PathMember::Memory(_))) {
            return Err("Path must end with a Memory");
        }

        Ok(Path(members))
    }

    /// Get the members of the path.
    pub fn members(&self) -> &[PathMember] {
        &self.0
    }

    /// To PathMemberIds
    pub fn to_ids(&self) -> Result<PathByIds, &'static str> {
        PathByIds::from_members(self.0.iter().map(|m| match m {
            PathMember::Room(room) => PathMemberIds::Room(room.id),
            PathMember::Pathway(pathway) => PathMemberIds::Pathway(pathway.id),
            PathMember::Memory(memory) => PathMemberIds::Memory(memory.id),
        }))
    }
}

impl NavigatorJson for Path {
    /// Agent-friendly JSON representation of the [`Path`]
    fn navigator_json(&self) -> serde_json::Value {
        use serde_json::Value;

        // Get a JSON array of member objects with indices for reference
        let members: Vec<_> = self
            .members()
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                if let Value::Object(mut map) = m.navigator_json() {
                    // For reference, include the index
                    map.insert("index".to_string(), Value::Number(i.into()));
                    Some(map)
                } else {
                    // This should never happen
                    debug_assert!(
                        false,
                        "PathMember did not serialize to an object."
                    );
                    #[cfg(feature = "log")]
                    log::warn!(
                        "PathMember did not serialize to an object: {}",
                        serde_json::json!(m)
                    );
                    None
                }
            })
            .collect();

        serde_json::json!({
            "path": members
        })
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

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::IsVariant)]
pub enum PathMember {
    /// A [`Room`] in the path
    Room(Room),
    /// A [`Pathway`] between two [`Room`]s in the path
    Pathway(Pathway),
    /// A [`Memory`] in the path (destination)
    Memory(Memory),
}

impl NavigatorJson for PathMember {
    fn navigator_json(&self) -> serde_json::Value {
        let variant = match self {
            PathMember::Room(_) => "room",
            PathMember::Pathway(_) => "pathway",
            PathMember::Memory(_) => "memory",
        };
        let member = match self {
            PathMember::Room(room) => room.navigator_json(),
            PathMember::Pathway(pathway) => pathway.agent_json(),
            PathMember::Memory(memory) => memory.agent_json(),
        };

        serde_json::json!({
            variant: member
        })
    }
}

/// A member in a [`Path`] taken through the [`MemoryPalace`], by id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
// TODO: the alignment here is 17 bytes, which might affect performance if we
// have a lot of heavy path traversals which is the core of the MemoryPalace
// tool. It might not matter but we could force a repr(C, align(8)) to make
// sure the alignment is 24 bytes instead of 17. More space, but potentially
// better performance. We should possibly benchmark if we're using this in a
// hot path.
// #[repr(C, align(8))] --- IGNORE ---
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
/// path. The path is a sequence of [`PathMemberIds`], which can be [`RoomId`],
/// [`PathwayId`], or [`MemoryId`].
#[derive(Debug, Clone, PartialEq, Eq, Hash, derive_more::Deref, Serialize)]
#[serde(transparent)]
pub struct PathByIds(Vec<PathMemberIds>);

impl PathByIds {
    /// From an iterable of [`PathMemberIds`]s
    pub fn from_members(
        members: impl IntoIterator<Item = PathMemberIds>,
    ) -> Result<Self, &'static str> {
        let members = members.into_iter().collect::<Vec<_>>();

        if !matches!(members.last(), Some(PathMemberIds::Memory(_))) {
            return Err("Path must end with a Memory");
        }

        Ok(PathByIds(members))
    }
}

impl<'de> Deserialize<'de> for PathByIds {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let members = Vec::<PathMemberIds>::deserialize(deserializer)?;
        Self::from_members(members).map_err(serde::de::Error::custom)
    }
}

impl IntoIterator for PathByIds {
    type Item = PathMemberIds;
    type IntoIter = std::vec::IntoIter<PathMemberIds>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a PathByIds {
    type Item = &'a PathMemberIds;
    type IntoIter = std::slice::Iter<'a, PathMemberIds>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
