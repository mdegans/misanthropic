//! [`Index`] and related types for addressing a [`CustomMethodDef`] or [`Content`](crate::prompt::message::Content)
//! [`Block`] inside a [`Prompt`].
//!
//! The motivating use is cache-breakpoint placement: [`Prompt::indices`] yields
//! every addressable position in Anthropic's cache-prefix order (tools →
//! system → messages), and [`Prompt::get_mut`] resolves one to a `&mut Block`
//! (or `&mut CustomMethodDef`) so a [`CacheControl`] can be dropped on it.
//!
//! [`CacheControl`]: crate::prompt::message::CacheControl
use super::{Prompt, message::Block, tool::CustomMethodDef};

/// An index into a [`Prompt`]. Addresses either a [`CustomMethodDef`] in
/// [`Prompt::tools`] or a [`Content`](crate::prompt::message::Content) [`Block`] in [`Prompt::system`] /
/// [`Prompt::messages`].
///
/// The derived [`Ord`] matches Anthropic's cache-prefix order: every
/// [`CustomMethodDef`] sorts before every [`Block`], system blocks before message
/// blocks. [`Prompt::indices`] yields indices in this order.
///
/// [`Prompt::tools`]: Prompt::tools
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Hash,
    derive_more::From,
    derive_more::IsVariant,
)]
pub enum Index {
    /// A [`CustomMethodDef`] in [`Prompt::tools`].
    Method(MethodIndex),
    /// A [`Content`](crate::prompt::message::Content) [`Block`] in [`Prompt::system`] or [`Prompt::messages`].
    Block(BlockIndex),
}

/// Index of a [`CustomMethodDef`] in [`Prompt::tools`].
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Hash,
    derive_more::Deref,
    derive_more::From,
    derive_more::Into,
)]
pub struct MethodIndex(pub usize);

/// Index of a [`Content`](crate::prompt::message::Content) [`Block`] in a [`Prompt`].
#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    PartialOrd,
    Eq,
    Ord,
    Hash,
    derive_more::IsVariant,
)]
pub enum BlockIndex {
    /// A [`Block`] in [`Prompt::system`].
    System(usize),
    /// A `(message, block)` pair in [`Prompt::messages`].
    Message((usize, usize)),
}

/// A shared reference to a [`CustomMethodDef`] or a [`Content`](crate::prompt::message::Content) [`Block`] in a
/// [`Prompt`], as returned by [`Prompt::get`].
pub enum IndexRef<'a> {
    /// Reference to a [`CustomMethodDef`] in [`Prompt::tools`].
    Method(&'a CustomMethodDef),
    /// Reference to a [`Content`](crate::prompt::message::Content) [`Block`] in a [`Prompt`].
    Block(&'a Block),
}

/// A mutable reference to a [`CustomMethodDef`] or a [`Content`](crate::prompt::message::Content) [`Block`] in a
/// [`Prompt`], as returned by [`Prompt::get_mut`].
pub enum IndexMut<'a> {
    /// Mutable reference to a [`CustomMethodDef`] in [`Prompt::tools`].
    Method(&'a mut CustomMethodDef),
    /// Mutable reference to a [`Content`](crate::prompt::message::Content) [`Block`] in a [`Prompt`].
    Block(&'a mut Block),
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for IndexRef<'_> {
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        match self {
            IndexRef::Method(method) => method.markdown_events_custom(options),
            IndexRef::Block(block) => block.markdown_events_custom(options),
        }
    }
}

impl Prompt {
    /// Resolve an [`Index`] to a shared reference, or [`None`] if it is out of
    /// bounds (or addresses [`Prompt::system`] / [`Prompt::tools`] when absent).
    pub fn get(&self, index: Index) -> Option<IndexRef<'_>> {
        match index {
            Index::Method(MethodIndex(i)) => self
                .tools
                .as_ref()?
                .get(i)?
                .as_method()
                .map(IndexRef::Method),
            Index::Block(BlockIndex::System(i)) => {
                self.system.as_ref()?.get(i).map(IndexRef::Block)
            }
            Index::Block(BlockIndex::Message((m, b))) => {
                self.messages.get(m)?.content.get(b).map(IndexRef::Block)
            }
        }
    }

    /// Resolve an [`Index`] to a mutable reference, or [`None`] if it is out of
    /// bounds (or addresses [`Prompt::system`] / [`Prompt::tools`] when absent).
    pub fn get_mut(&mut self, index: Index) -> Option<IndexMut<'_>> {
        match index {
            Index::Method(MethodIndex(i)) => self
                .tools
                .as_mut()?
                .get_mut(i)?
                .as_method_mut()
                .map(IndexMut::Method),
            Index::Block(BlockIndex::System(i)) => {
                self.system.as_mut()?.get_mut(i).map(IndexMut::Block)
            }
            Index::Block(BlockIndex::Message((m, b))) => self
                .messages
                .get_mut(m)?
                .content
                .get_mut(b)
                .map(IndexMut::Block),
        }
    }

    /// Iterate over every addressable [`Index`] in cache-prefix order:
    /// tools, then system blocks, then message blocks.
    pub fn indices(&self) -> impl Iterator<Item = Index> + '_ {
        // Only custom tools are addressable as a `MethodIndex`; server tools
        // carry their own `cache_control` and are skipped here.
        let tools = self
            .tools
            .iter()
            .flatten()
            .enumerate()
            .filter(|(_, t)| t.as_method().is_some())
            .map(|(i, _)| Index::Method(MethodIndex(i)));

        let system = (0..self.system.as_ref().map_or(0, |c| c.len()))
            .map(|i| Index::Block(BlockIndex::System(i)));

        let messages = self.messages.iter().enumerate().flat_map(|(m, msg)| {
            (0..msg.content.len())
                .map(move |b| Index::Block(BlockIndex::Message((m, b))))
        });

        tools.chain(system).chain(messages)
    }
}

impl std::ops::Index<MethodIndex> for Prompt {
    type Output = CustomMethodDef;

    /// # Panics
    /// - If [`Prompt::tools`] is absent, the index is out of bounds, or the
    ///   addressed tool is a [`ServerMethodDef`](crate::tool::ServerMethodDef) rather
    ///   than a custom [`CustomMethodDef`].
    fn index(&self, index: MethodIndex) -> &Self::Output {
        self.tools.as_ref().expect("no tools on this prompt")[index.0]
            .as_method()
            .expect(
                "tool at this index is a server tool, not a CustomMethodDef",
            )
    }
}

impl std::ops::IndexMut<MethodIndex> for Prompt {
    /// # Panics
    /// - If [`Prompt::tools`] is absent, the index is out of bounds, or the
    ///   addressed tool is a [`ServerMethodDef`](crate::tool::ServerMethodDef) rather
    ///   than a custom [`CustomMethodDef`].
    fn index_mut(&mut self, index: MethodIndex) -> &mut Self::Output {
        self.tools.as_mut().expect("no tools on this prompt")[index.0]
            .as_method_mut()
            .expect(
                "tool at this index is a server tool, not a CustomMethodDef",
            )
    }
}

impl std::ops::Index<BlockIndex> for Prompt {
    type Output = Block;

    /// # Panics
    /// - If the addressed [`Block`] (or [`Prompt::system`]) does not exist.
    fn index(&self, index: BlockIndex) -> &Self::Output {
        match index {
            BlockIndex::System(i) => {
                &self.system.as_ref().expect("no system on this prompt")[i]
            }
            BlockIndex::Message((m, b)) => &self.messages[m].content[b],
        }
    }
}

impl std::ops::IndexMut<BlockIndex> for Prompt {
    /// # Panics
    /// - If the addressed [`Block`] (or [`Prompt::system`]) does not exist.
    fn index_mut(&mut self, index: BlockIndex) -> &mut Self::Output {
        match index {
            BlockIndex::System(i) => {
                &mut self.system.as_mut().expect("no system on this prompt")[i]
            }
            BlockIndex::Message((m, b)) => &mut self.messages[m].content[b],
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_index_ordering() {
        assert!(
            [
                Index::Method(0.into()),
                Index::Method(1.into()),
                BlockIndex::System(0).into(),
                BlockIndex::System(1).into(),
                BlockIndex::Message((0, 0)).into(),
                BlockIndex::Message((0, 1)).into(),
                BlockIndex::Message((1, 0)).into()
            ]
            .is_sorted()
        )
    }

    #[test]
    fn test_block_index_ordering() {
        assert!(
            [
                // tools go here in cache
                BlockIndex::System(0),
                BlockIndex::System(1),
                BlockIndex::Message((0, 0)),
                BlockIndex::Message((0, 1)),
                BlockIndex::Message((1, 0))
            ]
            .is_sorted()
        )
    }

    #[test]
    fn indices_in_cache_prefix_order() {
        use crate::prompt::message::{Content, Role};
        use crate::tool::CustomMethodDef;

        let prompt = Prompt::default()
            .add_tool(CustomMethodDef {
                name: "a".into(),
                description: "a".into(),
                schema: serde_json::json!({}),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })
            .system(Content(vec!["sys0".into(), "sys1".into()]))
            .add_message((Role::User, "hi"))
            .unwrap();

        let indices: Vec<_> = prompt.indices().collect();

        // tools (1) → system blocks (2) → message blocks (1)
        assert_eq!(
            indices,
            vec![
                Index::Method(MethodIndex(0)),
                Index::Block(BlockIndex::System(0)),
                Index::Block(BlockIndex::System(1)),
                Index::Block(BlockIndex::Message((0, 0))),
            ]
        );
        // already sorted, matching the derived Ord
        assert!(indices.is_sorted());
    }

    #[test]
    fn get_and_get_mut_round_trip() {
        use crate::prompt::message::{Block, Content, Role};

        let mut prompt = Prompt::default()
            .system(Content::text("system"))
            .add_message((Role::User, "hello"))
            .unwrap();

        // get resolves a message block.
        let idx = Index::Block(BlockIndex::Message((0, 0)));
        assert!(matches!(prompt.get(idx), Some(IndexRef::Block(_))));

        // out-of-bounds resolves to None.
        assert!(
            prompt
                .get(Index::Block(BlockIndex::Message((9, 0))))
                .is_none()
        );
        assert!(prompt.get(Index::Method(MethodIndex(0))).is_none());

        // get_mut lets us drop a cache breakpoint.
        if let Some(IndexMut::Block(block)) = prompt.get_mut(idx) {
            block.cache();
        } else {
            panic!("expected a block");
        }
        assert!(prompt[BlockIndex::Message((0, 0))].is_cached());

        // panicking Index trait also works.
        assert!(matches!(prompt[BlockIndex::System(0)], Block::Text { .. }));
    }
}
