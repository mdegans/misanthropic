//! A [`Prompt`] wrapper that prevents mutation of the
//! [cache prefix](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching).
//!
//! The Anthropic prompt cache is keyed on the prefix: `tools` → `system` →
//! `messages`, in that order.  Mutating any field that participates in the
//! prefix (tools, system, tool_choice, thinking, model) after a cache entry
//! has been written silently invalidates the cache and turns every subsequent
//! request into a full-price cache *write* instead of a cheap *read*.
//!
//! [`CachedPrompt`] makes this class of bug a compile error: the inner
//! [`Prompt`] is private, and only operations that preserve the cache prefix
//! are exposed.
//!
//! # Cache-safe operations
//!
//! | Method | Why it's safe |
//! |---|---|
//! | [`push_message`] | Appends after the prefix; cache reads via lookback |
//! | [`cache`] | Adds a breakpoint — doesn't change content |
//! | [`set_max_tokens`] | Not part of the cache key |
//! | [`set_temperature`] | Not part of the cache key |
//! | [`set_top_k`] | Not part of the cache key |
//! | [`set_top_p`] | Not part of the cache key |
//! | [`set_stop_sequences`] | Not part of the cache key |
//! | [`set_metadata`] | Not part of the cache key |
//!
//! [`push_message`]: CachedPrompt::push_message
//! [`cache`]: CachedPrompt::cache
//! [`set_max_tokens`]: CachedPrompt::set_max_tokens
//! [`set_temperature`]: CachedPrompt::set_temperature
//! [`set_top_k`]: CachedPrompt::set_top_k
//! [`set_top_p`]: CachedPrompt::set_top_p
//! [`set_stop_sequences`]: CachedPrompt::set_stop_sequences
//! [`set_metadata`]: CachedPrompt::set_metadata
//!
//! # Cache-breaking fields (immutable after construction)
//!
//! | Field | Invalidates |
//! |---|---|
//! | `tools` / `functions` | Everything |
//! | `system` | System + messages cache |
//! | `tool_choice` | Messages cache |
//! | `thinking` | Messages cache |
//! | `model` | Everything (different model = different cache) |
//!
//! # Breakpoint budget
//!
//! The API supports up to 4 explicit cache breakpoints.  If more are present,
//! the API keeps the last 4.  This means you can freely call [`cache`] each
//! turn without tracking how many breakpoints exist.
//!
//! [`cache`]: CachedPrompt::cache

use std::{
    borrow::Cow,
    num::{NonZeroU16, NonZeroU32},
    ops::Deref,
};

use serde::{Deserialize, Serialize};

use super::message::CacheControl;
use super::{Message, Prompt, TurnOrderError};

/// A [`Prompt`] with an immutable cache prefix.
///
/// Created via [`From<Prompt>`] (which adds a cache breakpoint) or
/// [`CachedPrompt::uncached`] (which does not).
///
/// To deliberately break the cache (e.g. removing tools for a different
/// phase), call [`into_inner`] — the explicit escape hatch.
///
/// [`into_inner`]: CachedPrompt::into_inner
#[derive(Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct CachedPrompt<'a> {
    inner: Prompt<'a>,
}

impl std::fmt::Debug for CachedPrompt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedPrompt")
            .field("inner", &self.inner)
            .finish()
    }
}

// --- Construction -----------------------------------------------------------

impl<'a> From<Prompt<'a>> for CachedPrompt<'a> {
    /// Freeze the prompt and add a cache breakpoint on the last cacheable
    /// block.  This is the preferred constructor for most use cases.
    fn from(prompt: Prompt<'a>) -> Self {
        Self {
            inner: prompt.cache(),
        }
    }
}

impl<'a> CachedPrompt<'a> {
    /// Freeze the prompt *without* adding a cache breakpoint.
    ///
    /// Use this when the prompt already has explicit `cache_control` markers
    /// set during construction (e.g. on individual system blocks or tools).
    pub fn uncached(prompt: Prompt<'a>) -> Self {
        Self { inner: prompt }
    }
}

// --- Cache-safe mutations ---------------------------------------------------

impl<'a> CachedPrompt<'a> {
    /// Append a [`Message`] to the conversation.
    ///
    /// This is always cache-safe: new messages are appended after the prefix,
    /// and the API's 20-block lookback finds earlier cache entries.
    ///
    /// # Errors
    ///
    /// Returns [`TurnOrderError`] if the turn order would be violated
    /// (consecutive messages from the same role).
    pub fn push_message<M>(&mut self, message: M) -> Result<(), TurnOrderError>
    where
        M: Into<Message<'a>>,
    {
        self.inner.push_message(message)
    }

    /// Add a cache breakpoint on the last cacheable block.
    ///
    /// Call this after appending messages to extend the cached region.
    /// The API keeps only the last 4 breakpoints, so calling this every
    /// turn is safe.
    ///
    /// Uses the default 5-minute ephemeral TTL. For 1-hour TTL (useful
    /// for cache priming across an hourly batch cadence), use
    /// [`cache_1h`](CachedPrompt::cache_1h).
    pub fn cache(&mut self) {
        // Prompt::cache() is `fn cache(mut self) -> Self`, so we need to
        // temporarily take ownership.
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.cache();
    }

    /// Add a 1-hour cache breakpoint on the last cacheable block.
    ///
    /// Behaves identically to [`cache`](CachedPrompt::cache) but uses
    /// [`CacheControl::one_hour`](crate::prompt::message::CacheControl::one_hour).
    /// Useful when the priming write and the real requests may be
    /// separated by more than the default 5-minute window.
    pub fn cache_1h(&mut self) {
        let taken = std::mem::take(&mut self.inner);
        self.inner = taken.cache_1h();
    }

    /// Add a cache breakpoint on the last message, keeping at most `n`
    /// message-level breakpoints using a windowed strategy: keep the
    /// **first** and the **last (n-1)**, drop everything in between.
    ///
    /// The first breakpoint anchors the start of messages for the 20-block
    /// lookback window. The trailing breakpoints maximize cache hits from
    /// the most recent rounds.
    ///
    /// Breakpoints on tools and system blocks are untouched (they are part
    /// of the immutable prefix).
    ///
    /// # Budget accounting
    ///
    /// The API limits explicit `cache_control` blocks to 4 total. This
    /// method counts at message granularity: a message with any cached
    /// block counts as one toward `n`. When a message is dropped from
    /// the window, **all** its cached blocks are cleared. In normal usage
    /// ([`cache`] sets one breakpoint on the last block), each message has
    /// at most one cached block, so message-level and block-level counts
    /// are equivalent.
    ///
    /// # Typical usage
    ///
    /// Call [`cache`] once after building the initial prompt, then
    /// `cache_windowed(3)` after each tool-use round. With 1 breakpoint
    /// on the tools/system prefix, this uses all 4 API slots optimally.
    ///
    /// Uses the default 5-minute ephemeral TTL. For 1-hour TTL use
    /// [`cache_windowed_1h`](CachedPrompt::cache_windowed_1h), or pass
    /// an explicit [`CacheControl`] via
    /// [`cache_windowed_with`](CachedPrompt::cache_windowed_with).
    ///
    /// [`cache`]: CachedPrompt::cache
    pub fn cache_windowed(&mut self, n: usize) {
        self.cache_windowed_with(n, CacheControl::ephemeral());
    }

    /// Like [`cache_windowed`](CachedPrompt::cache_windowed) but uses a
    /// 1-hour TTL on the new breakpoint.
    ///
    /// Useful when rounds may be separated by more than the default
    /// 5-minute window — for example, a human-driven deliberation loop
    /// where the operator reads each response before calling the next
    /// round.
    pub fn cache_windowed_1h(&mut self, n: usize) {
        self.cache_windowed_with(n, CacheControl::one_hour());
    }

    /// Like [`cache_windowed`](CachedPrompt::cache_windowed) but lets the
    /// caller choose the [`CacheControl`] applied to the new breakpoint.
    ///
    /// Existing breakpoints on earlier messages retain whatever
    /// `CacheControl` they were originally given; only the newly-added
    /// breakpoint on the last message uses `cache_control`. Middle
    /// breakpoints that fall outside the window are removed regardless
    /// of their original TTL.
    pub fn cache_windowed_with(
        &mut self,
        n: usize,
        cache_control: CacheControl,
    ) {
        // 1. Add breakpoint on the last message with the requested TTL.
        if let Some(last) = self.inner.messages.last_mut() {
            last.content.cache_with(cache_control);
        }

        // 2. Find all message-level breakpoint indices.
        let cached_indices: Vec<usize> = self
            .inner
            .messages
            .iter()
            .enumerate()
            .filter(|(_, msg)| msg.content.has_cache())
            .map(|(i, _)| i)
            .collect();

        // 3. If over budget, keep first + last (n-1), drop the middle.
        if cached_indices.len() > n {
            let keep_trailing = n.saturating_sub(1);
            let drop_start = 1; // skip first
            let drop_end = cached_indices.len() - keep_trailing;
            for &idx in &cached_indices[drop_start..drop_end] {
                self.inner.messages[idx].content.uncache();
            }
        }
    }

    /// Set `max_tokens`.  Not part of the cache key.
    pub fn set_max_tokens(&mut self, max_tokens: NonZeroU32) {
        self.inner.max_tokens = max_tokens;
    }

    /// Set `temperature`.  Not part of the cache key.
    pub fn set_temperature(&mut self, temperature: Option<f32>) {
        self.inner.temperature = temperature;
    }

    /// Set `top_k`.  Not part of the cache key.
    pub fn set_top_k(&mut self, top_k: Option<NonZeroU16>) {
        self.inner.top_k = top_k;
    }

    /// Set `top_p`.  Not part of the cache key.
    pub fn set_top_p(&mut self, top_p: Option<f32>) {
        self.inner.top_p = top_p;
    }

    /// Set `stop_sequences`.  Not part of the cache key.
    pub fn set_stop_sequences(
        &mut self,
        stop_sequences: Option<Vec<Cow<'a, str>>>,
    ) {
        self.inner.stop_sequences = stop_sequences;
    }

    /// Set request `metadata`.  Not part of the cache key.
    pub fn set_metadata(
        &mut self,
        metadata: serde_json::Map<String, serde_json::Value>,
    ) {
        self.inner.metadata = metadata;
    }
}

// --- Conversions ------------------------------------------------------------

impl<'a> CachedPrompt<'a> {
    /// Consume the wrapper and return the inner [`Prompt`].
    ///
    /// **This is an explicit escape hatch.**  After calling this, the prompt
    /// can be freely mutated — including cache-breaking fields.  Use this
    /// when you deliberately need to change the prefix (e.g. removing tools
    /// for a reflect phase).
    pub fn into_inner(self) -> Prompt<'a> {
        self.inner
    }

    /// Convert to `'static` lifetime, mirroring [`Prompt::into_static`].
    pub fn into_static(self) -> CachedPrompt<'static> {
        CachedPrompt {
            inner: self.inner.into_static(),
        }
    }
}

// --- Read-only access -------------------------------------------------------

/// `Deref` provides read-only access to all [`Prompt`] fields.
///
/// There is intentionally **no** `DerefMut` — preventing direct mutation of
/// cache-prefix fields like `tool_choice` and `functions`.
impl<'a> Deref for CachedPrompt<'a> {
    type Target = Prompt<'a>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> AsRef<Prompt<'a>> for CachedPrompt<'a> {
    fn as_ref(&self) -> &Prompt<'a> {
        &self.inner
    }
}

// --- Serialization ----------------------------------------------------------

/// Serializes identically to the inner [`Prompt`], so this works with
/// [`Client::message`](crate::Client::message) which takes `P: Serialize`.
impl Serialize for CachedPrompt<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.serialize(serializer)
    }
}

/// Deserializes as a [`Prompt`] and wraps it *without* adding a cache
/// breakpoint (since the serialized form may already contain breakpoints).
impl<'de, 'a: 'de> Deserialize<'de> for CachedPrompt<'a> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Prompt::deserialize(deserializer).map(Self::uncached)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::message::Role;

    #[test]
    fn from_prompt_adds_cache_breakpoint() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let cached = CachedPrompt::from(prompt);

        // The system content should now have a cache_control marker.
        // (cache() falls through: no messages → caches system)
        match cached.system.as_ref().unwrap() {
            crate::prompt::message::Content::MultiPart(blocks) => {
                let last = blocks.last().unwrap();
                assert!(
                    last.is_cached(),
                    "expected cache_control on system block"
                );
            }
            crate::prompt::message::Content::SinglePart(_) => {
                panic!("expected MultiPart after cache()")
            }
        }
    }

    #[test]
    fn cache_1h_sets_one_hour_ttl() {
        use crate::prompt::message::{CacheControl, CacheTtl};

        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let mut cached = CachedPrompt::uncached(prompt);
        cached.cache_1h();

        // The system block should now carry a 1-hour cache_control.
        match cached.system.as_ref().unwrap() {
            crate::prompt::message::Content::MultiPart(blocks) => {
                let last = blocks.last().unwrap();
                let cc = match last {
                    crate::prompt::message::Block::Text {
                        cache_control,
                        ..
                    } => cache_control.as_ref().unwrap(),
                    _ => panic!("expected text block"),
                };
                assert_eq!(
                    cc,
                    &CacheControl::Ephemeral {
                        ttl: Some(CacheTtl::OneHour)
                    }
                );
            }
            _ => panic!("expected MultiPart after cache_1h()"),
        }
    }

    #[test]
    fn uncached_does_not_add_breakpoint() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let cached = CachedPrompt::uncached(prompt);

        // System should still be SinglePart — no cache() was called.
        assert!(
            cached.system.as_ref().unwrap().is_single_part(),
            "expected SinglePart (no cache breakpoint)"
        );
    }

    #[test]
    fn push_message_works() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

        cached
            .push_message((Role::User, "Hello"))
            .expect("first message should succeed");
        cached
            .push_message((Role::Assistant, "Hi there"))
            .expect("assistant response should succeed");

        assert_eq!(cached.messages.len(), 2);
    }

    #[test]
    fn push_message_enforces_turn_order() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

        cached
            .push_message((Role::User, "Hello"))
            .expect("first message should succeed");
        let result = cached.push_message((Role::User, "Hello again"));
        assert!(result.is_err(), "consecutive user messages should fail");
    }

    #[test]
    fn set_max_tokens_works() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

        cached.set_max_tokens(NonZeroU32::new(512).unwrap());
        assert_eq!(cached.max_tokens, NonZeroU32::new(512).unwrap());
    }

    #[test]
    fn into_inner_returns_prompt() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text("test")),
            ..Default::default()
        };

        let cached = CachedPrompt::from(prompt);
        let inner = cached.into_inner();

        // We can now mutate freely — this is the escape hatch.
        assert!(inner.system.is_some());
    }

    #[test]
    fn serialization_roundtrip() {
        let mut prompt = Prompt::default();
        prompt
            .push_message((Role::User, "test"))
            .expect("first message");

        let cached = CachedPrompt::from(prompt);
        let json = serde_json::to_string(&cached).expect("serialize");
        let deserialized: CachedPrompt<'_> =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(cached.messages.len(), deserialized.messages.len());
    }

    #[test]
    fn into_static_works() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text("test")),
            ..Default::default()
        };
        let cached = CachedPrompt::from(prompt);
        let _static_cached: CachedPrompt<'static> = cached.into_static();
    }

    #[test]
    fn deref_provides_read_access() {
        let prompt = Prompt {
            max_tokens: NonZeroU32::new(1024).unwrap(),
            ..Default::default()
        };
        let cached = CachedPrompt::from(prompt);

        // Can read via Deref
        assert_eq!(cached.max_tokens, NonZeroU32::new(1024).unwrap());
        assert!(cached.tool_choice.is_none());
        assert!(cached.functions.is_none());
    }

    #[test]
    fn cache_windowed_keeps_first_and_last() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::uncached(prompt);

        // Add 7 user/assistant message pairs (14 messages total)
        for i in 0..7 {
            cached
                .push_message((Role::User, format!("user {i}")))
                .unwrap();
            cached
                .push_message((Role::Assistant, format!("asst {i}")))
                .unwrap();
            // Mark end of each "round" with a cache breakpoint
            cached.cache();
        }

        // Should have 7 cached messages (the last of each pair)
        let cached_count = cached
            .messages
            .iter()
            .filter(|m| m.content.has_cache())
            .count();
        assert_eq!(cached_count, 7);

        // Now apply windowed(3) — should keep first + last 2 = 3
        cached.cache_windowed(3);

        let cached_indices: Vec<usize> = cached
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.content.has_cache())
            .map(|(i, _)| i)
            .collect();

        assert_eq!(
            cached_indices.len(),
            3,
            "should have exactly 3 breakpoints"
        );
        // First breakpoint is the earliest cached message
        assert_eq!(
            cached_indices[0], 1,
            "first breakpoint should be message 1 (asst 0)"
        );
        // Last two should be the most recent
        assert_eq!(
            cached_indices[cached_indices.len() - 1],
            13,
            "last breakpoint should be the final message"
        );
    }

    #[test]
    fn cache_windowed_1h_sets_one_hour_ttl_on_last_message() {
        use crate::prompt::message::{Block, CacheControl, CacheTtl, Content};

        let prompt = Prompt::default();
        let mut cached = CachedPrompt::uncached(prompt);

        cached.push_message((Role::User, "hello")).unwrap();
        cached.push_message((Role::Assistant, "hi")).unwrap();
        cached.cache_windowed_1h(2);

        // The last message's last block should carry a 1-hour TTL.
        let last_msg = cached.messages.last().unwrap();
        let last_block = match &last_msg.content {
            Content::MultiPart(blocks) => blocks.last().unwrap(),
            Content::SinglePart(_) => {
                panic!("expected MultiPart after cache_windowed_1h")
            }
        };
        let cc = match last_block {
            Block::Text { cache_control, .. } => {
                cache_control.as_ref().unwrap()
            }
            _ => panic!("expected text block"),
        };
        assert_eq!(
            cc,
            &CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            }
        );
    }

    #[test]
    fn cache_windowed_with_preserves_earlier_ttls() {
        use crate::prompt::message::{Block, CacheControl, CacheTtl, Content};

        let prompt = Prompt::default();
        let mut cached = CachedPrompt::uncached(prompt);

        // Round 1: mark with 1h TTL
        cached.push_message((Role::User, "round 1 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 1 asst"))
            .unwrap();
        cached.cache_windowed_1h(3);

        // Round 2: mark with 5m (default ephemeral)
        cached.push_message((Role::User, "round 2 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 2 asst"))
            .unwrap();
        cached.cache_windowed(3);

        // Round 3: mark with 1h again
        cached.push_message((Role::User, "round 3 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 3 asst"))
            .unwrap();
        cached.cache_windowed_1h(3);

        // All three rounds should still be cached; round 1 keeps 1h,
        // round 2 keeps 5m, round 3 is now 1h.
        let ttl_at = |idx: usize| -> CacheControl {
            let msg = &cached.messages[idx];
            let block = match &msg.content {
                Content::MultiPart(blocks) => blocks.last().unwrap(),
                Content::SinglePart(_) => panic!("expected MultiPart"),
            };
            match block {
                Block::Text { cache_control, .. } => {
                    cache_control.as_ref().unwrap().clone()
                }
                _ => panic!("expected text block"),
            }
        };

        // Messages 0..5 are 3 user/assistant pairs. cache_windowed marks
        // the *last* message of each round (index 1, 3, 5).
        assert_eq!(
            ttl_at(1),
            CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            },
            "round 1 should still be 1h"
        );
        assert_eq!(
            ttl_at(3),
            CacheControl::Ephemeral { ttl: None },
            "round 2 should be 5m (default)"
        );
        assert_eq!(
            ttl_at(5),
            CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            },
            "round 3 should be 1h"
        );
    }

    #[test]
    fn cache_windowed_no_op_when_under_budget() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::uncached(prompt);

        cached.push_message((Role::User, "hello")).unwrap();
        cached.push_message((Role::Assistant, "hi")).unwrap();
        cached.cache();

        // Only 1 cached message, budget is 3 — should be a no-op
        cached.cache_windowed(3);

        let cached_count = cached
            .messages
            .iter()
            .filter(|m| m.content.has_cache())
            .count();
        assert_eq!(cached_count, 1);
    }

    #[test]
    fn uncache_removes_breakpoint() {
        use crate::prompt::message::Content;

        let mut content = Content::text("hello");
        content.cache();
        assert!(content.has_cache());

        content.uncache();
        assert!(!content.has_cache());
    }
}
