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
//! # Construction
//!
//! Three constructors, differing only in whether they add a default cache
//! breakpoint on top of whatever the caller already placed:
//!
//! - [`From::from`] / [`Into::into`] — wrap exactly as-is, no breakpoint added.
//!   Use when the caller already set `cache_control` markers (inline, or
//!   via [`Prompt::cache`] / [`Prompt::cache_1h`] before the conversion).
//! - [`CachedPrompt::cached`] — wrap and add a 5-minute breakpoint.
//! - [`CachedPrompt::cached_1h`] — wrap and add a 1-hour breakpoint.
//!
//! See the [`CachedPrompt`] struct-level docs for the full rationale.
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
//! | [`set_auto_cache`] | Server-side trailing breakpoint — adds, never mutates |
//!
//! [`push_message`]: CachedPrompt::push_message
//! [`cache`]: CachedPrompt::cache
//! [`set_max_tokens`]: CachedPrompt::set_max_tokens
//! [`set_temperature`]: CachedPrompt::set_temperature
//! [`set_top_k`]: CachedPrompt::set_top_k
//! [`set_top_p`]: CachedPrompt::set_top_p
//! [`set_stop_sequences`]: CachedPrompt::set_stop_sequences
//! [`set_metadata`]: CachedPrompt::set_metadata
//! [`set_auto_cache`]: CachedPrompt::set_auto_cache
//!
//! # Cache-breaking fields (immutable after construction)
//!
//! | Field | Invalidates |
//! |---|---|
//! | `tools` / `methods` | Everything |
//! | `system` | System + messages cache |
//! | `tool_choice` | Messages cache |
//! | `thinking` | Messages cache |
//! | `model` | Everything (different model = different cache) |
//!
//! # Breakpoint budget
//!
//! Anthropic rejects a request carrying more than 4 `cache_control` markers
//! with a 400 ("A maximum of 4 blocks with cache_control may be provided.
//! Found 5.") — it does not keep the last 4. Each marked block counts,
//! across `tools` + `system` + `messages`, and so does the top-level
//! automatic slot ([`set_auto_cache`]), even when the last block is marked
//! too. (Probed against the free `count_tokens` endpoint, 2026-09-30.)
//!
//! So the wrapper keeps count: [`cache`] and [`cache_windowed`] never take a
//! request past 4. They slide a window over the messages — keeping the
//! newest message markers, evicting the oldest — and never touch the
//! `tools` / `system` markers or the automatic slot. Calling [`cache`] every
//! turn is safe, and loses no cache: an evicted marker's entry stays on the
//! server for its TTL, and the newer markers reach it through the API's
//! lookback (about 20 blocks back from each).
//!
//! Mixing TTLs has its own rules: a 1-hour marker must not come after a
//! 5-minute one, in `tools` → `system` → `messages` order with the automatic
//! slot last, and the automatic slot must match the TTL of a marker on the
//! block it lands on — so place 1-hour markers first. The 1-hour and
//! automatic placements return a [`CacheError`] rather than build a request
//! that breaks them, and the 5-minute ones never do (see
//! [`Prompt::check_cache`]).
//!
//! [`cache`]: CachedPrompt::cache
//! [`cache_windowed`]: CachedPrompt::cache_windowed

use std::{
    borrow::Cow,
    num::{NonZeroU16, NonZeroU32},
    ops::Deref,
};

use serde::{Deserialize, Serialize};

use super::message::CacheControl;
use super::{CacheError, Message, Prompt, TurnOrderError};

/// A [`Prompt`] with an immutable cache prefix.
///
/// # Construction
///
/// Three constructors, all equally explicit about what breakpoints the
/// resulting `CachedPrompt` carries:
///
/// | Constructor                | Adds a breakpoint? | When to use |
/// |----------------------------|---------------------|-------------|
/// | [`From<Prompt>`] / `.into()` | No                | The prompt already has its own `cache_control` markers (set inline during construction or via [`Prompt::cache`] / [`Prompt::cache_1h`] before the conversion) and you just want to lock down the prefix. |
/// | [`CachedPrompt::cached`]    | Yes, 5-minute TTL | You want the convenient default: wrap the prompt and add one 5-minute breakpoint on the last cacheable block. |
/// | [`CachedPrompt::cached_1h`] | Yes, 1-hour TTL   | Same as `cached` but the breakpoint uses a 1-hour TTL. |
///
/// # Why `From` does not add a breakpoint
///
/// An earlier design made `From<Prompt>` call [`Prompt::cache`] under the
/// hood. That turned `.into()` into a subtle footgun: a caller who had
/// already placed an explicit 1-hour marker (via `.cache_1h()` or an inline
/// `cache_control`) and then wrote `.into()` would silently have that marker
/// overwritten with a default 5-minute one — producing an Anthropic-side
/// "`ttl='1h' ... must not come after ttl='5m'`" error at submit time.
///
/// The current design splits the two intents apart:
///
/// - **Freeze, don't mark**: `Prompt::into()` / `CachedPrompt::from(prompt)`.
///   Exactly preserves whatever `cache_control` markers the caller placed.
/// - **Freeze and mark**: [`cached`](Self::cached) / [`cached_1h`](Self::cached_1h).
///   A convenience for the common case where the caller wants the wrapper
///   to pick the breakpoint location.
///
/// To deliberately break the cache (e.g. removing tools for a different
/// phase), call [`into_inner`] — the explicit escape hatch.
///
/// [`into_inner`]: CachedPrompt::into_inner
#[derive(Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct CachedPrompt {
    inner: Prompt,
}

impl std::fmt::Debug for CachedPrompt {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("CachedPrompt")
            .field("inner", &self.inner)
            .finish()
    }
}

// --- Construction -----------------------------------------------------------

impl From<Prompt> for CachedPrompt {
    /// Freeze the prompt into a [`CachedPrompt`] without touching its
    /// `cache_control` markers. Use this when the prompt already carries
    /// the breakpoints you want — either set inline at construction time
    /// (e.g. `Block::Text { cache_control: Some(CacheControl::one_hour()), ... }`)
    /// or placed via [`Prompt::cache`] / [`Prompt::cache_1h`] before the
    /// conversion.
    ///
    /// For the common "wrap and also add a breakpoint" case, use
    /// [`CachedPrompt::cached`] (5-minute TTL) or
    /// [`CachedPrompt::cached_1h`] (1-hour TTL).
    fn from(prompt: Prompt) -> Self {
        Self { inner: prompt }
    }
}

impl CachedPrompt {
    /// Freeze the prompt into a [`CachedPrompt`] **and** add a 5-minute
    /// cache breakpoint on the last cacheable block (via [`Prompt::cache`]).
    ///
    /// Equivalent to `CachedPrompt::from(prompt.cache())`.
    ///
    /// Use this when the prompt does not yet carry any explicit
    /// `cache_control` markers and you want the wrapper to place one at
    /// the default location (messages → system → tools, whichever has
    /// content first).
    ///
    /// For 1-hour TTL, use [`CachedPrompt::cached_1h`].
    /// For wrapping without adding any new breakpoint, use
    /// [`From::from`] / `.into()`.
    pub fn cached(prompt: Prompt) -> Self {
        Self {
            inner: prompt.cache(),
        }
    }

    /// Freeze the prompt into a [`CachedPrompt`] **and** add a 1-hour
    /// cache breakpoint on the last cacheable block (via [`Prompt::cache_1h`]).
    ///
    /// Equivalent to `prompt.cache_1h().map(CachedPrompt::from)`.
    ///
    /// Use this when priming or caching data that needs to survive longer
    /// than the default 5-minute window — for example, a prompt prefix
    /// that will be read by a batch of requests submitted over the next
    /// hour via the Anthropic Batch API.
    ///
    /// For 5-minute TTL, use [`CachedPrompt::cached`].
    /// For wrapping without adding any new breakpoint, use
    /// [`From::from`] / `.into()`.
    ///
    /// # Errors
    /// [`CacheError`] if the prompt already carries a 5-minute marker the
    /// new one would follow — see [`Prompt::cache_1h`].
    pub fn cached_1h(prompt: Prompt) -> Result<Self, CacheError> {
        Ok(Self {
            inner: prompt.cache_1h()?,
        })
    }
}

// --- Cache-safe mutations ---------------------------------------------------

impl CachedPrompt {
    /// Append a [`Message`] to the conversation.
    ///
    /// This is always cache-safe: new messages are appended after the prefix,
    /// and the API's 20-block lookback finds earlier cache entries.
    ///
    /// # Errors
    ///
    /// Returns [`TurnOrderError`] if the turn order would be violated
    /// (consecutive same-role turns, or a misplaced system turn).
    pub fn push_message<M>(
        &mut self,
        message: M,
    ) -> Result<&mut Self, TurnOrderError>
    where
        M: Into<Message>,
    {
        self.inner.push_message(message)?;
        Ok(self)
    }

    /// Add a cache breakpoint on the last cacheable block.
    ///
    /// Call this after appending messages to extend the cached region.
    /// Calling this every turn is safe: it never takes the request past
    /// Anthropic's 4-marker limit (see the [module docs](self)), keeping the
    /// newest message markers and evicting the oldest to make room.
    ///
    /// Uses the default 5-minute ephemeral TTL. For 1-hour TTL (useful
    /// for cache priming across an hourly batch cadence), use
    /// [`cache_1h`](CachedPrompt::cache_1h).
    pub fn cache(&mut self) -> &mut Self {
        let plan = self.inner.plan_end(CacheControl::ephemeral());
        self.inner.apply(plan);
        self
    }

    /// Add a 1-hour cache breakpoint on the last cacheable block.
    ///
    /// Behaves identically to [`cache`](CachedPrompt::cache) but uses
    /// [`CacheControl::one_hour`](crate::prompt::message::CacheControl::one_hour).
    /// Useful when the priming write and the real requests may be
    /// separated by more than the default 5-minute window.
    ///
    /// # Errors
    /// [`CacheError`] if a 5-minute marker would come before it; the prompt
    /// is unchanged.
    pub fn cache_1h(&mut self) -> Result<&mut Self, CacheError> {
        self.cache_with(CacheControl::one_hour())
    }

    /// [`cache`](CachedPrompt::cache) with a caller-provided
    /// [`CacheControl`] — see [`Prompt::cache_with`].
    ///
    /// # Errors
    /// [`CacheError`] if the request would break a rule
    /// [`Prompt::check_cache`] enforces; the prompt is unchanged.
    pub fn cache_with(
        &mut self,
        cache_control: CacheControl,
    ) -> Result<&mut Self, CacheError> {
        let plan = self.inner.plan_end(cache_control).checked()?;
        self.inner.apply(plan);
        Ok(self)
    }

    /// Place `n` cache breakpoints in a rolling trailing window across
    /// `messages`, spaced 2 positions apart, then enforce the API's hard
    /// 4-marker budget by evicting older message-level breakpoints.
    ///
    /// The window lands on indices `[len-1, len-3, …, len-1 - 2(n-1)]`,
    /// skipping any position that would fall before message 0. The 2-step
    /// spacing matches the typical "push assistant + push user_results
    /// per round" cadence: when this method is called again after another
    /// such pair is pushed, the new `len-1 - 2k` aligns with the previous
    /// call's `len-1 - 2(k-1)`, so an already-marked message gets re-marked
    /// (a no-op when present) rather than the marker jumping role.
    ///
    /// This pins the rolling window to the role of the trailing message at
    /// the *first* call's marker site. Subsequent calls in the same cadence
    /// keep the marker on the same role, which is what backends that key
    /// prefix re-use on the trailing-assistant render hash need to fire.
    ///
    /// # Budget enforcement
    ///
    /// Anthropic rejects a request with more than **4** `cache_control`
    /// markers, counting each marked block across `tools` + `system` +
    /// `messages` plus the automatic slot ([`set_auto_cache`]). The markers
    /// outside `messages` are a fixed cost this method never touches; the
    /// window gets what is left, newest position first, and older
    /// message-level markers are evicted to fit.
    ///
    /// A position already carrying a `cache_control` marker is left alone
    /// (its existing TTL is preserved); only freshly marked positions take
    /// the requested `cache_control`.
    ///
    /// # Typical usage
    ///
    /// Call [`cache`] once after building the initial prompt to mark the
    /// `tools` / `system` prefix, then `cache_windowed(2)` after each
    /// tool-use round. With 1 prefix marker + 2 message markers this fits
    /// inside the 4-budget with one slot left over.
    ///
    /// Uses the default 5-minute ephemeral TTL. For 1-hour TTL use
    /// [`cache_windowed_1h`](CachedPrompt::cache_windowed_1h), or pass
    /// an explicit [`CacheControl`] via
    /// [`cache_windowed_with`](CachedPrompt::cache_windowed_with).
    ///
    /// [`cache`]: CachedPrompt::cache
    /// [`set_auto_cache`]: CachedPrompt::set_auto_cache
    pub fn cache_windowed(&mut self, n: usize) {
        self.inner.cache_windowed(n);
    }

    /// Like [`cache_windowed`](CachedPrompt::cache_windowed) but uses a
    /// 1-hour TTL on the new breakpoint.
    ///
    /// Useful when rounds may be separated by more than the default
    /// 5-minute window — for example, a human-driven deliberation loop
    /// where the operator reads each response before calling the next
    /// round.
    ///
    /// # Errors
    /// [`CacheError`] if a 5-minute marker would come before a new one;
    /// the prompt is unchanged.
    pub fn cache_windowed_1h(&mut self, n: usize) -> Result<(), CacheError> {
        self.cache_windowed_with(n, CacheControl::one_hour())
    }

    /// Like [`cache_windowed`](CachedPrompt::cache_windowed) but lets the
    /// caller choose the [`CacheControl`] applied to freshly marked
    /// positions.
    ///
    /// Positions already carrying a marker retain whatever `CacheControl`
    /// they were originally given. When the 4-marker budget forces
    /// eviction, the window slides: the oldest message-level markers go
    /// first, and if the window alone doesn't fit, its oldest positions go
    /// too.
    ///
    /// # Errors
    /// [`CacheError`] if the request would break a rule
    /// [`Prompt::check_cache`] enforces — say, 1-hour markers after
    /// 5-minute ones. The prompt is unchanged.
    pub fn cache_windowed_with(
        &mut self,
        n: usize,
        cache_control: CacheControl,
    ) -> Result<(), CacheError> {
        // The algorithm lives on `Prompt` (promoted so plain prompts get
        // budget-aware windowed marking too); the frozen prefix is
        // unaffected — markers only ever move within `messages`.
        self.inner.cache_windowed_with(n, cache_control)
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
        stop_sequences: Option<Vec<Cow<'static, str>>>,
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

    /// Enable [automatic prompt caching]: the API places a breakpoint on the
    /// last cacheable block server-side, at request time, with the default
    /// 5-minute TTL. Cache-safe — it only ever *adds* a trailing breakpoint;
    /// the prefix content is untouched. Composes with the wrapper's manual
    /// breakpoints ([`cache`], [`cache_windowed`], …) under the API's shared
    /// 4-breakpoint budget, where it takes a slot of its own; if all 4 are
    /// placed, the oldest message marker makes way for it.
    ///
    /// # Errors
    /// [`CacheError::AutoMismatch`] when the last block carries a 1-hour
    /// marker (see [`Prompt::auto_cache`]); the prompt is unchanged.
    ///
    /// [automatic prompt caching]: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
    /// [`cache`]: CachedPrompt::cache
    /// [`cache_windowed`]: CachedPrompt::cache_windowed
    pub fn set_auto_cache(&mut self) -> Result<(), CacheError> {
        self.inner.set_auto_cache(CacheControl::ephemeral())
    }

    /// [`set_auto_cache`](CachedPrompt::set_auto_cache) with a 1-hour TTL.
    ///
    /// # Errors
    /// [`CacheError`] when any 5-minute marker is placed: the automatic
    /// slot counts as the last marker, after every block's. The prompt is
    /// unchanged.
    pub fn set_auto_cache_1h(&mut self) -> Result<(), CacheError> {
        self.inner.set_auto_cache(CacheControl::one_hour())
    }
}

// --- Conversions ------------------------------------------------------------

impl CachedPrompt {
    /// Consume the wrapper and return the inner [`Prompt`].
    ///
    /// **This is an explicit escape hatch.**  After calling this, the prompt
    /// can be freely mutated — including cache-breaking fields.  Use this
    /// when you deliberately need to change the prefix (e.g. removing tools
    /// for a reflect phase).
    pub fn into_inner(self) -> Prompt {
        self.inner
    }
}

// --- Read-only access -------------------------------------------------------

/// `Deref` provides read-only access to all [`Prompt`] fields.
///
/// There is intentionally **no** `DerefMut` — preventing direct mutation of
/// cache-prefix fields like `tool_choice` and `methods`.
impl Deref for CachedPrompt {
    type Target = Prompt;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl AsRef<Prompt> for CachedPrompt {
    fn as_ref(&self) -> &Prompt {
        &self.inner
    }
}

// --- Serialization ----------------------------------------------------------

/// Serializes identically to the inner [`Prompt`], so this works with
/// [`Client::message`](crate::Client::message) which takes `P: Serialize`.
impl Serialize for CachedPrompt {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.inner.serialize(serializer)
    }
}

/// Deserializes as a [`Prompt`] and wraps it via [`From`] (which preserves
/// any `cache_control` markers present in the serialized form exactly).
impl<'de> Deserialize<'de> for CachedPrompt {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Prompt::deserialize(deserializer).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prompt::message::Role;

    #[test]
    fn from_prompt_does_not_add_breakpoint() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let cached = CachedPrompt::from(prompt);

        // From must not add a cache breakpoint.
        assert!(
            !cached.system.as_ref().unwrap().has_cache(),
            "From must not add a breakpoint"
        );
    }

    #[test]
    fn cached_adds_5m_breakpoint() {
        use crate::prompt::message::CacheControl;

        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let cached = CachedPrompt::cached(prompt);

        // The system block should now carry a 5-minute cache_control.
        // (cache() falls through: no messages → caches system)
        let last = cached.system.as_ref().unwrap().last().unwrap();
        let cc = match last {
            crate::prompt::message::Block::Text { cache_control, .. } => {
                cache_control.as_ref().unwrap()
            }
            _ => panic!("expected text block"),
        };
        assert_eq!(cc, &CacheControl::Ephemeral { ttl: None });
    }

    #[test]
    fn cached_1h_adds_one_hour_breakpoint() {
        use crate::prompt::message::{CacheControl, CacheTtl};

        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let cached = CachedPrompt::cached_1h(prompt).unwrap();

        // The system block should now carry a 1-hour cache_control.
        let last = cached.system.as_ref().unwrap().last().unwrap();
        let cc = match last {
            crate::prompt::message::Block::Text { cache_control, .. } => {
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

    /// Regression test for a bug where the old `From<Prompt> for
    /// CachedPrompt` silently called `prompt.cache()` and would overwrite
    /// an inline 1h `cache_control` marker with a fresh 5m one — producing
    /// an Anthropic-side "ttl='1h' cache_control block must not come after
    /// a ttl='5m' cache_control block" error at submit time.
    ///
    /// The current `From` impl just wraps. This test confirms an inline
    /// 1h marker survives the conversion unchanged.
    #[test]
    fn from_preserves_inline_1h_marker() {
        use crate::prompt::message::{Block, CacheControl, CacheTtl, Content};

        let prompt = Prompt {
            system: Some(Content(vec![Block::Text {
                text: "You are a helpful assistant.".into(),
                citations: None,
                cache_control: Some(CacheControl::one_hour()),
            }])),
            ..Default::default()
        };

        let cached = CachedPrompt::from(prompt);

        let cc = match cached.system.as_ref().unwrap().last().unwrap() {
            Block::Text { cache_control, .. } => {
                cache_control.as_ref().unwrap()
            }
            _ => panic!("expected text block"),
        };
        assert_eq!(
            cc,
            &CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            },
            "From must preserve the inline 1h marker unchanged"
        );
    }

    #[test]
    fn cache_1h_on_mut_sets_one_hour_ttl() {
        use crate::prompt::message::{CacheControl, CacheTtl};

        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text(
                "You are a helpful assistant.",
            )),
            ..Default::default()
        };

        let mut cached = CachedPrompt::from(prompt);
        cached.cache_1h().unwrap();

        // The system block should now carry a 1-hour cache_control.
        let last = cached.system.as_ref().unwrap().last().unwrap();
        let cc = match last {
            crate::prompt::message::Block::Text { cache_control, .. } => {
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
        let deserialized: CachedPrompt =
            serde_json::from_str(&json).expect("deserialize");

        assert_eq!(cached.messages.len(), deserialized.messages.len());
    }

    #[test]
    fn cached_prompt_from_prompt() {
        let prompt = Prompt {
            system: Some(crate::prompt::message::Content::text("test")),
            ..Default::default()
        };
        let cached = CachedPrompt::from(prompt);
        let _: CachedPrompt = cached;
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
        assert!(cached.tools.is_none());
    }

    /// `cache` every turn slides the window: the newest markers stay, the
    /// oldest go, and a window over them has nothing left to change.
    #[test]
    fn cache_every_turn_slides_the_window() {
        let mut cached = CachedPrompt::from(Prompt::default());

        converse(&mut cached, 7, |cached| {
            cached.cache();
        });
        assert_eq!(marked(&cached), [7, 9, 11, 13]);

        cached.cache_windowed(3);
        assert_eq!(marked(&cached), [7, 9, 11, 13]);
    }

    /// The window's own positions outrank a newer marker outside it.
    #[test]
    fn cache_windowed_keeps_its_positions_before_newer_ones() {
        let mut cached = CachedPrompt::cached(Prompt::default().system("s"));
        converse(&mut cached, 7, |_| {});
        cached.inner.messages[12].content.cache();

        // `system` holds one slot; the window's three take the rest.
        cached.cache_windowed(3);

        assert_eq!(marked(&cached), [9, 11, 13]);
    }

    #[test]
    fn cache_windowed_fits_beside_a_system_marker() {
        use crate::prompt::message::{CacheControl, Content};

        // Prefix marker on system consumes 1 of the 4 budget slots. With
        // cache_windowed(3) the tail (3) uses the remaining 3, leaving 0
        // for any other message-level marker.
        let mut prompt = Prompt::default();
        let mut system = Content::text("system prompt");
        system.cache_with(CacheControl::ephemeral());
        prompt.system = Some(system);
        let mut cached = CachedPrompt::from(prompt);

        for i in 0..7 {
            cached
                .push_message((Role::User, format!("user {i}")))
                .unwrap();
            cached
                .push_message((Role::Assistant, format!("asst {i}")))
                .unwrap();
            cached.cache();
        }

        cached.cache_windowed(3);

        let cached_indices: Vec<usize> = cached
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.content.has_cache())
            .map(|(i, _)| i)
            .collect();

        assert_eq!(
            cached_indices,
            vec![9, 11, 13],
            "system marker holds 1 budget slot; the tail uses the other 3"
        );
        assert!(
            cached.inner.system.as_ref().unwrap().has_cache(),
            "system marker must not be touched by the windowed call"
        );
    }

    #[test]
    fn cache_windowed_skip_oob_positions_when_messages_shorter_than_window() {
        // Only 2 messages but cache_windowed(3) — tail set should be just
        // index 1 (the only in-bounds position from the [N, N-2, N-4]
        // sequence; N-2=-1 and N-4=-3 are skipped).
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);
        cached.push_message((Role::User, "hello")).unwrap();
        cached.push_message((Role::Assistant, "hi")).unwrap();

        cached.cache_windowed(3);

        let cached_indices: Vec<usize> = cached
            .messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.content.has_cache())
            .map(|(i, _)| i)
            .collect();
        assert_eq!(cached_indices, vec![1]);
    }

    #[test]
    fn cache_windowed_1h_sets_one_hour_ttl_on_last_message() {
        use crate::prompt::message::{Block, CacheControl, CacheTtl};

        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

        cached.push_message((Role::User, "hello")).unwrap();
        cached.push_message((Role::Assistant, "hi")).unwrap();
        cached.cache_windowed_1h(2).unwrap();

        // The last message's last block should carry a 1-hour TTL.
        let last_msg = cached.messages.last().unwrap();
        let last_block = last_msg.content.last().unwrap();
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
        use crate::prompt::message::{Block, CacheControl, CacheTtl};

        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

        // 1h markers first, then 5m: Anthropic rejects the reverse order.
        // Round 1: mark with 1h TTL
        cached.push_message((Role::User, "round 1 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 1 asst"))
            .unwrap();
        cached.cache_windowed_1h(3).unwrap();

        // Round 2: mark with 1h again
        cached.push_message((Role::User, "round 2 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 2 asst"))
            .unwrap();
        cached.cache_windowed_1h(3).unwrap();

        // Round 3: mark with 5m (default ephemeral), which re-marks the
        // earlier rounds' positions without touching their TTL.
        cached.push_message((Role::User, "round 3 user")).unwrap();
        cached
            .push_message((Role::Assistant, "round 3 asst"))
            .unwrap();
        cached.cache_windowed(3);

        // All three rounds should still be cached; rounds 1 and 2 keep 1h,
        // round 3 is 5m.
        let ttl_at = |idx: usize| -> CacheControl {
            let msg = &cached.messages[idx];
            let block = msg.content.last().unwrap();
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
            CacheControl::Ephemeral {
                ttl: Some(CacheTtl::OneHour)
            },
            "round 2 should still be 1h"
        );
        assert_eq!(
            ttl_at(5),
            CacheControl::Ephemeral { ttl: None },
            "round 3 should be 5m (default)"
        );
    }

    #[test]
    fn cache_windowed_no_op_when_under_budget() {
        let prompt = Prompt::default();
        let mut cached = CachedPrompt::from(prompt);

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

    /// The indices of `cached`'s messages carrying a marker.
    fn marked(cached: &CachedPrompt) -> Vec<usize> {
        let messages = cached.messages.iter().enumerate();
        messages
            .filter(|(_, m)| m.content.has_cache())
            .map(|(i, _)| i)
            .collect()
    }

    /// `cached` with `turns` more user/assistant pairs, calling `mark`
    /// after each, and asserting the budget holds every time.
    fn converse(
        cached: &mut CachedPrompt,
        turns: usize,
        mark: impl Fn(&mut CachedPrompt),
    ) {
        for i in 0..turns {
            cached.push_message((Role::User, format!("u{i}"))).unwrap();
            cached
                .push_message((Role::Assistant, format!("a{i}")))
                .unwrap();
            mark(cached);
            let serialized = serde_json::to_string(&*cached).unwrap();
            let on_wire = serialized.matches("\"cache_control\"").count();
            assert_eq!(on_wire, cached.cache_markers(), "turn {i}");
            assert!(on_wire <= 4, "turn {i}: {on_wire} markers");
        }
    }

    /// The automatic slot counts: with it and a system marker, `cache`
    /// every turn keeps the two newest message markers.
    #[test]
    fn cache_every_turn_counts_the_automatic_slot() {
        let system = Prompt::default().system("You are terse.");
        let mut cached = CachedPrompt::cached(system);
        cached.set_auto_cache().unwrap();

        converse(&mut cached, 6, |cached| {
            cached.cache();
        });

        assert_eq!(cached.cache_markers(), 4);
        assert_eq!(marked(&cached), [9, 11]);
    }

    /// A window wider than the budget keeps its newest positions.
    #[test]
    fn cache_windowed_never_exceeds_the_budget() {
        let mut cached = CachedPrompt::cached(Prompt::default().system("s"));
        cached.set_auto_cache().unwrap();

        converse(&mut cached, 6, |cached| cached.cache_windowed(5));

        assert_eq!(marked(&cached), [9, 11], "system + auto hold two");
    }

    /// A marker on each of a message's blocks counts, one per block.
    #[test]
    fn cache_counts_every_marked_block() {
        use crate::prompt::message::{Block, Message};

        let mut cached = CachedPrompt::from(Prompt::default());
        let blocks = ["a", "b", "c"].map(|text| {
            let mut block = Block::from(text);
            block.cache();
            block
        });
        let user = Message::from((Role::User, blocks.to_vec()));
        cached.push_message(user).unwrap();
        cached.push_message((Role::Assistant, "ok")).unwrap();
        assert_eq!(cached.cache_markers(), 3);

        cached.cache();
        cached.push_message((Role::User, "more")).unwrap();
        cached.cache();

        // The newest, then the newest of the rest: the assistant's marker
        // and the user's last two stay, and the user's first goes.
        assert_eq!(cached.cache_markers(), 4);
        let first = &cached.messages[0].content;
        let kept: Vec<bool> = first.iter().map(Block::is_cached).collect();
        assert_eq!(kept, [false, true, true]);
        assert!(cached.messages[1].content.has_cache());
        assert!(cached.messages[2].content.has_cache());
    }

    /// A prefix already holding every slot leaves no room: `cache` places
    /// nothing, on the messages or the prefix.
    #[test]
    fn cache_with_a_full_prefix_places_nothing() {
        use crate::prompt::message::{Block, Content};

        let full = Content(
            ["a", "b", "c", "d", "e"]
                .map(|text| {
                    let mut block = Block::from(text);
                    // The last is left for `cache` to try.
                    if text != "e" {
                        block.cache();
                    }
                    block
                })
                .to_vec(),
        );
        let prompt = Prompt {
            system: Some(full),
            ..Default::default()
        };

        let cached = CachedPrompt::cached(prompt.clone());
        assert_eq!(cached.cache_markers(), 4, "the system isn't re-marked");

        let mut cached = CachedPrompt::from(prompt);
        converse(&mut cached, 2, |cached| {
            cached.cache();
        });
        assert!(marked(&cached).is_empty());
    }

    /// With every slot on messages, the automatic slot takes the oldest's.
    #[test]
    fn set_auto_cache_makes_room() {
        let mut cached = CachedPrompt::from(Prompt::default());
        converse(&mut cached, 4, |cached| {
            cached.cache();
        });
        assert_eq!(marked(&cached), [1, 3, 5, 7]);

        cached.set_auto_cache().unwrap();

        assert_eq!(cached.cache_markers(), 4);
        assert_eq!(marked(&cached), [3, 5, 7]);
    }

    #[test]
    fn set_auto_cache_sets_top_level_cache_control() {
        let mut cached = CachedPrompt::cached(Prompt::default());
        assert!(cached.cache_control.is_none());

        cached.set_auto_cache().unwrap();
        let json = serde_json::to_value(&cached).unwrap();
        assert_eq!(json["cache_control"]["type"], "ephemeral");

        cached.set_auto_cache_1h().unwrap();
        let json = serde_json::to_value(&cached).unwrap();
        assert_eq!(json["cache_control"]["ttl"], "1h");
    }
}
