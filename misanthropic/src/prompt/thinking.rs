//! Anthropic native [`Thinking`] support, not to be confused with the `cot`
//! feature which works with all models.
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

/// Options for `Thinking` support. Requires Anthropic model support. As of now,
/// this is just Sonnet 3.7.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Thinking {
    /// Thinking budget in tokens. This must be at least 1024 tokens and at most
    /// `max_tokens` toke
    /// ns.
    pub budget_tokens: NonZeroU32,
    /// Thinking type.
    #[serde(rename = "type")]
    pub kind: Kind,
}

/// Thinking type.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Thinking enabled.
    #[default]
    Enabled,
}
