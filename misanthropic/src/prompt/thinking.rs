//! Anthropic native [`Thinking`] support, not to be confused with the `cot`
//! feature which works with all models.
use std::num::NonZeroU32;

use serde::{Deserialize, Serialize};

/// Extended thinking configuration, set on a [`Prompt`] via
/// [`Prompt::thinking`].
///
/// [`Adaptive`] is recommended on Claude 4 and *required* on Opus 4.7 and
/// newer; the fixed-budget [`Enabled`] mode is deprecated and rejected by
/// those models.
///
/// [`Prompt`]: crate::Prompt
/// [`Prompt::thinking`]: crate::Prompt::thinking
/// [`Adaptive`]: Self::Adaptive
/// [`Enabled`]: Self::Enabled
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Thinking {
    /// The model decides when and how much to think per request. Recommended
    /// on Claude 4 and required on Opus 4.7 and newer. Interleaved thinking
    /// between tool calls is enabled automatically.
    Adaptive {
        /// How thinking content is returned. `None` uses the model default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<Display>,
    },
    /// Fixed token budget. **Deprecated** by the API and rejected by Opus 4.7
    /// and newer — prefer [`Adaptive`]. Still valid on Sonnet 3.7 and older
    /// Claude 4 models.
    ///
    /// [`Adaptive`]: Self::Adaptive
    Enabled {
        /// Maximum tokens for internal reasoning. At least 1024, and normally
        /// less than `max_tokens` (it may exceed it with interleaved
        /// thinking).
        budget_tokens: NonZeroU32,
        /// How thinking content is returned. `None` uses the model default.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        display: Option<Display>,
    },
    /// Extended thinking explicitly disabled.
    Disabled,
}

impl Thinking {
    /// [`Adaptive`] thinking with the model's default [`Display`].
    ///
    /// [`Adaptive`]: Self::Adaptive
    pub const fn adaptive() -> Self {
        Self::Adaptive { display: None }
    }

    /// [`Enabled`] thinking with a fixed token budget and the model's default
    /// [`Display`]. Prefer [`Self::adaptive`] on current models.
    ///
    /// [`Enabled`]: Self::Enabled
    pub const fn enabled(budget_tokens: NonZeroU32) -> Self {
        Self::Enabled {
            budget_tokens,
            display: None,
        }
    }
}

/// How thinking content is returned in the response, set on [`Thinking`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "lowercase")]
pub enum Display {
    /// Summarized thinking text. Default on Claude 4 models except Opus 4.7
    /// and newer. Billed for the full thinking tokens, not the summary.
    Summarized,
    /// Empty thinking with an encrypted signature for multi-turn continuity.
    /// Default on Opus 4.7 and newer.
    Omitted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adaptive_round_trip() {
        let json = serde_json::to_value(Thinking::adaptive()).unwrap();
        assert_eq!(json, serde_json::json!({"type": "adaptive"}));
        assert_eq!(
            serde_json::from_value::<Thinking>(json).unwrap(),
            Thinking::adaptive()
        );
    }

    #[test]
    fn enabled_round_trip() {
        let thinking = Thinking::enabled(1024.try_into().unwrap());
        let json = serde_json::to_value(thinking).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(serde_json::from_value::<Thinking>(json).unwrap(), thinking);
    }

    #[test]
    fn disabled_round_trip() {
        let json = serde_json::to_value(Thinking::Disabled).unwrap();
        assert_eq!(json, serde_json::json!({"type": "disabled"}));
        assert_eq!(
            serde_json::from_value::<Thinking>(json).unwrap(),
            Thinking::Disabled
        );
    }

    #[test]
    fn display_serializes_when_set() {
        let thinking = Thinking::Adaptive {
            display: Some(Display::Omitted),
        };
        assert_eq!(
            serde_json::to_value(thinking).unwrap(),
            serde_json::json!({"type": "adaptive", "display": "omitted"})
        );
    }
}
