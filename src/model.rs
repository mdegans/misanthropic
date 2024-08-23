//! [`Model`] to use for inference.

use serde::{Deserialize, Serialize};

/// Model to use for inference. Note that **some features may limit choices**.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// Sonnet 3.5.
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Sonnet35,
    /// Opus 3.0.
    #[cfg(not(feature = "prompt-caching"))]
    #[serde(rename = "claude-3-opus-20240229")]
    Opus30,
    /// Sonnet 3.0
    #[cfg(not(feature = "prompt-caching"))]
    #[serde(rename = "claude-3-sonnet-20240229")]
    Sonnet30,
    /// Haiku 3.0.
    #[serde(rename = "claude-3-haiku-20240307")]
    Haiku30,
}
