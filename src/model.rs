//! [`Model`] to use for inference.
use serde::{Deserialize, Serialize};

/// Model to use for inference. Note that **some features may limit choices**.
#[derive(
    Debug,
    Default,
    Clone,
    Copy,
    Serialize,
    Deserialize,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
)]
#[serde(rename_all = "snake_case")]
pub enum Model {
    /// Sonnet 3.5 (latest)
    #[serde(rename = "claude-3-5-sonnet-latest")]
    Sonnet35,
    /// Sonnet 3.5 2024-06-20
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Sonnet35_20240620,
    /// Sonnet 3.0 2024-10-22
    #[serde(rename = "claude-3-sonnet-20241022")]
    Sonnet30_20241022,
    /// Opus 3.0 (latest)
    #[serde(rename = "claude-3-opus-latest")]
    Opus30,
    /// Opus 3.0 2024-02-29
    #[serde(rename = "claude-3-opus-20240229")]
    Opus30_20240229,
    /// Sonnet 3.0
    #[cfg(not(feature = "prompt-caching"))]
    #[serde(rename = "claude-3-sonnet-20240229")]
    Sonnet30,
    /// Haiku 3.0 (latest) This is the default model.
    #[default]
    #[serde(rename = "claude-3-haiku-latest")]
    Haiku30,
    /// Haiku 3.0 2024-03-07
    #[serde(rename = "claude-3-haiku-20240307")]
    Haiku30_20240307,
    /// Haiku 3.5 (latest)
    #[serde(rename = "claude-3-5-haiku-latest")]
    Haiku35,
}
