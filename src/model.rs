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
    /// Sonnet 3.5 2024-10-22
    #[serde(rename = "claude-3-5-sonnet-20241022")] 
    Sonnet35_20241022, 
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
    #[serde(rename = "claude-3-sonnet-20240229")]
    Sonnet30,
    /// Haiku 3.5 (latest)
    #[serde(rename = "claude-3-5-haiku-latest")]
    Haiku35,
    /// Haiku 3.5 2024-10-22
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Haiku35_20241022,
    /// Haiku 3.0 (latest) This is the default model.
    // Note: The `latest` tag is not yet supported by the API for Haiku 3.0, so
    // in the future this might point to a separate model. We can't use the same
    // serde tag for both, so there's only one option here for now.
    #[default]
    #[serde(
        rename = "claude-3-haiku-20240307",
        alias = "claude-3-haiku-latest"
    )]
    Haiku30,
}
