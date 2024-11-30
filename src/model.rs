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
// API reports; unknown variant `blabla`, expected one of
// * `claude-3-5-sonnet-latest`,
// * `claude-3-5-sonnet-20240620`,
// * `claude-3-sonnet-20241022`,
// * `claude-3-opus-latest`,
// * `claude-3-opus-20240229`,
// * `claude-3-sonnet-20240229`,
// * `claude-3-5-haiku-latest`,
// * `claude-3-5-haiku-20241022`,
// * `claude-3-haiku-20240307`,
// * `claude-3-haiku-latest`
//
// But docs say that `claude-3-5-sonnet-20241022` is a valid model, and the API
// does accept it. This appears to be a bug in the API. - mdegans
// https://docs.anthropic.com/en/docs/about-claude/models
//
// These does not exist at least for my API key. Last tried 11/27/2021.
// Anthropic(NotFound { message: "model: claude-3-haiku-latest" })
// - mdegans
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
    /// Opus 3.0 (latest)
    #[serde(rename = "claude-3-opus-latest")]
    Opus30,
    /// Opus 3.0 2024-02-29
    #[serde(rename = "claude-3-opus-20240229")]
    Opus30_20240229,
    /// Sonnet 3.0 2024-02-29
    #[serde(rename = "claude-3-sonnet-20240229")]
    Sonnet30,
    /// Haiku 3.5 (latest)
    #[serde(rename = "claude-3-5-haiku-latest")]
    Haiku35,
    /// Haiku 3.5 2024-10-22
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Haiku35_20241022,
    /// Haiku 3.0 (latest) This is the default model.
    // Note: It is documented that the `-latest` tag works, but last I tried it
    // the API rejected it. Last tried 11/27/2021.
    // Anthropic(NotFound { message: "model: claude-3-haiku-latest" })
    #[default]
    #[serde(
        rename = "claude-3-haiku-20240307",
        alias = "claude-3-haiku-latest"
    )]
    Haiku30,
}

impl Model {
    /// All available models.
    pub const ALL: &'static [Model] = &[
        Model::Sonnet35,
        Model::Sonnet35_20240620,
        Model::Sonnet35_20241022,
        Model::Opus30,
        Model::Opus30_20240229,
        Model::Sonnet30,
        Model::Haiku35,
        Model::Haiku35_20241022,
        Model::Haiku30,
    ];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{prompt::message::Role, Client, Prompt};

    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

    fn load_api_key() -> Option<String> {
        use std::fs::File;
        use std::io::Read;
        use std::path::Path;

        let mut file =
            File::open(Path::new(CRATE_ROOT).join("api.key")).ok()?;
        let mut key = String::new();
        file.read_to_string(&mut key).unwrap();
        Some(key.trim().to_string())
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_models_are_valid() {
        let key = load_api_key().expect("API key not found");
        let client = Client::new(key).unwrap();

        let mut prompt = Prompt::default()
            .add_message((Role::User, "Respond with just the parrot emoji."));

        for &model in Model::ALL {
            prompt.model = model;

            // If this fails (because a new model was added), it should be added
            // to the list of models above and the `latest` aliases should be
            // updated.
            let response = client.message(&prompt).await.unwrap();

            // If the mode is not a latest tag, we want to check it matches
            // the model we set.
            if !serde_json::to_string(&model).unwrap().contains("latest") {
                assert_eq!(response.model, model);
            }
        }
    }
}
