//! [`Model`] to use for inference.
use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// Model to use for inference, either a built-in Anthropic model or a custom
/// model.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(rename_all = "snake_case", untagged)]
pub enum Model<'a> {
    /// Anthropic model.
    Anthropic(AnthropicModel),
    /// Custom model.
    Custom(Cow<'a, str>),
}

impl<'a> Model<'a> {
    /// Get the name of the model.
    pub fn name(&'a self) -> &'a str {
        match self {
            Model::Anthropic(model) => match model {
                AnthropicModel::Sonnet35 => "claude-3-5-sonnet-latest",
                AnthropicModel::Sonnet35_20240620 => {
                    "claude-3-5-sonnet-20240620"
                }
                AnthropicModel::Sonnet35_20241022 => {
                    "claude-3-5-sonnet-20241022"
                }
                AnthropicModel::Opus30 => "claude-3-opus-latest",
                AnthropicModel::Opus30_20240229 => "claude-3-opus-20240229",
                AnthropicModel::Sonnet30 => "claude-3-sonnet-20240229",
                AnthropicModel::Haiku35 => "claude-3-5-haiku-latest",
                AnthropicModel::Haiku35_20241022 => "claude-3-5-haiku-20241022",
                AnthropicModel::Haiku30 => "claude-3-haiku-20240307",
            },
            Model::Custom(name) => name,
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    pub fn into_static(self) -> Model<'static> {
        match self {
            Model::Anthropic(model) => Model::Anthropic(model),
            Model::Custom(name) => Model::Custom(Cow::Owned(name.into_owned())),
        }
    }
}

impl<'a, T> From<T> for Model<'a>
where
    T: Into<Cow<'a, str>>,
{
    fn from(s: T) -> Self {
        // Unwrap can't panic because we have a catch-all variant.
        serde_json::from_str(&format!("\"{}\"", s.into())).unwrap()
    }
}

impl From<AnthropicModel> for Model<'_> {
    fn from(value: AnthropicModel) -> Self {
        Model::Anthropic(value)
    }
}

impl PartialEq<AnthropicModel> for Model<'_> {
    fn eq(&self, other: &AnthropicModel) -> bool {
        match self {
            Model::Anthropic(model) => model == other,
            Model::Custom(s) => s.as_ref() == other.name(),
        }
    }
}

impl<S> PartialEq<S> for Model<'_>
where
    S: AsRef<str>,
{
    fn eq(&self, other: &S) -> bool {
        match self {
            Model::Anthropic(model) => model.name() == other.as_ref(),
            Model::Custom(s) => s.as_ref() == other.as_ref(),
        }
    }
}

impl Default for Model<'_> {
    fn default() -> Self {
        Model::Anthropic(AnthropicModel::Haiku30)
    }
}

/// Choice of Anthropic models.
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
pub enum AnthropicModel {
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

impl AnthropicModel {
    /// All available models.
    pub const ALL: &'static [AnthropicModel] = &[
        AnthropicModel::Sonnet35,
        AnthropicModel::Sonnet35_20240620,
        AnthropicModel::Sonnet35_20241022,
        AnthropicModel::Opus30,
        AnthropicModel::Opus30_20240229,
        AnthropicModel::Sonnet30,
        AnthropicModel::Haiku35,
        AnthropicModel::Haiku35_20241022,
        AnthropicModel::Haiku30,
    ];

    /// Get the name of the model (what it serializes to).
    pub fn name(self) -> &'static str {
        // I don't like duplication, but this is fine for now.
        match self {
            AnthropicModel::Sonnet35 => "sonnet-3.5-latest",
            AnthropicModel::Sonnet35_20240620 => "sonnet-3.5-20240620",
            AnthropicModel::Sonnet35_20241022 => "sonnet-3.5-20241022",
            AnthropicModel::Opus30 => "opus-3.0-latest",
            AnthropicModel::Opus30_20240229 => "opus-3.0-20240229",
            AnthropicModel::Sonnet30 => "sonnet-3.0-20240229",
            AnthropicModel::Haiku35 => "haiku-3.5-latest",
            AnthropicModel::Haiku35_20241022 => "haiku-3.5-20241022",
            AnthropicModel::Haiku30 => "haiku-3.0-20240307",
        }
    }
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

    #[test]
    fn test_model_name() {
        assert_eq!(AnthropicModel::Sonnet35.name(), "sonnet-3.5-latest");
        assert_eq!(
            AnthropicModel::Sonnet35_20240620.name(),
            "sonnet-3.5-20240620"
        );
        assert_eq!(
            AnthropicModel::Sonnet35_20241022.name(),
            "sonnet-3.5-20241022"
        );
        assert_eq!(AnthropicModel::Opus30.name(), "opus-3.0-latest");
        assert_eq!(AnthropicModel::Opus30_20240229.name(), "opus-3.0-20240229");
        assert_eq!(AnthropicModel::Sonnet30.name(), "sonnet-3.0-20240229");
        assert_eq!(AnthropicModel::Haiku35.name(), "haiku-3.5-latest");
        assert_eq!(
            AnthropicModel::Haiku35_20241022.name(),
            "haiku-3.5-20241022"
        );
        assert_eq!(AnthropicModel::Haiku30.name(), "haiku-3.0-20240307");

        let model: Model = "custom_model".into();
        assert_eq!(model.name(), "custom_model");
        assert_eq!(model, "custom_model");
    }

    // Some of these overlap, but it's fine.

    #[test]
    fn test_model_into_static() {
        let model: Model = "custom_model".into();
        let model = model.into_static();
        assert_eq!(model, "custom_model");
    }

    #[test]
    fn test_model_conversion_from_model() {
        let model: Model = AnthropicModel::Sonnet35.into();
        assert_eq!(model, AnthropicModel::Sonnet35);
    }

    #[test]
    fn test_model_conversion_from_str() {
        // custom model
        let model: Model = "custom_model".into();
        assert_eq!(model, "custom_model");

        // known model
        let model: Model = "claude-3-5-sonnet-latest".into();
        assert_eq!(model, AnthropicModel::Sonnet35);
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_models_are_valid() {
        let key = load_api_key().expect("API key not found");
        let client = Client::new(key).unwrap();

        let mut prompt = Prompt::default()
            .add_message((Role::User, "Emit just the \"🙏\" emoji, please."));

        for &model in AnthropicModel::ALL {
            prompt.model = model.into();

            // If this fails (because a new model was added), it should be:
            // * added to the list of models above and
            // * the `latest` aliases should be updated
            // * the `name` method updated
            let response = client.message(&prompt).await.unwrap();

            // If the mode is not a latest tag, we want to check it matches
            // the model we set.
            if !serde_json::to_string(&model).unwrap().contains("latest") {
                assert_eq!(response.model, model);
            }
        }
    }
}
