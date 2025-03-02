//! [`Model`] to use for inference.
use std::borrow::Cow;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// All available models.
#[derive(Debug, Serialize, Deserialize, derive_more::Deref)]
#[serde(rename_all = "snake_case")]
pub struct Models<'a> {
    /// List of available models.
    data: Vec<Model<'a>>,
}

/// Model information.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Model<'a> {
    /// Model ID.
    pub id: Id<'a>,
    /// Display name.
    pub display_name: Cow<'a, str>,
    /// Created at.
    pub created_at: DateTime<Utc>,
}

/// Model ID.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case", untagged)]
pub enum Id<'a> {
    /// Anthropic model.
    Anthropic(AnthropicModel),
    /// Custom model id.
    Custom(Cow<'a, str>),
}

impl<'a> Id<'a> {
    /// Get the name of the model.
    pub fn name(&'a self) -> &'a str {
        match self {
            Id::Anthropic(model) => match model {
                AnthropicModel::Sonnet37 => "claude-3-7-sonnet-latest",
                AnthropicModel::Sonnet37_20250219 => {
                    "claude-3-7-sonnet-20250219"
                }
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
            Id::Custom(name) => name,
        }
    }

    /// Convert to a `'static` lifetime by taking ownership of the [`Cow`]
    pub fn into_static(self) -> Id<'static> {
        match self {
            Id::Anthropic(model) => Id::Anthropic(model),
            Id::Custom(name) => Id::Custom(Cow::Owned(name.into_owned())),
        }
    }
}

impl<'a, T> From<T> for Id<'a>
where
    T: Into<Cow<'a, str>>,
{
    fn from(s: T) -> Self {
        // Unwrap can't panic because we have a catch-all variant.
        serde_json::from_str(&format!("\"{}\"", s.into())).unwrap()
    }
}

impl From<AnthropicModel> for Id<'_> {
    fn from(value: AnthropicModel) -> Self {
        Id::Anthropic(value)
    }
}

impl PartialEq<AnthropicModel> for Id<'_> {
    fn eq(&self, other: &AnthropicModel) -> bool {
        match self {
            Id::Anthropic(model) => model == other,
            Id::Custom(s) => s.as_ref() == other.name(),
        }
    }
}

impl<S> PartialEq<S> for Id<'_>
where
    S: AsRef<str>,
{
    fn eq(&self, other: &S) -> bool {
        match self {
            Id::Anthropic(model) => model.name() == other.as_ref(),
            Id::Custom(s) => s.as_ref() == other.as_ref(),
        }
    }
}

impl Default for Id<'_> {
    fn default() -> Self {
        Id::Anthropic(AnthropicModel::Haiku30)
    }
}

/// Choice of Anthropic models.
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
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
    /// Sonnet 3.7 (latest)
    #[serde(rename = "claude-3-7-sonnet-latest")]
    Sonnet37,
    /// Sonnet 3.7 2025-02-19
    #[serde(rename = "claude-3-7-sonnet-20250219")]
    Sonnet37_20250219,
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
        AnthropicModel::Haiku30,
        AnthropicModel::Haiku35_20241022,
        AnthropicModel::Haiku35,
        AnthropicModel::Opus30_20240229,
        AnthropicModel::Opus30,
        AnthropicModel::Sonnet30,
        AnthropicModel::Sonnet35_20240620,
        AnthropicModel::Sonnet35_20241022,
        AnthropicModel::Sonnet35,
    ];

    /// Get the name of the model (what it serializes to).
    pub fn name(self) -> &'static str {
        // I don't like duplication, but this is fine for now.
        match self {
            AnthropicModel::Haiku30 => "haiku-3.0-20240307",
            AnthropicModel::Haiku35 => "haiku-3.5-latest",
            AnthropicModel::Haiku35_20241022 => "haiku-3.5-20241022",
            AnthropicModel::Opus30 => "opus-3.0-latest",
            AnthropicModel::Opus30_20240229 => "opus-3.0-20240229",
            AnthropicModel::Sonnet30 => "sonnet-3.0-20240229",
            AnthropicModel::Sonnet35 => "sonnet-3.5-latest",
            AnthropicModel::Sonnet35_20240620 => "sonnet-3.5-20240620",
            AnthropicModel::Sonnet35_20241022 => "sonnet-3.5-20241022",
            AnthropicModel::Sonnet37 => "sonnet-3.7-latest",
            AnthropicModel::Sonnet37_20250219 => "sonnet-3.7-20250219",
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
    fn test_model_deserialize() {
        const JSON:&[u8] = b"{\"data\":[{\"type\":\"model\",\"id\":\"claude-3-5-sonnet-20241022\",\"display_name\":\"Claude 3.5 Sonnet (New)\",\"created_at\":\"2024-10-22T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-5-haiku-20241022\",\"display_name\":\"Claude 3.5 Haiku\",\"created_at\":\"2024-10-22T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-5-sonnet-20240620\",\"display_name\":\"Claude 3.5 Sonnet (Old)\",\"created_at\":\"2024-06-20T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-haiku-20240307\",\"display_name\":\"Claude 3 Haiku\",\"created_at\":\"2024-03-07T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-opus-20240229\",\"display_name\":\"Claude 3 Opus\",\"created_at\":\"2024-02-29T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-sonnet-20240229\",\"display_name\":\"Claude 3 Sonnet\",\"created_at\":\"2024-02-29T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-2.1\",\"display_name\":\"Claude 2.1\",\"created_at\":\"2023-11-21T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-2.0\",\"display_name\":\"Claude 2.0\",\"created_at\":\"2023-07-11T00:00:00Z\"}],\"has_more\":false,\"first_id\":\"claude-3-5-sonnet-20241022\",\"last_id\":\"claude-2.0\"}";
        let models = serde_json::from_slice::<Models>(JSON).unwrap();
        assert_eq!(models.len(), 8);
    }

    #[test]
    fn test_id_name() {
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

        let model: Id = "custom_model".into();
        assert_eq!(model.name(), "custom_model");
        assert_eq!(model, "custom_model");
    }

    // Some of these overlap, but it's fine.

    #[test]
    fn test_id_into_static() {
        let model: Id = "custom_model".into();
        let model = model.into_static();
        assert_eq!(model, "custom_model");
    }

    #[test]
    fn test_id_conversion_from_anthropic_model() {
        let model: Id = AnthropicModel::Sonnet35.into();
        assert_eq!(model, AnthropicModel::Sonnet35);
    }

    #[test]
    fn test_id_conversion_from_str() {
        // custom model
        let model: Id = "custom_model".into();
        assert_eq!(model, "custom_model");

        // known model
        let model: Id = "claude-3-5-sonnet-latest".into();
        assert_eq!(model, AnthropicModel::Sonnet35);
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_ids_are_valid() {
        let key = load_api_key().expect("API key not found");
        let client = Client::new(key).unwrap();

        let mut prompt = Prompt::default()
            .add_message((Role::User, "Emit just the \"🙏\" emoji, please."))
            .unwrap();

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
