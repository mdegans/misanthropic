//! The [`GetEmbedding`] trait is used by various tools to get embeddings for
//! given text.
use std::{borrow::Cow, sync::Arc};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use zeroize::Zeroizing;

/// Default OpenAI model for text embeddings.
pub const DEFAULT_OPENAI_MODEL: &str = "text-embedding-ada-002";

/// A text embedding, A wrapper around a vector of floats.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextEmbedding {
    pub embedding: Arc<Vec<f32>>,
    pub model: Arc<String>,
}

/// True False Maybe
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TFM {
    True,
    False,
    Maybe,
}

fn error_to_string(err: &dyn std::error::Error) -> String {
    let mut s = String::new();
    s.push_str(&err.to_string());
    if let Some(source) = err.source() {
        s.push_str(&format!(": {}", error_to_string(source)));
    }
    s
}

/// Error for embedding retrieval.
#[allow(missing_docs)] // because very short and common, self describing, names.
#[derive(Debug, thiserror::Error, Serialize)]
#[error("Embedding error: {cause}")]
pub enum EmbeddingError {
    /// Reqwest error, such as network issues or invalid input.
    #[error("Request error: {0}")]
    #[serde(serialize_with = "error_to_string")]
    ReqwestError(#[from] reqwest::Error),
    /// Error caused by the embedding service, such as invalid API key or model.
    #[error("{} service error: {}", match is_fatal {
        TFM::True => "Fatal",
        TFM::False => "Non-fatal",
        TFM::Maybe => "Maybe fatal",
    }, message)]
    ServiceError {
        is_fatal: TFM,
        message: Cow<'static, str>,
    },
}

impl Into<Box<dyn std::error::Error + Send>> for EmbeddingError {
    fn into(self) -> Box<dyn std::error::Error + Send> {
        Box::new(self)
    }
}

/// Trait for getting embeddings from a text.
#[async_trait]
pub trait EmbeddingClient: Send {
    /// Get the embedding for a given text.
    async fn get_embedding(
        &self,
        text: &str,
    ) -> Result<TextEmbedding, EmbeddingError>;
    /// Get the name of the embedding client.
    fn name(&self) -> &'static str;
    /// Get the embedding size for the client.
    fn embedding_size(&self) -> usize;
    /// Get the model used by the client.
    fn model(&self) -> Arc<String>;
}
static_assertions::assert_obj_safe!(EmbeddingClient);

/// OpenAI embedding client.
#[derive(Clone)]
pub struct OpenAI {
    pub client: reqwest::Client,
    api_key: Arc<Zeroizing<String>>,
    pub model: Arc<String>,
    pub embedding_size: usize,
}

impl OpenAI {
    /// Create a new OpenAI client with the given `api_key`, `model`, and
    /// `size`. If you get the size messed up, it will panic.
    ///
    /// # Panics
    /// - The first time called if the size is not correct.
    pub fn new(api_key: String, model: String, size: usize) -> Self {
        OpenAI {
            client: reqwest::Client::new(),
            api_key: Zeroizing::new(api_key).into(),
            model: Arc::new(model),
            embedding_size: size,
        }
    }
}

#[async_trait]
impl EmbeddingClient for OpenAI {
    /// Get the embedding for a given text using OpenAI's API.
    async fn get_embedding(
        &self,
        text: &str,
    ) -> Result<TextEmbedding, EmbeddingError> {
        // TODO: Double check the API use below and pick a top notch model.
        let url = "https://api.openai.com/v1/embeddings";
        let response = self
            .client
            .post(url)
            .bearer_auth(self.api_key.as_str())
            .json(&serde_json::json!({
                "model": self.model.as_str(),
                "input": text,
            }))
            .send()
            .await?;

        if response.status().is_success() {
            let json: serde_json::Value = response.json().await?;
            // FIXME: Does this error handling need to be more robust? We are
            // assuming the api is well behaved or this could panic. We should
            // use `serde` instead. It's much more concise than this:
            let embedding: Arc<Vec<f32>> = json
                .get("data")
                .and_then(|v| v.as_array())
                .ok_or_else(|| EmbeddingError::ServiceError {
                    is_fatal: TFM::True,
                    message: "Invalid response format".into(),
                })?
                .get(0)
                .and_then(|v| v.get("embedding"))
                .and_then(|v| {
                    v.as_array().and_then(|arr| {
                        if arr.len() == self.embedding_size {
                            Some(arr)
                        } else {
                            None
                        }
                    })
                })
                .ok_or_else(|| EmbeddingError::ServiceError {
                    is_fatal: TFM::True,
                    message: "Embedding size mismatch".into(),
                })?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect::<Vec<f32>>()
                .into();

            Ok(TextEmbedding {
                embedding,
                model: self.model.clone(),
            })
        } else {
            let error: serde_json::Value = response.json().await?;
            Err(EmbeddingError::ServiceError {
                // We can't know, so this is why there is a Maybe.
                is_fatal: TFM::Maybe,
                // Serde itself would have to be broken for this to fail
                message: serde_json::to_string(&error).unwrap().into(),
            })
        }
    }

    #[doc = " Get the name of the embedding client."]
    fn name(&self) -> &'static str {
        "OpenAI"
    }

    #[doc = " Get the embedding size for the client."]
    fn embedding_size(&self) -> usize {
        self.embedding_size
    }

    #[doc = " Get the model used by the client."]
    fn model(&self) -> Arc<String> {
        self.model.clone()
    }
}

// TODO: Local embedding client implementation. We are able to run 60b models
// which are overkill for this use case, but the embeddings would be very high
// quality. How much this will help search is unknown. A cheap model may be good
// enough for Haiku to fill in the blanks. We should test. In any case there are
// very high quality local embedding models available.
// TODO: Caching embedding wrapper. This would be very useful
