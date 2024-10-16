//! [`Client`] for the Anthropic Messages API and related types.

use std::{env, num::NonZeroU16, sync::Arc};

use eventsource_stream::Eventsource;
use serde::{Deserialize, Serialize};

use crate::{key, response, Key};

/// Result type for the client. See also [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// Client for the Anthropic Messages API.
///
/// See [`Self::new`] for creating a new client and [`Self::message`] and
/// [`Self::stream`] to get started.
#[derive(Clone)]
pub struct Client {
    /// Inner [`reqwest::Client`]. Be aware that setting this to a custom client
    /// without the appropriate headers (such as `anthropic-version`) will
    /// result in rejected requests. It is **not necessary** to set the API key
    /// on a custom client.
    ///
    /// ## Note:
    /// - The API [`Key`] is **set automatically on requests**. Set
    ///   [`Self::key`] to change the [`Key`].
    /// - **Do not use** `client.inner.get` directly. Use [`Self::get`] instead
    ///   to safely set the API [`Key`] as sensitive.
    pub inner: reqwest::Client,
    /// Encrypted API [`Key`] for convenience. It can be set to a new [`Key`] to
    /// change the key used for requests.
    pub key: Arc<Key>,
}

/// Claude client. Uses the Messages API and the prompt caching beta.
impl Client {
    /// Version of the API. This is appended to the header as
    /// "anthropic-version".
    pub const ANTHROPIC_VERSION: &'static str = "2023-06-01";
    /// Beta we are using. This is appended to the header as "anthropic-beta".
    #[cfg(feature = "prompt-caching")]
    pub const BETA: &'static str = "prompt-caching-2024-07-31";
    /// Our user agent.
    pub const USER_AGENT: &'static str =
        concat!(env!("CARGO_PKG_NAME"), "-", env!("CARGO_PKG_VERSION"));
    /// Default URL for the Messages API.
    pub const DEFAULT_URL: &'static str =
        "https://api.anthropic.com/v1/messages";

    /// Create a new client from any type that can be converted into a [`Key`].
    ///
    /// ## Note:
    /// - It's safest to use a [`String`]. If you use a [`&str`] you must
    ///   zeroize it after creating the client.
    // misanthropic/src/client.rs
    pub fn new<K>(key: K) -> std::result::Result<Self, key::InvalidKeyLength>
    where
        K: TryInto<Key, Error = key::InvalidKeyLength>,
    {
        Ok(Self::from_key(key.try_into()?))
    }

    /// Create a new client with the given key.
    pub fn from_key(key: Key) -> Self {
        #[cfg(feature = "log")]
        {
            log::info!(concat!(
                "Creating ",
                env!("CARGO_PKG_NAME", " client...")
            ));
            log::debug!(concat!("Crate version: ", env!("CARGO_PKG_VERSION")));
            log::debug!("Anthropic version: {}", Self::ANTHROPIC_VERSION);
            #[cfg(feature = "beta")]
            log::debug!("Anthropic beta: {}", Self::BETA);
        }

        // Headers for all requests.
        let mut headers = reqwest::header::HeaderMap::new();

        // Content type needs to be set to JSON.
        headers.insert(
            reqwest::header::CONTENT_TYPE,
            reqwest::header::HeaderValue::from_static("application/json"),
        );

        // Anthropic version needs to be set.
        headers.insert(
            "anthropic-version",
            reqwest::header::HeaderValue::from_static(Self::ANTHROPIC_VERSION),
        );

        // Enable prompt caching beta.
        #[cfg(feature = "prompt-caching")]
        headers.insert(
            "anthropic-beta",
            reqwest::header::HeaderValue::from_static(Self::BETA),
        );

        Self {
            inner: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .unwrap(),
            key: Arc::new(key),
        }
    }

    /// Create a [`reqwest::RequestBuilder`] with the API key set as a sensitive
    /// header value.
    pub fn request_raw<U>(
        &self,
        method: reqwest::Method,
        url: U,
    ) -> reqwest::RequestBuilder
    where
        U: reqwest::IntoUrl,
    {
        #[cfg(feature = "log")]
        {
            log::debug!("{} request to {}", method, url.as_str());
        }

        let mut val =
            reqwest::header::HeaderValue::from_bytes(self.key.read().as_ref())
                .unwrap();
        val.set_sensitive(true);

        self.inner.request(method, url).header("x-api-key", val)
    }

    /// Send a GET request with the API key set as a sensitive header value.
    pub async fn get<U>(&self, url: U) -> reqwest::Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
    {
        self.request_raw(reqwest::Method::GET, url).send().await
    }

    /// Send a POST request with the API key set as a sensitive header value.
    pub async fn post<U, B>(
        &self,
        url: U,
        body: B,
    ) -> reqwest::Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
        B: serde::Serialize,
    {
        let req = self.request_raw(reqwest::Method::POST, url);

        #[cfg(feature = "log")]
        {
            if let Ok(json) = serde_json::to_string_pretty(&body) {
                log::debug!("Sending body:\n{}", json);
            } else {
                log::warn!("Could not serialize body. Request will fail.");
            }
        }

        req.json(&body).send().await
    }

    /// Post a request to the Messages API.
    ///
    /// `prompt` can be a [`Request`] (as an example) or anything that can be
    /// serialized but it should conform to the Messages API. The return will be
    /// either a [`Response`] of a single [`response::Message`] or a [`Stream`]
    /// of events depending on whether `stream` is set to `true` in the
    /// `prompt`.
    ///
    /// See also [`Self::message`] and [`Self::stream`] for convenience methods
    /// as well as [`Self::request_custom`] for a custom URL.
    ///
    /// [`Response`]: crate::Response
    /// [`Request`]: crate::prompt
    /// [`Message`]: crate::Message
    /// [`Stream`]: crate::Stream
    pub async fn request<P>(&self, prompt: P) -> Result<crate::Response>
    where
        P: Serialize,
    {
        self.request_custom(prompt, Self::DEFAULT_URL).await
    }

    /// Post a [`request`] to a custom URL. This is useful for testing or for
    /// using a different Messages compatible endpoint.
    ///
    /// [`request`]: Self::request
    pub async fn request_custom<P, U>(
        &self,
        prompt: P,
        url: U,
    ) -> Result<crate::Response>
    where
        P: Serialize,
        U: reqwest::IntoUrl,
    {
        let json = serde_json::to_value(prompt)?;
        let streaming = json["stream"].as_bool().unwrap_or(false);

        let response: reqwest::Response = self.post(url, json).await?;

        if response.status() != reqwest::StatusCode::OK {
            let error: AnthropicErrorWrapper = response.json().await?;

            // Error was sucessfully parsed from the API.
            return Err(error.error.into());
        }

        if streaming {
            // Get a stream and wrap it in our stream type.
            Ok(crate::Response::Stream {
                stream: crate::Stream::new(
                    response.bytes_stream().eventsource(),
                ),
            })
        } else {
            // Get a single response message.
            Ok(crate::Response::Message {
                message: response.json().await?,
            })
        }
    }

    /// Make a [`request`] to the Messages API forcing `stream=false`. This
    /// function will always return a single [`response::Message`].
    ///
    /// [`request`]: Self::request
    pub async fn message<P>(&self, prompt: P) -> Result<response::Message>
    where
        P: Serialize,
    {
        let mut json = serde_json::to_value(prompt)?;
        json["stream"] = serde_json::Value::Bool(false);

        if let crate::Response::Message { message } = self.request(json).await?
        {
            // We have a message.
            Ok(message)
        } else {
            // This should never really happen. If it does the server is
            // misbehaving. However as a policy we don't panic in this crate
            // except in `unwrap` functions like `unwrap_message`.
            Err(Error::UnexpectedResponse {
                message: "Expected a message, got a stream.",
            })
        }
    }

    /// Make a [`request`] to the Messages API forcing `stream=true`. This
    /// function will always return a [`crate::Stream`].
    ///
    /// [`request`]: Self::request
    pub async fn stream<P>(&self, prompt: P) -> Result<crate::Stream>
    where
        P: Serialize,
    {
        let mut json = serde_json::to_value(prompt)?;
        json["stream"] = serde_json::Value::Bool(true);

        if let crate::Response::Stream { stream } = self.request(json).await? {
            Ok(stream)
        } else {
            Err(Error::UnexpectedResponse {
                message: "Expected a stream, got a message.",
            })
        }
    }
}

/// [`Client`] error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// HTTP error.
    #[error("HTTP error: {0}")]
    HTTP(#[from] reqwest::Error),
    /// Data could not be parsed.
    #[error("Parse error: {0}")]
    Parse(#[from] serde_json::Error),
    /// Anthropic error.
    #[error("Anthropic error: {0}")]
    Anthropic(#[from] AnthropicError),
    /// Unexpected response from the API. These should never happen unless the
    /// server is misbehaving (for example, returning a stream when a message is
    /// expected).
    #[error("Unexpected response: {message}")]
    #[allow(missing_docs)]
    UnexpectedResponse { message: &'static str },
}

/// Anthropic error type.
#[derive(Debug, thiserror::Error, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
#[serde(tag = "type")]
#[allow(missing_docs)]
pub enum AnthropicError {
    #[error("invalid request (400): {message}")]
    #[serde(rename = "invalid_request_error")]
    InvalidRequest { message: String },
    #[error("authentication (401): {message}")]
    #[serde(rename = "authentication_error")]
    Authentication { message: String },
    #[error("permission (403): {message}")]
    #[serde(rename = "permission_error")]
    Permission { message: String },
    #[error("not found (404): {message}")]
    #[serde(rename = "not_found_error")]
    NotFound { message: String },
    #[error("request too large (413): {message}")]
    // This inconsistency is in the API.
    RequestTooLarge { message: String },
    #[error("rate limit (429): {message}")]
    #[serde(rename = "rate_limit_error")]
    RateLimit { message: String },
    #[error("api error (500): {message}")]
    #[serde(rename = "api_error")]
    API { message: String },
    #[error("overloaded (529): {message}")]
    #[serde(rename = "overloaded_error")]
    Overloaded { message: String },
    // Anthropic's API specifies they can add more error codes in the future.
    #[error("unknown error ({code}): {message}")]
    Unknown { code: NonZeroU16, message: String },
}

impl AnthropicError {
    /// Get the HTTP status code for the error.
    pub fn status(&self) -> NonZeroU16 {
        match self {
            Self::InvalidRequest { .. } => NonZeroU16::new(400).unwrap(),
            Self::Authentication { .. } => NonZeroU16::new(401).unwrap(),
            Self::Permission { .. } => NonZeroU16::new(403).unwrap(),
            Self::NotFound { .. } => NonZeroU16::new(404).unwrap(),
            Self::RequestTooLarge { .. } => NonZeroU16::new(413).unwrap(),
            Self::RateLimit { .. } => NonZeroU16::new(429).unwrap(),
            Self::API { .. } => NonZeroU16::new(500).unwrap(),
            Self::Overloaded { .. } => NonZeroU16::new(529).unwrap(),
            Self::Unknown { code, .. } => *code,
        }
    }
}

// This is because the API tags errors and there isn't a way to tag
// both fields with "type" *and* the enum itself so we must wrap it.
#[derive(Deserialize)]
#[serde(tag = "error")]
pub(crate) struct AnthropicErrorWrapper {
    pub(crate) error: AnthropicError,
}

#[cfg(test)]
mod tests {
    use futures::TryStreamExt;

    use super::*;

    // Test error deserialization.

    #[test]
    fn test_anthropic_error_deserialize() {
        const INVALID_REQUEST: &str =
            r#"{"type":"invalid_request_error","message":"Invalid request"}"#;
        let error: AnthropicError =
            serde_json::from_str(INVALID_REQUEST).unwrap();
        assert_eq!(
            error,
            AnthropicError::InvalidRequest {
                message: "Invalid request".to_string()
            }
        );

        const AUTHENTICATION: &str = r#"{"type":"authentication_error","message":"Authentication error"}"#;
        let error: AnthropicError =
            serde_json::from_str(AUTHENTICATION).unwrap();
        assert_eq!(
            error,
            AnthropicError::Authentication {
                message: "Authentication error".to_string()
            }
        );

        const PERMISSION: &str =
            r#"{"type":"permission_error","message":"Permission denied"}"#;
        let error: AnthropicError = serde_json::from_str(PERMISSION).unwrap();
        assert_eq!(
            error,
            AnthropicError::Permission {
                message: "Permission denied".to_string()
            }
        );

        const NOT_FOUND: &str =
            r#"{"type":"not_found_error","message":"Resource not found"}"#;
        let error: AnthropicError = serde_json::from_str(NOT_FOUND).unwrap();
        assert_eq!(
            error,
            AnthropicError::NotFound {
                message: "Resource not found".to_string()
            }
        );

        const REQUEST_TOO_LARGE: &str =
            r#"{"type":"request_too_large","message":"Request too large"}"#;
        let error: AnthropicError =
            serde_json::from_str(REQUEST_TOO_LARGE).unwrap();
        assert_eq!(
            error,
            AnthropicError::RequestTooLarge {
                message: "Request too large".to_string()
            }
        );

        const RATE_LIMIT: &str =
            r#"{"type":"rate_limit_error","message":"Rate limit exceeded"}"#;
        let error: AnthropicError = serde_json::from_str(RATE_LIMIT).unwrap();
        assert_eq!(
            error,
            AnthropicError::RateLimit {
                message: "Rate limit exceeded".to_string()
            }
        );

        const API: &str =
            r#"{"type":"api_error","message":"Internal server error"}"#;
        let error: AnthropicError = serde_json::from_str(API).unwrap();
        assert_eq!(
            error,
            AnthropicError::API {
                message: "Internal server error".to_string()
            }
        );

        const OVERLOADED: &str =
            r#"{"type":"overloaded_error","message":"Service overloaded"}"#;
        let error: AnthropicError = serde_json::from_str(OVERLOADED).unwrap();
        assert_eq!(
            error,
            AnthropicError::Overloaded {
                message: "Service overloaded".to_string()
            }
        );

        // Test wrapped error (we use this in the client). We only need test one
        // variant because the wrapper is the same for all.
        const INVALID_REQUEST_WRAPPED: &str = r#"{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "<string>"
  }
}"#;

        let error: AnthropicErrorWrapper =
            serde_json::from_str(INVALID_REQUEST_WRAPPED).unwrap();
        assert_eq!(
            error.error,
            AnthropicError::InvalidRequest {
                message: "<string>".to_string()
            }
        );
    }

    // Test the Client

    use crate::{prompt::message::Role, stream::FilterExt, Prompt};

    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

    // Note: This is a real key but it's been disabled. As is warned in the
    // docs above, do not use a string literal for a real key. There is no
    // TryFrom<&'static str> for Key for this reason.
    const FAKE_API_KEY: &str = "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";

    // Error message for when the API key is not found.
    const NO_API_KEY: &str = "API key not found. Create a file named `api.key` in the crate root with your API key.";

    // Load the API key from the `api.key` file in the crate root.
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
    fn test_client_new() {
        let client = Client::new(FAKE_API_KEY.to_string()).unwrap();
        assert_eq!(client.key.to_string(), FAKE_API_KEY);

        // Apparently there isn't a way to check if the headers have been set
        // on the client. Making a request returns a builder but the headers
        // are not exposed.
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_client_message() {
        let key = load_api_key().expect(NO_API_KEY);
        let client = Client::new(key).unwrap();

        let message = client
            .message(Prompt::default().messages([(
                Role::User,
                "Emit just the \"🙏\" emoji, please.",
            )]))
            .await
            .unwrap();

        assert_eq!(message.message.role, Role::Assistant);
        assert!(message.to_string().contains("🙏"));
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_client_stream() {
        let key = load_api_key().expect(NO_API_KEY);
        let client = Client::new(key).unwrap();

        let stream = client
            .stream(Prompt::default().messages([(
                Role::User,
                "Emit just the \"🙏\" emoji, please.",
            )]))
            .await
            .unwrap();

        let msg: String = stream
            .filter_rate_limit()
            .text()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(msg, "🙏");
    }
}
