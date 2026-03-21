//! [`Client`] for the Anthropic Messages API and related types.

#[allow(unused_imports)] // because lots of conditional compilation
use std::{collections::HashMap, env, num::NonZeroU16, sync::Arc};

#[cfg(feature = "client")]
use eventsource_stream::Eventsource;
#[cfg(feature = "client")]
use reqwest::Url;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[allow(unused_imports)] // because lots of conditional compilation
use crate::{Key, Prompt, key, model::Models, response};

#[cfg(all(feature = "batch", feature = "client"))]
use crate::batch::{self, IdentifiedBatchResult, Prompts};

/// Result type for the client. See also [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// FIXME: Prompt caching is now out of beta so we can remove the feature flag
/// for it. This will require a breaking change. Additionally, it should be
/// possible to set the beta version at runtime.

/// Client for the Anthropic Messages API. Cheap to clone.
///
/// See [`Self::new`] for creating a new client and [`Self::message`] and
/// [`Self::stream`] to get started.
#[derive(Clone)]
#[cfg(feature = "client")]
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
    /// Rate limiter. Defaults to 50 requests per minute (tier 1).
    #[cfg(feature = "rate-limiting")]
    pub rate_limiter: Option<
        Arc<
            governor::RateLimiter<
                governor::state::NotKeyed,
                governor::state::InMemoryState,
                governor::clock::DefaultClock,
                governor::middleware::NoOpMiddleware,
            >,
        >,
    >,
    /// Rate limit jitter. Defaults to [`Self::DEFAULT_JITTER_MS`].
    #[cfg(feature = "rate-limiting")]
    pub jitter: Option<governor::Jitter>,
    /// Custom endpoint for the Messages API. Defaults to [`Self::MESSAGES_URL`].
    pub messages_url: Arc<Url>,
    /// Custom endpoint for the Batch API. Defaults to [`Self::BATCH_URL`].
    pub batch_url: Arc<Url>,
    /// Custom endpoint for the Models API. Defaults to [`Self::MODELS_URL`].
    pub models_url: Arc<Url>,
}

/// Claude client. Uses the Messages API and the prompt caching beta.
#[cfg(feature = "client")]
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
    pub const MESSAGES_URL: &'static str =
        "https://api.anthropic.com/v1/messages";
    /// Default URL for the Batch API.
    pub const BATCH_URL: &'static str =
        "https://api.anthropic.com/v1/messages/batches/";
    /// Default URL for the Models API.
    pub const MODELS_URL: &'static str =
        "https://api.anthropic.com/v1/models?limit=1000";
    /// Default jitter in milliseconds for rate limiting (max).
    #[cfg(feature = "rate-limiting")]
    pub const DEFAULT_JITTER_MS: u64 = 20;

    /// Create a new [`Client`] from any type that can be converted into a
    /// [`Key`], like a [`String`] or a [`Vec`], but not a `&str`.
    // misanthropic/src/client.rs
    pub fn new<K>(key: K) -> std::result::Result<Self, key::InvalidKeyLength>
    where
        K: TryInto<Key, Error = key::InvalidKeyLength>,
    {
        Ok(Self::from_key(key.try_into()?))
    }

    /// Create a new [`Client`] with the given [`Key`].
    pub fn from_key(key: Key) -> Self {
        #[cfg(feature = "log")]
        {
            log::debug!(concat!(
                "Creating ",
                env!("CARGO_PKG_NAME"),
                " client..."
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
            #[cfg(feature = "rate-limiting")]
            rate_limiter: Some(Arc::new(governor::RateLimiter::direct(
                governor::Quota::per_minute(
                    std::num::NonZeroU32::new(50).unwrap(),
                ),
            ))),
            #[cfg(feature = "rate-limiting")]
            jitter: Some(governor::Jitter::up_to(
                std::time::Duration::from_millis(Self::DEFAULT_JITTER_MS),
            )),
            messages_url: Arc::new(Url::parse(Self::MESSAGES_URL).unwrap()),
            batch_url: Arc::new(Url::parse(Self::BATCH_URL).unwrap()),
            models_url: Arc::new(Url::parse(Self::MODELS_URL).unwrap()),
        }
    }

    /// Set [`Quota`] for the [`RateLimiter`].
    ///
    /// [`Quota`]: governor::Quota
    /// [`RateLimiter`]: governor::RateLimiter
    #[cfg(feature = "rate-limiting")]
    pub fn set_rate_limit(&mut self, quota: governor::Quota) {
        self.rate_limiter =
            Some(Arc::new(governor::RateLimiter::direct(quota)));
    }

    /// Set [`Jitter`] for the [`RateLimiter`].
    ///
    /// [`Jitter`]: governor::Jitter
    /// [`RateLimiter`]: governor::RateLimiter
    #[cfg(feature = "rate-limiting")]
    pub fn set_rate_limit_jitter(&mut self, jitter: governor::Jitter) {
        self.jitter = Some(jitter);
    }

    /// Create a [`reqwest::RequestBuilder`] with the API key set as a sensitive
    /// header value. **Does not check rate limiting**.
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
            if !matches!(method, reqwest::Method::POST) {
                // We log in the post method for POST requests.
                log::debug!("{}({})", method, url.as_str());
            }
        }

        #[allow(clippy::useless_asref)] // because conditional compilation
        let mut val =
            reqwest::header::HeaderValue::from_bytes(self.key.read().as_ref())
                .unwrap();
        val.set_sensitive(true);

        self.inner.request(method, url).header("x-api-key", val)
    }

    /// Await the rate limiter. It is not necessary to call this manually unless
    /// you are doing something custom with [`request_raw`] or similar.
    ///
    /// [`request_raw`]: Self::request_raw
    #[cfg(feature = "rate-limiting")]
    pub async fn await_rate_limiter(&self) {
        if let Some(limiter) = self.rate_limiter.as_ref() {
            if let Some(jitter) = self.jitter {
                limiter.until_ready_with_jitter(jitter).await;
            } else {
                limiter.until_ready().await;
            }
        }
    }

    /// Send a GET request with the API key set as a sensitive header value.
    /// Returns a [`reqwest::Result`] for maximum flexibility.
    pub async fn get_raw<U>(&self, url: U) -> reqwest::Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
    {
        #[cfg(feature = "rate-limiting")]
        {
            self.await_rate_limiter().await;
        }

        #[cfg(feature = "log")]
        {
            log::debug!("GET:{}", url.as_str());
        }

        self.request_raw(reqwest::Method::GET, url).send().await
    }

    /// Same as [`Self::get_raw`] but returns a crate [`Result`] instead of a
    /// [`reqwest`] result. Parses [`AnthropicError`]s.s
    async fn get<U>(&self, url: U) -> Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
    {
        let response = self.get_raw(url).await?;

        if response.status() != reqwest::StatusCode::OK {
            let error: AnthropicErrorWrapper = response.json().await?;

            // Error was sucessfully parsed from the API.
            return Err(error.error.into());
        }

        Ok(response)
    }

    /// Send a POST request with the API key set as a sensitive header value.
    /// Returns a [`reqwest::Result`] for maximum flexibility.
    pub async fn post_raw<U, B>(
        &self,
        url: U,
        body: B,
    ) -> reqwest::Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
        B: serde::Serialize,
    {
        let url: reqwest::Url = url.into_url()?;

        #[cfg(feature = "rate-limiting")]
        {
            self.await_rate_limiter().await;
        }

        #[cfg(feature = "log")]
        {
            if let Ok(json) = serde_json::to_string_pretty(&body) {
                log::debug!("POST({}):{}", &url, json);
            } else {
                log::warn!("Could not serialize body. Request will fail.");
            }
        }

        let req = self.request_raw(reqwest::Method::POST, url);

        req.json(&body).send().await
    }

    /// Same as [`Self::post_raw`] but returns a crate [`Result`] instead of a
    /// [`reqwest`] result. Parses [`AnthropicError`]s.
    pub async fn post<U, B>(&self, url: U, body: B) -> Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
        B: serde::Serialize,
    {
        let response = self.post_raw(url, body).await?;

        if response.status() != reqwest::StatusCode::OK {
            let error: AnthropicErrorWrapper = response.json().await?;

            // Error was sucessfully parsed from the API.
            return Err(error.error.into());
        }

        Ok(response)
    }

    /// Get all available [`Models`] from the API. [`Models`] is a thin wrapper
    /// around a `Vec` of [`Model`]s and derefs to it.
    ///
    /// [`Model``]: misanthropic::model::Model
    pub async fn models(&self) -> Result<Models<'_>> {
        let response = self.get(self.models_url.as_str()).await?;
        let body = response.text().await?;

        match serde_json::from_str(&body) {
            Ok(models) => {
                #[cfg(feature = "log")]
                {
                    log::debug!("RECV:{}", body);
                }

                Ok(models)
            }
            Err(e) => {
                #[cfg(feature = "log")]
                {
                    log::error!(
                        "ERROR:Could not parse {} from JSON: {:#?}",
                        stringify!(response::Model),
                        body
                    );
                }

                Err(e.into())
            }
        }
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
    pub async fn request<P>(&self, prompt: P) -> Result<crate::Response<'_>>
    where
        P: Serialize,
    {
        self.request_custom(prompt, self.messages_url.as_str())
            .await
    }

    /// Post a [`request`] to a custom URL. This is useful for testing or for
    /// using a different Messages compatible endpoint.
    ///
    /// [`request`]: Self::request
    pub async fn request_custom<P, U>(
        &self,
        prompt: P,
        url: U,
    ) -> Result<crate::Response<'_>>
    where
        P: Serialize,
        U: reqwest::IntoUrl,
    {
        let json = serde_json::to_value(prompt)?;
        let streaming = json["stream"].as_bool().unwrap_or(false);
        let response: reqwest::Response = self.post(url, json).await?;

        if streaming {
            #[cfg(feature = "log")]
            {
                log::debug!("RECV:Stream");
            }

            // Get a stream and wrap it in our stream type.
            Ok(crate::Response::Stream {
                stream: crate::Stream::new(
                    response.bytes_stream().eventsource(),
                ),
            })
        } else {
            // Get body as JSON.
            let body = response.bytes().await?;

            // Get a single response message.
            Ok(crate::Response::Message {
                message: match serde_json::from_slice(&body) {
                    Ok(msg) => {
                        #[cfg(feature = "log")]
                        {
                            log::debug!(
                                "RECV:{}",
                                serde_json::to_string_pretty(&msg).unwrap()
                            );
                        }

                        msg
                    }
                    Err(e) => {
                        #[cfg(feature = "log")]
                        {
                            log::error!(
                                "ERROR:Could not parse {} from JSON: {:#?}",
                                stringify!(response::Message),
                                body
                            );
                        }

                        return Err(e.into());
                    }
                },
            })
        }
    }

    /// Make a [`request`] to the Messages API forcing `stream=false`. This
    /// function will always return a single [`response::Message`].
    ///
    /// [`request`]: Self::request
    pub async fn message<P>(&self, prompt: P) -> Result<response::Message<'_>>
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

            let error = Error::UnexpectedResponse {
                message: "Expected a message, got a stream.",
            };

            #[cfg(feature = "log")]
            {
                log::error!("{}", error);
            }

            Err(error)
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
            #[cfg(feature = "log")]
            {
                log::error!("Expected a stream, got a message.");
            }

            Err(Error::UnexpectedResponse {
                message: "Expected a stream, got a message.",
            })
        }
    }

    /// Make a batch request of [`Prompts`] to the Messages API. This allows
    /// sending multiple prompts at once at lower cost. Batches take up to 24
    /// hours to process.
    ///
    /// Unique [`batch::Id`]s are generated for each Prompt in the batch. These
    /// can be used to track the progress of the batch.
    ///
    /// [`Prompts`]: crate::batch::Prompts
    #[cfg(feature = "batch")]
    pub async fn batch<'a, P>(&self, prompts: P) -> Result<batch::Pending<'a>>
    where
        P: IntoIterator<Item = Prompt<'a>>,
    {
        let prompts: Prompts<'a> = prompts.into_iter().collect();
        let meta = self
            .post(self.batch_url.as_str(), &prompts)
            .await?
            .json()
            .await?;

        Ok(batch::Pending { prompts, meta })
    }

    /// Same as [`Self::batch`] but with user-supplied [`batch::Id`]s. Duplicate
    /// [`batch::Id`]s will be overwritten in the order they are supplied.
    #[cfg(feature = "batch")]
    pub async fn tagged_batch<'a, It, Id>(
        &self,
        prompts: It,
    ) -> Result<batch::Pending<'a>>
    where
        It: IntoIterator<Item = (Id, Prompt<'a>)>,
        Id: Into<batch::Id>,
    {
        let prompts: Prompts<'a> = prompts.into_iter().collect();
        let meta = self
            .post(self.batch_url.as_str(), &prompts)
            .await?
            .json()
            .await?;

        Ok(batch::Pending { prompts, meta })
    }

    /// Poll the status of a [`Pending`] batch request. This update the metadata
    /// with the latest status of the [`Batch`]. If the batch is ready, the
    /// results are downloaded and returned in a [`batch::Ready`] variant.
    ///
    /// [`Batch`]: batch::Batch
    #[cfg(feature = "batch")]
    pub async fn batch_poll<'a>(
        &self,
        mut pending: batch::Pending<'a>,
    ) -> Result<batch::Batch<'a>> {
        use batch::{Batch, Ready};

        // Craft the URL for the batch.
        let url = Url::parse(self.batch_url.as_str())
            .unwrap()
            .join(pending.meta.id.as_str())
            .unwrap();

        // Update the metadata with the latest status.
        pending.meta = self.get(url).await?.json().await?;

        // Check if we're done.
        if let Some(url) = pending.results_url() {
            // Download the json lines file with `IdentifiedBatchResult`s.
            let response = self.get(url.clone()).await?.text().await?;

            // Create a new hashmap to store the results.
            let mut results = HashMap::new();

            for line in response.lines() {
                match serde_json::from_str::<serde_json::Value>(line)
                    .and_then(|v| serde_json::from_value::<IdentifiedBatchResult>(v))
                {
                    Ok(IdentifiedBatchResult { id, result }) => {
                        // We do need to check for this to maintain the Ready
                        // invariant that every result has a corresponding
                        // prompt (or it will panic).
                        if pending.prompts.contains_key(&id) {
                            results.insert(id, result);
                        } else {
                            #[cfg(feature = "log")]
                            {
                                log::warn!(
                                    "Received result for unknown ID `{}`: {}",
                                    id,
                                    serde_json::to_string_pretty(&result)
                                        .unwrap()
                                );
                            }
                        }
                    }
                    #[allow(unused_variables)]
                    Err(e) => {
                        // This should almost never happen. If it does
                        // the server is likely misbehaving.
                        #[cfg(feature = "log")]
                        {
                            log::error!(
                                "Could not parse line from batch result `{}` because: {}",
                                line,
                                e
                            );
                        }
                    }
                }
            }

            #[cfg(feature = "log")]
            if results.len() != pending.prompts.len() {
                log::warn!(
                    "Expected {} results, got {}.",
                    pending.prompts.len(),
                    results.len()
                );
            }

            // The batch is now ready.
            return Ok(Batch::Ready(Ready { pending, results }));
        }

        Ok(Batch::Pending(pending))
    }
}

#[cfg(feature = "client")]
impl From<Key> for Client {
    fn from(key: Key) -> Self {
        Self::from_key(key)
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

/// Some of the errors don't implment `Serialize` so we need to do it manually.
impl Serialize for Error {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::HTTP(e) => {
                json!({ "type": "http", "message": e.to_string() })
                    .serialize(serializer)
            }
            Self::Parse(e) => {
                json!({ "type": "parse", "message": e.to_string() })
                    .serialize(serializer)
            }
            Self::Anthropic(e) => {
                // With the `AnthropicError` we can serialize it directly, yay!
                json!({ "type": "anthropic", "message": e.to_string(), "error": e,  }).serialize(serializer)
            }
            Self::UnexpectedResponse { message } => {
                json!({ "type": "unexpected_response", "message": message })
                    .serialize(serializer)
            }
        }
    }
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
    #[error("billing: {message}")]
    #[serde(rename = "billing_error")]
    Billing { message: String },
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
    #[error("timeout: {message}")]
    #[serde(rename = "timeout_error")]
    Timeout { message: String },
    // Anthropic's API specifies they can add more error codes in the future.
    #[error("unknown error ({code}): {message}")]
    Unknown { code: NonZeroU16, message: String },
}

impl AnthropicError {
    /// Get the HTTP status code for the error.
    pub fn status(&self) -> Option<NonZeroU16> {
        match self {
            Self::InvalidRequest { .. } => Some(NonZeroU16::new(400).unwrap()),
            Self::Authentication { .. } => Some(NonZeroU16::new(401).unwrap()),
            Self::Billing { .. } => None,
            Self::Permission { .. } => Some(NonZeroU16::new(403).unwrap()),
            Self::NotFound { .. } => Some(NonZeroU16::new(404).unwrap()),
            Self::RequestTooLarge { .. } => Some(NonZeroU16::new(413).unwrap()),
            Self::RateLimit { .. } => Some(NonZeroU16::new(429).unwrap()),
            Self::API { .. } => Some(NonZeroU16::new(500).unwrap()),
            Self::Overloaded { .. } => Some(NonZeroU16::new(529).unwrap()),
            Self::Unknown { code, .. } => Some(*code),
            Self::Timeout { .. } => None,
        }
    }
}

// This is because the API tags errors and there isn't a way to tag
// both fields with "type" *and* the enum itself so we must wrap it.
#[derive(Deserialize)]
#[serde(tag = "error")]
#[cfg(feature = "client")]
pub(crate) struct AnthropicErrorWrapper {
    pub(crate) error: AnthropicError,
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
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
        #[cfg(feature = "client")]
        const INVALID_REQUEST_WRAPPED: &str = r#"{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "<string>"
  }
}"#;

        #[cfg(feature = "client")]
        {
            let error: AnthropicErrorWrapper =
                serde_json::from_str(INVALID_REQUEST_WRAPPED).unwrap();
            assert_eq!(
                error.error,
                AnthropicError::InvalidRequest {
                    message: "<string>".to_string()
                }
            );
        }
    }

    // Test the Client
    #[cfg(feature = "client")]
    use crate::{Prompt, prompt::message::Role, stream::FilterExt};

    // Note: This is a real key but it's been disabled. As is warned in the
    // docs above, do not use a string literal for a real key. There is no
    // TryFrom<&'static str> for Key for this reason.
    #[cfg(feature = "client")]
    const FAKE_API_KEY: &str = "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";

    #[cfg(feature = "client")]
    use crate::utils::load_api_key;

    #[cfg(feature = "log")]
    fn init_log() {
        let mut log_builder = env_logger::Builder::from_default_env();
        log_builder
            .filter(None, log::LevelFilter::Debug)
            .try_init()
            .ok();
    }

    #[test]
    #[cfg(feature = "client")]
    fn test_client_new() {
        #[cfg(feature = "log")]
        init_log();

        let client = Client::new(FAKE_API_KEY.to_string()).unwrap();
        assert_eq!(client.key.to_string(), FAKE_API_KEY);

        // Apparently there isn't a way to check if the headers have been set
        // on the client. Making a request returns a builder but the headers
        // are not exposed.
    }

    #[tokio::test]
    #[cfg(feature = "client")]
    #[ignore = "This test requires a real API key."]
    async fn test_client_message() {
        #[cfg(feature = "log")]
        init_log();

        let key = load_api_key().await;
        let client = Client::new(key).unwrap();

        let message = client
            .message(Prompt::default().set_messages([(
                Role::User,
                "Emit just the \"🙏\" emoji, please.",
            )]))
            .await
            .unwrap();

        assert_eq!(message.inner.role, Role::Assistant);
        assert!(message.to_string().contains("🙏"));
    }

    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_client_stream() {
        #[cfg(feature = "log")]
        init_log();

        let key = load_api_key().await;
        let client = Client::new(key).unwrap();

        let stream = client
            .stream(Prompt::default().set_messages([(
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

    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_client_models() {
        #[cfg(feature = "log")]
        init_log();

        let key = load_api_key().await;
        let client = Client::new(key).unwrap();

        let models = client.models().await.unwrap();
        assert!(!models.is_empty());
        for model in models.iter() {
            dbg!(&model);
            assert!(!model.display_name.is_empty());
        }
    }
}
