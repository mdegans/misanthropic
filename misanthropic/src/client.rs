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
    /// Custom endpoint for the Messages API. Defaults to [`Self::MESSAGES_URL`].
    pub messages_url: Arc<Url>,
    /// Custom endpoint for the Batch API. Defaults to [`Self::BATCH_URL`].
    pub batch_url: Arc<Url>,
    /// Custom endpoint for the Models API. Defaults to [`Self::MODELS_URL`].
    pub models_url: Arc<Url>,
    /// Custom endpoint for the token counting API. Defaults to
    /// [`Self::COUNT_TOKENS_URL`].
    pub count_tokens_url: Arc<Url>,
}

/// Claude client. Uses the Messages API.
#[cfg(feature = "client")]
impl Client {
    /// Version of the API. This is appended to the header as
    /// "anthropic-version".
    pub const ANTHROPIC_VERSION: &'static str = "2023-06-01";
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
    /// Default URL for the token counting API.
    pub const COUNT_TOKENS_URL: &'static str =
        "https://api.anthropic.com/v1/messages/count_tokens";

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

        Self {
            inner: reqwest::Client::builder()
                .default_headers(headers)
                .build()
                .unwrap(),
            key: Arc::new(key),
            messages_url: Arc::new(Url::parse(Self::MESSAGES_URL).unwrap()),
            batch_url: Arc::new(Url::parse(Self::BATCH_URL).unwrap()),
            models_url: Arc::new(Url::parse(Self::MODELS_URL).unwrap()),
            count_tokens_url: Arc::new(
                Url::parse(Self::COUNT_TOKENS_URL).unwrap(),
            ),
        }
    }

    /// Set a custom base URL for all API endpoints.
    ///
    /// Replaces the scheme, host, and port of all endpoint URLs while
    /// preserving their paths and query strings. Useful for pointing at
    /// Ollama's Anthropic-compatible endpoint, proxies, or test servers.
    ///
    /// ```rust,no_run
    /// # use misanthropic::Client;
    /// let client = Client::new("x".repeat(108))?
    ///     .with_base_url("http://localhost:11434")?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn with_base_url(
        mut self,
        base: &str,
    ) -> std::result::Result<Self, url::ParseError> {
        let base = Url::parse(base)?;

        let rebase = |endpoint: &Url| -> Arc<Url> {
            let mut new = base.clone();
            new.set_path(endpoint.path());
            new.set_query(endpoint.query());
            Arc::new(new)
        };

        self.messages_url = rebase(&self.messages_url);
        self.batch_url = rebase(&self.batch_url);
        self.models_url = rebase(&self.models_url);
        self.count_tokens_url = rebase(&self.count_tokens_url);

        Ok(self)
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

    /// Send a GET request with the API key set as a sensitive header value.
    /// Returns a [`reqwest::Result`] for maximum flexibility.
    pub async fn get_raw<U>(&self, url: U) -> reqwest::Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
    {
        #[cfg(feature = "log")]
        {
            log::debug!("GET:{}", url.as_str());
        }

        self.request_raw(reqwest::Method::GET, url).send().await
    }

    /// Send a GET and return the response body as a `String`.
    ///
    /// On non-OK statuses, attempts to parse the body as an
    /// [`AnthropicError`]. If that fails (e.g. because an edge proxy
    /// returned an HTML error page, a Cloudflare challenge, a
    /// rate-limit plaintext response, or anything else the API
    /// doesn't normally emit), surfaces an [`Error::NonJsonResponse`]
    /// carrying the HTTP status and the first few KB of the body.
    /// This keeps the opaque reqwest `"error decoding response body"`
    /// out of callers' error chains and gives operators something to
    /// grep for when an upstream misbehaves.
    ///
    /// When the `log` feature is enabled, the full body is emitted at
    /// `debug!` level — matching the `RECV:{body}` pattern used by
    /// `request_json` for POST responses. Previously GETs logged only
    /// the request URL.
    async fn get<U>(&self, url: U) -> Result<String>
    where
        U: reqwest::IntoUrl,
    {
        let response = self.get_raw(url).await?;
        let status = response.status();
        let body = response.text().await?;

        if status != reqwest::StatusCode::OK {
            // Error path: try the documented Anthropic shape first.
            return match serde_json::from_str::<AnthropicErrorWrapper>(&body) {
                Ok(wrapper) => Err(wrapper.error.into()),
                Err(_parse_err) => {
                    #[cfg(feature = "log")]
                    {
                        log::error!(
                            "ERROR:non-JSON error body (status {}): {}",
                            status.as_u16(),
                            truncate_body(&body),
                        );
                    }
                    Err(Error::NonJsonResponse {
                        status: status.as_u16(),
                        body: truncate_body(&body).into_owned(),
                    })
                }
            };
        }

        #[cfg(feature = "log")]
        {
            log::debug!("RECV:{}", body);
        }

        Ok(body)
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
    ///
    /// On non-OK statuses, attempts to parse the body as an
    /// [`AnthropicError`]. If that fails (edge proxy returned HTML,
    /// plaintext rate-limit notice, Cloudflare challenge, …), surfaces
    /// an [`Error::NonJsonResponse`] with the HTTP status and a
    /// truncated body snippet — mirroring [`Self::get`]'s behaviour so
    /// operators get the same diagnostic signal for POST and GET
    /// failures.
    pub async fn post<U, B>(&self, url: U, body: B) -> Result<reqwest::Response>
    where
        U: reqwest::IntoUrl,
        B: serde::Serialize,
    {
        let response = self.post_raw(url, body).await?;

        if response.status() != reqwest::StatusCode::OK {
            let status = response.status();
            let body = response.text().await?;
            return match serde_json::from_str::<AnthropicErrorWrapper>(&body) {
                Ok(wrapper) => Err(wrapper.error.into()),
                Err(_parse_err) => {
                    #[cfg(feature = "log")]
                    {
                        log::error!(
                            "ERROR:non-JSON error body (status {}): {}",
                            status.as_u16(),
                            truncate_body(&body),
                        );
                    }
                    Err(Error::NonJsonResponse {
                        status: status.as_u16(),
                        body: truncate_body(&body).into_owned(),
                    })
                }
            };
        }

        Ok(response)
    }

    /// Get all available [`Models`] from the API. [`Models`] is a thin wrapper
    /// around a `Vec` of [`Model`]s and derefs to it.
    ///
    /// [`Model``]: misanthropic::model::Model
    pub async fn models(&self) -> Result<Models<'_>> {
        // `get` now reads the body, logs it, and surfaces non-JSON
        // error bodies as `Error::NonJsonResponse`, so we only have to
        // deal with parse failures on an OK body.
        let body = self.get(self.models_url.as_str()).await?;

        match serde_json::from_str(&body) {
            Ok(models) => Ok(models),
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
    /// Unique [`batch::Id`]s are generated for each prompt in the batch. These
    /// can be used to track the progress of the batch.
    ///
    /// The prompt type `P` can be any [`Serialize`] type — typically
    /// [`Prompt`] or [`CachedPrompt`](crate::CachedPrompt).
    ///
    /// [`Prompts`]: crate::batch::Prompts
    #[cfg(feature = "batch")]
    pub async fn batch<P>(
        &self,
        prompts: impl IntoIterator<Item = P>,
    ) -> Result<batch::Pending<P>>
    where
        P: Serialize,
    {
        let prompts: Prompts<P> = prompts.into_iter().collect();
        let response = self.post(self.batch_url.as_str(), &prompts).await?;
        let meta = parse_body(response, "batch::Metadata").await?;

        Ok(batch::Pending { prompts, meta })
    }

    /// Same as [`Self::batch`] but with user-supplied [`batch::Id`]s. Duplicate
    /// [`batch::Id`]s will be overwritten in the order they are supplied.
    #[cfg(feature = "batch")]
    pub async fn tagged_batch<P, It, Id>(
        &self,
        prompts: It,
    ) -> Result<batch::Pending<P>>
    where
        P: Serialize,
        It: IntoIterator<Item = (Id, P)>,
        Id: Into<batch::Id>,
    {
        let prompts: Prompts<P> = prompts.into_iter().collect();
        let response = self.post(self.batch_url.as_str(), &prompts).await?;
        let meta = parse_body(response, "batch::Metadata").await?;

        Ok(batch::Pending { prompts, meta })
    }

    /// Poll the status of a [`Pending`] batch request. This update the metadata
    /// with the latest status of the [`Batch`]. If the batch is ready, the
    /// results are downloaded and returned in a [`batch::Ready`] variant.
    ///
    /// The prompt type `P` does not need to be [`Serialize`] — polling only
    /// downloads results, it never re-serializes prompts.
    ///
    /// [`Batch`]: batch::Batch
    #[cfg(feature = "batch")]
    pub async fn batch_poll<P>(
        &self,
        mut pending: batch::Pending<P>,
    ) -> Result<batch::Batch<P>> {
        use batch::{Batch, Ready};

        // Craft the URL for the batch.
        let url = Url::parse(self.batch_url.as_str())
            .unwrap()
            .join(pending.meta.id.as_str())
            .unwrap();

        // Update the metadata with the latest status. `get` returns the
        // body as text and already surfaces non-JSON error bodies via
        // `Error::NonJsonResponse`, so any failure here is an actual
        // parse error on a body that was at least plausible JSON.
        let meta_body = self.get(url).await?;
        pending.meta = serde_json::from_str(&meta_body)?;

        // Check if we're done.
        if let Some(url) = pending.results_url() {
            // Download the json lines file with `IdentifiedBatchResult`s.
            let response = self.get(url.clone()).await?;

            // Create a new hashmap to store the results.
            let mut results = HashMap::new();

            for line in response.lines() {
                match serde_json::from_str::<serde_json::Value>(line).and_then(
                    |v| serde_json::from_value::<IdentifiedBatchResult>(v),
                ) {
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

    /// Count the number of input tokens in a prompt without creating a message.
    ///
    /// This calls the `/v1/messages/count_tokens` endpoint and returns the
    /// `input_tokens` count. Useful for estimating costs or making decisions
    /// about prompt construction before sending a full request.
    pub async fn count_tokens<P>(&self, prompt: P) -> Result<u32>
    where
        P: Serialize,
    {
        #[derive(Deserialize)]
        struct TokenCount {
            input_tokens: u32,
        }

        let response =
            self.post(self.count_tokens_url.as_str(), prompt).await?;
        let count: TokenCount = parse_body(response, "TokenCount").await?;

        Ok(count.input_tokens)
    }
}

#[cfg(feature = "client")]
impl From<Key> for Client {
    fn from(key: Key) -> Self {
        Self::from_key(key)
    }
}

/// Maximum number of bytes of a non-JSON response body to preserve in
/// [`Error::NonJsonResponse`] and debug logs. Bodies longer than this are
/// truncated with a `... [N more bytes]` suffix so operators still know
/// how much content was elided.
#[cfg(any(feature = "client", test))]
const NON_JSON_BODY_SNIPPET_LEN: usize = 2048;

/// Read `response` body as text and parse it as JSON, logging the raw
/// body (truncated to [`NON_JSON_BODY_SNIPPET_LEN`]) at `error!` level
/// on parse failure.
///
/// Use this instead of `response.json().await?` on any post-OK-status
/// body where the caller needs structured output. The built-in
/// [`reqwest::Response::json`] path surfaces only the opaque
/// `"error decoding response body"` message and discards the bytes
/// before they can be logged, leaving upstream schema drift or edge-
/// proxy error bodies undiagnosable without a packet capture.
///
/// `context` identifies the call site in the log message (for
/// example, `"batch::Metadata"` or `"TokenCount"`). It is unused when
/// the `log` feature is disabled.
#[cfg(feature = "client")]
async fn parse_body<T>(
    response: reqwest::Response,
    #[cfg_attr(not(feature = "log"), allow(unused_variables))] context: &str,
) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    let body = response.text().await?;
    match serde_json::from_str::<T>(&body) {
        Ok(value) => Ok(value),
        Err(e) => {
            #[cfg(feature = "log")]
            {
                log::error!(
                    "ERROR:Could not parse {} from JSON: {}",
                    context,
                    truncate_body(&body),
                );
            }
            Err(e.into())
        }
    }
}

/// Truncate `body` to at most [`NON_JSON_BODY_SNIPPET_LEN`] bytes at a
/// valid UTF-8 boundary, appending a `... [N more bytes]` suffix if the
/// body was longer.
#[cfg(any(feature = "client", test))]
fn truncate_body(body: &str) -> std::borrow::Cow<'_, str> {
    if body.len() <= NON_JSON_BODY_SNIPPET_LEN {
        return std::borrow::Cow::Borrowed(body);
    }
    // Walk backward from the budget to find a valid UTF-8 boundary so we
    // don't slice through a multi-byte character.
    let mut cut = NON_JSON_BODY_SNIPPET_LEN;
    while cut > 0 && !body.is_char_boundary(cut) {
        cut -= 1;
    }
    let remaining = body.len() - cut;
    std::borrow::Cow::Owned(format!(
        "{}... [{remaining} more bytes]",
        &body[..cut]
    ))
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
    /// The server returned a non-OK status whose body could not be parsed
    /// as an [`AnthropicError`] — most commonly a proxy or edge layer
    /// (Cloudflare, gateway timeout) returning HTML or plaintext instead
    /// of the documented JSON error shape. The body is truncated to
    /// [`NON_JSON_BODY_SNIPPET_LEN`] bytes so the error fits in a log
    /// line.
    #[error("non-JSON error response (status {status}): {body}")]
    #[allow(missing_docs)]
    NonJsonResponse { status: u16, body: String },
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
            Self::NonJsonResponse { status, body } => json!({
                "type": "non_json_response",
                "status": status,
                "body": body,
            })
            .serialize(serializer),
            Self::UnexpectedResponse { message } => {
                json!({ "type": "unexpected_response", "message": message })
                    .serialize(serializer)
            }
        }
    }
}

/// Anthropic error type.
///
/// Deserialization route:
/// - Matches on the `type` string field using the explicit
///   `#[serde(rename = "...")]` aliases below when populated by the
///   custom [`Deserialize`] impl.
/// - Any `type` value not matching a known variant falls through to
///   [`Self::Unknown`] with `code: None` and `message` carrying both
///   the unrecognized type name and the body's `message`. This is the
///   genuine catch-all — previously a `type` value Anthropic hadn't
///   documented at crate release time (e.g. a new `gateway_timeout`
///   variant) would fail deserialization entirely and surface as
///   reqwest's opaque `"error decoding response body"`.
///
/// `#[serde(tag = "type", rename_all = "snake_case")]` remains in the
/// derive attributes purely for the `Serialize` path — we still emit
/// variants in Anthropic's documented shape.
#[derive(Debug, thiserror::Error, Serialize, PartialEq)]
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
    /// Fallback for any error `type` not covered by the variants above
    /// — either an Anthropic error kind introduced after this crate
    /// was released, or a synthetic error constructed by the crate
    /// itself (e.g. batch results for cancelled/expired items).
    ///
    /// `code` is `None` when constructed via the deserialize fallback
    /// (no HTTP status can be inferred from a `type` string alone).
    /// Synthetic constructors can pass `Some(code)` when they have a
    /// meaningful status to carry.
    #[error("unknown error{}: {message}", code.map(|c| format!(" ({c})")).unwrap_or_default())]
    Unknown {
        code: Option<NonZeroU16>,
        message: String,
    },
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
            Self::Unknown { code, .. } => *code,
            Self::Timeout { .. } => None,
        }
    }
}

/// Custom `Deserialize` impl so unknown `type` values fall through to
/// [`AnthropicError::Unknown`] instead of failing deserialization.
/// The derive'd impl would reject anything outside the explicit
/// variant list — and because `Unknown` requires a `code` field that
/// real Anthropic bodies don't carry, it couldn't serve as a fallback
/// on its own.
impl<'de> Deserialize<'de> for AnthropicError {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;

        let value = serde_json::Value::deserialize(deserializer)?;
        let type_name =
            value.get("type").and_then(|v| v.as_str()).ok_or_else(|| {
                D::Error::custom("AnthropicError body missing 'type' field")
            })?;
        let message = value
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(match type_name {
            "invalid_request_error" => Self::InvalidRequest { message },
            "authentication_error" => Self::Authentication { message },
            "billing_error" => Self::Billing { message },
            "permission_error" => Self::Permission { message },
            "not_found_error" => Self::NotFound { message },
            "request_too_large" => Self::RequestTooLarge { message },
            "rate_limit_error" => Self::RateLimit { message },
            "api_error" => Self::API { message },
            "overloaded_error" => Self::Overloaded { message },
            "timeout_error" => Self::Timeout { message },
            unknown => Self::Unknown {
                code: None,
                message: format!("{unknown}: {message}"),
            },
        })
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

    // Test body truncation for NonJsonResponse payloads.

    #[test]
    fn test_truncate_body_shorter_than_limit_is_borrowed() {
        let body = "short response";
        let out = truncate_body(body);
        assert_eq!(out.as_ref(), body);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_truncate_body_exact_limit_is_borrowed() {
        let body = "a".repeat(NON_JSON_BODY_SNIPPET_LEN);
        let out = truncate_body(&body);
        assert_eq!(out.len(), NON_JSON_BODY_SNIPPET_LEN);
        assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    }

    #[test]
    fn test_truncate_body_over_limit_is_owned_with_suffix() {
        let body = "a".repeat(NON_JSON_BODY_SNIPPET_LEN + 500);
        let out = truncate_body(&body);
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        assert!(out.starts_with(&"a".repeat(NON_JSON_BODY_SNIPPET_LEN)));
        assert!(out.ends_with("[500 more bytes]"));
    }

    #[test]
    fn test_truncate_body_does_not_split_utf8() {
        // Build a body where a multi-byte character straddles the limit.
        // Each `é` is 2 bytes. Pad with 2046 ASCII then add `é` so the
        // boundary falls inside the `é`.
        let mut body = "a".repeat(NON_JSON_BODY_SNIPPET_LEN - 1);
        body.push('é');
        body.push('é');
        let out = truncate_body(&body);
        // The truncate walks backward from 2048; the char at byte 2047
        // starts at 2047 and ends at 2049, so the walk lands at 2047.
        // Result must still be valid UTF-8.
        assert!(matches!(out, std::borrow::Cow::Owned(_)));
        // If this panics, we sliced through a multi-byte char.
        let _: &str = out.as_ref();
    }

    #[test]
    fn test_non_json_response_error_serializes() {
        let err = Error::NonJsonResponse {
            status: 502,
            body: "<html>502 Bad Gateway</html>".to_string(),
        };
        let json = serde_json::to_value(&err).unwrap();
        assert_eq!(json["type"], "non_json_response");
        assert_eq!(json["status"], 502);
        assert_eq!(json["body"], "<html>502 Bad Gateway</html>");
    }

    #[test]
    fn test_non_json_response_error_display() {
        let err = Error::NonJsonResponse {
            status: 502,
            body: "<html>oops</html>".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("502"), "missing status: {msg}");
        assert!(msg.contains("<html>oops</html>"), "missing body: {msg}");
    }

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

    // Test that unknown error types fall through to Unknown rather
    // than failing deserialization (the failure mode observed 2026-04-18
    // where reqwest surfaced an opaque "error decoding response body"
    // and the raw Anthropic response was gone).

    #[test]
    fn test_unknown_error_type_falls_through() {
        const UNKNOWN: &str = r#"{"type":"gateway_timeout_error","message":"upstream timed out"}"#;
        let error: AnthropicError = serde_json::from_str(UNKNOWN).unwrap();
        assert_eq!(
            error,
            AnthropicError::Unknown {
                code: None,
                message: "gateway_timeout_error: upstream timed out"
                    .to_string()
            }
        );
        // `status()` returns None for deserialize-fallback Unknowns —
        // we have no way to infer HTTP status from a type string alone.
        assert_eq!(error.status(), None);
    }

    #[test]
    fn test_unknown_error_type_without_message_field() {
        // Anthropic could (legally, by their own docs) ship an error
        // body missing the `message` field. Don't fail the whole
        // deserialize for that — default to empty string.
        const UNKNOWN: &str = r#"{"type":"brand_new_error"}"#;
        let error: AnthropicError = serde_json::from_str(UNKNOWN).unwrap();
        assert_eq!(
            error,
            AnthropicError::Unknown {
                code: None,
                message: "brand_new_error: ".to_string()
            }
        );
    }

    #[test]
    fn test_anthropic_error_missing_type_field_is_rejected() {
        // But a body with no `type` at all is still a hard fail —
        // that's definitionally not an Anthropic error shape. The
        // outer `NonJsonResponse` / `parse_body` paths in
        // `client::post` / `client::get` will catch it and preserve
        // the raw body.
        const BAD: &str = r#"{"message":"where's my type?"}"#;
        assert!(serde_json::from_str::<AnthropicError>(BAD).is_err());
    }

    #[test]
    fn test_synthetic_unknown_preserves_code() {
        // Synthetic constructors (batch result for cancelled/expired,
        // stream assertion fallbacks) still supply a meaningful status
        // code via `Some(...)`. Confirm `status()` returns it.
        let e = AnthropicError::Unknown {
            code: Some(NonZeroU16::new(408).unwrap()),
            message: "Batch result expired".to_string(),
        };
        assert_eq!(e.status(), Some(NonZeroU16::new(408).unwrap()));
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

    #[test]
    #[cfg(feature = "client")]
    fn test_with_base_url() {
        let client = Client::new(FAKE_API_KEY.to_string())
            .unwrap()
            .with_base_url("http://localhost:11434")
            .unwrap();

        assert_eq!(
            client.messages_url.as_str(),
            "http://localhost:11434/v1/messages"
        );
        assert_eq!(
            client.batch_url.as_str(),
            "http://localhost:11434/v1/messages/batches/"
        );
        assert_eq!(client.models_url.path(), "/v1/models");
        assert_eq!(client.models_url.query(), Some("limit=1000"));
        assert_eq!(
            client.count_tokens_url.as_str(),
            "http://localhost:11434/v1/messages/count_tokens"
        );

        // Invalid URL should error.
        let client = Client::new(FAKE_API_KEY.to_string()).unwrap();
        assert!(client.with_base_url("not a url").is_err());
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
            .text()
            .try_collect()
            .await
            .unwrap();

        assert_eq!(msg, "🙏");
    }

    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_client_count_tokens() {
        #[cfg(feature = "log")]
        init_log();

        let key = load_api_key().await;
        let client = Client::new(key).unwrap();

        let count = client
            .count_tokens(
                Prompt::default().set_messages([(Role::User, "Hello, world!")]),
            )
            .await
            .unwrap();

        assert!(count > 0, "Token count should be positive");
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
