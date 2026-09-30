//! [`MockTransport`] — a scripted, recording [`Transport`] for offline tests
//! of code generic over one, plus [`Reply`] builders for the canned
//! [`response::Message`]s it hands back.
//!
//! Every [`send`](Transport::send) is recorded as the JSON that would have
//! hit the wire, in call order, and answered from a FIFO script of
//! [`Outcome`]s — falling back to a closure, if one was given with
//! [`MockTransport::with`]. An exhausted script with no fallback **panics**:
//! a missing reply is a bug in the test, and an error could be swallowed by
//! the retry or fallback logic under test instead of failing it.
//!
//! ```
//! use misanthropic::{
//!     Prompt, Transport,
//!     client::AnthropicError,
//!     mock::{self, MockTransport},
//!     prompt::message::Role,
//! };
//! # futures::executor::block_on(async {
//! let mock = MockTransport::new()
//!     .then(mock::text("Hello!").usage(10, 2).cache_read(100))
//!     .then(mock::tool_use("vote", serde_json::json!({ "vote": "yes" })))
//!     .then(mock::http_error(529, Some(3)));
//!
//! let prompt = Prompt::default().add_message((Role::User, "Hi"))?;
//! let reply = mock.send(&prompt).await?;
//! assert_eq!(reply.usage.cache_read_input_tokens, Some(100));
//! assert!(mock.send(&prompt).await?.tool_use().is_some());
//! let err = mock.send(&prompt).await.unwrap_err();
//! assert!(matches!(
//!     err,
//!     misanthropic::client::Error::Anthropic(AnthropicError::Overloaded {
//!         retry_after: Some(3),
//!         ..
//!     })
//! ));
//!
//! assert_eq!(mock.len(), 3);
//! assert_eq!(mock.last().unwrap()["messages"][0]["content"][0]["text"], "Hi");
//! # Ok::<_, Box<dyn std::error::Error>>(())
//! # }).unwrap();
//! ```

use std::{
    borrow::Cow,
    collections::VecDeque,
    num::NonZeroUsize,
    sync::{
        Mutex, MutexGuard, PoisonError,
        atomic::{AtomicUsize, Ordering},
    },
};

use serde::Serialize;

use crate::{
    Prompt, Quirks, Transport,
    client::{self, AnthropicError},
    model,
    prompt::{AssistantMessage, message::Content},
    response::{self, StopDetails, StopReason},
    tool,
};

/// A responder closure — see [`MockTransport::with`].
type Responder<P> = Box<dyn Fn(&P) -> Outcome + Send + Sync>;

/// A scripted, recording [`Transport`]. See the [module docs](self).
///
/// `P` is the prompt type served — [`Prompt`] by default, or
/// [`CachedPrompt`](crate::CachedPrompt). Share one between tasks by
/// reference or through an [`Arc`](std::sync::Arc); both satisfy
/// `T: Transport<P>`.
pub struct MockTransport<P = Prompt> {
    script: Mutex<VecDeque<Outcome>>,
    responder: Option<Responder<P>>,
    requests: Mutex<Vec<serde_json::Value>>,
    events: Mutex<Vec<Event>>,
    in_flight: AtomicUsize,
    peak_in_flight: AtomicUsize,
    yields: usize,
    quirks: Quirks,
    concurrency: NonZeroUsize,
    models: Vec<model::ModelInfo>,
}

/// A [`send`](Transport::send) starting or finishing, by request index —
/// see [`MockTransport::events`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Event {
    /// Request `n` was recorded and its [`Outcome`] chosen.
    Start(usize),
    /// Request `n` is about to return.
    End(usize),
}

/// What one [`send`](Transport::send) returns. Anything that converts —
/// a [`Reply`], a [`response::Message`], a [`client::Error`] or
/// [`AnthropicError`], a `Result` of the two, or bare text — can be
/// scripted with [`MockTransport::then`].
// Test scaffolding in a short queue: boxing the message would cost every
// `match` a deref to save a few hundred bytes per scripted reply.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum Outcome {
    /// Succeed with this message.
    Message(response::Message),
    /// Fail with this error.
    Error(client::Error),
}

impl Outcome {
    /// As the `Result` [`send`](Transport::send) returns.
    pub fn into_result(self) -> Result<response::Message, client::Error> {
        match self {
            Self::Message(message) => Ok(message),
            Self::Error(error) => Err(error),
        }
    }
}

impl From<response::Message> for Outcome {
    fn from(message: response::Message) -> Self {
        Self::Message(message)
    }
}

impl From<Reply> for Outcome {
    fn from(reply: Reply) -> Self {
        Self::Message(reply.build())
    }
}

impl From<client::Error> for Outcome {
    fn from(error: client::Error) -> Self {
        Self::Error(error)
    }
}

impl From<AnthropicError> for Outcome {
    fn from(error: AnthropicError) -> Self {
        Self::Error(error.into())
    }
}

impl From<Result<response::Message, client::Error>> for Outcome {
    fn from(result: Result<response::Message, client::Error>) -> Self {
        result.map_or_else(Self::Error, Self::Message)
    }
}

impl From<&'static str> for Outcome {
    fn from(text: &'static str) -> Self {
        self::text(text).into()
    }
}

impl From<String> for Outcome {
    fn from(text: String) -> Self {
        self::text(text).into()
    }
}

impl<P> MockTransport<P> {
    /// An empty script. Add replies with [`then`](Self::then).
    pub fn new() -> Self {
        Self {
            script: Mutex::default(),
            responder: None,
            requests: Mutex::default(),
            events: Mutex::default(),
            in_flight: AtomicUsize::new(0),
            peak_in_flight: AtomicUsize::new(0),
            yields: 0,
            quirks: Quirks::default(),
            concurrency: NonZeroUsize::MIN,
            models: Vec::new(),
        }
    }

    /// Answer requests with `outcomes`, in order.
    pub fn scripted<O, Os>(outcomes: Os) -> Self
    where
        O: Into<Outcome>,
        Os: IntoIterator<Item = O>,
    {
        let mock = Self::new();
        *lock(&mock.script) = outcomes.into_iter().map(Into::into).collect();
        mock
    }

    /// Answer each request by calling `respond` with the prompt. Scripted
    /// outcomes, if any are added with [`then`](Self::then), still go
    /// first; `respond` answers once they run out, so it never exhausts.
    ///
    /// ```
    /// # use misanthropic::{CachedPrompt, Transport, mock::{self, MockTransport}};
    /// let mock = MockTransport::with(|p: &CachedPrompt| {
    ///     mock::text(format!("{} messages", p.messages.len()))
    /// });
    /// ```
    pub fn with<F, O>(respond: F) -> Self
    where
        F: Fn(&P) -> O + Send + Sync + 'static,
        O: Into<Outcome>,
    {
        Self {
            responder: Some(Box::new(move |p| respond(p).into())),
            ..Self::new()
        }
    }

    /// Append `outcome` to the script.
    pub fn then(self, outcome: impl Into<Outcome>) -> Self {
        self.push(outcome);
        self
    }

    /// Append `outcome` to the script through a shared reference — e.g.
    /// while the transport is borrowed by the code under test.
    pub fn push(&self, outcome: impl Into<Outcome>) {
        lock(&self.script).push_back(outcome.into());
    }

    /// Yield to the executor `n` times between each send's
    /// [`Start`](Event::Start) and [`End`](Event::End), so concurrent sends
    /// overlap and [`events`](Self::events) shows whether they did.
    /// Runtime-agnostic; `0` (the default) returns without yielding.
    pub fn yields(mut self, n: usize) -> Self {
        self.yields = n;
        self
    }

    /// Report these [`Quirks`] from [`Transport::quirks`].
    pub fn with_quirks(mut self, quirks: Quirks) -> Self {
        self.quirks = quirks;
        self
    }

    /// Report this from [`Transport::max_concurrency`], which bounds the
    /// default [`send_batch`](Transport::send_batch).
    pub fn with_concurrency(mut self, concurrency: NonZeroUsize) -> Self {
        self.concurrency = concurrency;
        self
    }

    /// Report these [`ModelInfo`](model::ModelInfo)s from
    /// [`Transport::models`] (none by default) — e.g. to give a driver a
    /// model's `max_tokens` ceiling.
    pub fn with_models(
        mut self,
        models: impl IntoIterator<Item = model::ModelInfo>,
    ) -> Self {
        self.models = models.into_iter().collect();
        self
    }

    /// Every request so far, as wire JSON, in the order the sends started.
    pub fn requests(&self) -> Vec<serde_json::Value> {
        lock(&self.requests).clone()
    }

    /// The most recent request, as wire JSON.
    pub fn last(&self) -> Option<serde_json::Value> {
        lock(&self.requests).last().cloned()
    }

    /// How many requests have been sent.
    pub fn len(&self) -> usize {
        lock(&self.requests).len()
    }

    /// Whether nothing has been sent.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Scripted outcomes not yet consumed — `0` once the script has been
    /// used up exactly.
    pub fn remaining(&self) -> usize {
        lock(&self.script).len()
    }

    /// Every [`Event`] so far, in the order it happened. With
    /// [`yields`](Self::yields), this is how a test asserts that one call
    /// finished before the rest started (or that they overlapped).
    pub fn events(&self) -> Vec<Event> {
        lock(&self.events).clone()
    }

    /// The most sends that were in flight at once.
    pub fn peak_in_flight(&self) -> usize {
        self.peak_in_flight.load(Ordering::SeqCst)
    }
}

impl<P> Default for MockTransport<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P> std::fmt::Debug for MockTransport<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MockTransport")
            .field("requests", &self.len())
            .field("remaining", &self.remaining())
            .field("responder", &self.responder.is_some())
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl<P> Transport<P> for MockTransport<P>
where
    P: Serialize + Send + Sync,
{
    type Error = client::Error;

    async fn send(&self, prompt: &P) -> Result<response::Message, Self::Error> {
        let request = serde_json::to_value(prompt)?;
        // Record and pop under one lock, so request `n` is always paired
        // with the `n`th scripted outcome, however sends interleave.
        let (index, scripted) = {
            let mut requests = lock(&self.requests);
            requests.push(request);
            (requests.len() - 1, lock(&self.script).pop_front())
        };
        let outcome = match (scripted, &self.responder) {
            (Some(outcome), _) => outcome,
            (None, Some(respond)) => respond(prompt),
            (None, None) => panic!(
                "MockTransport: script exhausted at request {index} (zero-\
                 based) with no fallback — script another reply with \
                 `then`/`push`, or answer the rest with `MockTransport::with`"
            ),
        };

        lock(&self.events).push(Event::Start(index));
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.peak_in_flight.fetch_max(now, Ordering::SeqCst);
        for _ in 0..self.yields {
            yield_now().await;
        }
        self.in_flight.fetch_sub(1, Ordering::SeqCst);
        lock(&self.events).push(Event::End(index));

        outcome.into_result()
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        Ok(self.models.iter().cloned().collect())
    }

    fn quirks(&self) -> Quirks {
        self.quirks
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        self.concurrency
    }
}

/// Lock, ignoring poison: one panicking test task shouldn't turn every
/// later accessor into a second, misleading panic.
fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Pending once, then ready — a runtime-agnostic `yield_now`.
async fn yield_now() {
    let mut yielded = false;
    futures::future::poll_fn(move |cx| {
        if yielded {
            std::task::Poll::Ready(())
        } else {
            yielded = true;
            cx.waker().wake_by_ref();
            std::task::Poll::Pending
        }
    })
    .await
}

/// Builder for a canned [`response::Message`]. Start from [`text`],
/// [`tool_use`], [`refusal`], [`max_tokens`] or [`message`]; converts into an
/// [`Outcome`] or, with [`build`](Self::build), a [`response::Message`].
///
/// Usage defaults to zero tokens; set it with [`usage`](Self::usage),
/// [`cache_read`](Self::cache_read) and [`cache_write`](Self::cache_write),
/// or wholesale with [`counts`](Self::counts).
#[derive(Clone, Debug)]
pub struct Reply {
    message: response::Message,
}

/// A reply of `text`, stopping at [`EndTurn`](StopReason::EndTurn).
pub fn text(text: impl Into<crate::CowStr>) -> Reply {
    message(
        response::Message::builder(
            model::Model::default(),
            AssistantMessage::text(text),
        )
        .stop_reason(StopReason::EndTurn)
        .build(),
    )
}

/// A reply calling tool `name` with `input`, stopping at
/// [`ToolUse`](StopReason::ToolUse). The call gets a fresh unique id; use
/// [`Reply::call`] to choose one.
pub fn tool_use(
    name: impl Into<Cow<'static, str>>,
    input: serde_json::Value,
) -> Reply {
    empty().tool_use(name, input)
}

/// A [`Refusal`](StopReason::Refusal) with no content, carrying
/// [`StopDetails`]. Chain it after content for a refusal that came with
/// some: `mock::tool_use(..).refusal(..)`.
pub fn refusal(
    category: impl Into<Cow<'static, str>>,
    explanation: impl Into<Cow<'static, str>>,
) -> Reply {
    empty().refusal(category, explanation)
}

/// A reply of `partial` text, cut off at
/// [`MaxTokens`](StopReason::MaxTokens).
pub fn max_tokens(partial: impl Into<crate::CowStr>) -> Reply {
    text(partial).stop_reason(StopReason::MaxTokens)
}

/// Start from an existing `message` — e.g. a captured fixture — to adjust
/// its usage or stop reason.
pub fn message(message: response::Message) -> Reply {
    Reply { message }
}

/// A scripted failure with `error` — anything convertible to a
/// [`client::Error`], such as an [`AnthropicError`].
pub fn error(error: impl Into<client::Error>) -> Outcome {
    Outcome::Error(error.into())
}

/// A scripted failure as the [`Client`](crate::Client) reports a non-OK
/// `status` response: the [`AnthropicError`] variant the API sends for
/// it, carrying `retry_after` (seconds) where that variant can (`429`,
/// `529`); any other status becomes [`AnthropicError::Unknown`] with the
/// status as its code.
pub fn http_error(status: u16, retry_after: Option<u64>) -> Outcome {
    let message = format!("mock HTTP {status}");
    error(match status {
        400 => AnthropicError::InvalidRequest { message },
        401 => AnthropicError::Authentication { message },
        403 => AnthropicError::Permission { message },
        404 => AnthropicError::NotFound { message },
        413 => AnthropicError::RequestTooLarge { message },
        429 => AnthropicError::RateLimit {
            message,
            retry_after,
        },
        500 => AnthropicError::API { message },
        529 => AnthropicError::Overloaded {
            message,
            retry_after,
        },
        code => AnthropicError::Unknown {
            code: std::num::NonZeroU16::new(code),
            message,
        },
    })
}

/// A reply with no content and no stop reason.
fn empty() -> Reply {
    message(
        response::Message::builder(
            model::Model::default(),
            AssistantMessage::from(Content(Vec::new())),
        )
        .build(),
    )
}

impl Reply {
    /// Append a text block.
    pub fn text(mut self, text: impl Into<crate::CowStr>) -> Self {
        self.message.inner.content.push(text.into());
        self
    }

    /// Append a call to tool `name` with `input`, under a fresh unique id,
    /// and stop at [`ToolUse`](StopReason::ToolUse).
    pub fn tool_use(
        self,
        name: impl Into<Cow<'static, str>>,
        input: serde_json::Value,
    ) -> Self {
        let id = format!("toolu_mock_{}", uuid::Uuid::new_v4().simple());
        self.call(tool::Use::new(name, input).with_id(id))
    }

    /// Append `call` as given — its id included — and stop at
    /// [`ToolUse`](StopReason::ToolUse).
    pub fn call(mut self, call: tool::Use) -> Self {
        self.message.inner.content.push(call);
        self.stop_reason(StopReason::ToolUse)
    }

    /// Stop at [`Refusal`](StopReason::Refusal) with these
    /// [`StopDetails`].
    pub fn refusal(
        mut self,
        category: impl Into<Cow<'static, str>>,
        explanation: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.message.stop_details = Some(Box::new(StopDetails {
            category: Some(category.into()),
            explanation: Some(explanation.into()),
        }));
        self.stop_reason(StopReason::Refusal)
    }

    /// Set the [`StopReason`].
    pub fn stop_reason(mut self, stop_reason: StopReason) -> Self {
        self.message.stop_reason = Some(stop_reason);
        self
    }

    /// Set the input and output token counts.
    pub fn usage(mut self, input: u64, output: u64) -> Self {
        self.message.usage.input_tokens = input;
        self.message.usage.output_tokens = output;
        self
    }

    /// Set the tokens read from the prompt cache.
    pub fn cache_read(mut self, tokens: u64) -> Self {
        self.message.usage.cache_read_input_tokens = Some(tokens);
        self
    }

    /// Set the tokens written to the prompt cache.
    pub fn cache_write(mut self, tokens: u64) -> Self {
        self.message.usage.cache_creation_input_tokens = Some(tokens);
        self
    }

    /// Replace the token counts wholesale.
    pub fn counts(mut self, counts: response::TokenCounts) -> Self {
        self.message.usage.counts = counts;
        self
    }

    /// Set the [`Model`](model::Model) the reply claims to be from.
    pub fn model(mut self, model: impl Into<model::Model>) -> Self {
        self.message.model = model.into();
        self
    }

    /// Set the message [`id`](response::Message::id).
    pub fn id(mut self, id: impl Into<Cow<'static, str>>) -> Self {
        self.message.id = id.into();
        self
    }

    /// Finish.
    pub fn build(self) -> response::Message {
        self.message
    }
}

impl From<Reply> for response::Message {
    fn from(reply: Reply) -> Self {
        reply.build()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::executor::block_on;

    use super::*;
    use crate::{CachedPrompt, prompt::message::Role};

    fn prompt(text: &'static str) -> Prompt {
        Prompt::default().add_message((Role::User, text)).unwrap()
    }

    fn user_text(request: &serde_json::Value) -> &str {
        request["messages"][0]["content"][0]["text"]
            .as_str()
            .unwrap()
    }

    #[test]
    fn text_reply() {
        let mock = MockTransport::new().then("Hello!");
        let reply = block_on(mock.send(&prompt("Hi"))).unwrap();
        assert_eq!(reply.stop_reason, Some(StopReason::EndTurn));
        assert_eq!(reply.inner.content.len(), 1);
        assert!(reply.to_string().ends_with("Hello!"));
    }

    #[test]
    fn tool_use_reply() {
        let mock = MockTransport::new()
            .then(tool_use("vote", serde_json::json!({ "vote": "yes" })))
            .then(
                text("Calling.").call(
                    tool::Use::new("vote", serde_json::json!({}))
                        .with_id("toolu_1"),
                ),
            );

        let first = block_on(mock.send(&prompt("a"))).unwrap();
        let call = first.tool_use().unwrap();
        assert_eq!(first.stop_reason, Some(StopReason::ToolUse));
        assert_eq!(call.name, "vote");
        assert_eq!(call.input["vote"], "yes");
        assert!(call.id.starts_with("toolu_mock_"));

        let second = block_on(mock.send(&prompt("b"))).unwrap();
        assert_eq!(second.inner.content.len(), 2);
        assert_eq!(second.tool_use().unwrap().id, "toolu_1");
    }

    #[test]
    fn refusal_reply() {
        let mock = MockTransport::new()
            .then(refusal("cyber", "Flagged."))
            .then(
                tool_use("vote", serde_json::json!({})).refusal("bio", "No."),
            );

        let bare = block_on(mock.send(&prompt("a"))).unwrap();
        assert_eq!(bare.stop_reason, Some(StopReason::Refusal));
        assert!(bare.inner.content.is_empty());
        let details = bare.stop_details.unwrap();
        assert_eq!(details.category.as_deref(), Some("cyber"));
        assert_eq!(details.explanation.as_deref(), Some("Flagged."));

        // A refusal can carry content; the stop reason is the refusal's.
        let with_call = block_on(mock.send(&prompt("b"))).unwrap();
        assert_eq!(with_call.stop_reason, Some(StopReason::Refusal));
        assert!(with_call.inner.tool_use().is_some());
    }

    #[test]
    fn max_tokens_reply() {
        let mock = MockTransport::new().then(max_tokens("Partial"));
        let reply = block_on(mock.send(&prompt("a"))).unwrap();
        assert_eq!(reply.stop_reason, Some(StopReason::MaxTokens));
        assert!(reply.to_string().ends_with("Partial"));
    }

    #[test]
    fn full_message_round_trips_unchanged() {
        let captured = text("Fixture").id("msg_fixture").build();
        let mock = MockTransport::new().then(captured.clone());
        let reply = block_on(mock.send(&prompt("a"))).unwrap();
        assert_eq!(reply, captured);
    }

    #[test]
    fn usage_and_cache_counts() {
        let mut counts = response::TokenCounts::new(7, 8);
        counts.cache_creation_input_tokens = Some(9);
        let mock = MockTransport::new()
            .then(text("a").usage(10, 20).cache_read(300).cache_write(40))
            .then(text("b").counts(counts));

        let first = block_on(mock.send(&prompt("a"))).unwrap();
        assert_eq!(first.usage.input_tokens, 10);
        assert_eq!(first.usage.output_tokens, 20);
        assert_eq!(first.usage.cache_read_input_tokens, Some(300));
        assert_eq!(first.usage.cache_creation_input_tokens, Some(40));

        let second = block_on(mock.send(&prompt("b"))).unwrap();
        assert_eq!(second.usage.counts, counts);
    }

    #[test]
    fn errors_including_http_status_and_retry_after() {
        let mock = MockTransport::new()
            .then(http_error(429, Some(7)))
            .then(http_error(529, None))
            .then(http_error(502, Some(1)))
            .then(error(AnthropicError::InvalidRequest {
                message: "bad".into(),
            }));

        let anthropic = |e| match e {
            client::Error::Anthropic(e) => e,
            other => panic!("expected an AnthropicError, got {other:?}"),
        };
        let rate = anthropic(block_on(mock.send(&prompt("a"))).unwrap_err());
        assert_eq!(rate.status().map(|s| s.get()), Some(429));
        assert_eq!(rate.retry_after(), Some(std::time::Duration::from_secs(7)));

        let overloaded =
            anthropic(block_on(mock.send(&prompt("b"))).unwrap_err());
        assert_eq!(overloaded.status().map(|s| s.get()), Some(529));
        assert_eq!(overloaded.retry_after(), None);

        // No variant for 502: the status survives on `Unknown`, and
        // `retry_after` has nowhere to go.
        let gateway = anthropic(block_on(mock.send(&prompt("c"))).unwrap_err());
        assert_eq!(gateway.status().map(|s| s.get()), Some(502));
        assert!(matches!(gateway, AnthropicError::Unknown { .. }));

        let bad = anthropic(block_on(mock.send(&prompt("d"))).unwrap_err());
        assert_eq!(bad.status().map(|s| s.get()), Some(400));
        assert_eq!(mock.len(), 4, "failed sends are recorded too");
    }

    #[test]
    fn records_requests_in_order() {
        let mock = MockTransport::scripted(["1", "2", "3"]);
        assert!(mock.is_empty());
        for text in ["one", "two", "three"] {
            block_on(mock.send(&prompt(text))).unwrap();
        }
        let requests = mock.requests();
        let texts: Vec<_> = requests.iter().map(user_text).collect();
        assert_eq!(texts, ["one", "two", "three"]);
        assert_eq!(mock.len(), 3);
        assert_eq!(mock.last().as_ref().map(user_text), Some("three"));
        assert_eq!(mock.remaining(), 0);
    }

    #[test]
    fn records_the_exact_wire_json() {
        let prompt = prompt("Hi").system("Be brief.");
        let mock = MockTransport::new().then("ok");
        block_on(mock.send(&prompt)).unwrap();
        assert_eq!(mock.last(), Some(serde_json::to_value(&prompt).unwrap()));
    }

    #[test]
    fn responder_answers_after_the_script() {
        let mock = MockTransport::with(|p: &Prompt| {
            text(format!("echo: {}", p.messages[0].content))
        })
        .then("scripted");

        let first = block_on(mock.send(&prompt("a"))).unwrap();
        assert!(first.to_string().ends_with("scripted"));
        for _ in 0..3 {
            let echoed = block_on(mock.send(&prompt("b"))).unwrap();
            assert!(echoed.to_string().ends_with("echo: b"), "{echoed}");
        }
    }

    #[test]
    #[should_panic(expected = "script exhausted at request 1")]
    fn exhausted_script_panics() {
        let mock = MockTransport::new().then("only");
        block_on(mock.send(&prompt("a"))).unwrap();
        let _ = block_on(mock.send(&prompt("b")));
    }

    #[test]
    fn push_through_a_shared_reference() {
        let mock = Arc::new(MockTransport::<Prompt>::new());
        let shared = Arc::clone(&mock);
        shared.push("late");
        assert_eq!(mock.remaining(), 1);
        block_on(shared.send(&prompt("a"))).unwrap();
        assert_eq!(mock.len(), 1);
    }

    #[test]
    fn serves_cached_prompts() {
        fn generic<T: Transport<CachedPrompt>>(t: &T) -> response::Message {
            let prompt = CachedPrompt::cached(prompt("Hi"));
            block_on(t.send(&prompt)).unwrap()
        }

        let mock = MockTransport::new().then(text("ok").cache_read(12));
        let reply = generic(&mock);
        assert_eq!(reply.usage.cache_read_input_tokens, Some(12));
        let request = mock.last().unwrap();
        assert_eq!(user_text(&request), "Hi");
        assert!(
            request.to_string().contains("cache_control"),
            "the cached prompt's markers are on the wire: {request}"
        );
    }

    #[test]
    fn events_show_serial_sends_do_not_overlap() {
        let mock = MockTransport::scripted(["a", "b"]).yields(3);
        block_on(mock.send(&prompt("a"))).unwrap();
        block_on(mock.send(&prompt("b"))).unwrap();
        assert_eq!(
            mock.events(),
            [
                Event::Start(0),
                Event::End(0),
                Event::Start(1),
                Event::End(1)
            ]
        );
        assert_eq!(mock.peak_in_flight(), 1);
    }

    #[test]
    fn send_batch_honors_concurrency_and_pairs_outcomes() {
        let mock = MockTransport::scripted((0..8).map(|i| i.to_string()))
            .yields(2)
            .with_concurrency(NonZeroUsize::new(3).unwrap());
        let prompts: Vec<Prompt> = (0..8).map(|_| prompt("x")).collect();
        let refs: Vec<&Prompt> = prompts.iter().collect();

        let results = block_on(mock.send_batch(&refs)).unwrap();

        for (i, result) in results.into_iter().enumerate() {
            assert!(result.unwrap().to_string().ends_with(&i.to_string()));
        }
        assert_eq!(mock.peak_in_flight(), 3);
        // Overlap is visible: the second send starts before the first ends.
        let events = mock.events();
        assert_eq!(&events[..2], [Event::Start(0), Event::Start(1)]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_sends_from_many_threads() {
        const N: usize = 64;
        let mock = Arc::new(
            MockTransport::with(|p: &Prompt| {
                text(p.messages[0].content.to_string())
            })
            .yields(4),
        );

        let tasks: Vec<_> = (0..N)
            .map(|i| {
                let mock = Arc::clone(&mock);
                tokio::spawn(async move {
                    let text = i.to_string();
                    let prompt = Prompt::default()
                        .add_message((Role::User, text.clone()))
                        .unwrap();
                    let reply = mock.send(&prompt).await.unwrap();
                    assert!(reply.to_string().ends_with(&text));
                })
            })
            .collect();
        for task in tasks {
            task.await.unwrap();
        }

        assert_eq!(mock.len(), N);
        let mut seen: Vec<usize> = mock
            .requests()
            .iter()
            .map(|r| user_text(r).parse().unwrap())
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..N).collect::<Vec<_>>());
        let events = mock.events();
        assert_eq!(events.len(), 2 * N);
        for n in 0..N {
            let start = events.iter().position(|e| *e == Event::Start(n));
            let end = events.iter().position(|e| *e == Event::End(n));
            assert!(start < end, "request {n}: {events:?}");
        }
    }

    #[test]
    fn quirks_are_reported() {
        let quirks = Quirks {
            cache_markers_ignored: true,
            ..Quirks::default()
        };
        let mock = MockTransport::<Prompt>::new().with_quirks(quirks);
        assert!(Transport::<Prompt>::quirks(&mock).cache_markers_ignored);
    }

    static_assertions::assert_impl_all!(
        MockTransport<Prompt>: Transport<Prompt>, Send, Sync
    );
    static_assertions::assert_impl_all!(
        MockTransport<CachedPrompt>: Transport<CachedPrompt>, Send, Sync
    );
}
