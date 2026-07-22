//! [`Transport`] — how a prompt travels to an inference endpoint and comes
//! back one assembled assistant [`response::Message`], plus the endpoint's
//! behavioral [`Quirks`], as data.
//!
//! This is the minimal call shape: no retry, no middleware, no streaming.
//! Richer traits downstream (e.g. agentkit's `Inference`) extend it, and
//! chat drivers are generic over it, so a local inference engine plugs in
//! wherever a [`Client`] does.
//!
//! [`Client`]: crate::Client

use std::num::NonZeroUsize;

use futures::StreamExt;
use serde::{Deserialize, Serialize};

use crate::{model, response};

/// Per-endpoint behavioral quirks, as data. `Default` (all `false`) is
/// canonical Anthropic behavior; a `true` flag is a deviation the caller
/// can act on. See [`Transport::quirks`].
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize,
)]
#[non_exhaustive]
pub struct Quirks {
    /// Cache breakpoints belong on the assistant message, before the next
    /// user turn (blallama: the hash side-table keys on the end-of-assistant
    /// render)
    pub breakpoint_after_assistant: bool,
    /// Cache markers are ignored entirely (ollama: byte-prefix KV cache)
    pub cache_markers_ignored: bool,
    /// `tool_choice` semantics aren't honored (ollama: `Auto` forces a tool
    /// call; `Any`/`None` unsupported) — guaranteeing a text turn requires
    /// stripping `tools`, at cache cost
    pub tool_choice_not_respected: bool,
    /// Changing `output_config` does *not* invalidate the prefix cache (a
    /// blallama improvement; Anthropic re-prefills)
    pub output_config_cache_safe: bool,
    /// Usage reports no cache stats (ollama) — hit-rate logging is noise
    pub cache_stats_unreported: bool,
}

/// How a prompt travels to an inference endpoint and comes back one
/// assembled assistant [`response::Message`].
///
/// Implementations serialize the prompt; they never mutate it. `P` is the
/// prompt type — [`Prompt`](crate::Prompt) by default, and an
/// implementation may also serve [`CachedPrompt`](crate::CachedPrompt) or
/// any other `Serialize` request shape it understands.
#[async_trait::async_trait]
pub trait Transport<P = crate::Prompt>: Send + Sync
where
    P: Serialize + Send + Sync,
{
    /// Transport failure. A plain [`std::error::Error`] bound — retry
    /// classification and other policy belong to richer downstream traits.
    type Error: std::error::Error + Send + Sync + 'static;

    /// One request, one assembled [`response::Message`].
    async fn send(&self, prompt: &P) -> Result<response::Message, Self::Error>;

    /// Fan `prompts` out through [`send`](Self::send), results **aligned to
    /// input order**; the outer `Err` is a whole-submission failure. The
    /// default is bounded by [`max_concurrency`](Self::max_concurrency);
    /// transports with a cheaper native batch override it.
    async fn send_batch(
        &self,
        prompts: &[&P],
    ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error> {
        let limit = self.max_concurrency().get();
        // Materialize the (lazy) futures eagerly so the closure is invoked
        // at the method's concrete lifetimes — a closure left inside
        // `stream`/`buffered` can't satisfy the HRTB an `async_trait`
        // default imposes. `buffered` then drives the ready-made futures
        // in order, bounded by `limit`.
        let futs: Vec<_> = prompts.iter().map(|&p| self.send(p)).collect();
        Ok(futures::stream::iter(futs).buffered(limit).collect().await)
    }

    /// The [`Models`](model::Models) this transport can serve.
    async fn models(&self) -> Result<model::Models, Self::Error>;

    /// The endpoint's behavioral [`Quirks`]. The default is canonical
    /// Anthropic behavior.
    fn quirks(&self) -> Quirks {
        Quirks::default()
    }

    /// How many requests this transport should run at once. `1` (the
    /// default) forces serial execution — in general, with the default
    /// Anthropic tier and with local models, that default is optimal.
    fn max_concurrency(&self) -> NonZeroUsize {
        NonZeroUsize::MIN
    }
}

#[cfg(feature = "client")]
#[async_trait::async_trait]
impl Transport for crate::Client {
    type Error = crate::client::Error;

    async fn send(
        &self,
        prompt: &crate::Prompt,
    ) -> Result<response::Message, Self::Error> {
        self.message(prompt).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        crate::Client::models(self).await
    }
}

#[cfg(feature = "client")]
#[async_trait::async_trait]
impl Transport<crate::CachedPrompt> for crate::Client {
    type Error = crate::client::Error;

    async fn send(
        &self,
        prompt: &crate::CachedPrompt,
    ) -> Result<response::Message, Self::Error> {
        self.message(prompt).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        crate::Client::models(self).await
    }
}

// Forwarding impls for the pointers a *type-erased* transport travels in.
// [`Transport`] is dyn-compatible per prompt type, but `dyn Transport<…>` is
// unsized, so anything generic over `T: Transport` — [`Chat`] among them —
// takes the pointer, not the object. Without these, erasure compiles right
// up until the first `Chat::new`.
//
// `Arc` is the load-bearing one: a multi-agent driver hands N chat loops a
// clone apiece of one shared endpoint. `Box` is here because it is what a
// reader reaches for first when erasing a single owner.
//
// Every method forwards, the defaulted ones included. Inheriting the
// defaults instead would silently downgrade an implementor's `send_batch`,
// `quirks`, or `max_concurrency` the moment it was boxed — a native batch
// endpoint quietly falling back to serial `send`s, or a local engine's
// quirks reverting to canonical Anthropic behavior, with nothing at the
// call site to suggest the pointer changed the answer.
//
// [`Chat`]: crate::chat::Chat
#[async_trait::async_trait]
impl<P, T> Transport<P> for std::sync::Arc<T>
where
    P: Serialize + Send + Sync,
    T: Transport<P> + ?Sized,
{
    type Error = T::Error;

    async fn send(&self, prompt: &P) -> Result<response::Message, Self::Error> {
        (**self).send(prompt).await
    }

    async fn send_batch(
        &self,
        prompts: &[&P],
    ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error> {
        (**self).send_batch(prompts).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        (**self).models().await
    }

    fn quirks(&self) -> Quirks {
        (**self).quirks()
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        (**self).max_concurrency()
    }
}

#[async_trait::async_trait]
impl<P, T> Transport<P> for Box<T>
where
    P: Serialize + Send + Sync,
    T: Transport<P> + ?Sized,
{
    type Error = T::Error;

    async fn send(&self, prompt: &P) -> Result<response::Message, Self::Error> {
        (**self).send(prompt).await
    }

    async fn send_batch(
        &self,
        prompts: &[&P],
    ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error> {
        (**self).send_batch(prompts).await
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        (**self).models().await
    }

    fn quirks(&self) -> Quirks {
        (**self).quirks()
    }

    fn max_concurrency(&self) -> NonZeroUsize {
        (**self).max_concurrency()
    }
}

#[cfg(feature = "client")]
static_assertions::assert_impl_all!(crate::Client: Send, Sync, Clone);
#[cfg(feature = "client")]
static_assertions::assert_impl_all!(
    crate::client::Error: std::error::Error, Send, Sync
);

#[cfg(test)]
pub(crate) mod tests {
    use std::{
        collections::VecDeque,
        sync::{
            Mutex,
            atomic::{AtomicUsize, Ordering},
        },
    };

    use super::*;

    /// A scripted [`Transport`] — pops pre-built responses in order. Shared
    /// with the chat driver's tests.
    pub(crate) struct Script {
        pub(crate) responses: Mutex<VecDeque<response::Message>>,
        pub(crate) quirks: Quirks,
        pub(crate) concurrency: NonZeroUsize,
        in_flight: AtomicUsize,
        pub(crate) peak_in_flight: AtomicUsize,
    }

    /// The [`Script`] ran out of responses.
    #[derive(Debug, thiserror::Error)]
    #[error("script exhausted")]
    pub(crate) struct ScriptExhausted;

    impl Script {
        pub(crate) fn new(
            responses: impl IntoIterator<Item = response::Message>,
        ) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().collect()),
                quirks: Quirks::default(),
                concurrency: NonZeroUsize::MIN,
                in_flight: AtomicUsize::new(0),
                peak_in_flight: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl Transport for Script {
        type Error = ScriptExhausted;

        async fn send(
            &self,
            _prompt: &crate::Prompt,
        ) -> Result<response::Message, Self::Error> {
            let n = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak_in_flight.fetch_max(n, Ordering::SeqCst);
            // Yield so overlapping sends can actually overlap under
            // `buffered` before we decrement.
            yield_once().await;
            let popped = self.responses.lock().unwrap().pop_front();
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            popped.ok_or(ScriptExhausted)
        }

        async fn models(&self) -> Result<model::Models, Self::Error> {
            Ok(model::Models::from(vec![]))
        }

        fn quirks(&self) -> Quirks {
            self.quirks
        }

        fn max_concurrency(&self) -> NonZeroUsize {
            self.concurrency
        }
    }

    /// Pending exactly once, then ready — lets `buffered` interleave the
    /// scripted sends so the peak-in-flight counter means something.
    async fn yield_once() {
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

    fn canned(text: &str) -> response::Message {
        serde_json::from_value(serde_json::json!({
            "id": "test",
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": "test-model",
            "stop_reason": "end_turn",
            "stop_sequence": null,
        }))
        .unwrap()
    }

    #[test]
    fn quirks_default_is_canonical_anthropic() {
        let q = Quirks::default();
        assert!(!q.breakpoint_after_assistant);
        assert!(!q.cache_markers_ignored);
        assert!(!q.tool_choice_not_respected);
        assert!(!q.output_config_cache_safe);
        assert!(!q.cache_stats_unreported);
    }

    #[test]
    fn send_batch_preserves_input_order() {
        let script = Script::new(["one", "two", "three"].map(canned));
        let prompts: Vec<crate::Prompt> =
            (0..3).map(|_| crate::Prompt::default()).collect();
        let refs: Vec<&crate::Prompt> = prompts.iter().collect();

        let results =
            futures::executor::block_on(script.send_batch(&refs)).unwrap();

        let texts: Vec<String> = results
            .into_iter()
            .map(|r| r.unwrap().to_string())
            .collect();
        // `Display` renders the markdown role heading; the payload is the
        // part that proves ordering.
        for (text, expected) in texts.iter().zip(["one", "two", "three"]) {
            assert!(
                text.ends_with(expected),
                "expected {text:?} to end with {expected:?}"
            );
        }
    }

    #[test]
    fn send_batch_respects_max_concurrency() {
        let mut script = Script::new((0..8).map(|i| canned(&i.to_string())));
        script.concurrency = NonZeroUsize::new(3).unwrap();
        let prompts: Vec<crate::Prompt> =
            (0..8).map(|_| crate::Prompt::default()).collect();
        let refs: Vec<&crate::Prompt> = prompts.iter().collect();

        let results =
            futures::executor::block_on(script.send_batch(&refs)).unwrap();

        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|r| r.is_ok()));
        assert!(
            script.peak_in_flight.load(Ordering::SeqCst) <= 3,
            "peak in-flight {} exceeded max_concurrency 3",
            script.peak_in_flight.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn send_batch_surfaces_per_item_errors() {
        let script = Script::new([canned("only")]);
        let prompts: Vec<crate::Prompt> =
            (0..2).map(|_| crate::Prompt::default()).collect();
        let refs: Vec<&crate::Prompt> = prompts.iter().collect();

        let results =
            futures::executor::block_on(script.send_batch(&refs)).unwrap();

        assert!(results[0].is_ok());
        assert!(results[1].is_err());
    }

    /// The point of the smart-pointer impls: an *erased* transport still
    /// satisfies `T: Transport`, so it can be handed to anything generic
    /// over one (`Chat`, a reactor) — and `Arc` clones share the endpoint.
    #[test]
    fn erased_transport_sends_through_the_pointer() {
        fn generic_send<T: Transport>(transport: T) -> String {
            let prompt = crate::Prompt::default();
            futures::executor::block_on(transport.send(&prompt))
                .unwrap()
                .to_string()
        }

        let erased: std::sync::Arc<
            dyn Transport<crate::Prompt, Error = ScriptExhausted>,
        > = std::sync::Arc::new(Script::new(["one", "two"].map(canned)));

        assert!(generic_send(erased.clone()).ends_with("one"));
        // The clone drew from the *same* script, not a fresh one.
        assert!(generic_send(erased).ends_with("two"));

        let boxed: Box<dyn Transport<crate::Prompt, Error = ScriptExhausted>> =
            Box::new(Script::new([canned("boxed")]));
        assert!(generic_send(boxed).ends_with("boxed"));
    }

    /// Forwarding covers the *defaulted* methods too. Inheriting the trait
    /// defaults here would silently serialize a batching endpoint and
    /// reset a local engine's quirks to canonical Anthropic behavior the
    /// moment it was erased.
    #[test]
    fn erased_transport_forwards_overridden_defaults() {
        let mut script = Script::new((0..8).map(|i| canned(&i.to_string())));
        script.concurrency = NonZeroUsize::new(3).unwrap();
        script.quirks.tool_choice_not_respected = true;

        let erased: std::sync::Arc<
            dyn Transport<crate::Prompt, Error = ScriptExhausted>,
        > = std::sync::Arc::new(script);

        assert_eq!(erased.max_concurrency(), NonZeroUsize::new(3).unwrap());
        assert!(erased.quirks().tool_choice_not_respected);

        // `send_batch` forwards rather than re-deriving from the pointer's
        // own `max_concurrency` — same bound, honored through the vtable.
        let prompts: Vec<crate::Prompt> =
            (0..8).map(|_| crate::Prompt::default()).collect();
        let refs: Vec<&crate::Prompt> = prompts.iter().collect();
        let results =
            futures::executor::block_on(erased.send_batch(&refs)).unwrap();

        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}
