//! Test support: the wire-legality invariants [`Chat`](super::Chat) upholds,
//! written independently of the driver's own bookkeeping, and [`Checked`], a
//! [`Transport`] that asserts them on every request it forwards.

use std::sync::{Arc, Mutex};

use crate::{
    Prompt, Quirks, Transport, model,
    prompt::message::{Message, Role},
    response::{self, TokenCounts},
};

/// The roles of `prompt`'s turns as one letter each (`U`, `A`, `S`) — a
/// compact shape to assert and to print.
pub(crate) fn roles(prompt: &Prompt) -> String {
    prompt
        .messages
        .iter()
        .map(|m| match m.role {
            Role::User => 'U',
            Role::Assistant => 'A',
            Role::System => 'S',
        })
        .collect()
}

/// `prompt`'s turns, one line each, for a failure message.
fn render(prompt: &Prompt) -> String {
    prompt
        .messages
        .iter()
        .map(|m| format!("  {}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether `tail` is a turn only the model can follow: anything but an
/// assistant turn, or an assistant turn paused on a server tool.
fn awaits_model(tail: &Message) -> bool {
    tail.role != Role::Assistant
        || tail.unfinished_server_tool_uses().next().is_some()
}

/// Assert legal turn order and that no turn is empty — the API rejects an
/// empty turn anywhere but a final assistant prefill, which [`Chat`] never
/// sends.
///
/// [`Chat`]: super::Chat
fn assert_well_formed(prompt: &Prompt) {
    if let Err(error) = prompt.check_turn_order() {
        panic!("illegal turn order: {error}\n{}", render(prompt));
    }
    assert!(
        prompt.messages.iter().all(|m| !m.content.is_empty()),
        "an empty turn:\n{}",
        render(prompt)
    );
}

/// Assert that `prompt` is a request the Messages API accepts, as far as
/// turn placement goes: [well formed](assert_well_formed), a tail the model
/// answers (no prefill, no unanswered client call), and no
/// [`System`](Role::System) turn for a known model without the role. A
/// custom model (a local server) may accept one, so it is not held to that.
pub(crate) fn assert_request_legal(prompt: &Prompt) {
    assert_well_formed(prompt);
    let tail = prompt
        .messages
        .last()
        .unwrap_or_else(|| panic!("a request with no messages"));
    assert!(
        awaits_model(tail),
        "the request's tail doesn't await the model:\n{}",
        render(prompt)
    );
    assert!(
        tail.role != Role::Assistant || tail.tool_uses().next().is_none(),
        "the request's tail carries unanswered client calls:\n{}",
        render(prompt)
    );
    if matches!(prompt.model, model::Model::Anthropic(_))
        && !prompt.model.supports_system_role()
    {
        assert!(
            prompt.messages.iter().all(|m| m.role != Role::System),
            "a system turn sent to {}, which has no system role:\n{}",
            prompt.model,
            render(prompt)
        );
    }
}

/// Assert that the caller can carry on from a `prompt` [`Chat`] handed back
/// (`Ok` or inside a [`chat::Error`]): [well formed](assert_well_formed),
/// with a tail that either awaits the model — so it must be a legal request,
/// which a resumed `Chat` answers first — or takes the caller's next beat.
///
/// [`Chat`]: super::Chat
/// [`chat::Error`]: super::Error
pub(crate) fn assert_handback_legal(prompt: &Prompt) {
    assert_well_formed(prompt);
    let Some(tail) = prompt.messages.last() else {
        return; // nothing yet: any beat may open it
    };
    if awaits_model(tail) {
        return assert_request_legal(prompt);
    }
    let beat = Message::from((Role::User, "next"));
    if let Err(error) = tail.may_precede(&beat) {
        panic!("the next beat can't follow the hand-back: {error}");
    }
}

/// What a [`Checked`] transport saw, in order.
#[derive(Default)]
pub(crate) struct Log {
    /// Every prompt sent, as sent.
    pub(crate) sent: Vec<Prompt>,
    /// Every response received (failed sends have none).
    pub(crate) received: Vec<response::Message>,
}

impl Log {
    /// The received responses' token counts, summed — what a
    /// [`track_usage`](super::Chat::track_usage) sink should hold.
    pub(crate) fn usage(&self) -> TokenCounts {
        self.received
            .iter()
            .fold(TokenCounts::default(), |mut sum, r| {
                sum += r.usage.counts;
                sum
            })
    }
}

/// A [`Transport`] that [asserts](assert_request_legal) every request is
/// legal before forwarding it, and logs what went out and came back.
#[derive(Clone)]
pub(crate) struct Checked<T> {
    inner: T,
    log: Arc<Mutex<Log>>,
}

impl<T> Checked<T> {
    pub(crate) fn new(inner: T) -> Self {
        Self {
            inner,
            log: Arc::default(),
        }
    }

    /// The log so far.
    pub(crate) fn log(&self) -> std::sync::MutexGuard<'_, Log> {
        self.log.lock().unwrap()
    }
}

impl<T> Checked<T> {
    /// Assert `prompt` is legal, then log it as sent.
    fn check(&self, prompt: &Prompt) {
        assert_request_legal(prompt);
        self.log().sent.push(prompt.clone());
    }
}

/// Every method forwards to the inner transport — a default left in place
/// would test the trait's behavior, not the transport's.
#[async_trait::async_trait]
impl<T: Transport> Transport for Checked<T> {
    type Error = T::Error;

    async fn send(
        &self,
        prompt: &Prompt,
    ) -> Result<response::Message, Self::Error> {
        self.check(prompt);
        let response = self.inner.send(prompt).await?;
        self.log().received.push(response.clone());
        Ok(response)
    }

    async fn send_batch(
        &self,
        prompts: &[&Prompt],
    ) -> Result<Vec<Result<response::Message, Self::Error>>, Self::Error> {
        prompts.iter().for_each(|prompt| self.check(prompt));
        let responses = self.inner.send_batch(prompts).await?;
        let received = responses.iter().filter_map(|r| r.as_ref().ok());
        self.log().received.extend(received.cloned());
        Ok(responses)
    }

    async fn models(&self) -> Result<model::Models, Self::Error> {
        self.inner.models().await
    }

    fn quirks(&self) -> Quirks {
        self.inner.quirks()
    }

    fn max_concurrency(&self) -> std::num::NonZeroUsize {
        self.inner.max_concurrency()
    }
}

#[test]
fn checks_catch_illegal_shapes() {
    use crate::tool::Use;

    let panics = |f: &dyn Fn()| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err()
    };
    let user = || Message::from((Role::User, "hi"));
    let mut dangling = Message::from((Role::Assistant, "calling"));
    dangling
        .content
        .push(Use::new("f", serde_json::Value::Null).with_id("a"));

    // A prefill and an unanswered call are hand-back-legal only if the next
    // beat can follow: the prefill can, the dangling call can't.
    let prefill = Prompt::user("hi").add_message((Role::Assistant, "ok"));
    let prefill = prefill.unwrap();
    assert!(panics(&|| assert_request_legal(&prefill)));
    assert_handback_legal(&prefill);

    let unanswered = Prompt {
        messages: vec![user(), dangling],
        ..Prompt::default()
    };
    assert!(panics(&|| assert_request_legal(&unanswered)));
    assert!(panics(&|| assert_handback_legal(&unanswered)));

    // A system turn to a model without one; fine for a custom model.
    let noted = Prompt::user("hi").add_message((Role::System, "note"));
    let noted = noted.unwrap().model(crate::Id::Sonnet46);
    assert!(panics(&|| assert_request_legal(&noted)));
    assert_request_legal(&noted.clone().model(crate::Id::Opus48));
    assert_request_legal(&noted.model("local.gguf"));

    // An empty turn, even one the order allows.
    let mut empty = Prompt::user("hi");
    empty.messages.push(Message::from((Role::Assistant, "")));
    empty.messages[1].content.clear();
    empty.messages.push(user());
    assert!(panics(&|| assert_request_legal(&empty)));
}

#[cfg(feature = "mock")]
#[test]
fn checked_forwards_every_method() {
    use crate::mock::{self, MockTransport};

    let quirks = Quirks {
        tool_choice_not_respected: true,
        ..Quirks::default()
    };
    let three = std::num::NonZeroUsize::new(3).unwrap();
    let mock = MockTransport::new()
        .with_quirks(quirks)
        .with_concurrency(three)
        .then(mock::text("one"))
        .then(mock::text("two"));
    let checked = Checked::new(mock);
    let prompts = [Prompt::user("a"), Prompt::user("b")];

    let replies = futures::executor::block_on(
        checked.send_batch(&prompts.iter().collect::<Vec<_>>()),
    )
    .unwrap();

    assert_eq!(replies.len(), 2);
    let log = checked.log();
    assert_eq!((log.sent.len(), log.received.len()), (2, 2));
    assert_eq!(checked.quirks(), quirks);
    assert_eq!(checked.max_concurrency(), three);
}
