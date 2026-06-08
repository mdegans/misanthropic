//! Tool *callbacks*: a [`Tool`] pushes free-standing [`Content`] into the
//! conversation through a [`Mailbox`], and the driver drains the pushes via
//! [`Notifications`].
//!
//! Tool use is a *pair* — a [`Use`](crate::tool::Use) answered by one
//! [`Result`](crate::tool::Result) in the next message. A [`Notification`] is
//! the *other* shape: a beat a tool emits on its own schedule (a backgrounded
//! job reporting in, a periodic reminder), seated by the driver as a [`Message`]
//! at whichever [`Role`] the model supports.
//!
//! A [`Mailbox`] *owns* a channel: a [`Tool`] sends through it and the owner
//! takes the consumer end once via [`subscribe`](Mailbox::subscribe). Clones and
//! [`derive`](Mailbox::derive)d children are **send-only** (an `mpsc` has one
//! receiver). A standalone tool holds its own mailbox and exposes the receiver
//! through [`Tool::subscribe`](crate::tool::Tool::subscribe); a
//! [`ToolBox`](crate::tool::ToolBox) hands each of its tools a derived,
//! send-only mailbox and owns the single aggregate receiver itself.
//!
//! [`Message`]: crate::prompt::message::Message

use std::sync::Arc;

use futures::{StreamExt, channel::mpsc, stream::FusedStream};

use crate::prompt::message::{Content, Role};

/// A free-standing push into the conversation, outside the
/// [`Use`](crate::tool::Use)/[`Result`](crate::tool::Result) pairing.
///
/// The [`source`](Self::source) is stamped by the emitting [`Mailbox`], so a
/// tool cannot impersonate another — routing may trust it. [`preferred_roles`]
/// is a *preference*: the driver picks the first the current model supports (see
/// [`Prompt::resolve_role`](crate::Prompt::resolve_role)).
///
/// [`preferred_roles`]: Self::preferred_roles
#[derive(Debug, Clone)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Notification {
    /// Authoritative origin, stamped by the [`Mailbox`]. A nested
    /// [`ToolBox`](crate::tool::ToolBox) composes it as `parent/child/leaf`.
    pub source: Arc<str>,
    /// The pushed content.
    pub content: Content,
    /// Roles the tool would like this seated as, best first. The driver resolves
    /// it against the model; an empty list defers entirely to the driver.
    pub preferred_roles: Vec<Role>,
}

/// [`Mailbox::send`] failed because every [`Notifications`] consumer has been
/// dropped. Carries the un-sent [`Notification`] back so the caller may retry or
/// discard.
#[derive(Debug)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct MailboxClosed(pub Notification);

impl std::fmt::Display for MailboxClosed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("notification channel closed (no subscriber)")
    }
}

impl std::error::Error for MailboxClosed {}

/// An outbox owning a [`Notification`] channel. A [`Tool`](crate::tool::Tool)
/// [`send`](Self::send)s through it (sync — callable from a spawned task, an SSE
/// follower, or inside [`call`](crate::tool::Tool::call)), and the owner takes
/// the consumer end once via [`subscribe`](Self::subscribe).
///
/// [`Clone`] and `derive` produce **send-only** handles: an
/// `mpsc` has exactly one receiver, so only the original owner can subscribe.
pub struct Mailbox {
    source: Arc<str>,
    tx: mpsc::UnboundedSender<Notification>,
    /// The consumer end, until [`subscribe`](Self::subscribe) takes it. `None`
    /// on a send-only handle (a clone or a `derive`d child).
    rx: Option<mpsc::UnboundedReceiver<Notification>>,
}

impl Clone for Mailbox {
    /// A **send-only** clone — the receiver is not shared (an `mpsc` has exactly
    /// one). [`subscribe`](Self::subscribe) first if you also need the consumer
    /// end, then clone for a task that only pushes.
    fn clone(&self) -> Self {
        Self {
            source: Arc::clone(&self.source),
            tx: self.tx.clone(),
            rx: None,
        }
    }
}

impl Mailbox {
    /// Create a fresh mailbox owning a new channel, stamping `source`. The owner
    /// takes the consumer end once via [`subscribe`](Self::subscribe); clones and
    /// `derive`d children are send-only.
    pub fn new(source: impl Into<Arc<str>>) -> Self {
        let (tx, rx) = mpsc::unbounded();
        Self {
            source: source.into(),
            tx,
            rx: Some(rx),
        }
    }

    /// The authoritative source string this mailbox stamps onto every
    /// [`Notification`].
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Push a [`Notification`], stamping the [`source`](Self::source). Errors
    /// only if every consumer has dropped — see [`MailboxClosed`].
    pub fn send(
        &self,
        content: impl Into<Content>,
        preferred_roles: impl Into<Vec<Role>>,
    ) -> Result<(), MailboxClosed> {
        let note = Notification {
            source: Arc::clone(&self.source),
            content: content.into(),
            preferred_roles: preferred_roles.into(),
        };
        self.tx
            .unbounded_send(note)
            .map_err(|e| MailboxClosed(e.into_inner()))
    }

    /// Take the consumer end of this mailbox's channel, **once**. `None` if it
    /// has already been taken, or this is a send-only handle (a clone or a
    /// `derive`d child).
    ///
    /// Not generally called directly: a driver gets [`Notifications`] from a
    /// tool via [`Tool::subscribe`](crate::tool::Tool::subscribe), which
    /// delegates here.
    pub fn subscribe(&mut self) -> Option<Notifications> {
        self.rx.take().map(Notifications::new)
    }

    /// A send-only child handle on the same channel, re-stamped with `source` —
    /// for a [`ToolBox`](crate::tool::ToolBox) to give each tool a mailbox whose
    /// source composes under the box's path.
    pub(crate) fn derive(&self, source: impl Into<Arc<str>>) -> Self {
        Self {
            source: source.into(),
            tx: self.tx.clone(),
            rx: None,
        }
    }
}

/// The single consumer end of a [`Mailbox`], handed out once by
/// [`subscribe`](Mailbox::subscribe) (usually via
/// [`Tool::subscribe`](crate::tool::Tool::subscribe)). Drain it with
/// [`recv`](Self::recv) (await), [`try_recv`](Self::try_recv) (non-blocking), or
/// as a [`Stream`](futures::Stream); all are cancel-safe under `select!`.
pub struct Notifications {
    rx: mpsc::UnboundedReceiver<Notification>,
}

impl Notifications {
    pub(crate) fn new(rx: mpsc::UnboundedReceiver<Notification>) -> Self {
        Self { rx }
    }

    /// Await the next push. `None` once every [`Mailbox`] sender has dropped
    /// (a [`ToolBox`](crate::tool::ToolBox) drops its own in
    /// [`teardown_tools`](crate::tool::ToolBox::teardown_tools)).
    pub async fn recv(&mut self) -> Option<Notification> {
        self.rx.next().await
    }

    /// The next already-queued push without blocking, or why there isn't one
    /// ([`Empty`](TryRecvError::Empty) vs [`Closed`](TryRecvError::Closed)).
    pub fn try_recv(&mut self) -> Result<Notification, TryRecvError> {
        self.rx.try_recv().map_err(Into::into)
    }
}

/// Why [`Notifications::try_recv`] had nothing to return.
#[derive(thiserror::Error, Debug)]
pub enum TryRecvError {
    /// No push queued, but the channel is still open.
    #[error("no notifications ready")]
    Empty,
    /// Every sender has dropped; no further pushes will arrive.
    #[error("notifications channel closed")]
    Closed,
}

impl From<mpsc::TryRecvError> for TryRecvError {
    fn from(value: mpsc::TryRecvError) -> Self {
        match value {
            mpsc::TryRecvError::Empty => Self::Empty,
            mpsc::TryRecvError::Closed => Self::Closed,
        }
    }
}

impl futures::Stream for Notifications {
    type Item = Notification;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_next_unpin(cx)
    }
}

impl FusedStream for Notifications {
    fn is_terminated(&self) -> bool {
        self.rx.is_terminated()
    }
}
