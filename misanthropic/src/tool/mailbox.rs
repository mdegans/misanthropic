//! Tool *callbacks*: a [`Tool`] pushes free-standing [`Content`] into the
//! conversation through a [`Mailbox`], and the driver drains the pushes via
//! [`Notifications`].
//!
//! Tool use is a *pair* — a [`Use`](crate::tool::Use) answered by one
//! [`Result`](crate::tool::Result) in the next message. A [`Notification`] is
//! the *other* shape: a beat a tool emits on its own schedule (a backgrounded
//! job reporting in, a periodic reminder), seated by the driver as a [`Message`]
//! at whichever [`Role`] the model supports. The [`ToolBox`] owns the single
//! consumer end; each [`Tool`] gets a [`Mailbox`] clone via
//! [`Tool::connect`](crate::tool::Tool::connect) and pushes whenever it likes.
//!
//! [`Message`]: crate::prompt::message::Message
//! [`ToolBox`]: crate::tool::ToolBox

use std::sync::Arc;

use futures::{StreamExt, channel::mpsc};

use crate::prompt::message::{Content, Role};

/// A free-standing push into the conversation, outside the
/// [`Use`](crate::tool::Use)/[`Result`](crate::tool::Result) pairing.
///
/// The [`source`](Self::source) is stamped by the emitting [`Mailbox`] (which
/// the [`ToolBox`](crate::tool::ToolBox) minted with the tool's own name), so a
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

/// An outbox handed to a [`Tool`](crate::tool::Tool) by its
/// [`ToolBox`](crate::tool::ToolBox) via
/// [`connect`](crate::tool::Tool::connect). Cheaply [`Clone`]able; every clone
/// stamps the same [`source`](Notification::source). [`send`](Self::send) is
/// sync, so it's callable from a spawned task, an SSE follower, or inside
/// [`call`](crate::tool::Tool::call).
#[derive(Clone)]
pub struct Mailbox {
    source: Arc<str>,
    tx: mpsc::UnboundedSender<Notification>,
}

impl Mailbox {
    /// Build a mailbox stamping `source`, pushing into `tx`.
    pub(crate) fn new(
        source: impl Into<Arc<str>>,
        tx: mpsc::UnboundedSender<Notification>,
    ) -> Self {
        Self {
            source: source.into(),
            tx,
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

    /// Decompose into the channel sender and the source path, for a nested
    /// [`ToolBox`](crate::tool::ToolBox) to adopt and re-stamp under.
    pub(crate) fn into_parts(
        self,
    ) -> (mpsc::UnboundedSender<Notification>, Arc<str>) {
        (self.tx, self.source)
    }
}

/// The single consumer end of a [`ToolBox`](crate::tool::ToolBox)'s outbox,
/// handed out once by [`subscribe`](crate::tool::ToolBox::subscribe). Drain it
/// with [`recv`](Self::recv) (await) or [`try_recv`](Self::try_recv)
/// (non-blocking); both are cancel-safe under `select!`.
pub struct Notifications {
    rx: mpsc::UnboundedReceiver<Notification>,
}

impl Notifications {
    pub(crate) fn new(rx: mpsc::UnboundedReceiver<Notification>) -> Self {
        Self { rx }
    }

    /// Await the next push. `None` once every [`Mailbox`] **and** the owning
    /// [`ToolBox`](crate::tool::ToolBox)'s own sender have dropped (the box
    /// drops its sender in
    /// [`teardown_tools`](crate::tool::ToolBox::teardown_tools)).
    pub async fn recv(&mut self) -> Option<Notification> {
        self.rx.next().await
    }

    /// The next already-queued push, if any — never blocks.
    pub fn try_recv(&mut self) -> Option<Notification> {
        self.rx.try_recv().ok()
    }
}
