use std::sync::Arc;

use model::request::Request;
use tokio::sync::{
    mpsc::{Receiver, Sender},
    Mutex,
};

use crate::Prompt;

/// App state. Cheap to clone. Thread-safe.
#[derive(Clone)]
pub struct AppState {
    pub to_events: Sender<Request>,
    pub from_user: Arc<Mutex<Receiver<Request>>>,
    pub client: misanthropic::Client,
    pub prompt: Arc<Mutex<Prompt>>,
}

static_assertions::assert_impl_all!(AppState: Send, Sync);

impl From<misanthropic::Client> for AppState {
    fn from(client: misanthropic::Client) -> Self {
        let (to_events, from_user) = tokio::sync::mpsc::channel(10);

        AppState {
            to_events,
            // Single consumer, so owned so, Arc.
            from_user: Arc::new(Mutex::new(from_user)),
            client,
            prompt: Arc::new(crate::prompt::default().into()),
        }
    }
}
