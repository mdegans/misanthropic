use std::sync::Arc;

use tokio::sync::{
    mpsc::{Receiver, Sender},
    Mutex,
};

use crate::{Prompt, UserMessage};

/// App state. Cheap to clone. Thread-safe.
#[derive(Clone)]
pub struct AppState {
    pub to_events: Sender<UserMessage>,
    pub from_user: Arc<Mutex<Receiver<UserMessage>>>,
    pub client: misanthropic::Client,
    pub prompt: Arc<Mutex<Prompt>>,
}
