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
