use axum::{
    routing::{get, post},
    Router,
};
use shuttle_runtime::SecretStore;
use std::sync::Arc;
use tokio::sync::{
    mpsc::{Receiver, Sender},
    Mutex, RwLock,
};

pub mod endpoints;
pub mod prompt;

type UserMessage = misanthropic::prompt::message::UserMessage<'static>;
type Prompt = misanthropic::Prompt<'static>;
type Message = misanthropic::prompt::message::Message<'static>;

/// App state. Cheap to clone. Thread-safe.
#[derive(Clone)]
pub struct AppState {
    to_events: Sender<UserMessage>,
    from_user: Arc<Mutex<Receiver<UserMessage>>>,
    client: misanthropic::Client,
    // We take the prompt when we need it
    prompt: Arc<RwLock<Option<Prompt>>>,
}

pub enum State {
    None,
    Some(AppState),
}

// Router setup example
pub fn create_router(secrets: SecretStore) -> Router {
    let client = misanthropic::Client::new(
        secrets
            .get("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set in a Secrets.toml file."),
    )
    .unwrap();
    let prompt = prompt::default();

    let (to_events, from_user) = tokio::sync::mpsc::channel(10);

    let state = AppState {
        to_events,
        // Single consumer, so owned so, Arc.
        from_user: Arc::new(Mutex::new(from_user)),
        client,
        prompt: Arc::new(RwLock::new(Some(prompt))),
    };

    Router::new()
        .route("/events", get(endpoints::events_stream))
        .route("/message", post(endpoints::message_post))
        .with_state(state)
}

#[shuttle_runtime::main]
async fn main(
    #[shuttle_runtime::Secrets] secrets: SecretStore,
) -> shuttle_axum::ShuttleAxum {
    let router = create_router(secrets);

    Ok(router.into())
}

#[cfg(test)]
mod tests {}
