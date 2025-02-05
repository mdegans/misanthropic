use std::convert::Infallible;

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    BoxError, Json,
};
use futures::{Stream, StreamExt};
use misanthropic::{client, prompt};
use std::pin::pin;

use crate::{AppState, Prompt, UserMessage};

/// Accept a message from the user.
#[axum::debug_handler]
pub async fn message_post(
    State(state): State<AppState>,
    Json(message): Json<UserMessage>,
) -> impl IntoResponse {
    // Panic is not possible here because the channel is never closed because
    // AppState owns the channel and it is never dropped until the program
    // exits.
    state.to_events.send(message).await.unwrap();

    StatusCode::PROCESSING
}

/// Stream events from the user and the AI. Also updates the prompt with any
/// new messages, maintaining the turn order.
pub async fn events_stream(
    State(state): State<AppState>,
) -> Sse<impl Stream<Item = Result<Event, BoxError>>> {
    let prompt = if let Some(prompt) = state.prompt.write().await.take() {
        // Take the prompt so we can update it with new messages.
        prompt
    } else {
        let err_stream = futures::stream::once(async { Err("Chat in progress.".into()) }).boxed();

        return Sse::new(err_stream);
    };
    // We

    todo!("Update the prompt with new messages and stream them.")
}
