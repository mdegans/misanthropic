use std::{ops::Deref, sync::Arc};

use axum::{
    extract::State,
    http::StatusCode,
    response::{
        sse::{Event, Sse},
        IntoResponse,
    },
    routing::{get, post},
    BoxError, Json, Router,
};
use futures::{pin_mut, Stream, StreamExt};
use serde_json::json;
use shuttle_runtime::SecretStore;
use tokio::sync::Mutex;

use misanthropic::{prompt::message::Role, stream::FilterExt};

use crate::{AppState, UserMessage};

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
    // Get the prompt. If it's in use, return a busy message.
    let stream = async_stream::try_stream! {
        // Lock the prompt and the user message channel. A more sophisticated
        // system could lock and unlock these as needed, but then we'd have to
        // handle many more error cases.
        let mut from_user = state.from_user.lock().await;
        let mut prompt = state.prompt.lock().await;

        let mut assistant_message = None;
        let mut interrupt_message = None;

        loop {
            // If we were interrupted, we need to handle the partial message.
            if let Some(user_message) = interrupt_message.take() {
                // User interrupted. There should be a partial assistant message
                // unless the assistant hasn't responded yet.
                if let Some(assistant_message) = assistant_message.take() {
                    // Unwrap can't panic because we have exclusive access to the
                    // prompt and we just verified the turn order.
                    prompt.push_message(assistant_message).unwrap();
                    prompt.push_message(user_message).unwrap();
                } else if let Err(e) = prompt.push_message(user_message) {
                    // The user interrupted before the assistant responded. This
                    // is very unlikely and a bug in the frontend if it happens.
                    yield Event::default()
                        .event("turn_order_error")
                        .json_data(json!({
                            "type": "turn_order_error",
                            "data": e,
                        }))
                        .unwrap();
                    break;
                }
            }

            // Yield a copy of the prompt. In production, this would be a bad
            // idea because the prompt could be large and this struct includes
            // the system prompt. This is just for demonstration purposes and so
            // the code is easier to understand. If somehow the client is
            // desynchronized, this will help us debug it. If the client needs
            // to reconnect, it will get the full prompt to resynchronize. A
            // fancier system could have a user event to request a full prompt.
            yield Event::default()
                .json_data(json!({
                    "type": "prompt",
                    "data": prompt.deref(),
                }))
                .unwrap();

            if prompt.messages.last().is_none_or(|m| m.role == Role::Assistant) {
                // If the last message is a user message, we don't want to await
                // a new message from the user just yet, because we must
                // maintain the turn order. It would be the assistant's turn to
                // respond. We'll wait for the assistant to respond first before
                // taking from the user message channel again.
                let user_message = from_user.recv().await.unwrap();
                // Unwrap can't panic because we just verified the turn order
                // and we have exclusive access to the prompt.
                prompt.push_message(user_message).unwrap();
            }
            // Agent's turn to respond. We have guaranteed that the last message
            // in the prompt is a user message because there are only two roles.

            // Get a streaming response from the Anthropic AI. This will include
            // full messages and tool use events.
            let stream = match state.client.stream(prompt.deref()).await {
                Ok(stream) => stream
                    // Anthropic is sending *us* messages so this is not very
                    // important, however with many users it might be useful to
                    // include rate limit errors.
                    .filter_rate_limit()
                    // Adds a full message event, `Event::Message`
                    .with_message_ip(&mut assistant_message)
                    // Adds a tool use event, `Event::ToolUse`
                    .with_tool_use(),
                Err(e) => {
                    // Something went wrong getting a stream from Anthropic. We
                    // should really handle the individual errors here, since
                    // some are recoverable.
                    yield Event::default()
                        .event("misanthropic_client_error")
                        .json_data(json!({
                            "type": "misanthropic_client_error",
                            "data": e.to_string(),
                        }))
                        .unwrap();
                    break;
                }
            };

            pin_mut!(stream);

            while let Some(event) = stream.next().await {
                // Listen for an interrupt signal from the user. We could join
                // this with `stream.next()` but it's easier to understand this
                // way. Very small latency difference since we must wait for the
                // next event but there are many of these per second.
                if let Ok(user_message) = from_user.try_recv() {
                    // We can't take the partial message here because the stream
                    // owns a mutable reference to it. We'll just store the user
                    // message here and handle it on the next iteration.
                    interrupt_message = Some(user_message);
                    break;
                }

                match event {
                    Ok(misanthropic::stream::Event::Message { message }) => {
                        // A complete message from the AI. We'll add it to the
                        // prompt. The next iteration of the loop will send it
                        // to the user, who should have an identical copy if
                        // the frontend is assembling the events correctly.
                        // Unwrap can't panic because we have exclusive access
                        // to the prompt and we just verified the turn order.
                        prompt.push_message(message).unwrap();
                        // TODO: Handle tool use here. Technically, as is, the
                        // API can handle tool use on the client side since the
                        // tool use response is a user message.
                    }
                    Ok(event) => {
                        // Any other event we'll just send to the user,
                        // including tool use.
                        yield Event::default()
                            .event("stream_event")
                            .json_data(json!({
                                "type": "stream_event",
                                "data": event,
                            }))
                            .unwrap();
                    }
                    Err(e) => {
                        // Something went wrong getting an event from the stream.
                        yield Event::default()
                            .event("stream_error")
                            .json_data(json!({
                                "type": "stream_error",
                                "data": e.to_string(),
                            }))
                            .unwrap();
                    }
                }
            }
        }
    };

    Sse::new(stream)
}

pub fn create_router(secrets: SecretStore) -> Router {
    let client = misanthropic::Client::new(
        secrets
            .get("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set in a Secrets.toml file."),
    )
    .unwrap();
    let prompt = crate::prompt::default();

    let (to_events, from_user) = tokio::sync::mpsc::channel(10);

    let state = AppState {
        to_events,
        // Single consumer, so owned so, Arc.
        from_user: Arc::new(Mutex::new(from_user)),
        client,
        prompt: Arc::new(Mutex::new(prompt)),
    };

    Router::new()
        .route("/events", get(events_stream))
        .route("/message", post(message_post))
        // frontend goes here
        .with_state(state)
}
