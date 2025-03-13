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

use model::request::Request;

/// Accept a message from the user.
#[axum::debug_handler]
pub async fn message_post(
    State(state): State<AppState>,
    Json(request): Json<model::request::Request>,
) -> impl IntoResponse {
    // Panic is not possible here because the channel is never closed because
    // AppState owns the channel and it is never dropped until the program
    // exits.
    state.to_events.send(request).await.unwrap();

    StatusCode::ACCEPTED
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

        let mut assistant_message: Option<misanthropic::response::Message> = None;
        let mut interrupt_message: Option<UserMessage> = None;

        loop {
            // If we were interrupted, we need to handle the partial message.
            if let Some(user_message) = interrupt_message.take() {
                // User interrupted. There should be a partial assistant message
                // unless the assistant hasn't responded yet.
                if let Some(assistant_message) = assistant_message.take() {
                    // Assistant was interrupted. If the assistant just started
                    // thinking there is a chance that the first thought block
                    // is empty and the API will not accept unsigned thoughts.
                    if let Some(assistant_message) = assistant_message.remove_incomplete_thought() {
                        // There is still at least one block in the message, so
                        // we can push both new messages to the prompt.
                        prompt.push_message(assistant_message).unwrap();
                        prompt.push_message(user_message).unwrap();
                    } else {
                        // User interrupted before the assistant responded and
                        // the Assistant thought block is empty. This user was
                        // very quick! We'll handle this edge case by merging
                        // the user message onto any previous message, which,
                        // unless we screwed up, should be a user message.
                        if let Some(last) = prompt.messages.last_mut() {
                            for block in user_message {
                                last.content.push(block);
                            }
                        }
                    }
                } else if let Err(e) = prompt.push_message(user_message) {
                    // The user interrupted before the assistant responded. This
                    // is very unlikely and a bug in the frontend if it happens.
                    let response = model::response::Response::Err(
                        model::response::Error::TurnOrder { error: e },
                    );
                    yield Event::default()
                        .event("response")
                        .json_data(response)
                        .unwrap();
                    break;
                }
            }

            if prompt.messages.last().is_none_or(|m| m.role == Role::Assistant) {
                // If the last message is a user message, we don't want to await
                // a new message from the user just yet, because we must
                // maintain the turn order. It would be the assistant's turn to
                // respond. We'll wait for the assistant to respond first before
                // taking from the user message channel again.
                let user_message = match from_user.recv().await.unwrap() {
                    Request::GetPrompt => {
                        yield Event::default()
                            .event("response")
                            // We are doing this to avoid a clone.
                            // This is supposed to be a Response::Ok(Prompt(prompt))
                            .json_data(json!({
                                "Ok": {
                                    "prompt": prompt.deref(),
                                },
                            }))
                            .unwrap();
                        continue;
                    },
                    Request::SetPrompt(new) => {
                        // This is not a fantastic idea in production because
                        // letting the user specify the prompt, including the
                        // system prompt, will absolutely lead to assholes
                        // abusing the system and getting your API key banned.
                        *prompt = new;
                        continue;
                    }
                    Request::UserMessage(user_message) => {
                        // Echo the user message back to the user. Input
                        // checks should be done here.
                        yield Event::default()
                            .event("response")
                            .json_data(json!({
                                "Ok": {
                                    "user_message": &user_message,
                                },
                            }))
                            .unwrap();
                        user_message
                    },
                };
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
                    let response = model::response::Response::Err(
                        model::response::Error::MisanthropicClient {
                            message: e.to_string(),
                        },
                    );
                    yield Event::default()
                        .event("response")
                        .json_data(response)
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
                if let Ok(request) = from_user.try_recv() {
                    // Handle user interrupt.
                    let user_message = match request {
                        Request::GetPrompt => {
                            yield Event::default()
                                .event("response")
                                .json_data(json!({
                                    "Ok": {
                                        "prompt": prompt.deref(),
                                    },
                                }))
                                .unwrap();
                            continue;
                        },
                        Request::SetPrompt(new) => {
                            *prompt = new;
                            continue;
                        }
                        Request::UserMessage(user_message) => {
                            yield Event::default()
                                .event("response")
                                .json_data(json!({
                                    "Ok": {
                                        "user_message": &user_message,
                                    },
                                }))
                                .unwrap();
                            user_message
                        },
                    };



                    // We can't take the partial message here because the stream
                    // owns a mutable reference to it. We'll just store the user
                    // message here and handle it on the next iteration.
                    interrupt_message = Some(user_message);
                    break;
                }

                match event {
                    Ok(event) => {
                        // Forward the message to the client.
                        let response = model::response::Response::Ok(
                            model::response::Success::Stream(event),
                        );
                        yield Event::default()
                            .event("response")
                            .json_data(&response)
                            .unwrap();

                        let event = response.unwrap().unwrap_stream();

                        // Handle it with the same code that the client uses.
                        prompt.handle_stream_event(event).unwrap();
                    }
                    Err(e) => {
                        // Something went wrong getting an event from the stream.
                        let response = model::response::Response::Err(
                            model::response::Error::Stream {
                                message: e.to_string(),
                            },
                        );
                        yield Event::default()
                            .event("response")
                            .json_data(response)
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

    let router = Router::new()
        .route("/events", get(events_stream))
        .route("/message", post(message_post));
    // frontend goes here
    #[cfg(debug_assertions)]
    {
        use axum::http::Method;

        let cors = tower_http::cors::CorsLayer::new()
            // Allow localhost.
            .allow_origin(tower_http::cors::Any)
            // Allow Get and post
            .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
            // Allow content type headers
            .allow_headers([axum::http::HeaderName::from_static(
                "content-type",
            )]);

        router.layer(cors).with_state(state)
    }
    #[cfg(not(debug_assertions))]
    {
        router.with_state(state)
    }
}
