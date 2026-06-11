use std::ops::Deref;

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

use misanthropic::{prompt::message::Role, stream::FilterExt};

use crate::{AppState, AssistantMessage, UserMessage};

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
    log::debug!("Creating event stream");

    // Get the prompt. If it's in use, return a busy message.
    let stream = async_stream::try_stream! {
        // Lock the prompt and the user message channel. A more sophisticated
        // system could lock and unlock these as needed, but then we'd have to
        // handle many more error cases.
        let mut from_user = state.from_user.lock().await;
        let mut prompt = state.prompt.lock().await;

        log::info!("Starting event stream");
        log::info!("Prompt: {}", serde_json::to_string_pretty(prompt.deref()).unwrap());

        let mut assistant_message: Option<misanthropic::response::Message> = None;
        let mut interrupt_message: Option<UserMessage> = None;

        // Backoff for when Anthropic rejects the prompt. We retry the same
        // prompt in place (without tearing down the stream) after a growing
        // delay so a hard-failing prompt can't hammer the API on reconnect.
        const MAX_BACKOFF: std::time::Duration =
            std::time::Duration::from_secs(60);
        let mut backoff = std::time::Duration::from_secs(1);

        loop {
            // If we were interrupted, we need to handle the partial message.
            if let Some(user_message) = interrupt_message.take() {
                log::debug!(
                    "Handling user message: {:?}",
                    serde_json::to_string_pretty(&user_message).unwrap()
                );
                // User interrupted. There should be a partial assistant message
                // unless the assistant hasn't responded yet.
                if let Some(assistant_message) = assistant_message.take() {
                    log::debug!(
                        "Already have partial assistant message: {:?}",
                        serde_json::to_string_pretty(&assistant_message).unwrap()
                    );
                    // Assistant was interrupted. If the assistant just started
                    // thinking there is a chance that the first thought block
                    // is empty and the API will not accept unsigned thoughts.
                    if let Some(assistant_message) = assistant_message.remove_incomplete_thought() {
                        log::debug!(
                            // TODO: In the future, if the model is not sonnet
                            // 3.7 with built-in thought block support, we can
                            // terminate the thought block instead of removing.
                            // (because *in this configuration* thoughts must be
                            // signed by Anthropic).
                            "Any incomplete thought was removed and a message is still left over: {:?}",
                            serde_json::to_string_pretty(&assistant_message).unwrap()
                        );
                        // There is still at least one block in the message, so
                        // we can push both new messages to the prompt.
                        prompt.push_message(assistant_message).unwrap();
                        prompt.push_message(user_message).unwrap();
                    } else {
                        log::debug!(
                            "After removing thought, the assistant message is empty. This is not allowed, so we'll merge the user message onto the previous message."
                        );
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
                        log::debug!(
                            "Merge complete. User message: {:?}",
                            serde_json::to_string_pretty(prompt.messages.last().unwrap()).unwrap()
                        )
                    }
                } else if let Err(e) = prompt.push_message(user_message) {
                    // The assistant did not get a chance to even start
                    // responding. This is posisble but unlikely. The client
                    // will have to handle this case.
                    log::error!(
                        "User possibly responded too fast. Error pushing user message to prompt: {e}",
                    );

                    let response = model::response::Response::Err(
                        model::response::Error::TurnOrder {
                            message: e.to_string(),
                        },
                    );
                    yield Event::default()
                        .event("response")
                        .json_data(response)
                        .unwrap();
                    break;
                }
            }

            if prompt.messages.last().is_none_or(|m| m.role == Role::Assistant) {
                log::debug!("User's turn to respond");

                // If the last message is a user message, we don't want to await
                // a new message from the user just yet, because we must
                // maintain the turn order. It would be the assistant's turn to
                // respond. We'll wait for the assistant to respond first before
                // taking from the user message channel again.
                let user_message = match from_user.recv().await.unwrap() {
                    Request::GetPrompt => {
                        log::info!(
                            "GetPrompt {}",
                            serde_json::to_string_pretty(prompt.deref()).unwrap()
                        );
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
                        log::info!("SetPrompt: {}", serde_json::to_string_pretty(&new).unwrap());
                        *prompt = new;
                        continue;
                    }
                    Request::UserMessage(user_message) => {
                        log::info!("UserMessage: {}", serde_json::to_string_pretty(&user_message).unwrap());
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
                // We have a user message and the prompt is ready for it.

                // This can't panic because we just checked that the last
                // message was a user message or that there were no messages in
                // which case the push is allowed.
                log::debug!("Pushing user message");
                prompt.push_message(user_message).unwrap();
            }
            // Final message is a user message. It is the Agent's turn to
            // respond. We have guaranteed that the last message in the prompt
            // is a user message because there are only two roles.

            // Get a streaming response from the Anthropic AI. This will include
            // full messages and tool use events.
            let stream = match state.client.stream(prompt.deref()).await {
                Ok(stream) => stream
                    // Adds a full message event, `Event::Message` and tool use.
                    .with_message_ip(&mut assistant_message),
                Err(e) => {
                    log::error!("Error getting stream from Anthropic: {e}");
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

                    // Don't break: breaking ends the stream, the client
                    // reconnects, sends GetPrompt, and the same rejected prompt
                    // (still the last user turn) is re-streamed immediately,
                    // hammering the API. Instead back off and retry the same
                    // prompt in place. A new message from the client cancels the
                    // wait so the user can recover (e.g. edit the prompt via
                    // SetPrompt) instead of being stuck behind the backoff.
                    //
                    // TODO: the client should grow a modal so the user can edit
                    // or drop the offending prompt rather than only waiting.
                    tokio::select! {
                        _ = tokio::time::sleep(backoff) => {
                            backoff = (backoff * 2).min(MAX_BACKOFF);
                        }
                        // `recv` only returns `None` if the channel closes,
                        // which can't happen (AppState owns the sender).
                        request = from_user.recv() => match request.unwrap() {
                            Request::GetPrompt => {
                                yield Event::default()
                                    .event("response")
                                    .json_data(json!({
                                        "Ok": { "prompt": prompt.deref() },
                                    }))
                                    .unwrap();
                            }
                            Request::SetPrompt(new) => {
                                // User likely fixed the prompt; retry it now.
                                *prompt = new;
                                backoff = std::time::Duration::from_secs(1);
                            }
                            Request::UserMessage(user_message) => {
                                yield Event::default()
                                    .event("response")
                                    .json_data(json!({
                                        "Ok": { "user_message": &user_message },
                                    }))
                                    .unwrap();
                                interrupt_message = Some(user_message);
                            }
                        },
                    }
                    continue;
                }
            };
            log::info!("Got Stream from Anthropic");
            // Successfully reached Anthropic; reset the rejection backoff.
            backoff = std::time::Duration::from_secs(1);

            pin_mut!(stream);

            while let Some(event) = stream.next().await {
                log::debug!("Event: {}", serde_json::to_string(&event).unwrap());
                // Listen for an interrupt signal from the user. We could join
                // this with `stream.next()` but it's easier to understand this
                // way. Very small latency difference since we must wait for the
                // next event but there are many of these per second.
                if let Ok(request) = from_user.try_recv() {
                    // Handle user interrupt.
                    let user_message = match request {
                        Request::GetPrompt => {
                            log::info!(
                                "GetPrompt: {}",
                                serde_json::to_string_pretty(prompt.deref()).unwrap()
                            );
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
                            log::info!("SetPrompt: {}", serde_json::to_string_pretty(&new).unwrap());
                            *prompt = new;
                            break;
                        }
                        Request::UserMessage(user_message) => {
                            log::info!("UserMessage: {}", serde_json::to_string_pretty(&user_message).unwrap());
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

                        // `response` was just built as `Ok(Stream(event))`; we
                        // round-trip through it to get the moved `event` back
                        // after serializing, so the literal unwrap is deliberate.
                        #[allow(clippy::unnecessary_literal_unwrap)]
                        let event = response.unwrap().unwrap_stream();

                        // The client rebuilds its own prompt from the raw stream
                        // events; ours only needs *complete* messages for the
                        // next request. Push the synthesized `Event::Message`
                        // (the whole assistant turn) as a typed
                        // `AssistantMessage` rather than rebuilding it
                        // incrementally with `handle_stream_event`. The latter
                        // tracked the turn here *and* in `assistant_message`, so
                        // a tool result racing in before `MessageStop` pushed the
                        // assistant twice (a `TurnOrderError`). On interrupt
                        // (break before `MessageStop`) the partial turn is pushed
                        // from `assistant_message` instead — and since
                        // `with_message_ip` `take`s the accumulator exactly at
                        // `MessageStop`, only one of the two paths ever fires.
                        if let misanthropic::stream::Event::Message { message } =
                            event
                        {
                            let assistant = AssistantMessage::from(message);
                            if let Err(e) = prompt.push_message(assistant) {
                                log::error!(
                                    "Turn order error pushing assistant message: {e}"
                                );
                            }
                        }
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
    log::debug!("Creating router");
    let client = misanthropic::Client::new(
        secrets
            .get("ANTHROPIC_API_KEY")
            .expect("ANTHROPIC_API_KEY must be set in a Secrets.toml file."),
    )
    .unwrap();
    let state = AppState::from(client);

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
