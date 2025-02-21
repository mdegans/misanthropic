use dioxus::prelude::*;

use misanthropic::{exports::futures::StreamExt, html::ToHtml, prompt::Prompt};

const BLOG_CSS: Asset = asset!("/assets/styling/blog.css");

#[component]
pub fn Chat() -> Element {
    // launch an async task to pull state from the server's event stream
    let mut prompt = use_signal(Prompt::default);
    let _stream_task = use_resource(move || async move {
        let client = crate::client::Client::new();
        let mut stream = match client.stream().await {
            Ok(stream) => stream,
            Err(e) => {
                log::error!("Failed to connect to server: {}", e);
                return;
            }
        };

        while let Some(event) = stream.next().await {
            use crate::client;
            match event {
                Ok(event) => match event {
                    client::Event::Stream(event) => {
                        // The most common event is a stream event forwarded
                        // from Anthropic. This is where the magic happens.
                        if let Err(e) =
                            prompt.write().handle_stream_event(event)
                        {
                            log::error!("Failed to handle stream event: {}", e);
                        }
                    }
                    client::Event::Prompt(new_prompt) => {
                        // Update the prompt with the new prompt. This should
                        // match the one assembled by the stream. This handles
                        // desyncs and initial state.
                        prompt.set(new_prompt);
                    }
                    client::Event::StreamError(e) => {
                        // This indicates a problem with the stream. Normally
                        // this should not reach the user.
                        log::error!("Stream error: {}", e);
                    }
                    client::Event::MisanthropicClientError(e) => {
                        // This indicates a problem with the Anthropic client on
                        // the server side. Normally this should not reach the
                        // user.
                        log::error!("Misanthropic client error: {}", e);
                    }
                    client::Event::TurnOrderError(e) => {
                        // This should not happen. It indicates a logic error.
                        // We could panic but it's better to log and continue.
                        log::error!("Turn order error: {}", e);
                    }
                },
                // This is a problem with the stream itself. Can't reach the
                // backend, etc.
                Err(e) => log::error!("Error: {}", e),
            };
        }
    });

    rsx! {
        document::Link { rel: "stylesheet", href: BLOG_CSS }

        div {
            h1 { "Chat" }
            div { class: "chat",
                for message in prompt.read().messages.iter() {
                    // Santization safety: `misanthropic` escapes HTML in
                    // messages properly, so we can trust the HTML here.
                    div { class: "message", role: message.role.as_str(), {message.html_verbose()} }
                }
            }
        }
    }
}
