use dioxus::prelude::*;

use misanthropic::{
    dioxus::{
        opts::{self, HeadingLevel},
        IntoElement, Options,
    },
    exports::futures::StreamExt,
    prompt::Prompt,
};

use crate::utils::sleep_ms;

const CSS: Asset = asset!("/assets/styling/chat.css");
/// The client should be a global resource. It will be shared across all views.
/// and it's the easiest way to manage things like rate limiting and connection
/// pooling.
static CLIENT: GlobalSignal<crate::client::Client> =
    GlobalSignal::new(crate::client::Client::new);
/// Global flags to toggle the visibility of different types of messages.
static SHOW_THOUGHT: GlobalSignal<bool> = GlobalSignal::new(|| false);
static SHOW_TOOL_USE: GlobalSignal<bool> = GlobalSignal::new(|| false);
static SHOW_SYSTEM: GlobalSignal<bool> = GlobalSignal::new(|| false);

#[cfg(debug_assertions)]
fn make_prompt() -> Prompt<'static> {
    use misanthropic::{
        json,
        prompt::{
            message::{Content, Role},
            Message,
        },
        tool, AnthropicModel, Tool,
    };
    use AnthropicModel::*;

    Prompt::default()
        .model(Sonnet35)
        .add_tool(Tool {
            name: "python".into(),
            description: "Run a Python script.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Python script to run.",
                    },
                },
                "required": ["script"],
            }),
            cache_control: None,
        })
        // Inform the assistant about their limitations.
        .set_system(include_str!("../../../../misanthropic/examples/python_system.md"))
        .add_system(format!("## Python Environment\n\n{}", "3.12"))
        // The example has some examples of the Assistant using Python and some
        // without to help guide the assistant to use Python when necessary and
        // not when it isn't. The more examples here, with more varied prompts,
        // the better the Assistant will be at this.
        .set_messages([
            Message {
                role: Role::User,
                content: "Write a haiku about Python.".into(),
            },
            Message {
                role: Role::Assistant,
                content: "Elegant syntax\rPowerful and versatile\nPython, my delight.".into(),
            },
            Message {
                role: Role::User,
                content: "Count the number of r's in 'strawberry'".into(),
            },
            Message {
                role: Role::Assistant,
                content: Content::MultiPart(vec![
                    r#"<thinking>I can't do that myself, but I can run a Python script to count the number of r's in "strawberry". The user did not specify case sensitivity so I will default to case insensitive.</thinking>"#.into(),
                    tool::Use {
                        id: "calibration_000".into(),
                        name: "python".into(),
                        input: json!({
                            "script": r#"print("strawberry".lower().count("r"))"#
                        }),
                        cache_control: None
                    }.into()
                ]),
            },
            tool::Result {
                tool_use_id: "calibration_000".into(),
                content: "3".into(),
                is_error: false,
                cache_control: None,
            }.into(),
            (Role::Assistant, r#"The number of r's in "strawberry" is 3.""#).into(),
            (Role::User, "List the permutations of the first four letters of the alphabet.").into(),
            Message {
                role: Role::Assistant,
                content: Content::MultiPart(vec![
                    r#"<thinking>This request is complex enough to need Python. I should use the itertools module for this..</thinking>"#.into(),
                    tool::Use {
                        id: "calibration_001".into(),
                        name: "python".into(),
                        input: json!({
                            "script": r#"import itertools\nprint(','.join("".join(t) for t in itertools.permutations(('a', 'b', 'c', 'd'))))"#
                        }),
                        cache_control: None
                    }.into()
                ]),
            },
            tool::Result {
                tool_use_id: "calibration_001".into(),
                content: "abcd,abdc,acbd,acdb,adbc,adcb,bacd,badc,bcad,bcda,bdac,bdca,cabd,cadb,cbad,cbda,cdab,cdba,dabc,dacb,dbac,dbca,dcab,dcba".into(),
                is_error: false,
                cache_control: None
            }.into(),
            (Role::Assistant, "The permutations of the first four letters of the alphabet are:\n\nabcd, abdc, acbd, acdb, adbc, adcb, bacd, badc, bcad, bcda, bdac, bdca, cabd, cadb, cbad, cbda, cdab, cdba, dabc, dacb, dbac, dbca, dcab, dcba.").into(),
            (Role::User, "What is the capital of France?").into(),
            (Role::Assistant, "Paris.").into(),
            (Role::User, "Thanks for all your help. I have to go now.").into(),
            (Role::Assistant, "You're welcome. Have a great day!<narrator>A new user enters the chat</narrator>").into(),
        ]).unwrap()
        // Insert cache breakpoint. It won't do anything in this example, but if
        // the system prompt and examples are very long, it can be useful to
        // cache everything up to the user input.
        .cache()
}

#[component]
pub fn Chat() -> Element {
    // Our prompt is a shared resource. When it is written to, it will update
    // the view. As streaming events come in, they will update the prompt.
    let mut prompt = use_signal(make_prompt);
    let mut connected = use_signal(|| false);
    let mut failures = use_signal(|| 0);
    let mut shift_held = use_signal(|| false);
    let mut options = use_signal(Options::default);
    // Our long-running task is the stream task. It will run until the
    // component is dropped, connecting with the
    let _stream_task = use_resource(move || async move {
        let mut stream = loop {
            match CLIENT.read().stream().await {
                Ok(stream) => {
                    connected.set(true);
                    break stream;
                }
                Err(e) => {
                    log::error!("Failed to connect to stream: {}", e);
                    *failures.write() += 1;
                    // Wait a second before trying again.
                    sleep_ms(1000).await;
                }
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

        // If the infinite stream ends, we're disconnected.
        connected.set(false);
    });

    let mut input_buffer = use_signal(|| String::new());

    // if !*connected.read() {
    //     return rsx! {
    //         document::Link { rel: "stylesheet", href: CSS }

    //         div {
    //             h1 { "Chat" }
    //             div { class: "chat",
    //                 div { class: "message", role: "system",
    //                     {
    //                         // Do our little indicator dance.
    //                         format!(
    //                             "Connecting{}",
    //                             std::iter::repeat('.').take(*failures.read() % 4).collect::<String>(),
    //                         )
    //                     }
    //                 }
    //             }
    //         }
    //     };
    // }

    rsx! {
        document::Stylesheet { href: CSS }

        div {
            class: "chat",
            {prompt.read().into_element_custom(1337, &options.read())}
        }

        div {
            class: "input",
            form {
                onsubmit: move |e| async move {
                    if let Err(e) = CLIENT.read().send(e.value()).await {
                        log::error!("Failed to send message: {}", e);
                    } else {
                        input_buffer.write().clear();
                    }
                },
                textarea {
                    class: "input-box",
                    placeholder: "Type your message...",
                    autofocus: true,
                    value: "{input_buffer}",
                    oninput: move |e| {
                        input_buffer.set(e.value());
                    },
                    onkeydown: move |e| async move {
                        if e.key() == Key::Shift {
                            shift_held.set(true);
                            return;
                        }

                        if e.key() == Key::Enter && !*shift_held.read() {
                            e.prevent_default();
                            let message = input_buffer.read().to_string();
                            if !message.trim().is_empty() {
                                if let Err(e) = CLIENT.read().send(message).await {
                                    connected.set(false);
                                    log::error!(
                                        "Failed to send message to backend: {}",
                                        e
                                    );
                                } else {
                                    input_buffer.write().clear();
                                }
                            }
                        }
                    },
                    onkeyup: move |e| {
                        if e.key() == Key::Shift {
                            shift_held.set(false);
                        }
                    }
                }
            }
        }

        div {
            button {
                class: "toggle",
                class: if *SHOW_THOUGHT.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *SHOW_THOUGHT.read();
                    *SHOW_THOUGHT.write() = !val;

                    if !val {
                        options.write().thought = opts::Thought::Show {
                            class: "thought show".into()
                        }
                    } else {
                        options.write().thought = opts::Thought::Placeholder {
                            class: "thought placeholder".into()
                        };
                    }
                },
                "Thoughts"
            }
            button {
                class: "toggle",
                class: if *SHOW_TOOL_USE.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *SHOW_TOOL_USE.read();
                    *SHOW_TOOL_USE.write() = !val;

                    if !val {
                        options.write().tool_use = opts::ToolUse::Show {
                            show_name: Some(HeadingLevel::H3),
                            class: "tool-use".into()
                        };
                        options.write().tool_result = opts::ToolResult::Show {
                            error: "tool-result error".into(),
                            ok: "tool-result ok".into()
                        };
                    } else {
                        options.write().tool_use = opts::ToolUse::Hidden;
                        options.write().tool_result = opts::ToolResult::Hidden;
                    }
                },
                "Tool Use"
            }
            button {
                class: "toggle",
                class: if *SHOW_SYSTEM.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *SHOW_SYSTEM.read();
                    *SHOW_SYSTEM.write() = !val;

                    if !val {
                        options.write().system = opts::System::Show {
                            class: "system".into()
                        }
                    } else {
                        options.write().system = opts::System::Hidden;
                    }
                },
                "System"
            }
        }
    }
}
