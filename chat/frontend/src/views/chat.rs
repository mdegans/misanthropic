use std::ops::Deref;

use dioxus::{html::HasFileData, prelude::*};

use misanthropic::{
    dioxus::{
        opts::{self, HeadingLevel},
        IntoElement, Options,
    },
    exports::{
        base64::{self, engine::GeneralPurpose, Engine},
        futures::StreamExt,
    },
    prompt::{
        message::{Block, Image, MediaType, UserMessage},
        Prompt,
    },
};
use model::{request::Request, response::Success};

use crate::utils::sleep_ms;

const CSS: Asset = asset!("/assets/styling/chat.css");
/// The client should be a global resource. It will be shared across all views.
/// and it's the easiest way to manage things like rate limiting and connection
/// pooling.
static CLIENT: GlobalSignal<crate::client::Client> =
    GlobalSignal::new(crate::client::Client::new);
const BASE64: GeneralPurpose = base64::engine::general_purpose::STANDARD;

/// A test prompt for testing the chat view.
#[cfg(debug_assertions)]
fn make_prompt() -> Prompt<'static> {
    use misanthropic::{
        json,
        prompt::{
            message::{Block, Content, Role},
            Message,
        },
        tool, AnthropicModel, Spec,
    };
    use AnthropicModel::*;

    Prompt::default()
        .model(Sonnet35)
        .add_tool(Spec {
            name: "python".into(),
            description: "Run a Python script.".into(),
            schema: json!({
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
                    // Regular, plain old, legacy thinking block. When displayed
                    // with `ThoughtsOrSpeech`, it will be styled as a thought.
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
                    // Anthropic provided `Thought` blocks should have the same
                    // exact styling as the Assistant's thoughts. So now "old"
                    // models have feature parity with the new ones, at least
                    // visually. However it is possible to *not* use Anthropic's
                    // `Thought` blocks even with new models, writing your own
                    // system prompt, giving your own `<thinking>` instructions.
                    //
                    // There may or may not be a performance hit for this,
                    // depending on your prompt and application. The option is
                    // here for flexibility.
                    Block::Thought { thought: "This request is complex enough to need Python. I should use the itertools module for this...".into(), signature: "...".into() },
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

#[cfg(not(debug_assertions))]
fn make_prompt() -> Prompt<'static> {
    // The server will send us the prompt. This is just a placeholder.
    Prompt::default()
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
    let mut show_thought = use_signal(|| false);
    let mut show_tool_use = use_signal(|| false);
    let mut show_system = use_signal(|| false);
    let mut dragged_over = use_signal(|| true);
    let mut dragged_file_supported = use_signal(|| false);
    let mut attachments = use_signal(|| Vec::<Block>::new());
    let mut ready_json = use_signal(|| None);

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

        // Stream connected. Get the prompt.
        if let Err(e) = CLIENT.read().send(Request::GetPrompt).await {
            connected.set(false);
            log::error!("Failed to request prompt because: {}", e);
        };

        // Stream events until disconnected.
        while let Some(event) = stream.next().await {
            match event {
                Ok(event) => {
                    log::debug!("Event: {:?}", event);

                    match event {
                        Ok(Success::Stream(event)) => {
                            // The most common event is a stream event forwarded
                            // from Anthropic. This is where the magic happens.
                            if let Err(e) =
                                // A Prompt and Vec<Message> both implement
                                // `HandleStreamEvent`. In a real app, the
                                // latter is likely more useful on the client
                                // side.
                                //
                                // The trait can also be implemented for custom
                                // types, like a wrapper for a `Vec<Message>`
                                // with class invariant checks.
                                prompt.write().handle_stream_event(event)
                            {
                                log::error!(
                                    "Failed to handle stream event: {}",
                                    e
                                );
                            }
                        }
                        Ok(Success::Prompt(new)) => {
                            // This handles desyncs and initial state.
                            prompt.set(new);
                        }
                        Ok(Success::UserMessage(message)) => {
                            // This is a message from the user. It should be
                            // displayed in the chat.
                            if let Err(e) = prompt.write().push_message(message)
                            {
                                log::error!("Failed to push message: {}", e);
                                // no clue how we got here but do know how to
                                // fix it
                                if let Err(ce) =
                                    CLIENT.read().send(Request::GetPrompt).await
                                {
                                    connected.set(false);
                                    log::error!(
                                        "Failed to get prompt after error `{e}` because: {}",
                                        ce
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("{e}");
                        }
                    }
                }
                // This is a problem with the stream itself. Can't reach the
                // backend, etc.
                Err(e) => log::error!("Error: {}", e),
            };
        }

        // If the infinite stream ends, we're disconnected.
        connected.set(false);
    });

    let mut input_buffer = use_signal(|| String::new());

    if !*connected.read() {
        return rsx! {
            document::Link { rel: "stylesheet", href: CSS }

            div {
                class: "chat connecting",
                // Do our little indicator dance.
                {format!(
                    "Connecting{}",
                    std::iter::repeat('.').take(*failures.read() % 4).collect::<String>(),
                )}
            }
        };
    }

    rsx! {
        document::Stylesheet { href: CSS }

        div {
            class: "chat",
            // Renders the entire prompt. This is the main view of the chat.
            {prompt.read().into_element_custom(1337, &options.read())}
        }

        div {
            class: "input",
            form {
                textarea {
                    class: "input-box",
                    class: if *dragged_over.read() {
                        "dragged-over"
                    } else { "" },
                    class: if *dragged_file_supported.read() {
                        "dragged-file-supported"
                    } else { "" },
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
                            let content = input_buffer.read().to_string();
                            if !content.trim().is_empty() {
                                let mut message = UserMessage::from(content);
                                message.content_mut().extend(
                                    attachments.write().drain(..)
                                );
                                if let Err(e) = CLIENT.read().send(message).await {
                                    connected.set(false);
                                    log::error!(
                                        "Failed to send message to backend: {}",
                                        e
                                    );
                                } else {
                                    // Sucessfully sent the message.
                                    input_buffer.write().clear();
                                }
                            }
                        }
                    },
                    onkeyup: move |e| {
                        if e.key() == Key::Shift {
                            shift_held.set(false);
                        }
                    },
                    ondragover: move |e| {
                        e.prevent_default();
                        dragged_over.set(true);
                        if let Some(files) = e.files() {
                            let filenames = files.files();
                            if filenames.len() != 1 {
                                dragged_file_supported.set(false);
                                return;
                            }
                            let filename = &filenames[0];
                            if MediaType::is_supported(filename) {
                                dragged_file_supported.set(true);
                            } else {
                                dragged_file_supported.set(false);
                            }
                        }
                    },
                    ondragleave: move |e| {
                        e.prevent_default();
                        dragged_over.set(false);
                        dragged_file_supported.set(false);
                    },
                    ondrop: move |e| async move {
                        e.prevent_default();
                        dragged_over.set(false);
                        dragged_file_supported.set(false);
                        if let Some(files) = e.files() {
                            let filenames = files.files();
                            if filenames.len() == 0 {
                                log::warn!("No files dropped.");
                                return;
                            } else if filenames.len() > 1 {
                                log::warn!("Only one file can be dropped at a time.");
                                return;
                            }

                            // len is 1
                            let filename = &filenames[0];
                            let format = if let Some(format) =  MediaType::detect(&filename) {
                                format
                            } else {
                                log::warn!("Unsupported file type.");
                                return;
                            };

                            let data = if let Some(data) = files.read_file(filename).await {
                                data
                            } else {
                                log::warn!("Failed to read file.");
                                return;
                            };

                            if data.is_empty() {
                                log::warn!("Empty file.");
                                return;
                            }
                            // We have a file data with a supported format. Load
                            // it and push it to the attachments.

                            let image = Image::from_compressed(format, data);
                            attachments.write().push(image.into());
                        }
                    }
                }
            }
        }

        div {
            class: "attachments",
            {attachments.read().iter().map(|block| {
                block.into_element_custom(1338, &options.read())
            })}
        }

        div {
            button {
                class: "toggle",
                class: if *show_thought.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *show_thought.read();
                    *show_thought.write() = !val;

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
                class: if *show_tool_use.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *show_tool_use.read();
                    *show_tool_use.write() = !val;

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
                class: if *show_system.read() { "active" } else { "" },
                onmousedown: move |e| {
                    e.prevent_default();
                    let val = *show_system.read();
                    *show_system.write() = !val;

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
            // This is kind of hacky, but it's actually the cleanest way to
            // do this without incomprehensible web_sys code.
            button {
                class: "toggle",
                class: "save",
                class: if ready_json.read().is_some() { "ready" } else { "" },
                onmouseover: move |e| {
                    e.prevent_default();
                    let val = serde_wasm_bindgen::to_value(
                        prompt.read().deref()
                    ).unwrap();

                    let json = js_sys::JSON::stringify(&val).unwrap();
                    ready_json.write().replace(String::from(json));
                },
                onmouseleave: move |e| {
                    e.prevent_default();
                    ready_json.write().take();
                },
                a {
                    href: if let Some(json) = ready_json.read().deref() {
                        Some(format!(
                            "data:application/json;base64,{}",
                            BASE64.encode(json.as_bytes())
                        ))
                    } else {
                        None
                    },
                    download: "prompt.json",
                    "Download"
                }
            }
            // button {
            //     class: "toggle",
            //     class: "load",

            // }
        }
    }
}

// <a download="prompt.json" href="data:application/json;base64,">Download</a>
