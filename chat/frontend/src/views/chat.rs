use std::ops::Deref;

use dioxus::{
    html::{mo, HasFileData},
    prelude::*,
};

use dioxus_sdk::storage::use_persistent;
use misanthropic::{
    dioxus::{
        opts::{self, HeadingLevel},
        IntoElement, Options,
    },
    exports::{
        base64::{self, engine::GeneralPurpose, Engine},
        futures::StreamExt,
        serde_json,
    },
    json,
    prompt::{
        message::{Block, Image, MediaType, UserMessage},
        Prompt,
    },
    tool::Tool,
    Spec,
};
use model::{request::Request, response::Success, toolbox};
use wasm_bindgen::{prelude::Closure, JsCast};

use crate::utils::sleep_ms;

const CSS: Asset = asset!("/assets/styling/chat.css");
/// The client should be a global resource. It will be shared across all views.
/// and it's the easiest way to manage things like rate limiting and connection
/// pooling.
static CLIENT: GlobalSignal<crate::client::Client> =
    GlobalSignal::new(crate::client::Client::new);
const BASE64: GeneralPurpose = base64::engine::general_purpose::STANDARD;
static DEFAULT_DRAG_CLOSURE: GlobalSignal<
    Closure<dyn FnMut(web_sys::DragEvent)>,
> = GlobalSignal::new(|| {
    Closure::wrap(Box::new(|e: web_sys::DragEvent| e.prevent_default())
        as Box<dyn FnMut(_)>)
});

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
    // Our signals. This is reactive state management. When these signals are
    // updated, the component will re-render. Signals are a Copy type, and use
    // automatic dependency tracking to determine when to re-render.
    let mut attachments = use_signal(|| Vec::<Block>::new());
    let mut connected = use_signal(|| false);
    let mut dragged_file_supported = use_signal(|| false);
    let mut dragged_over = use_signal(|| true);
    let mut failures = use_signal(|| 0);
    let mut options = use_signal(Options::default);
    let mut prompt = use_signal(make_prompt);
    let mut ready_json = use_signal(|| None);
    let mut shift_held = use_signal(|| false);
    let mut show_system = use_signal(|| false);
    let mut show_thought = use_signal(|| false);
    let mut show_tool_use = use_signal(|| false);
    let specs = use_signal(|| {
        let specs: Vec<Spec> = toolbox::create().specs().collect();
        specs
    });
    let mut toolbox_state =
        use_persistent("toolbox-state", || serde_json::Value::Null);
    let mut toolbox = use_signal(|| {
        let mut toolbox = toolbox::create();
        log::info!("Loading toolbox state.");
        if let Err(e) = toolbox.load_json(toolbox_state.peek().clone()) {
            log::error!("`Toolbox::load_json` had error(s): {e}");
        } else {
            log::info!("Toolbox state loaded.")
        }

        toolbox
    });

    // Suppress the default drag and drop behavior.
    use_effect(|| {
        // o3-mini's suggestion, thank you! I could not figure out why on drop a
        // photo would open in a new tab. This is a solution. Instead of trying
        // to track down the problem, we just globally suppress the default
        // behavior.
        let window = web_sys::window().expect("no global `window` exists");
        window
            .add_event_listener_with_callback(
                "dragover",
                DEFAULT_DRAG_CLOSURE.read().as_ref().unchecked_ref(),
            )
            .unwrap();
        window
            .add_event_listener_with_callback(
                "drop",
                DEFAULT_DRAG_CLOSURE.read().as_ref().unchecked_ref(),
            )
            .unwrap();
    });

    // Our long-running task is the stream task. It will run until the
    // component is dropped, connecting with the
    let _stream_task = use_resource(move || async move {
        loop {
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
                log::debug!("Event: {:?}", event);
                match event {
                    Ok(event) => {
                        match event {
                            // The most common event is a stream event,
                            // forwarded from Anthropic. We handle tool use on
                            // the client side.
                            Ok(Success::Stream(event)) => {
                                // If the event is a tool use event, we handle
                                // things differently. The tools run on the
                                // client side.
                                if let misanthropic::stream::Event::ToolUse {
                                    tool_use,
                                } = &event
                                {
                                    log::info!("Tool use: {:?}", tool_use);
                                    // A tool has been used.
                                    let result =
                                        toolbox.write().call(tool_use.clone());
                                    toolbox_state
                                        .set(toolbox.peek().save_json());
                                    log::info!("Tool result: {:?}", result);

                                    // We send the result back to the server.
                                    if let Err(e) = CLIENT
                                        .read()
                                        .send(Request::UserMessage(
                                            result.clone().into(),
                                        ))
                                        .await
                                    {
                                        connected.set(false);
                                        log::error!(
                                            "Failed to set prompt after tool use: {}",
                                            e
                                        );
                                    } else {
                                        // Sucessfully sent. The server will
                                        // send it back in a UserMessage.
                                    }
                                }

                                if let Err(e) =
                                    // A Prompt and Vec<Message> both implement
                                    // `HandleStreamEvent`. In a real app, the
                                    // latter is likely more useful on the
                                    // client side.
                                    //
                                    // The trait can also be implemented for
                                    // custom types, like a wrapper for a
                                    // `Vec<Message>` with class invariant
                                    // checks.
                                    prompt
                                        .write()
                                        .handle_stream_event(event)
                                {
                                    log::error!(
                                        "Failed to handle stream event: {}",
                                        e
                                    );
                                }
                            }
                            Ok(Success::Prompt(mut new)) => {
                                // Update tools.
                                new.tools = Some(specs.read().clone());

                                // Update the prompt with the tools.
                                if let Err(e) = toolbox.peek().setup(&mut new) {
                                    log::error!(
                                        "`Toolbox::setup` had error(s): {e}"
                                    );
                                } else {
                                    // Send it back to the server.

                                    log::info!("Toolbox setup sucessful.")
                                }

                                // We updated tools and applied state to the
                                // prompt, so we need to update the server.
                                // Don't let the client set the system prompt in
                                // production. It's a security risk.
                                log::info!("Sending ToolBox to backend.");
                                if let Err(e) = CLIENT
                                    .read()
                                    .send(Request::SetPrompt(new.clone()))
                                    .await
                                {
                                    connected.set(false);
                                    log::error!(
                                        "Failed to update Toolbox on backend: {}",
                                        e
                                    );
                                    continue; // do not set the prompt
                                }

                                prompt.set(new.clone());
                            }
                            Ok(Success::UserMessage(message)) => {
                                // This is a message from the user. It should be
                                // displayed in the chat.
                                if let Err(e) =
                                    prompt.write().push_message(message)
                                {
                                    log::error!(
                                        "Failed to push message: {}",
                                        e
                                    );
                                    // no clue how we got here but do know how to
                                    // fix it
                                    if let Err(ce) = CLIENT
                                        .read()
                                        .send(Request::GetPrompt)
                                        .await
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
                                sleep_ms(1000).await;
                            }
                        }
                    }
                    // This is a problem with the stream itself. Can't reach the
                    // backend, Anthropic rejected the prompt, etc.
                    Err(e) => {
                        log::error!("Error: {}", e);
                    }
                };
            }

            // If the infinite stream ends, we're disconnected.
            connected.set(false);
        }
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
                    placeholder: "Type your message or drag a .json file here to load a chat...\n\n...You can also drag images here to attach them.",
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
                            if MediaType::is_supported(filename) || filename.ends_with(".json") {
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
                        e.stop_propagation();
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

                            // Is the file a JSON file? We need to load it, set
                            // the tools, and send the updated prompt to the
                            // backend.
                            if filename.ends_with(".json") {
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

                                let mut json: serde_json::Value = match serde_json::from_slice(&data) {
                                    Ok(prompt) => prompt,
                                    Err(e) => {
                                        log::warn!("Failed to deserialize JSON: {}", e);
                                        return;
                                    }
                                };

                                // Load any prompt from the JSON file.
                                if let serde_json::Value::Object(new_prompt) = json["prompt"].take() {
                                    log::info!("Loading new prompt.");
                                    let mut new_prompt: Prompt = match serde_json::from_value(
                                        new_prompt.into()
                                    ) {
                                        Ok(prompt) => prompt,
                                        Err(e) => {
                                            log::warn!("Failed to deserialize prompt: {}", e);
                                            return;
                                        }
                                    };

                                    // Tool use has a 1:1 relationship with the
                                    // app's current capabilities, not any given
                                    // prompt or backend so it wouldn't make
                                    // sense for the user to be able to specify
                                    // tools that don't exist, and may have
                                    // changed since the original prompt was
                                    // created. This is always overwritten.
                                    new_prompt.tools.replace(specs.read().clone());

                                    // Update tools.
                                    new_prompt.tools = Some(specs.read().clone());

                                    // Update the prompt with the tools.
                                    if let Err(e) = toolbox.peek().setup(&mut new_prompt) {
                                        log::error!(
                                            "`Toolbox::setup` had error(s): {e}"
                                        );
                                    } else {
                                        // Send it back to the server.

                                        log::info!("Toolbox setup sucessful.")
                                    }

                                    match CLIENT.read().send(
                                        Request::SetPrompt(new_prompt.clone())
                                    ).await {
                                        Ok(_) => {
                                            prompt.set(new_prompt);
                                        }
                                        Err(e) => {
                                            log::error!("Failed to set prompt: {}", e);
                                        }
                                    }
                                }

                                // Load any tool state from the JSON file.
                                if let serde_json::Value::Object(tool_state) = json["tool_state"].take() {
                                    log::info!("Loading tool state.");
                                    if let Err(e) = toolbox.write().load_json(tool_state.into()) {
                                        log::warn!("Failed to load tool state: {}", e);
                                    } else {
                                        log::info!("Tool state loaded.")
                                    }
                                    toolbox_state.set(toolbox.peek().save_json());
                                }

                                return;
                            }


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
                // Serialize on hover.
                onmouseover: move |e| {
                    e.prevent_default();
                    let json = serde_json::to_string_pretty(&json!({
                        "prompt": prompt.read().deref(),
                        "tool_state": toolbox.read().save_json(),
                    })).unwrap();
                    ready_json.write().replace(json);
                },
                onmouseleave: move |e| {
                    e.prevent_default();
                    ready_json.write().take();
                },
                // Download the JSON.
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
                    "Save"
                }
            }
        }
    }
}

// <a download="prompt.json" href="data:application/json;base64,">Download</a>
