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
        serde_json,
    },
    prompt::{
        message::{Block, DocumentSource, Image, MediaType, UserMessage},
        Prompt,
    },
    tool::Tool,
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
/// `localStorage` key under which the toolbox state is persisted across
/// sessions. `localStorage` (not `sessionStorage`): the notepad's whole point
/// is recalling notes from *other* sessions, so state must survive a tab or
/// browser close and be shared across tabs.
const TOOLBOX_STATE_KEY: &str = "toolbox-state";
static DEFAULT_DRAG_CLOSURE: GlobalSignal<
    Closure<dyn FnMut(web_sys::DragEvent)>,
> = GlobalSignal::new(|| {
    Closure::wrap(Box::new(|e: web_sys::DragEvent| e.prevent_default())
        as Box<dyn FnMut(_)>)
});

/// A dropped `.json` file: either a saved conversation or saved tool state.
///
/// Distinguished by shape via serde `untagged` (the same way we sniff image
/// types) since Save writes the two as separate files. The shapes are mutually
/// exclusive: a tool-state file fails the `Prompt` arm because its `tools` is a
/// map rather than the array `Prompt::tools` (serialized as `tools`) expects,
/// and a prompt fails the `ToolState` arm because it has no top-level `name`.
#[derive(serde::Deserialize)]
#[serde(untagged)]
enum Dropped {
    /// A saved conversation. Tried first as the common case.
    Prompt(Box<Prompt>),
    /// Saved toolbox state, mirroring [`ToolBox::save_json`]'s `{name, tools}`.
    ToolState(ToolStateFile),
}

/// Mirror of the toolbox's `save_json` shape, used only to sniff dropped files.
#[derive(serde::Deserialize, serde::Serialize)]
struct ToolStateFile {
    name: String,
    tools: serde_json::Map<String, serde_json::Value>,
}

/// Whether a toolbox `save_json` value carries no actual state — every tool
/// serialized to `null`, `[]`, or `{}`.
///
/// An empty toolbox is perfectly valid in general (so the library doesn't
/// forbid it), but the demo must never *persist* one over existing browser
/// storage: a transient empty (e.g. a save firing before notes are restored)
/// would otherwise wipe real notes that survive across sessions.
fn is_empty_tool_state(state: &serde_json::Value) -> bool {
    state
        .get("tools")
        .and_then(serde_json::Value::as_object)
        .map(|tools| {
            tools.values().all(|v| match v {
                serde_json::Value::Null => true,
                serde_json::Value::Array(a) => a.is_empty(),
                serde_json::Value::Object(o) => o.is_empty(),
                _ => false,
            })
        })
        .unwrap_or(true)
}

/// Serialize the toolbox and write it to `localStorage` — unless it's
/// [`is_empty_tool_state`], in which case we leave existing storage untouched.
/// Signals are `Copy`, so the toolbox is taken by value.
///
/// Writes go straight to `localStorage` via [`crate::utils::storage_set`]
/// rather than through `dioxus_sdk`'s reactive `use_storage`, whose spawned
/// watcher task failed to flush our `.set()` under the 0.7 runtime (issue #66).
async fn persist_tool_state(mut toolbox: Signal<misanthropic::tool::ToolBox>) {
    let saved = toolbox.write().save_json().await;
    if is_empty_tool_state(&saved) {
        log::debug!("Tool state is empty; leaving stored state untouched.");
        return;
    }
    match serde_json::to_string(&saved) {
        Ok(json) => crate::utils::storage_set(TOOLBOX_STATE_KEY, &json),
        Err(e) => log::warn!("Failed to serialize tool state: {e}"),
    }
}

/// A test prompt for testing the chat view.
#[cfg(debug_assertions)]
fn make_prompt() -> Prompt {
    use misanthropic::{
        json,
        prompt::{
            message::{Block, Content, Role},
            Message,
        },
        tool::{self, CustomMethodDef},
        Id,
    };
    use Id::*;

    Prompt::default()
        .model(Sonnet35)
        .add_tool(CustomMethodDef {
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
            strict: Some(true),
            defer_loading: None,
            allowed_callers: None,
        })
        // Inform the assistant about their limitations.
        .system(include_str!("../../../../misanthropic/examples/python_system.md"))
        .add_system(format!("## Python Environment\n\n{}", "3.12"))
        // The example has some examples of the Assistant using Python and some
        // without to help guide the assistant to use Python when necessary and
        // not when it isn't. The more examples here, with more varied prompts,
        // the better the Assistant will be at this.
        .messages([
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
                content: Content(vec![
                    // Regular, plain old, legacy thinking block. When displayed
                    // with `ThoughtsOrSpeech`, it will be styled as a thought.
                    r#"<thinking>I can't do that myself, but I can run a Python script to count the number of r's in "strawberry". The user did not specify case sensitivity so I will default to case insensitive.</thinking>"#.into(),
                    tool::Use::new(
                        "python",
                        json!({
                            "script": r#"print("strawberry".lower().count("r"))"#
                        }),
                    )
                    .with_id("calibration_000")
                    .into()
                ]),
            },
            tool::Result::new("calibration_000", "3").into(),
            (Role::Assistant, r#"The number of r's in "strawberry" is 3.""#).into(),
            (Role::User, "List the permutations of the first four letters of the alphabet.").into(),
            Message {
                role: Role::Assistant,
                content: Content(vec![
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
                    tool::Use::new(
                        "python",
                        json!({
                            "script": r#"import itertools\nprint(','.join("".join(t) for t in itertools.permutations(('a', 'b', 'c', 'd'))))"#
                        }),
                    )
                    .with_id("calibration_001")
                    .into()
                ]),
            },
            tool::Result::new(
                "calibration_001",
                "abcd,abdc,acbd,acdb,adbc,adcb,bacd,badc,bcad,bcda,bdac,bdca,cabd,cadb,cbad,cbda,cdab,cdba,dabc,dacb,dbac,dbca,dcab,dcba",
            ).into(),
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
fn make_prompt() -> Prompt {
    // The server will send us the prompt. This is just a placeholder.
    Prompt::default()
}

#[component]
pub fn Chat() -> Element {
    // Our signals. This is reactive state management. When these signals are
    // updated, the component will re-render. Signals are a Copy type, and use
    // automatic dependency tracking to determine when to re-render.
    let mut attachments = use_signal(Vec::<Block>::new);
    let mut connected = use_signal(|| false);
    let mut dragged_file_supported = use_signal(|| false);
    let mut dragged_over = use_signal(|| true);
    let mut failures = use_signal(|| 0);
    let mut options = use_signal(Options::default);
    let mut prompt = use_signal(make_prompt);
    let mut ready_json = use_signal(|| None);
    let mut ready_tool_json = use_signal(|| None);
    let mut shift_held = use_signal(|| false);
    let mut show_system = use_signal(|| false);
    let mut show_thought = use_signal(|| false);
    let mut show_tool_use = use_signal(|| false);
    // Created un-loaded; persisted state is restored asynchronously as the
    // first step of the stream task (see below), before the first prompt is
    // requested. Doing it here would require `block_on`, which only works for
    // tools whose `load_json` never actually awaits.
    let mut toolbox = use_signal(toolbox::create);

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
        // Restore persisted tool state once, before connecting. Read directly
        // from `localStorage` (not a reactive signal) so this resource doesn't
        // subscribe to anything written on every tool call — subscribing would
        // restart the whole stream each time. Outside the reconnect `loop` so a
        // reconnect never clobbers in-session tool state with the last snapshot.
        let stored = crate::utils::storage_get(TOOLBOX_STATE_KEY)
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        log::info!(
            "Restoring tool state from storage ({}).",
            if stored.is_null() { "empty" } else { "present" }
        );
        if let Err(e) = toolbox.write().load_json(stored).await {
            log::error!("`Toolbox::load_json` had error(s): {e}");
        } else {
            log::info!("Toolbox state loaded.");
        }

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
                                    let result = toolbox
                                        .write()
                                        .call(tool_use.clone())
                                        .await;
                                    persist_tool_state(toolbox).await;
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
                                // Install our tools: overwrites `methods` with
                                // the toolbox's definitions and runs each tool's
                                // `on_init` (e.g. Notepad injects its notes).
                                if let Err(e) =
                                    toolbox.write().prepare(&mut new).await
                                {
                                    log::error!(
                                        "`Toolbox::prepare` had error(s): {e}"
                                    );
                                } else {
                                    log::info!("Toolbox prepared.")
                                }

                                // Run the per-turn hook. A no-op for Notepad
                                // (notes apply on init only, to keep the cache
                                // warm) but wires the lifecycle for tools that
                                // do need per-turn context. Must run before
                                // SetPrompt so any mutation reaches the backend.
                                if let Err(e) = toolbox
                                    .write()
                                    .update_turn_context(&mut new)
                                    .await
                                {
                                    log::error!(
                                        "`Toolbox::update_turn_context` had error(s): {e}"
                                    );
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

    let mut input_buffer = use_signal(String::new);

    if !*connected.read() {
        return rsx! {
            document::Link { rel: "stylesheet", href: CSS }

            div {
                class: "chat connecting",
                // Do our little indicator dance.
                {format!(
                    "Connecting{}",
                    ".".repeat(*failures.read() % 4),
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
                // Dioxus 0.7 submits forms by default (0.6 prevented it); we
                // drive sending from the textarea's Enter handler, so suppress
                // the native submit/reload here.
                onsubmit: move |e| e.prevent_default(),
                textarea {
                    class: "input-box",
                    class: if *dragged_over.read() {
                        "dragged-over"
                    } else { "" },
                    class: if *dragged_file_supported.read() {
                        "dragged-file-supported"
                    } else { "" },
                    placeholder: "Type your message or drag a .json file here to load a chat...\n\n...You can also drag images, PDFs, or text files here to attach them.",
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
                        // Dioxus 0.7: `files()` returns `Vec<FileData>` directly
                        // (0.6 returned an `Option<FileEngine>`).
                        let files = e.files();
                        if files.len() != 1 {
                            dragged_file_supported.set(false);
                            return;
                        }
                        let filename = files[0].name();
                        if MediaType::is_supported(&filename)
                            || filename.ends_with(".json")
                            || filename.ends_with(".pdf")
                            || filename.ends_with(".txt")
                        {
                            dragged_file_supported.set(true);
                        } else {
                            dragged_file_supported.set(false);
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
                        // Dioxus 0.7: `files()` returns `Vec<FileData>` directly
                        // (0.6 returned an `Option<FileEngine>`), and each file
                        // is read with `FileData::read_bytes`.
                        let files = e.files();
                        if files.is_empty() {
                            log::warn!("No files dropped.");
                            return;
                        } else if files.len() > 1 {
                            log::warn!("Only one file can be dropped at a time.");
                            return;
                        }
                        // len is 1
                        let file = &files[0];
                        let filename = file.name();

                        // Is the file a JSON file? We need to load it, set
                        // the tools, and send the updated prompt to the
                        // backend.
                        if filename.ends_with(".json") {
                            let data = match file.read_bytes().await {
                                Ok(data) => data,
                                Err(e) => {
                                    log::warn!("Failed to read file: {e}");
                                    return;
                                }
                            };

                            if data.is_empty() {
                                log::warn!("Empty file.");
                                return;
                            }

                            // Sniff whether this is a saved conversation or
                            // saved tool state by shape (Save writes them as
                            // separate files). See `Dropped`.
                            match serde_json::from_slice::<Dropped>(&data) {
                                Ok(Dropped::Prompt(new_prompt)) => {
                                    log::info!("Loading new prompt.");
                                    let mut new_prompt = *new_prompt;

                                    // Tool definitions track the app's
                                    // current capabilities, not whatever the
                                    // loaded (possibly foreign or older)
                                    // prompt carried. `prepare` overwrites
                                    // `methods` and runs each tool's
                                    // `on_init`, so old/foreign prompts just
                                    // work.
                                    if let Err(e) = toolbox.write().prepare(&mut new_prompt).await {
                                        log::error!("`Toolbox::prepare` had error(s): {e}");
                                    } else {
                                        log::info!("Toolbox prepared.");
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
                                Ok(Dropped::ToolState(state)) => {
                                    log::info!("Loading tool state.");
                                    let value = match serde_json::to_value(state) {
                                        Ok(value) => value,
                                        Err(e) => {
                                            log::warn!("Failed to re-encode tool state: {}", e);
                                            return;
                                        }
                                    };
                                    if let Err(e) = toolbox.write().load_json(value).await {
                                        log::warn!("Failed to load tool state: {}", e);
                                    } else {
                                        log::info!("Tool state loaded.");
                                    }
                                    persist_tool_state(toolbox).await;

                                    // Re-apply to the current prompt so newly
                                    // loaded notes appear in the system block,
                                    // then push the update to the backend.
                                    let mut new_prompt = prompt.peek().clone();
                                    if let Err(e) = toolbox.write().prepare(&mut new_prompt).await {
                                        log::error!("`Toolbox::prepare` had error(s): {e}");
                                    }
                                    match CLIENT.read().send(
                                        Request::SetPrompt(new_prompt.clone())
                                    ).await {
                                        Ok(_) => {
                                            prompt.set(new_prompt);
                                        }
                                        Err(e) => {
                                            log::error!("Failed to set prompt after tool state load: {}", e);
                                        }
                                    }
                                }
                                Err(e) => {
                                    log::warn!("Dropped .json is neither a prompt nor tool state: {}", e);
                                }
                            }

                            return;
                        }

                        // PDF: base64-encode and attach as a document with
                        // citations enabled so the model can cite it.
                        if filename.ends_with(".pdf") {
                            let data = match file.read_bytes().await {
                                Ok(data) => data,
                                Err(e) => {
                                    log::warn!("Failed to read file: {e}");
                                    return;
                                }
                            };
                            if data.is_empty() {
                                log::warn!("Empty file.");
                                return;
                            }
                            let doc = Block::document_with_citations(
                                DocumentSource::from_base64(BASE64.encode(&data)),
                            );
                            attachments.write().push(doc);
                            return;
                        }

                        // Plain text: attach as a text document with citations
                        // enabled (auto-chunked into sentences server-side).
                        if filename.ends_with(".txt") {
                            let data = match file.read_bytes().await {
                                Ok(data) => data,
                                Err(e) => {
                                    log::warn!("Failed to read file: {e}");
                                    return;
                                }
                            };
                            if data.is_empty() {
                                log::warn!("Empty file.");
                                return;
                            }
                            let text = String::from_utf8_lossy(&data).into_owned();
                            let doc = Block::document_with_citations(
                                DocumentSource::from_text(text),
                            );
                            attachments.write().push(doc);
                            return;
                        }

                        // Image files.
                        let format = if let Some(format) =  MediaType::detect(&filename) {
                            format
                        } else {
                            log::warn!("Unsupported file type.");
                            return;
                        };

                        let data = match file.read_bytes().await {
                            Ok(data) => data,
                            Err(e) => {
                                log::warn!("Failed to read file: {e}");
                                return;
                            }
                        };

                        if data.is_empty() {
                            log::warn!("Empty file.");
                            return;
                        }

                        let image = Image::from_compressed(format, data);
                        attachments.write().push(image.into());
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
            //
            // Prompt and tool state are saved separately: the prompt is just
            // conversation data, while tool state lives in browser storage and
            // is only exported on demand. Drag either file back in to load it
            // (the drop handler sniffs which is which).
            button {
                class: "toggle",
                class: "save",
                class: if ready_json.read().is_some() { "ready" } else { "" },
                // Serialize on hover.
                onmouseover: move |e| {
                    e.prevent_default();
                    let json = serde_json::to_string_pretty(
                        prompt.read().deref()
                    ).unwrap();
                    ready_json.write().replace(json);
                },
                onmouseleave: move |e| {
                    e.prevent_default();
                    ready_json.write().take();
                },
                // Download the JSON.
                a {
                    href: ready_json.read().deref().as_ref().map(|json| {
                        format!(
                            "data:application/json;base64,{}",
                            BASE64.encode(json.as_bytes())
                        )
                    }),
                    download: "prompt.json",
                    "Save"
                }
            }
            button {
                class: "toggle",
                class: "save",
                class: if ready_tool_json.read().is_some() { "ready" } else { "" },
                // Serialize on hover.
                onmouseover: move |e| async move {
                    e.prevent_default();
                    let json = serde_json::to_string_pretty(
                        &toolbox.write().save_json().await
                    ).unwrap();
                    ready_tool_json.write().replace(json);
                },
                onmouseleave: move |e| {
                    e.prevent_default();
                    ready_tool_json.write().take();
                },
                // Download the JSON.
                a {
                    href: ready_tool_json.read().deref().as_ref().map(|json| {
                        format!(
                            "data:application/json;base64,{}",
                            BASE64.encode(json.as_bytes())
                        )
                    }),
                    download: "tool_state.json",
                    "Save Tools"
                }
            }
        }
    }
}

// <a download="prompt.json" href="data:application/json;base64,">Download</a>
