//! [`Notepad`] [`tool`].
//!
//! [`tool`]: super
use crate::{prompt::message::Block, Prompt};

use super::{Method, Tool, Use};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const NOTEPAD_INSTRUCTIONS: &str = r#"<notepad_instructions>What follows in `notepad` tags are `note`s you took in other sessions using the `notepad` tool.</notepad_instructions>"#;

/// A `Notepad` tool for an [`Assistant`] to take persistent notes.
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[derive(Default, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Notepad<'a> {
    /// Notes taken by the [`Assistant`].
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    notes: Vec<crate::CowStr<'a>>,
}

impl<'a> Notepad<'a> {
    const NAME: &'static str = stringify!(Notepad);

    /// Creates a new `Notepad` tool.
    pub fn new() -> Self {
        Self { notes: Vec::new() }
    }
}

#[async_trait::async_trait]
impl<'a> Tool for Notepad<'a> {
    fn name(&self) -> &str {
        Self::NAME
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new(std::iter::once(
            Method::builder("Notepad::push")
                .description("Take a note for the next chat.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "note": {
                            "type": "string",
                            "description": "The note to take."
                        }
                    },
                    "required": ["note"]
                }))
                .build()
                .unwrap(),
        ))
    }

    async fn call<'c>(&mut self, call: Use<'c>) -> super::Result<'c> {
        #[cfg(feature = "log")]
        log::debug!("Notepad call: {:?}", serde_json::to_string_pretty(&call));
        if !call.name.ends_with("Notepad::push") {
            #[cfg(feature = "log")]
            log::error!("Invalid tool name.");
            return super::Result {
                tool_use_id: call.id,
                content:
                    "`Notepad::push` is the only method available on `Notepad`"
                        .into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            };
        }

        let mut map = if let Value::Object(map) = call.input {
            map
        } else {
            let detail = serde_json::to_string(&call.input).unwrap();
            #[cfg(feature = "log")]
            log::error!("`input` not an object: {detail}");
            return super::Result {
                tool_use_id: call.id,
                content: format!(
                    "`input` must be an object. This should be impossible is probably the developer's fault. Got: `{}`",
                    detail
                ).into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            };
        };

        if let Some(Value::String(note)) = map.remove("note") {
            if note.contains("<notepad>") || note.contains("</notepad>") {
                #[cfg(feature = "log")]
                log::error!("Injection attack detected. `<notepad>` or `</notepad>` in note.");
                return super::Result {
                    tool_use_id: call.id,
                    content: "You cannot put `<notepad>` or `</notepad>` in your note.".into(),
                    is_error: true,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                };
            }

            if note.contains("<note>") || note.contains("</note>") {
                #[cfg(feature = "log")]
                log::error!("Agent goofed and put a note tag in their note.");
                return super::Result {
                    tool_use_id: call.id,
                    content: "You cannot put `<note>` or `</note>` in your note. `notepad` will handle it.".into(),
                    is_error: true,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                };
            }

            #[cfg(feature = "log")]
            log::debug!("Note taken: {}", note);
            self.notes.push(note.into());
        } else {
            #[cfg(feature = "log")]
            log::error!("`note` not a string.");
            return super::Result {
                tool_use_id: call.id,
                content: "`note` must be a string.".into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            };
        }

        super::Result {
            tool_use_id: call.id,
            content: "Note taken.".into(),
            is_error: false,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    /// Save notepad state. Now async to support potential future IO operations.
    async fn save_json(&mut self) -> serde_json::Value {
        json!(self)
    }

    /// Load notepad state. Now async to support potential future IO operations.
    async fn load_json(
        &mut self,
        json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        let new: Notepad =
            serde_json::from_value(json).map_err(|e| e.to_string())?;

        for note in &new.notes {
            if note.contains("<notepad>") || note.contains("</notepad>") {
                return Err("Injection attack detected. Notepad contains `<notepad>` tags.".into());
            }
            if note.contains("<note>") || note.contains("</note>") {
                return Err("Notepad contains forbidden tags (`<note>` is not necessary).".into());
            }
        }

        self.notes = new.notes;
        Ok(())
    }

    /// Setup [`Prompt`] by updating the notepad block in the system prompt.
    ///
    /// O(n) where n is the length of the system prompt.
    // This would be O(1), but Anthropic won't let us stuff as much metadata as
    // we want in Prompt::metadata. We found this out the hard way.
    fn apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Check for the presence of a `<notepad>` tags in the system prompt.
        for note in &self.notes {
            if note.contains("<notepad>") || note.contains("</notepad>") {
                // This should never happen unless the developer allows the user
                // to supply the notepad and a compromised prompt is used. It is
                // possible to Deserialize or otherwise craft such a Notepad.
                return Err("Injection attack detected. Notepad is compromised and contains forbidden tags.".into());
            }
            if note.contains("<note>") || note.contains("</note>") {
                // This should never happen unless the developer allows the user
                // to supply the notepad and a compromised prompt is used. It is
                // possible to Deserialize or otherwise craft such a Notepad.
                return Err("Notepad contains forbidden tags.".into());
            }
        }
        // Notepad does not contain forbidden tags.

        // Write the text to the prompt, returning true if the text was written.
        // Text is only written where <notepad_instructions> is found.
        let write_text = |text: &mut crate::CowStr| -> bool {
            #[cfg(feature = "langsan")]
            if text.contains("<notepad_instructions>") {
                // This is the correct block. Overwrite it.
                let mut new: crate::CowStr = String::new().into();

                new.push_str(NOTEPAD_INSTRUCTIONS);
                new.push_str("<notepad>");
                for note in &self.notes {
                    new.push_str("<note>");
                    new.push_str(note);
                    new.push_str("</note>");
                }
                new.push_str("</notepad>");

                *text = new;

                true
            } else {
                false
            }
            #[cfg(not(feature = "langsan"))]
            if text.contains("<notepad_instructions>") {
                // Regular old std::borrow::Cow<str>
                text.to_mut().clear();
                text.to_mut().push_str(NOTEPAD_INSTRUCTIONS);
                text.to_mut().push_str("<notepad>");
                for note in &self.notes {
                    text.to_mut().push_str("<note>");
                    text.to_mut().push_str(note);
                    text.to_mut().push_str("</note>");
                }
                text.to_mut().push_str("</notepad>");

                true
            } else {
                false
            }
        };

        if let Some(system) = &mut prompt.system {
            // Existing system prompt. Try to find the notepad instructions.
            for block in system.iter_mut() {
                if let Block::Text { text, .. } = block {
                    if write_text(text) {
                        return Ok(());
                    }
                }
            }

            // Not found. Append it to existing system prompt.
            let mut text: crate::CowStr = "<notepad_instructions>".into();
            write_text(&mut text);
            system.push(text);
            return Ok(());
        }

        // Not found. No existing system prompt. Create a new one.
        let mut text: crate::CowStr = "<notepad_instructions>".into();
        write_text(&mut text);
        prompt.system.replace(text.into());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use crate::tool::ToolBox;

    use super::*;

    #[test]
    fn test_notepad_name() {
        let notepad = Notepad::new();
        assert_eq!(notepad.name(), stringify!(Notepad));
    }

    #[test]
    fn test_notepad_functions() {
        let notepad = Notepad::new();
        let function = notepad.methods().next().unwrap();
        assert!(function.name.starts_with(stringify!(Notepad)));
        assert!(function.name.ends_with("::push"));
        assert_eq!(
            function.description,
            Cow::Borrowed("Take a note for the next chat.")
        );
        assert_eq!(
            function.schema,
            json!({
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "The note to take."
                    }
                },
                "required": ["note"]
            })
        );
    }

    #[tokio::test]
    async fn test_notepad_call() {
        let mut notepad = Notepad::new();
        let call = Use {
            id: "abcd".into(),
            name: "Notepad::push".into(),
            input: json!({
                "note": "Hello, world!"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };
        let result = notepad.call(call).await;
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert_eq!(result.is_error, false);
        assert_eq!(notepad.notes.len(), 1);
        assert_eq!(notepad.notes[0].as_ref(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_notepad_save_load_json() {
        let mut notepad = Notepad::new();
        notepad.notes.push("Hello, world!".into());
        let json = notepad.save_json().await;
        let mut notepad2 = Notepad::new();
        notepad2.load_json(json).await.unwrap();
        assert_eq!(notepad.notes, notepad2.notes);
    }

    #[tokio::test]
    async fn test_notepad_in_toolbox() {
        let mut toolbox = ToolBox::default().add(Notepad::new());
        for method in toolbox.methods() {
            assert_eq!(method.name, "toolbox::Notepad::push");
        }
        let call = Use {
            id: "abcd".into(),
            name: "toolbox::Notepad::push".into(),
            input: json!({
                "note": "Hello, world!"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };
        let result = toolbox.call(call).await;
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert_eq!(result.is_error, false);

        let json = toolbox.save_json().await;
        let mut toolbox2 = ToolBox::default().add(Notepad::new());
        toolbox2.load_json(json).await.unwrap();

        let notepad = toolbox2
            .tool_name_to_tool
            .get_mut(Notepad::new().name())
            .unwrap();
        let json = notepad.save_json().await;
        let mut notepad2 = Notepad::new();
        notepad2.load_json(json).await.unwrap();
        assert_eq!(notepad2.notes.len(), 1);
        assert_eq!(notepad2.notes[0].as_ref(), "Hello, world!");
    }

    #[test]
    fn test_notepad_setup_injection_attack() {
        const FORBIDDEN: &[&str] =
            &["<notepad>", "</notepad>", "<note>", "</note>"];

        for &seq in FORBIDDEN {
            let mut notepad = Notepad::new();
            notepad.notes.push(seq.into());
            let mut prompt = Prompt::default();
            let result = notepad.apply_to_prompt(&mut prompt);
            assert!(result.is_err());
        }
    }

    // Test with no existing block.
    #[test]
    fn test_notepad_setup_no_existing_block() {
        let mut notepad = Notepad::new();
        notepad.notes.push("I am test code! Whee!".into());
        let mut prompt =
            Prompt::default().set_system("You are a test code! Whee!");
        notepad.apply_to_prompt(&mut prompt).unwrap();

        // The block should have been appended.
        assert_eq!(prompt.system.as_ref().unwrap().len(), 2);
        if let Block::Text { text, .. } = prompt.system.unwrap().last().unwrap()
        {
            assert_eq!(
                text.as_ref(),
                "<notepad_instructions>What follows in `notepad` tags are `note`s you took in other sessions using the `notepad` tool.</notepad_instructions><notepad><note>I am test code! Whee!</note></notepad>"
            );
        } else {
            panic!("Expected a text block.");
        }
    }

    // Test with existing block.
    #[test]
    fn test_notepad_setup_existing_block() {
        let mut notepad = Notepad::new();
        notepad.notes.push("I am test code! Whee!".into());
        let mut prompt = Prompt::default().set_system(
            "<notepad_instructions>What follows in `notepad` tags are `note`s you took in other sessions using the `notepad` tool.</notepad_instructions><notepad><note>Existing note.</note></notepad>",
        );
        notepad.apply_to_prompt(&mut prompt).unwrap();

        // The block should have been replaced.
        assert_eq!(prompt.system.as_ref().unwrap().len(), 1);
        if let Block::Text { text, .. } = prompt.system.unwrap().last().unwrap()
        {
            assert_eq!(
                text.as_ref(),
                "<notepad_instructions>What follows in `notepad` tags are `note`s you took in other sessions using the `notepad` tool.</notepad_instructions><notepad><note>I am test code! Whee!</note></notepad>"
            );
        } else {
            panic!("Expected a text block.");
        }
    }
}
