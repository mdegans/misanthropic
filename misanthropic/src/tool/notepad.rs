//! [`Notepad`] [`tool`], implemented on the typed-tool layer.
//!
//! [`tool`]: super
use crate::{
    Prompt,
    prompt::message::{Block, Content},
};

use super::{ErasedMethod, Method, Methods, ToolArgs};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

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
    /// Creates a new `Notepad` tool.
    pub fn new() -> Self {
        Self { notes: Vec::new() }
    }
}

/// Arguments for the `push` [`Method`]: take a note.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Push {
    /// The note to take.
    note: String,
}

impl ToolArgs for Push {
    const NAME: &'static str = "push";
    const DESCRIPTION: &'static str = "Take a note for the next chat.";
}

/// Arguments for the `clear` [`Method`]: a no-arg method (proves heterogeneous
/// `Args` coexist on one tool).
#[derive(Debug, Deserialize, JsonSchema)]
pub struct Clear {}

impl ToolArgs for Clear {
    const NAME: &'static str = "clear";
    const DESCRIPTION: &'static str = "Erase all saved notes.";
}

/// The `push` method.
struct PushMethod;

#[async_trait::async_trait]
impl<'a> Method<Notepad<'a>> for PushMethod {
    type Args = Push;

    async fn run(
        &self,
        state: &mut Notepad<'a>,
        args: Push,
    ) -> std::result::Result<Content<'static>, Content<'static>> {
        let note = args.note;

        if note.contains("<notepad>") || note.contains("</notepad>") {
            #[cfg(feature = "log")]
            log::error!(
                "Injection attack detected. `<notepad>` or `</notepad>` in note."
            );
            return Err(
                "You cannot put `<notepad>` or `</notepad>` in your note."
                    .into(),
            );
        }

        if note.contains("<note>") || note.contains("</note>") {
            #[cfg(feature = "log")]
            log::error!("Agent goofed and put a note tag in their note.");
            return Err("You cannot put `<note>` or `</note>` in your note. `notepad` will handle it.".into());
        }

        #[cfg(feature = "log")]
        log::debug!("Note taken: {}", note);
        state.notes.push(note.into());

        Ok("Note taken.".into())
    }
}

/// The `clear` method.
struct ClearMethod;

#[async_trait::async_trait]
impl<'a> Method<Notepad<'a>> for ClearMethod {
    type Args = Clear;

    async fn run(
        &self,
        state: &mut Notepad<'a>,
        _args: Clear,
    ) -> std::result::Result<Content<'static>, Content<'static>> {
        state.notes.clear();
        Ok("Notes cleared.".into())
    }
}

#[async_trait::async_trait]
impl<'a> Methods for Notepad<'a> {
    const NAME: &'static str = stringify!(Notepad);

    fn methods(&self) -> Vec<Box<dyn ErasedMethod<Self>>> {
        vec![
            Box::new(PushMethod) as Box<dyn ErasedMethod<Self>>,
            Box::new(ClearMethod),
        ]
    }

    /// Save notepad state.
    async fn save_json(&mut self) -> serde_json::Value {
        json!(self)
    }

    /// Load notepad state.
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

    async fn on_init(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Set up the notepad instructions and initial state.
        self.sync_apply_to_prompt(prompt).map_err(|e| {
            let error_string = e.to_string();
            Box::new(std::io::Error::other(error_string))
                as Box<dyn std::error::Error + Send + Sync>
        })
    }

    async fn on_turn(
        &mut self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Update the notepad content (notes may have been added).
        self.sync_apply_to_prompt(prompt).map_err(|e| {
            let error_string = e.to_string();
            Box::new(std::io::Error::other(error_string))
                as Box<dyn std::error::Error + Send + Sync>
        })
    }
}

impl<'a> Notepad<'a> {
    /// Synchronous version of apply_to_prompt for internal use
    fn sync_apply_to_prompt(
        &self,
        prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Check for the presence of a `<notepad>` tags in the system prompt.
        for note in &self.notes {
            if note.contains("<notepad>") || note.contains("</notepad>") {
                return Err("Injection attack detected. Notepad is compromised and contains forbidden tags.".into());
            }
            if note.contains("<note>") || note.contains("</note>") {
                return Err("Notepad contains forbidden tags.".into());
            }
        }

        // Write the text to the prompt, returning true if the text was written.
        let write_text = |text: &mut crate::CowStr| -> bool {
            #[cfg(feature = "langsan")]
            if text.contains("<notepad_instructions>") {
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
    use super::*;
    use crate::tool::{Methods, Tool, ToolBox, Typed, Use};

    #[test]
    fn test_notepad_name() {
        assert_eq!(Typed(Notepad::new()).name(), stringify!(Notepad));
    }

    #[test]
    fn test_notepad_push_definition() {
        let defs = Typed(Notepad::new()).definitions();
        let push = defs
            .iter()
            .find(|d| d.name == "Notepad__push")
            .expect("push method present");
        assert_eq!(push.description, "Take a note for the next chat.");
        let props = push.schema["properties"].as_object().unwrap();
        assert!(props.contains_key("note"));
        assert_eq!(props["note"]["type"], "string");
    }

    #[tokio::test]
    async fn test_notepad_call() {
        let mut notepad = Typed(Notepad::new());
        let result = notepad
            .call(Use {
                id: "abcd".into(),
                name: "Notepad__push".into(),
                input: json!({ "note": "Hello, world!" }),
                cache_control: None,
            })
            .await;
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert!(!result.is_error);
        assert_eq!(notepad.0.notes.len(), 1);
        assert_eq!(notepad.0.notes[0].as_ref(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_notepad_clear() {
        let mut notepad = Typed(Notepad::new());
        notepad.0.notes.push("scratch".into());
        let result = notepad
            .call(Use {
                id: "abcd".into(),
                name: "Notepad__clear".into(),
                input: json!({}),
                cache_control: None,
            })
            .await;
        assert!(!result.is_error);
        assert!(notepad.0.notes.is_empty());
    }

    #[tokio::test]
    async fn test_notepad_call_injection_rejected() {
        let mut notepad = Typed(Notepad::new());
        let result = notepad
            .call(Use {
                id: "abcd".into(),
                name: "Notepad__push".into(),
                input: json!({ "note": "<notepad>evil</notepad>" }),
                cache_control: None,
            })
            .await;
        assert!(result.is_error);
        assert!(notepad.0.notes.is_empty());
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
        let mut toolbox = ToolBox::default().add_typed(Notepad::new());

        let names: Vec<_> = toolbox
            .definitions()
            .into_iter()
            .map(|d| d.name.into_owned())
            .collect();
        assert!(names.contains(&"toolbox__Notepad__push".to_string()));
        assert!(names.contains(&"toolbox__Notepad__clear".to_string()));

        let result = toolbox
            .call(Use {
                id: "abcd".into(),
                name: "toolbox__Notepad__push".into(),
                input: json!({ "note": "Hello, world!" }),
                cache_control: None,
            })
            .await;
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert!(!result.is_error);

        let json = toolbox.save_json().await;
        let mut toolbox2 = ToolBox::default().add_typed(Notepad::new());
        toolbox2.load_json(json).await.unwrap();

        // Round-trip the inner Notepad's state back out.
        let tool = toolbox2.tool_name_to_tool.get_mut("Notepad").unwrap();
        let json = tool.save_json().await;
        let mut notepad = Notepad::new();
        notepad.load_json(json).await.unwrap();
        assert_eq!(notepad.notes.len(), 1);
        assert_eq!(notepad.notes[0].as_ref(), "Hello, world!");
    }

    #[tokio::test]
    async fn test_notepad_setup_injection_attack() {
        const FORBIDDEN: &[&str] =
            &["<notepad>", "</notepad>", "<note>", "</note>"];

        for &seq in FORBIDDEN {
            let mut notepad = Notepad::new();
            notepad.notes.push(seq.into());
            let mut prompt = Prompt::default();
            let result = notepad.on_init(&mut prompt).await;
            assert!(result.is_err());
        }
    }

    // Test with no existing block.
    #[tokio::test]
    async fn test_notepad_setup_no_existing_block() {
        let mut notepad = Notepad::new();
        notepad.notes.push("I am test code! Whee!".into());
        let mut prompt =
            Prompt::default().set_system("You are a test code! Whee!");
        notepad.on_init(&mut prompt).await.unwrap();

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
    #[tokio::test]
    async fn test_notepad_setup_existing_block() {
        let mut notepad = Notepad::new();
        notepad.notes.push("I am test code! Whee!".into());
        let mut prompt = Prompt::default().set_system(
            "<notepad_instructions>What follows in `notepad` tags are `note`s you took in other sessions using the `notepad` tool.</notepad_instructions><notepad><note>Existing note.</note></notepad>",
        );
        notepad.on_init(&mut prompt).await.unwrap();

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
