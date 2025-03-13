//! [`Notepad`] [`tool`].
//!
//! [`tool`]: super
use super::{Spec, Tool, Use};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use std::borrow::Cow;

/// A `Notepad` tool for an [`Assistant`] to take persistent notes.
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[derive(Serialize, Deserialize)]
#[serde(transparent)]
pub struct Notepad<'a> {
    /// Notes taken by the [`Assistant`].
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    pub notes: Vec<Cow<'a, str>>,
}

impl<'a> Notepad<'a> {
    /// Creates a new `Notepad` tool.
    pub fn new() -> Self {
        Self { notes: Vec::new() }
    }
}

impl<'a> Tool for Notepad<'a> {
    fn name(&self) -> &str {
        "notepad"
    }

    fn spec(&self) -> Spec<'static> {
        Spec::builder(self.name().to_string())
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
            .unwrap()
    }

    fn call<'c>(&mut self, mut call: Use<'c>) -> super::Result<'c> {
        if call.name != self.name() {
            return super::Result {
                tool_use_id: call.id,
                content: "Invalid tool name.".into(),
                is_error: true,
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            };
        }

        if let Value::String(note) = call.input["note"].take() {
            self.notes.push(note.into());
        }

        super::Result {
            tool_use_id: call.id,
            content: "Note taken.".into(),
            is_error: false,
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        }
    }

    fn save_json(&self) -> serde_json::Value {
        json!(self)
    }

    fn load_json(
        &mut self,
        _json: serde_json::Value,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let new: Notepad = serde_json::from_value(_json)?;
        self.notes = new.notes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tool::ToolBox;

    use super::*;

    #[test]
    fn test_notepad_name() {
        let notepad = Notepad::new();
        assert_eq!(notepad.name(), "notepad");
    }

    #[test]
    fn test_notepad_spec() {
        let notepad = Notepad::new();
        let spec = notepad.spec();
        assert_eq!(spec.name, "notepad");
        assert_eq!(
            spec.description,
            Cow::Borrowed("Take a note for the next chat.")
        );
        assert_eq!(
            spec.schema,
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

    #[test]
    fn test_notepad_call() {
        let mut notepad = Notepad::new();
        let call = Use {
            id: "abcd".into(),
            name: "notepad".into(),
            input: json!({
                "note": "Hello, world!"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };
        let result = notepad.call(call);
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert_eq!(result.is_error, false);
        assert_eq!(notepad.notes.len(), 1);
        assert_eq!(notepad.notes[0], "Hello, world!");
    }

    #[test]
    fn test_notepad_save_load_json() {
        let mut notepad = Notepad::new();
        notepad.notes.push("Hello, world!".into());
        let json = notepad.save_json();
        let mut notepad2 = Notepad::new();
        notepad2.load_json(json).unwrap();
        assert_eq!(notepad.notes, notepad2.notes);
    }

    #[test]
    fn test_notepad_in_toolbox() {
        let mut toolbox = ToolBox::default().add(Notepad::new());
        let call = Use {
            id: "abcd".into(),
            name: "notepad".into(),
            input: json!({
                "note": "Hello, world!"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };
        let result = toolbox.call(call);
        assert_eq!(result.tool_use_id, "abcd");
        assert_eq!(result.content, "Note taken.".into());
        assert_eq!(result.is_error, false);

        let json = toolbox.save_json();
        let mut toolbox2 = ToolBox::default().add(Notepad::new());
        toolbox2.load_json(json).unwrap();

        let notepad = toolbox2.get("notepad").unwrap();
        let json = notepad.save_json();
        let mut notepad2 = Notepad::new();
        notepad2.load_json(json).unwrap();
        assert_eq!(notepad2.notes.len(), 1);
        assert_eq!(notepad2.notes[0], "Hello, world!");
    }
}
