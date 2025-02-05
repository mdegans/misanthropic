use crate::{Message, Prompt};

const DEFAULT_PROMPT_JSON: &str = include_str!("prompt/default.json");

fn load_default() -> Prompt {
    serde_json::from_str(DEFAULT_PROMPT_JSON).unwrap()
}

// Parse once and store the result in a static variable.
lazy_static::lazy_static! {
    static ref DEFAULT: Prompt = load_default();
}

pub fn default_messages_len() -> usize {
    DEFAULT.messages.len()
}

pub fn default() -> Prompt {
    DEFAULT.clone()
}

/// Get new messages from the prompt (excluding any in the default prompt).
pub fn get_new_messages(p: &Prompt) -> &[Message] {
    p.messages[default_messages_len()..].as_ref()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_default_prompt() {
        let chat = Prompt::default();

        // get path to crate root
        let crate_root = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let path =
            std::path::Path::new(&crate_root).join("src/prompt/default.json");
        let json = serde_json::to_string_pretty(&chat).unwrap();
        std::fs::write(path, json).unwrap();
    }

    #[test]
    fn test_default() {
        let prompt = default();
        assert_eq!(prompt.messages.len(), 0);
    }
}
