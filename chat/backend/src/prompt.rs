use crate::Prompt;

const DEFAULT_PROMPT_JSON: &str = include_str!("prompt/default.json");

fn load_default() -> Prompt {
    serde_json::from_str(DEFAULT_PROMPT_JSON).unwrap()
}

// Parse once and store the result in a static variable. This is so we don't
// have to parse the JSON every time we want to get the default prompt.
lazy_static::lazy_static! {
    static ref DEFAULT: Prompt = load_default();
}

pub fn default() -> Prompt {
    DEFAULT.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let prompt = default();
        assert_eq!(prompt.messages.len(), 0);
    }
}
