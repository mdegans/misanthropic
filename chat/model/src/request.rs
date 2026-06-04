use misanthropic::prompt;
use serde::{Deserialize, Serialize};

/// Request to the backend.
#[derive(Debug, Deserialize, Serialize, derive_more::From)]
#[serde(rename_all = "snake_case")]
pub enum Request {
    /// Get the prompt.
    GetPrompt,
    /// Set the prompt.
    SetPrompt(prompt::Prompt),
    /// User message.
    UserMessage(prompt::UserMessage),
}
