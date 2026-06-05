use misanthropic::prompt;
use serde::{Deserialize, Serialize};

/// Request to the backend.
// One `SetPrompt(Prompt)` variant dwarfs the unit/message variants, but a
// single request is in flight at a time and boxing would ripple through the
// `From` derive and every construction site for no real gain in a demo.
#[allow(clippy::large_enum_variant)]
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
