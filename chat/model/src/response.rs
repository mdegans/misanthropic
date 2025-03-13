use serde::{Deserialize, Serialize};

use misanthropic::prompt;

/// Response from to the backend.
pub type Response = Result<Success, Error>;

/// Success from the backend.
#[derive(Debug, Deserialize, Serialize, derive_more::Unwrap)]
#[serde(rename_all = "snake_case")]
pub enum Success {
    /// The server-side copy of the prompt, considered the source of truth.
    // If there is ever a desync, this should be reported with repro steps. The
    // same handling code runs on the server and client, so this should never
    // happen.
    Prompt(prompt::Prompt<'static>),
    /// [`mianthropic::stream::Event`] forwarded from Anthropic
    Stream(misanthropic::stream::Event),
    /// The user message was successfully processed.
    UserMessage(prompt::UserMessage<'static>),
}

/// Error from the backend.
#[derive(
    Debug, Deserialize, Serialize, thiserror::Error, derive_more::Display,
)]
#[serde(rename_all = "snake_case")]
pub enum Error {
    /// Turn order error
    TurnOrder {
        /// Error
        #[from]
        error: prompt::TurnOrderError,
    },
    /// Misantropic client error
    MisanthropicClient {
        /// Error message
        message: String,
    },
    /// Stream error
    Stream {
        /// Error message
        message: String,
    },
}
