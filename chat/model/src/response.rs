use serde::{Deserialize, Serialize};

use misanthropic::prompt;

/// Response from to the backend.
pub type Response = Result<Success, Error>;

/// Success from the backend.
#[derive(Debug, Deserialize, Serialize, derive_more::Unwrap)]
#[serde(rename_all = "snake_case")]
pub enum Success {
    /// The server-side copy of the prompt, considered the source of truth.
    /// Except for tools. The toolbox lives in the frontend.
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
    /// [`TurnOrderError`] when appending a [`Message`] to the [`Prompt`].
    ///
    /// [`TurnOrderError`]: prompt::TurnOrderError
    /// [`Message`]: prompt::Message
    /// [`Prompt`]: prompt::Prompt
    TurnOrder {
        /// Error
        #[from]
        error: prompt::TurnOrderError,
    },
    /// [`misanthropic::client::Error`] (connection related).
    MisanthropicClient {
        /// Error message
        message: String,
    },
    /// [`stream::Error`]
    ///
    /// [`stream::Error`]: misanthropic::stream::Error
    Stream {
        /// Error message
        message: String,
    },
}
