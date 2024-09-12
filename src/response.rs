//! [`Response`] types for the [Anthropic Messages API].
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use derive_more::derive::IsVariant;

pub(crate) mod message;
pub use message::{Message, StopReason, Usage};

use crate::request;

/// Sucessful API response from the [Anthropic Messages API].
///
/// [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
#[derive(IsVariant)]
pub enum Response {
    /// Single [`response::Message`] from the API.
    ///
    /// [`response::Message`]: Message
    Message {
        #[allow(missing_docs)]
        message: self::Message,
    },
    /// [`Stream`] of [`Event`]s (message delta, etc.).
    ///
    /// [`Stream`]: crate::Stream
    /// [`Event`]: crate::stream::Event
    Stream {
        #[allow(missing_docs)]
        stream: crate::Stream,
    },
}

impl Response {
    /// Convert a [`Response::Stream`] variant into a [`crate::Stream`].
    pub fn into_stream(self) -> Option<crate::Stream> {
        match self {
            Self::Stream { stream } => Some(stream),
            _ => None,
        }
    }

    /// Unwrap a [`Response::Stream`] variant into a [`crate::Stream`].
    ///
    /// # Panics
    /// - If the variant is not a [`Response::Stream`].
    pub fn unwrap_stream(self) -> crate::Stream {
        self.into_stream()
            .expect("`Response` is not a `Stream` variant.")
    }

    /// Unwrap a [`Response::Message`] variant into a [`request::Message`]. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// # Panics
    /// - If the variant is not a [`Response::Message`].
    ///
    /// [`response::Message`]: self::Message
    pub fn unwrap_message(self) -> request::Message {
        self.into_message()
            .expect("`Response` is not a `Message` variant.")
    }

    /// Get the [`request::Message`] from a [`Response::Message`] variant. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// [`response::Message`]: self::Message
    pub fn message(&self) -> Option<&request::Message> {
        match self {
            Self::Message { message, .. } => Some(&message.message),
            _ => None,
        }
    }

    /// Convert a [`Response::Message`] variant into a [`request::Message`]. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// [`response::Message`]: self::Message
    pub fn into_message(self) -> Option<request::Message> {
        match self {
            Self::Message { message, .. } => Some(message.message),
            _ => None,
        }
    }

    /// Convert a [`Response::Message`] variant into a [`response::Message`].
    ///
    /// [`response::Message`]: self::Message
    pub fn into_response_message(self) -> Option<Message> {
        match self {
            Self::Message { message, .. } => Some(message),
            _ => None,
        }
    }

    /// Get the [`response::Message`] from a [`Response::Message`] variant.
    ///
    /// [`response::Message`]: self::Message
    pub fn response_message(&self) -> Option<&Message> {
        match self {
            Self::Message { message, .. } => Some(message),
            _ => None,
        }
    }

    /// Unwrap a [`Response::Message`] variant into a [`response::Message`].
    ///
    /// # Panics
    /// - If the variant is not a [`Response::Message`].
    ///
    /// [`response::Message`]: self::Message
    pub fn unwrap_response_message(self) -> Message {
        self.into_response_message()
            .expect("`Response` is not a `Message` variant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub const RESPONSE_JSON: &str = r#"{
  "content": [
    {
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "model": "claude-3-5-sonnet-20240620",
  "role": "assistant",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "input_tokens": 2095,
    "output_tokens": 503
  }
}"#;

    #[test]
    fn deserialize_response_message() {
        let message: Message = serde_json::from_str(RESPONSE_JSON).unwrap();
        assert_eq!(message.message.content.len(), 1);
        assert_eq!(message.id, "msg_013Zva2CMHLNnXjNJJKqJ2EF");
        assert_eq!(message.model, crate::Model::Sonnet35);
        assert!(matches!(message.stop_reason, Some(StopReason::EndTurn)));
        assert_eq!(message.stop_sequence, None);
        assert_eq!(message.usage.input_tokens, 2095);
        assert_eq!(message.usage.output_tokens, 503);
    }
}
