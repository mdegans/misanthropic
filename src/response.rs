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
pub enum Response<'a> {
    /// Single [`response::Message`] from the API.
    ///
    /// [`response::Message`]: Message
    Message {
        #[allow(missing_docs)]
        message: self::Message<'a>,
    },
    /// [`Stream`] of [`Event`]s (message delta, etc.).
    ///
    /// [`Stream`]: crate::Stream
    /// [`Event`]: crate::stream::Event
    Stream {
        #[allow(missing_docs)]
        stream: crate::Stream<'a>,
    },
}

impl<'a> Response<'a> {
    /// Convert a [`Response::Stream`] variant into a [`crate::Stream`].
    pub fn into_stream(self) -> Option<crate::Stream<'a>> {
        match self {
            Self::Stream { stream } => Some(stream),
            _ => None,
        }
    }

    /// Unwrap a [`Response::Stream`] variant into a [`crate::Stream`].
    ///
    /// # Panics
    /// - If the variant is not a [`Response::Stream`].
    pub fn unwrap_stream(self) -> crate::Stream<'a> {
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
    pub fn unwrap_message(self) -> request::Message<'a> {
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
    pub fn into_message(self) -> Option<request::Message<'a>> {
        match self {
            Self::Message { message, .. } => Some(message.message),
            _ => None,
        }
    }

    /// Convert a [`Response::Message`] variant into a [`response::Message`].
    ///
    /// [`response::Message`]: self::Message
    pub fn into_response_message(self) -> Option<Message<'a>> {
        match self {
            Self::Message { message, .. } => Some(message),
            _ => None,
        }
    }

    /// Get the [`response::Message`] from a [`Response::Message`] variant.
    ///
    /// [`response::Message`]: self::Message
    pub fn response_message(&self) -> Option<&Message<'a>> {
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
    pub fn unwrap_response_message(self) -> Message<'a> {
        self.into_response_message()
            .expect("`Response` is not a `Message` variant.")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::borrow::Cow;

    const TEST_ID: &str = "test_id";

    const CONTENT: &str = "Hello, world!";

    const RESPONSE: Response = Response::Message {
        message: Message {
            id: Cow::Borrowed(TEST_ID),
            message: request::Message {
                role: request::message::Role::User,
                content: request::message::Content::SinglePart(Cow::Borrowed(
                    CONTENT,
                )),
            },
            model: crate::Model::Sonnet35,
            stop_reason: None,
            stop_sequence: None,
            usage: Usage {
                input_tokens: 1,
                #[cfg(feature = "prompt-caching")]
                cache_creation_input_tokens: Some(2),
                #[cfg(feature = "prompt-caching")]
                cache_read_input_tokens: Some(3),
                output_tokens: 4,
            },
        },
    };

    #[test]
    fn test_into_stream() {
        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.into_stream().is_some());
        assert!(RESPONSE.into_stream().is_none());
    }

    #[test]
    fn test_unwrap_stream() {
        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        let _stream = response.unwrap_stream();
    }

    #[test]
    #[should_panic]
    fn test_unwrap_stream_panics() {
        let _panic = RESPONSE.unwrap_stream();
    }

    #[test]
    fn test_unwrap_message() {
        assert_eq!(
            RESPONSE.unwrap_message().content.to_string(),
            "Hello, world!"
        );
    }

    #[test]
    #[should_panic]
    fn test_unwrap_message_panics() {
        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        let _panic = response.unwrap_message();
    }

    #[test]
    fn test_message() {
        assert_eq!(
            RESPONSE.message().unwrap().content.to_string(),
            "Hello, world!"
        );

        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.message().is_none());
    }

    #[test]
    fn test_into_message() {
        assert_eq!(
            RESPONSE.into_message().unwrap().content.to_string(),
            "Hello, world!"
        );

        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.into_message().is_none());
    }

    #[test]
    fn test_into_response_message() {
        assert_eq!(
            RESPONSE
                .into_response_message()
                .unwrap()
                .message
                .content
                .to_string(),
            "Hello, world!"
        );

        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.into_response_message().is_none());
    }

    #[test]
    fn test_response_message() {
        assert_eq!(
            RESPONSE
                .response_message()
                .unwrap()
                .message
                .content
                .to_string(),
            "Hello, world!"
        );

        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.response_message().is_none());
    }

    #[test]
    fn test_unwrap_response_message() {
        assert_eq!(
            RESPONSE
                .unwrap_response_message()
                .message
                .content
                .to_string(),
            "Hello, world!"
        );
    }

    #[test]
    #[should_panic]
    fn test_unwrap_response_message_panics() {
        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        let _panic = response.unwrap_response_message();
    }
}
