//! [`Response`] types for the [Anthropic Messages API].
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use derive_more::derive::IsVariant;

pub(crate) mod message;
pub use message::{JsonError, Message, StopReason, Usage};

use crate::prompt;

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

    /// Unwrap a [`Response::Message`] variant into a [`prompt::message`]. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// # Panics
    /// - If the variant is not a [`Response::Message`].
    ///
    /// [`response::Message`]: self::Message
    pub fn unwrap_message(self) -> prompt::Message {
        self.into_message()
            .expect("`Response` is not a `Message` variant.")
    }

    /// Get the [`prompt::message`] from a [`Response::Message`] variant. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// [`response::Message`]: self::Message
    pub fn message(&self) -> Option<&prompt::Message> {
        match self {
            Self::Message { message, .. } => Some(&message.inner),
            _ => None,
        }
    }

    /// Convert a [`Response::Message`] variant into a [`prompt::message`]. Use
    /// this if you don't care about [`response::Message`] metadata.
    ///
    /// [`response::Message`]: self::Message
    pub fn into_message(self) -> Option<prompt::Message> {
        match self {
            Self::Message { message, .. } => Some(message.inner.into()),
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

    const TEST_ID: &str = "test_id";

    const CONTENT: &str = "Hello, world!";

    fn create_response() -> Response {
        Response::Message {
            message: Message {
                id: TEST_ID.into(),
                inner: prompt::AssistantMessage {
                    inner: prompt::Message {
                        role: prompt::message::Role::Assistant,
                        content: CONTENT.into(),
                    },
                },
                model: crate::Id::Sonnet35.into(),
                stop_reason: None,
                stop_sequence: None,
                usage: Usage {
                    input_tokens: 1,
                    cache_creation_input_tokens: Some(2),
                    cache_read_input_tokens: Some(3),
                    output_tokens: 4,
                    server_tool_use: None,
                },
                container: None,
            },
        }
    }

    #[test]
    fn test_into_stream() {
        let mock_stream = crate::stream::tests::mock_stream(include_str!(
            "../test/data/sse.stream.txt"
        ));

        let response = Response::Stream {
            stream: mock_stream,
        };

        assert!(response.into_stream().is_some());
        assert!(create_response().into_stream().is_none());
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
        let _panic = create_response().unwrap_stream();
    }

    #[test]
    fn test_unwrap_message() {
        assert_eq!(
            create_response().unwrap_message().content.to_string(),
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
            create_response().message().unwrap().content.to_string(),
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
            create_response()
                .into_message()
                .unwrap()
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

        assert!(response.into_message().is_none());
    }

    #[test]
    fn test_into_response_message() {
        assert_eq!(
            create_response()
                .into_response_message()
                .unwrap()
                .inner
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
            create_response()
                .response_message()
                .unwrap()
                .inner
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
            create_response()
                .unwrap_response_message()
                .inner
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

    #[test]
    fn test_usage_serde() {
        const A: &str = r#"{"output_tokens":89}"#;

        let deserialized: Usage = serde_json::from_str(A).unwrap();

        assert_eq!(deserialized.input_tokens, 0);
        assert_eq!(deserialized.output_tokens, 89);

        const B: &str = r#"{"input_tokens":1,"output_tokens":2}"#;

        let deserialized: Usage = serde_json::from_str(B).unwrap();

        assert_eq!(deserialized.input_tokens, 1);
        assert_eq!(deserialized.output_tokens, 2);
    }

    #[test]
    fn test_usage_add() {
        use crate::response::message::ServerToolUsage;

        let mut a = Usage {
            input_tokens: 1,
            cache_creation_input_tokens: Some(2),
            cache_read_input_tokens: Some(3),
            output_tokens: 4,
            server_tool_use: Some(ServerToolUsage {
                web_search_requests: 1,
                web_fetch_requests: 100,
                tool_search_requests: 10,
            }),
        };

        let b = Usage {
            input_tokens: 5,
            cache_creation_input_tokens: Some(6),
            cache_read_input_tokens: Some(7),
            output_tokens: 8,
            server_tool_use: Some(ServerToolUsage {
                web_search_requests: 2,
                web_fetch_requests: 200,
                tool_search_requests: 20,
            }),
        };

        a += b;

        assert_eq!(a.input_tokens, 6);
        assert_eq!(a.cache_creation_input_tokens, Some(8));
        assert_eq!(a.cache_read_input_tokens, Some(10));
        assert_eq!(a.output_tokens, 12);
        assert_eq!(
            a.server_tool_use,
            Some(ServerToolUsage {
                web_search_requests: 3,
                web_fetch_requests: 300,
                tool_search_requests: 30,
            })
        );
    }
}
