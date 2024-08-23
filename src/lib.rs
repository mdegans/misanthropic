#![warn(missing_docs)]
//! `misanthropic` is a crate providing ergonomic access to the [Anthropic
//! Messages API].
//!
//! To get started, create a [`Client`] with your API key and use it to send
//! [`Request`]s to the API. The API will return a [`Response`] with the
//! [`response::Message`] or a [`Stream`] of [`stream::Event`]s.
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
//!
//! See the `examples` directory for more detailed usage.
// Because I can't get the example scraping to work. TODO: Fix this.

pub mod key;
pub use key::Key;

pub mod client;
pub use client::Client;

pub mod model;
pub use model::Model;

pub mod request;
pub use request::Request;

pub mod stream;
pub use stream::Stream;

pub mod tool;
pub use tool::Tool;

pub mod response;
pub use response::Response;

/// Re-exports of commonly used crates to avoid version conflicts and reduce
/// dependency bloat.
pub mod exports {
    pub use base64;
    pub use eventsource_stream;
    pub use futures;
    #[cfg(feature = "image")]
    pub use image;
    #[cfg(feature = "log")]
    pub use log;
    pub use memsecurity;
    pub use reqwest;
    pub use serde;
    pub use serde_json;
}

/// Re-export of `serde_json::json!` for convenience because this is used
/// frequently.
pub use exports::serde_json::json;