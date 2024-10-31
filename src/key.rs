//! [`Key`] is a wrapper around an Anthropic API key.

#[cfg(feature = "memsecurity")]
mod encrypted;
#[cfg(feature = "memsecurity")]
pub use encrypted::{InvalidKeyLength, Key};
#[cfg(not(feature = "memsecurity"))]
mod unencrypted;
#[cfg(not(feature = "memsecurity"))]
pub use unencrypted::{InvalidKeyLength, Key};
