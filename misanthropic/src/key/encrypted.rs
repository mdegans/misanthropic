//! Encrypted [`Key`] management for Anthropic API keys.

// TODO: Remove this dependency for wasm32 and find an alternative. It's not a
// super idea to use this in a web app but wasm32 also has server use cases.
// This is the only thing that prevents this from building on wasm32.
use memsecurity::zeroize::Zeroizing;

/// The length of an Anthropic API key in bytes.
pub const LEN: usize = 108;

#[cfg(feature = "shuttle")]
impl From<Error> for shuttle_runtime::Error {
    fn from(e: Error) -> Self {
        shuttle_runtime::Error::Custom(e.into())
    }
}

/// `Error` type for [`Key`] operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The key length was invalid.
    #[error("Invalid key length: {actual} (expected {LEN})")]
    InvalidLength {
        /// The incorrect actual length of the key.
        actual: usize,
    },
    /// Vec<u8> was not valid UTF-8.
    #[error("Key is not valid UTF-8")]
    InvalidUtf8,
}

/// Stores an Anthropic API key securely. The API key is encrypted in memory.
/// The object features a [`Display`] implementation that can be used to write
/// out the key. **Be sure to zeroize whatever you write it to**. Prefer
/// [`Key::read`] if you want a return value that will automatically zeroize
/// the key on drop.
///
/// # Invariants
/// - The key is always valid UTF-8.
///
/// [`Display`]: std::fmt::Display
pub struct Key {
    // FIXME: `memsecurity` does not build on wasm32. Find a solution for web.
    // The `keyring` crate may work, but I'm likewise not sure if it builds on
    // wasm32. It's not listed in the platforms so likely not.
    mem: memsecurity::EncryptedMem,
}

impl std::fmt::Debug for Key {
    /// Write out the key in debug format. This is not recommended for production
    /// use as it will leak the key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key { <encrypted> }")
    }
}

impl Key {
    /// The length of an Anthropic API key in bytes.
    pub const LEN: usize = LEN;

    /// Read the key. The key is zeroized on drop.
    pub fn read(&self) -> memsecurity::ZeroizeBytes {
        // We want to upwrap if decryption fails because that indicates a
        // catastrophic failure of the encryption system.
        self.mem.decrypt().unwrap()
    }
}

impl TryFrom<String> for Key {
    type Error = Error;

    /// Create a new key from a string securely. The string is zeroized after
    /// conversion.
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::try_from(s.into_bytes())
    }
}

impl TryFrom<Vec<u8>> for Key {
    type Error = Error;

    /// Create a new key from a Vec<u8> securely. The Vec<u8> is zeroized after
    /// conversion.
    fn try_from(v: Vec<u8>) -> Result<Self, Self::Error> {
        Self::try_from(Zeroizing::new(v)) // take ownership and ensure zeroize on drop
    }
}

impl TryFrom<Zeroizing<Vec<u8>>> for Key {
    type Error = Error;

    /// Create a new key from a Zeroizing<Vec<u8>> securely. The Vec<u8> is
    /// zeroized after conversion.
    fn try_from(v: Zeroizing<Vec<u8>>) -> Result<Self, Self::Error> {
        if v.len() != LEN {
            let actual = v.len();
            return Err(Error::InvalidKeyLength { actual });
        }

        // We need to ensure valid UTF-8
        if std::str::from_utf8(&v).is_err() {
            return Err(Error::InvalidUtf8);
        }

        let mut mem = memsecurity::EncryptedMem::new();

        // Same reasoning as in `read` about unwrap here.
        mem.encrypt(&v).unwrap();

        Ok(Self { mem })
    }
}

impl TryFrom<Zeroizing<String>> for Key {
    type Error = Error;

    /// Create a new key from a Zeroizing<String> securely. The string is
    /// zeroized after conversion.
    fn try_from(s: Zeroizing<String>) -> Result<Self, Self::Error> {
        Self::try_from(s.into_bytes())
    }
}

impl std::fmt::Display for Key {
    /// Write out the key. Make sure to zeroize whatever you write it to if at
    /// all possible.
    ///
    /// Prefer [`Self::read`] if you want a return value that will automatically
    /// zeroize the key on drop.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Zeroized on drop
        let key = self.read();
        // Unwrap can never panic because a Key can only be created from a
        // String which is guaranteed to be valid UTF-8.
        let key_str = std::str::from_utf8(key.as_ref()).unwrap();
        write!(f, "{}", key_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: This is a real key but it's been disabled. As is warned in the
    // docs above, do not use a string literal for a real key. There is no
    // TryFrom<&'static str> for Key for this reason.
    const API_KEY: &str = "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";

    #[test]
    fn test_key() {
        let key = Key::try_from(API_KEY.to_string()).unwrap();
        let key_str = key.to_string();
        assert_eq!(key_str, API_KEY);
    }

    #[test]
    fn test_invalid_key_length() {
        let key = "test_key".to_string();
        let err = Key::try_from(key).unwrap_err();
        assert_eq!(err.to_string(), "Invalid key length: 8 (expected 108)");
    }
}
