//! Encrypted [`Key`] management for Anthropic API keys.

// TODO: Remove this dependency for wasm32 and find an alternative. It's not a
// super idea to use this in a web app but wasm32 also has server use cases.
// This is the only thing that prevents this from building on wasm32.
use memsecurity::zeroize::Zeroizing;

/// The length of an Anthropic API key in bytes.
pub const LEN: usize = 108;

/// Error for when a key is not 108 bytes.
#[derive(Debug, thiserror::Error)]
#[error("Invalid key length: {actual} (expected {LEN})")]
pub struct InvalidKeyLength {
    /// The incorrect actual length of the key.
    pub actual: usize,
}

/// Stores an Anthropic API key securely. The API key is encrypted in memory.
/// The object features a [`Display`] implementation that can be used to write
/// out the key. **Be sure to zeroize whatever you write it to**. Prefer
/// [`Key::read`] if you want a return value that will automatically zeroize
/// the key on drop.
///
/// [`Display`]: std::fmt::Display
#[derive(Debug)]
pub struct Key {
    // FIXME: `memsecurity` does not build on wasm32. Find a solution for web.
    // The `keyring` crate may work, but I'm likewise not sure if it builds on
    // wasm32. It's not listed in the platforms so likely not.
    mem: memsecurity::EncryptedMem,
}

impl Key {
    /// Read the key. The key is zeroized on drop.
    pub fn read(&self) -> memsecurity::ZeroizeBytes {
        self.mem.decrypt().unwrap()
    }
}

impl TryFrom<String> for Key {
    type Error = InvalidKeyLength;

    /// Create a new key from a string securely. The string is zeroized after
    /// conversion.
    fn try_from(s: String) -> Result<Self, Self::Error> {
        // This just unwraps the internal Vec<u8> so the data can still be
        // zeroized.
        let v = Zeroizing::new(s.into_bytes());
        if v.len() != LEN {
            let actual = v.len();
            return Err(InvalidKeyLength { actual });
        }

        let mut mem = memsecurity::EncryptedMem::new();

        // Unwrap is desirable here because this can only fail if encryption
        // is broken, which is a catastrophic failure.
        mem.encrypt(&v).unwrap();

        Ok(Self { mem })
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
