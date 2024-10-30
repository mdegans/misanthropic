//! [`Key`] management for Anthropic API keys.
use zeroize::{ZeroizeOnDrop, Zeroizing};

/// The length of an Anthropic API key in bytes.
pub const LEN: usize = 108;

/// Type alias for an Anthropic API key.
type Arr = [u8; LEN];

/// Error for when a key is not 108 bytes.
#[derive(Debug, thiserror::Error)]
#[error("Invalid key length: {actual} (expected {LEN})")]
pub struct InvalidKeyLength {
    /// The incorrect actual length of the key.
    pub actual: usize,
}

/// Stores an Anthropic API key securely. The object features a [`Display`]
/// implementation that can be used to write out the key. **Be sure to zeroize
/// whatever you write it to**. The key is zeroized on drop.
///
/// [`Display`]: std::fmt::Display
#[derive(Debug, ZeroizeOnDrop)]
pub struct Key {
    mem: Arr,
}

impl Key {
    /// Read the key. The key is zeroized on drop.
    // We can't return a &str becuase the other implementation of Key::read
    // returns a memsecurity::ZeroizeBytes, and we can't return a reference to
    // that because it's a temporary value, so this returns a slice instead,
    // which has more or less the same public API.
    pub fn read(&self) -> &[u8] {
        &self.mem
    }
}

impl TryFrom<String> for Key {
    type Error = InvalidKeyLength;

    /// Create a new key from a string securely. The string is zeroized after
    /// conversion.
    fn try_from(s: String) -> Result<Self, Self::Error> {
        let v = Zeroizing::new(s.into_bytes());
        let mut arr: Arr = [0; LEN];
        if v.len() != LEN {
            let actual = v.len();
            return Err(InvalidKeyLength { actual });
        }

        arr.copy_from_slice(&v);
        Ok(Key { mem: arr })
    }
}

impl std::fmt::Display for Key {
    /// Write out the key. Make sure to zeroize whatever you write it to if at
    /// all possible.
    ///
    /// Prefer [`Self::read`] if you want a return value that will automatically
    /// zeroize the key on drop.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Unwrap can never panic because a Key can only be created from String
        // whic is guaranteed to be valid UTF-8.
        let key_str = std::str::from_utf8(self.read()).unwrap();
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
