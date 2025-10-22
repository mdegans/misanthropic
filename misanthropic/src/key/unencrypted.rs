//! Unencrypted [`Key`] management for Anthropic API keys.
use zeroize::Zeroizing;

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

#[cfg(feature = "shuttle")]
impl From<InvalidKeyLength> for shuttle_runtime::Error {
    fn from(e: InvalidKeyLength) -> Self {
        shuttle_runtime::Error::Custom(e.into())
    }
}

/// Stores an Anthropic API key securely. The object features a [`Display`]
/// implementation that can be used to write out the key. **Be sure to zeroize
/// whatever you write it to**. The key is zeroized on drop.
///
/// # Invariants
/// - The key is always valid UTF-8.
/// [`Display`]: std::fmt::Display
#[repr(transparent)] // why not
pub struct Key {
    mem: Zeroizing<Arr>,
}

impl std::fmt::Debug for Key {
    /// Write out the key in debug format. This is not recommended for production
    /// use as it will leak the key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Key { <unencrypted> }")
    }
}

impl Key {
    /// The length of an Anthropic API key in bytes.
    pub const LEN: usize = LEN;

    /// Read a reference to the key bytes. Guaranteed to be UTF-8.
    pub fn read(&self) -> &[u8] {
        &self.mem.as_slice()
    }
}

/// Errors that can occur when creating a [`Key`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The key length was invalid.
    #[error(transparent)]
    InvalidLength(#[from] InvalidKeyLength),
    /// Vec<u8> was not valid UTF-8.
    #[error("Key is not valid UTF-8")]
    InvalidUtf8,
}

impl TryFrom<String> for Key {
    type Error = Error;

    /// Create a new key from a String securely. The string is consumed and
    /// zeroized on drop.
    fn try_from(s: String) -> Result<Self, Self::Error> {
        Self::try_from(Zeroizing::new(s))
    }
}

impl TryFrom<Zeroizing<String>> for Key {
    type Error = Error;

    /// Create a new key from a Zeroizing<String> securely. The string is
    /// zeroized on drop.
    fn try_from(s: Zeroizing<String>) -> Result<Self, Self::Error> {
        // Directly convert to Vec<u8> with `into_bytes` is not possible because
        // it moves out of a deref. So convert to String first and then move
        // into a zeroizing Vec<u8>.
        Self::try_from(s.to_string().into_bytes())
    }
}

impl TryFrom<Vec<u8>> for Key {
    type Error = Error;

    /// Create a new key from a Vec<u8> securely. The Vec<u8> is zeroized after
    /// conversion.
    fn try_from(v: Vec<u8>) -> Result<Self, Self::Error> {
        let v = Zeroizing::new(v); // take ownership and ensure zeroize on drop
        // Check UTF-8 validity
        if std::str::from_utf8(&v).is_err() {
            return Err(Error::InvalidUtf8);
        }
        // Check length
        if v.len() != LEN {
            let actual = v.len();
            return Err(Error::InvalidLength(InvalidKeyLength { actual }));
        }
        // v is valid, copy into array
        let mut arr: Zeroizing<Arr> = Zeroizing::new([0; LEN]);
        arr.copy_from_slice(&v);
        Ok(Key { mem: arr }) // v is zeroized on drop
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
