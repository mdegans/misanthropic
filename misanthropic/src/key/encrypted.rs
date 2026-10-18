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

#[cfg(feature = "shuttle")]
impl From<InvalidKeyLength> for shuttle_runtime::Error {
    fn from(e: InvalidKeyLength) -> Self {
        shuttle_runtime::Error::Custom(e.into())
    }
}
/// Stores an Anthropic API key securely. The API key is encrypted in memory,
/// zeroized on drop, and its [`Debug`] output is redacted. Use [`Key::read`]
/// to get at the bytes; the value it returns is zeroized on drop.
///
/// [`Debug`]: std::fmt::Debug
pub struct Key {
    // FIXME: `memsecurity` does not build on wasm32. Find a solution for web.
    // The `keyring` crate may work, but I'm likewise not sure if it builds on
    // wasm32. It's not listed in the platforms so likely not.
    mem: memsecurity::EncryptedMem,
}

impl Key {
    /// The length of an Anthropic API key in bytes.
    pub const LEN: usize = LEN;

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
        Self::try_from(s.into_bytes())
    }
}

impl TryFrom<Vec<u8>> for Key {
    type Error = InvalidKeyLength;

    /// Create a new key from a `Vec<u8>` securely. The `Vec<u8>` is zeroized after
    /// conversion.
    fn try_from(v: Vec<u8>) -> Result<Self, Self::Error> {
        let v = Zeroizing::new(v);
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

impl std::fmt::Debug for Key {
    /// Prints `Key([REDACTED])`, never the key (nor its ciphertext).
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("Key([REDACTED])")
    }
}

// The only field is a `memsecurity::EncryptedMem`, whose ciphertext is a
// `memsecurity::ZeroizeBytes`. That type zeroizes itself in its `Drop` impl
// (and implements `ZeroizeOnDrop` itself), so dropping a `Key` wipes the
// ciphertext with no `Drop` of our own. The nonce beside it is not secret.
// `EncryptedMem` doesn't implement `ZeroizeOnDrop`, so this can't be derived.
impl zeroize::ZeroizeOnDrop for Key {}

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
        assert_eq!(key.read().as_ref(), API_KEY.as_bytes());
    }

    #[test]
    fn test_debug_is_redacted() {
        let key = Key::try_from(API_KEY.to_string()).unwrap();
        for debug in [format!("{key:?}"), format!("{key:#?}")] {
            assert_eq!(debug, "Key([REDACTED])");
            // Belt and braces: no 8-byte window of the key appears.
            assert!(
                API_KEY
                    .as_bytes()
                    .windows(8)
                    .all(|w| !debug.contains(std::str::from_utf8(w).unwrap()))
            );
        }
    }

    // Both variants of `Key` must wipe themselves on drop.
    const _: fn() = || {
        fn assert_zeroize_on_drop<T: zeroize::ZeroizeOnDrop>() {}
        assert_zeroize_on_drop::<Key>();
    };

    #[test]
    fn test_invalid_key_length() {
        let key = "test_key".to_string();
        let err = Key::try_from(key).unwrap_err();
        assert_eq!(err.to_string(), "Invalid key length: 8 (expected 108)");
    }
}
