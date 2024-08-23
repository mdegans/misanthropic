//! [`Key`] management for Anthropic API keys.

// TODO: Remove this dependency for wasm32 and find an alternative. It's not a
// super idea to use this in a web app but wasm32 also has server use cases.
// This is the only thing that prevents this from building on wasm32.
use memsecurity::zeroize::Zeroize;

/// The length of an Anthropic API key in bytes.
pub const LEN: usize = 108;

/// Type alias for an Anthropic API key.
pub type Arr = [u8; LEN];

/// Error for when a key is not the correct [`key::LEN`].
///
/// [`key::LEN`]: LEN
#[derive(Debug, thiserror::Error)]
#[error("Invalid key length: {0} (expected {LEN})")]
pub struct InvalidKeyLength(usize);

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
        Self::try_from(s.into_bytes())
    }
}

impl TryFrom<Vec<u8>> for Key {
    type Error = InvalidKeyLength;

    /// Create a new key from a byte vector securely. The vector is zeroized
    /// after conversion.
    fn try_from(mut v: Vec<u8>) -> Result<Self, Self::Error> {
        let mut arr: Arr = [0; LEN];
        if v.len() != LEN {
            v.zeroize();
            return Err(InvalidKeyLength(v.len()));
        }

        arr.copy_from_slice(&v);
        let ret = Ok(Self::from(arr));

        v.zeroize();

        ret
    }
}

impl From<Arr> for Key {
    /// Create a new key from a [`key::Arr`] byte array securely. The array is
    /// zeroized.
    ///
    /// [`key::Arr`]: Arr
    fn from(mut arr: Arr) -> Self {
        let mut mem = memsecurity::EncryptedMem::new();

        // Unwrap is desirable here because this can only fail if encryption
        // is broken, which is a catastrophic failure.
        mem.encrypt(&arr).unwrap();
        arr.zeroize();

        Self { mem }
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
        // Unwrap can never panic because a Key can only be created from a str
        // or String which are guaranteed to be valid UTF-8.
        let key_str = std::str::from_utf8(key.as_ref()).unwrap();
        write!(f, "{}", key_str)
    }
}
