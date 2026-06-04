#[cold]
#[inline(always)]
pub(crate) fn cold_path() {}

/// Assert an exact serde round-trip of a wire fixture, and return the parsed
/// value for further assertions.
///
/// `fixture` is raw JSON captured from the API (see `test/data/README.md` for
/// the capture discipline). This deserializes it into `T`, re-serializes, and
/// asserts the result equals the captured input at the value level.
///
/// No response type uses `#[serde(deny_unknown_fields)]` — an undocumented,
/// renamed, or mis-tagged wire field is therefore *silently dropped* on the way
/// in. This assertion is the offline guard that catches that: the
/// re-serialized value would be missing the field (or carry a wrong tag) and
/// the comparison fails loudly in `just test`, with no API call.
#[cfg(test)]
pub(crate) fn roundtrip<T>(fixture: &str) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let captured: serde_json::Value =
        serde_json::from_str(fixture).expect("fixture is valid JSON");
    let parsed: T = serde_json::from_value(captured.clone())
        .expect("fixture deserializes into the target type");
    assert_eq!(
        serde_json::to_value(&parsed).expect("re-serializes to JSON"),
        captured,
        "wire round-trip mismatch: a field was dropped, renamed, added, or \
         mis-tagged. No response type uses deny_unknown_fields, so this \
         assertion is the only offline guard against wire drift.",
    );
    parsed
}

#[cfg(all(test, feature = "client"))]
pub(crate) const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

// Load the API key from the `api.key` file in the crate root.
#[cfg(all(test, feature = "client"))]
pub(crate) async fn load_api_key() -> String {
    use std::path::Path;

    let path = Path::new(CRATE_ROOT).join("api.key");
    tokio::fs::read_to_string(path)
        .await
        .ok()
        .unwrap()
        .trim()
        .to_string()
}
