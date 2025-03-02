#[cold]
#[inline(always)]
pub(crate) fn cold_path() {}

#[cfg(test)]
pub(crate) const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

// Load the API key from the `api.key` file in the crate root.
#[cfg(test)]
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
