//! `serde_json::json!` is banned in library code: untyped JSON hides wire drift
//! and typos that a `Serialize` type catches at compile time. Tests (a file's
//! trailing `#[cfg(test)]` module, `tests/`, examples) may use it to build
//! wire-shaped inputs.
//!
//! This is a source scan rather than clippy's `disallowed_macros`, which
//! also fires on `#[derive(JsonSchema)]` (schemars expands to `json!`) with
//! no way to allow a derive's generated impl short of a module-wide allow.

use std::path::{Path, PathBuf};

/// Library source roots, relative to this crate. Siblings are skipped when
/// absent (e.g. testing a published tarball).
const ROOTS: &[&str] = &["src", "../misanthropic-derive/src", "../bashd/src"];

fn rust_files(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .flat_map(|path| match path.is_dir() {
            true => rust_files(&path),
            false if path.extension().is_some_and(|e| e == "rs") => {
                vec![path]
            }
            false => vec![],
        })
        .collect()
}

/// `json!` uses in `source` above its test module, as 1-indexed lines.
fn library_uses(source: &str) -> Vec<usize> {
    let lines: Vec<&str> = source.lines().collect();
    let test_module = lines
        .windows(2)
        .position(|pair| {
            let (attr, item) = (pair[0].trim(), pair[1].trim_start());
            attr.starts_with("#[cfg(")
                && attr.contains("test")
                && item.contains("mod ")
                && item.ends_with('{')
        })
        .unwrap_or(lines.len());

    lines[..test_module]
        .iter()
        .enumerate()
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
        .filter(|(_, line)| {
            line.match_indices("json!").any(|(i, _)| {
                // Not part of a longer name (e.g. `json_schema!`).
                !line[..i]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
            })
        })
        .map(|(i, _)| i + 1)
        .collect()
}

#[test]
fn library_code_does_not_use_json_macro() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let offenders: Vec<String> = ROOTS
        .iter()
        .map(|root| crate_dir.join(root))
        .filter(|root| root.is_dir())
        .flat_map(|root| rust_files(&root))
        .flat_map(|file| {
            let source = std::fs::read_to_string(&file).unwrap();
            library_uses(&source)
                .into_iter()
                .map(move |line| format!("{}:{line}", file.display()))
        })
        .collect();

    assert!(
        offenders.is_empty(),
        "`json!` in library code; build a typed `Serialize` value instead:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn scanner_finds_only_library_uses() {
    let source = r#"
fn wire() -> Value {
    json!({ "a": 1 })
}
// json!({ "commented": true })
/// `json!` in a doc example
fn schema() { json_schema!({}) }
fn other() { serde_json::json!(null) }

#[cfg(test)]
mod tests {
    fn fixture() { json!({}) }
}
"#;
    assert_eq!(library_uses(source), [3, 8]);
}
