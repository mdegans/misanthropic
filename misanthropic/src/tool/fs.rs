//! Shared filesystem helpers for the client-executed predefined tools whose
//! model commands operate on files — [`memory`](super::memory) (and other
//! file-oriented backends to come). The load-bearing, copy-hostile pieces
//! (path-jailing, line numbering, unique-match finding, line insertion) live
//! here so each backend keeps only its own tool-specific error strings and
//! directory presentation.
//!
//! Everything here is pure (no I/O, no async): each function takes an
//! already-read string or a path and returns plain data, leaving the typed
//! errors and trained, model-facing strings to the caller.

use std::path::{Component, Path, PathBuf};

/// Map a model-supplied `path` onto `root`, returning `None` if it would escape
/// the jail (via `..`, an absolute path, or a virtual-root lookalike).
///
/// `virtual_root` is the wire root the model addresses files under when the
/// tool uses one (the memory tool's `/memories`); pass `None` when paths are
/// taken as plain root-relative (the text editor). A matched `virtual_root` is
/// stripped; an absolute path that is *not* under it is rejected.
pub(crate) fn resolve_jailed(
    root: &Path,
    path: &str,
    virtual_root: Option<&str>,
) -> Option<PathBuf> {
    let rel: &str = match virtual_root {
        Some(vroot) => match path.strip_prefix(vroot) {
            // Exactly the root, e.g. "/memories".
            Some("") => "",
            // "/memories/foo" → "foo".
            Some(rest) if rest.starts_with('/') => rest.trim_start_matches('/'),
            // A "/memoriesX" lookalike (or any other absolute path): escape.
            _ if path.starts_with('/') => return None,
            // A relative path is taken as-is, under the root.
            None => path,
            // strip_prefix matched yet `path` is absolute — caught just above.
            Some(_) => return None,
        },
        // No virtual root: paths are root-relative; reject absolutes.
        None if path.starts_with('/') => return None,
        None => path,
    };
    // Any non-`Normal` component (`..`, a root, a drive prefix) could climb out
    // of `root`, so refuse before joining.
    Path::new(rel)
        .components()
        .all(|c| matches!(c, Component::Normal(_) | Component::CurDir))
        .then(|| root.join(rel))
}

/// Render `content` with 6-wide, right-aligned, 1-indexed line numbers and a
/// tab separator — the coordinate system the model writes `insert` and
/// `str_replace` against. An optional inclusive 1-indexed `[start, end]` range
/// trims the output (clamped so `start >= 1`).
pub(crate) fn with_line_numbers(
    content: &str,
    range: Option<[u64; 2]>,
) -> String {
    let (start, end) = match range {
        Some([start, end]) => (start.max(1), end),
        None => (1, u64::MAX),
    };
    content
        .lines()
        .enumerate()
        .map(|(i, line)| (i as u64 + 1, line))
        .filter(|(n, _)| *n >= start && *n <= end)
        .map(|(n, line)| format!("{n:6}\t{line}\n"))
        .collect()
}

/// The 1-indexed line numbers at which `needle` occurs verbatim in `content`.
/// None (`[]`), unique (one entry), or ambiguous (several) drives the caller's
/// `str_replace` branch and its error message.
pub(crate) fn match_lines(content: &str, needle: &str) -> Vec<usize> {
    content
        .match_indices(needle)
        .map(|(offset, _)| content[..offset].matches('\n').count() + 1)
        .collect()
}

/// Insert `text` after (0-indexed-from-top) line `insert_line` (`0` =
/// beginning), returning the rewritten content, or `None` if `insert_line`
/// exceeds the file's line count. A single trailing newline on `text` is
/// dropped (the join re-adds breaks) and the result ends with a newline.
pub(crate) fn insert_after(
    content: &str,
    insert_line: u64,
    text: &str,
) -> Option<String> {
    let mut lines: Vec<&str> = content.lines().collect();
    if insert_line as usize > lines.len() {
        return None;
    }
    let text = text.strip_suffix('\n').unwrap_or(text);
    lines.insert(insert_line as usize, text);
    let mut out = lines.join("\n");
    out.push('\n');
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_jailed_virtual_root() {
        let root = Path::new("/srv/notes");
        // The bare root and paths under it map onto `root`.
        assert_eq!(
            resolve_jailed(root, "/memories", Some("/memories")),
            Some(root.to_path_buf())
        );
        assert_eq!(
            resolve_jailed(root, "/memories/a.md", Some("/memories")),
            Some(root.join("a.md"))
        );
        // A relative path is taken as-is under the root.
        assert_eq!(
            resolve_jailed(root, "a.md", Some("/memories")),
            Some(root.join("a.md"))
        );
        // Escapes: traversal, a foreign absolute path, a root lookalike.
        for evil in ["/memories/../escape.md", "/etc/passwd", "/memoriesX/a"] {
            assert_eq!(
                resolve_jailed(root, evil, Some("/memories")),
                None,
                "{evil}"
            );
        }
    }

    #[test]
    fn resolve_jailed_root_relative() {
        let root = Path::new("/work");
        assert_eq!(
            resolve_jailed(root, "primes.py", None),
            Some(root.join("primes.py"))
        );
        assert_eq!(
            resolve_jailed(root, "src/lib.rs", None),
            Some(root.join("src/lib.rs"))
        );
        // Absolutes and traversal escape the jail.
        for evil in ["/etc/passwd", "../../x", "a/../../b"] {
            assert_eq!(resolve_jailed(root, evil, None), None, "{evil}");
        }
    }

    #[test]
    fn line_numbers_and_range() {
        let body = "alpha\nbeta\ngamma\n";
        assert_eq!(
            with_line_numbers(body, None),
            "     1\talpha\n     2\tbeta\n     3\tgamma\n"
        );
        assert_eq!(with_line_numbers(body, Some([2, 2])), "     2\tbeta\n");
        // A start below 1 is clamped.
        assert_eq!(with_line_numbers(body, Some([0, 1])), "     1\talpha\n");
    }

    #[test]
    fn match_lines_counts_occurrences() {
        assert_eq!(match_lines("dup\ndup\nx\n", "dup"), vec![1, 2]);
        assert_eq!(match_lines("a\nb\nc\n", "b"), vec![2]);
        assert!(match_lines("a\nb\n", "zzz").is_empty());
    }

    #[test]
    fn insert_after_positions_and_bounds() {
        assert_eq!(
            insert_after("a\nb\n", 0, "top\n"),
            Some("top\na\nb\n".to_string())
        );
        assert_eq!(
            insert_after("a\nb\n", 2, "end"),
            Some("a\nb\nend\n".to_string())
        );
        // Out of range.
        assert_eq!(insert_after("a\nb\n", 3, "x"), None);
    }
}
