//! Client-side execution of the [memory tool] ([`ServerMethodDef::Memory`]).
//!
//! The memory tool is *predefined* (you add it by versioned name via
//! [`Memory::latest`], no schema of your own) but *client-executed*: the model
//! emits an ordinary [`Use`] whose [`input`](Use::input) is one of a small set
//! of file operations, and you run it against storage you control. This module
//! provides the typed [`Command`](crate::tool::memory::Command) those inputs deserialize into and
//! [`FsMemoryBackend`], a filesystem-backed reference executor jailed to a
//! single directory.
//!
//! ```no_run
//! # #[cfg(feature = "memory-fs")] // backend is feature-gated; doc isn't
//! # async fn f() -> Result<(), Box<dyn std::error::Error>> {
//! use misanthropic::{Prompt, tool::{Memory, Tool, memory::FsMemoryBackend}};
//!
//! let mut backend = FsMemoryBackend::new("./memories").await?;
//! let mut prompt = Prompt::default().add_tool(Memory::latest());
//! // ... when an assistant `tool_use` named "memory" arrives as `call`:
//! # let call: misanthropic::tool::Use = todo!();
//! let result = backend.call(call).await; // typed dispatch + canonical reply
//! # let _ = (result, &mut prompt); Ok(())
//! # }
//! ```
//!
//! [memory tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
//! [`ServerMethodDef::Memory`]: crate::tool::ServerMethodDef::Memory
//! [`Memory::latest`]: crate::tool::Memory::latest
//! [`Use`]: crate::tool::Use

use std::borrow::Cow;
#[cfg(feature = "memory-fs")]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[cfg(feature = "memory-fs")]
use super::{MethodDef, ServerMethodDef, Tool, Use, fs};

/// The wire root the model addresses memory files under (`/memories`), which a
/// backend maps onto its real directory. Only the (optional) filesystem backend
/// uses it in code; the typed `Command` layer just references it in docs.
#[cfg_attr(not(feature = "memory-fs"), allow(dead_code))]
const MEMORY_ROOT: &str = "/memories";

/// A typed memory-tool command, deserialized from a memory [`Use`]'s
/// [`input`](Use::input).
///
/// A known/unknown union (à la [`model::Model`]/[`Caller`]): commands this crate
/// has typed support for land in [`Known`]; anything else (e.g. a newer
/// memory-tool version) round-trips through [`Unknown`](Command::Unknown)
/// rather than failing to deserialize a live response.
///
/// [`model::Model`]: crate::model::Model
/// [`Caller`]: crate::tool::Caller
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(untagged)]
pub enum Command {
    /// A command with typed support.
    Known(Known),
    /// An unrecognized command, kept verbatim so it round-trips. Handle it
    /// yourself or return an error to the model.
    Unknown {
        /// The raw `command` discriminant.
        command: Cow<'static, str>,
        /// The command's remaining fields.
        #[serde(flatten)]
        rest: serde_json::Map<String, serde_json::Value>,
    },
}

/// The memory commands this crate models, internally tagged on `command`. See
/// the [memory tool] reference for the full semantics of each.
///
/// [memory tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Known {
    /// List a directory (up to two levels) or show a file's contents.
    View {
        /// Path under `MEMORY_ROOT`, e.g. `/memories/notes.md`.
        path: Cow<'static, str>,
        /// Optional inclusive 1-indexed `[start, end]` line range for files.
        #[serde(skip_serializing_if = "Option::is_none")]
        view_range: Option<[u64; 2]>,
    },
    /// Create a new file (errors if it already exists).
    Create {
        /// Destination path under `MEMORY_ROOT`.
        path: Cow<'static, str>,
        /// Full contents of the new file.
        file_text: Cow<'static, str>,
    },
    /// Replace the single verbatim occurrence of `old_str` with `new_str`.
    StrReplace {
        /// Path of the file to edit.
        path: Cow<'static, str>,
        /// Text to find — must appear exactly once.
        old_str: Cow<'static, str>,
        /// Replacement text.
        new_str: Cow<'static, str>,
    },
    /// Insert text after a given (0-indexed-from-top) line.
    Insert {
        /// Path of the file to edit.
        path: Cow<'static, str>,
        /// Line after which to insert (`0` = beginning), in `[0, n_lines]`.
        insert_line: u64,
        /// Text to insert.
        insert_text: Cow<'static, str>,
    },
    /// Delete a file or directory (recursively).
    Delete {
        /// Path to remove.
        path: Cow<'static, str>,
    },
    /// Rename/move a file or directory (errors if the destination exists).
    Rename {
        /// Source path.
        old_path: Cow<'static, str>,
        /// Destination path.
        new_path: Cow<'static, str>,
    },
}

impl TryFrom<serde_json::Value> for Command {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

/// Why a memory operation failed. Its [`Display`](std::fmt::Display) is written
/// for the model — it says what went wrong and (where useful) how to recover,
/// matching the strings the memory tool is trained on.
#[cfg(feature = "memory-fs")]
#[derive(Debug)]
pub enum MemoryError {
    /// A path escaped `MEMORY_ROOT` (traversal / absolute / prefix).
    Traversal(String),
    /// The path does not exist.
    NotFound(String),
    /// A `create` target already exists.
    AlreadyExists(String),
    /// `old_str` did not appear in the file.
    NoMatch {
        /// The text that was searched for.
        old_str: String,
        /// The file searched.
        path: String,
    },
    /// `old_str` appeared more than once (ambiguous).
    MultipleMatches {
        /// The text that was searched for.
        old_str: String,
        /// 1-indexed lines where it occurred.
        lines: Vec<usize>,
    },
    /// `insert_line` was out of range.
    InvalidLine {
        /// The offending line number.
        insert_line: u64,
        /// The file's line count.
        n_lines: usize,
    },
    /// A `rename` destination already exists.
    DestExists(String),
    /// A write targeted a disallowed file extension.
    Extension {
        /// The rejected path.
        path: String,
        /// The allowed extensions.
        allowed: Vec<String>,
    },
    /// A command this crate does not model.
    UnknownCommand(String),
    /// An underlying I/O error.
    Io(std::io::Error),
}

#[cfg(feature = "memory-fs")]
impl std::fmt::Display for MemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Traversal(p) => write!(
                f,
                "Error: the path {p} is outside the memory directory. Paths \
                 must stay within {MEMORY_ROOT}."
            ),
            Self::NotFound(p) => write!(
                f,
                "The path {p} does not exist. Please provide a valid path."
            ),
            Self::AlreadyExists(p) => {
                write!(f, "Error: File {p} already exists")
            }
            Self::NoMatch { old_str, path } => write!(
                f,
                "No replacement was performed, old_str `{old_str}` did not \
                 appear verbatim in {path}."
            ),
            Self::MultipleMatches { old_str, lines } => {
                let lines = lines
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "No replacement was performed. Multiple occurrences of \
                     old_str `{old_str}` in lines: {lines}. Please ensure it \
                     is unique"
                )
            }
            Self::InvalidLine {
                insert_line,
                n_lines,
            } => write!(
                f,
                "Error: Invalid `insert_line` parameter: {insert_line}. It \
                 should be within the range of lines of the file: \
                 [0, {n_lines}]"
            ),
            Self::DestExists(p) => {
                write!(f, "Error: The destination {p} already exists")
            }
            Self::Extension { path, allowed } => write!(
                f,
                "Error: cannot write {path}: only these file extensions are \
                 allowed in memory: {}",
                allowed.join(", ")
            ),
            Self::UnknownCommand(c) => {
                write!(f, "Error: unsupported memory command `{c}`")
            }
            Self::Io(e) => write!(f, "Error: {e}"),
        }
    }
}

#[cfg(feature = "memory-fs")]
impl std::error::Error for MemoryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "memory-fs")]
impl From<std::io::Error> for MemoryError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A filesystem-backed [memory tool] executor, jailed to a single directory and
/// (by default) to markdown files. Implements [`Tool`] so a memory `tool_use`
/// dispatches through [`Tool::call`] exactly like a custom tool — it just
/// contributes no [`definitions`](Tool::definitions) of its own, since the
/// definition is added with [`Memory::latest`](crate::tool::Memory::latest).
///
/// Drop it into a [`ToolBox`](crate::tool::ToolBox) like any other tool, or
/// **use it directly** by calling [`Tool::call`] with the memory `tool_use`.
/// Unlike a custom tool, it contributes a [`Server`](crate::tool::MethodDef)
/// def (via [`definitions`](Tool::definitions)) and is routed by its fixed bare
/// wire name `"memory"` rather than namespaced — so a `ToolBox` installs the
/// def *and* dispatches the resulting `tool_use` back here, with no per-tool
/// special-casing. There is no per-conversation setup to defer to a lifecycle
/// hook — the root directory is created eagerly in [`new`](Self::new).
///
/// Every path the model sends is mapped from `MEMORY_ROOT` onto [`root`] and
/// validated to stay within it (no `..`, absolute, or prefix escapes).
///
/// [memory tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
/// [`root`]: FsMemoryBackend::root
#[cfg(feature = "memory-fs")]
#[derive(Clone, Debug)]
pub struct FsMemoryBackend {
    /// The real directory standing in for `MEMORY_ROOT`.
    root: PathBuf,
    /// File extensions the model may create/keep (lowercased, no dot). Empty
    /// means "any".
    extensions: Vec<String>,
}

#[cfg(feature = "memory-fs")]
impl FsMemoryBackend {
    /// A backend rooted at `root` (created if missing), restricted to markdown
    /// (`.md`) files. Override the extension allowlist with
    /// [`with_extensions`](Self::with_extensions).
    pub async fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        tokio::fs::create_dir_all(&root).await?;
        Ok(Self {
            root,
            extensions: vec!["md".to_string()],
        })
    }

    /// Restrict writable files to these extensions (lowercased, no leading
    /// dot). An empty list allows any extension.
    pub fn with_extensions<I, S>(mut self, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extensions = extensions
            .into_iter()
            .map(|e| e.into().trim_start_matches('.').to_ascii_lowercase())
            .collect();
        self
    }

    /// The real directory standing in for `MEMORY_ROOT`.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Map a model-supplied path onto [`root`](Self::root), rejecting anything
    /// that would escape it. The memory tool addresses files under the virtual
    /// `MEMORY_ROOT`; see [`fs::resolve_jailed`].
    fn resolve(&self, path: &str) -> Result<PathBuf, MemoryError> {
        fs::resolve_jailed(&self.root, path, Some(MEMORY_ROOT))
            .ok_or_else(|| MemoryError::Traversal(path.to_string()))
    }

    /// Whether `path`'s extension is permitted to be written.
    fn extension_allowed(&self, path: &Path) -> bool {
        if self.extensions.is_empty() {
            return true;
        }
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| self.extensions.contains(&e.to_ascii_lowercase()))
            .unwrap_or(false)
    }

    /// Run one [`Command`], returning the canonical model-facing string.
    async fn run(&self, command: Command) -> Result<String, MemoryError> {
        match command {
            Command::Known(Known::View { path, view_range }) => {
                self.view(&path, view_range).await
            }
            Command::Known(Known::Create { path, file_text }) => {
                self.create(&path, &file_text).await
            }
            Command::Known(Known::StrReplace {
                path,
                old_str,
                new_str,
            }) => self.str_replace(&path, &old_str, &new_str).await,
            Command::Known(Known::Insert {
                path,
                insert_line,
                insert_text,
            }) => self.insert(&path, insert_line, &insert_text).await,
            Command::Known(Known::Delete { path }) => self.delete(&path).await,
            Command::Known(Known::Rename { old_path, new_path }) => {
                self.rename(&old_path, &new_path).await
            }
            Command::Unknown { command, .. } => {
                Err(MemoryError::UnknownCommand(command.into_owned()))
            }
        }
    }

    /// Whether `path` is an existing regular file (async stat).
    async fn is_file(&self, path: &Path) -> bool {
        tokio::fs::metadata(path)
            .await
            .map(|m| m.is_file())
            .unwrap_or(false)
    }

    async fn view(
        &self,
        path: &str,
        range: Option<[u64; 2]>,
    ) -> Result<String, MemoryError> {
        let resolved = self.resolve(path)?;
        match tokio::fs::metadata(&resolved).await {
            Ok(meta) if meta.is_dir() => self.list_dir(path, &resolved).await,
            Ok(meta) if meta.is_file() => {
                let content = tokio::fs::read_to_string(&resolved).await?;
                Ok(format!(
                    "Here's the content of {path} with line numbers:\n{}",
                    fs::with_line_numbers(&content, range)
                ))
            }
            _ => Err(MemoryError::NotFound(path.to_string())),
        }
    }

    /// A JSON directory listing (two levels deep) carrying size and an RFC3339
    /// modified time per entry — the latter so the model can distrust stale
    /// notes. Hidden entries and `node_modules` are skipped; files are filtered
    /// to the allowed [`extensions`](Self::extensions) (directories always
    /// shown so the model can navigate).
    async fn list_dir(
        &self,
        virtual_path: &str,
        real: &Path,
    ) -> Result<String, MemoryError> {
        // Level 1, then each immediate subdirectory for level 2 — depth-first
        // to two levels, iteratively (async recursion would need boxing).
        let mut entries = Vec::new();
        let subdirs = self.collect_dir(real, &mut entries).await?;
        for sub in subdirs {
            self.collect_dir(&sub, &mut entries).await?;
        }
        let listing = serde_json::json!({
            "directory": virtual_path.trim_end_matches('/'),
            "entries": entries,
        });
        Ok(serde_json::to_string_pretty(&listing)
            .unwrap_or_else(|_| "[]".to_string()))
    }

    /// Append `dir`'s filtered, name-sorted entries to `out` and return its
    /// immediate subdirectories (for the caller to descend one more level).
    /// Hidden entries and `node_modules` are skipped; files are filtered to the
    /// allowed [`extensions`](Self::extensions) (directories always shown).
    async fn collect_dir(
        &self,
        dir: &Path,
        out: &mut Vec<serde_json::Value>,
    ) -> Result<Vec<PathBuf>, MemoryError> {
        let mut read_dir = tokio::fs::read_dir(dir).await?;
        let mut items = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            items.push(entry);
        }
        items.sort_by_key(|e| e.file_name());

        let mut subdirs = Vec::new();
        for entry in items {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || name == "node_modules" {
                continue;
            }
            let meta = match entry.metadata().await {
                Ok(m) => m,
                Err(_) => continue,
            };
            let path = entry.path();
            let is_dir = meta.is_dir();
            if !is_dir && !self.extension_allowed(&path) {
                continue;
            }
            out.push(serde_json::json!({
                "path": self.virtual_path(&path),
                "kind": if is_dir { "dir" } else { "file" },
                "size": human_size(meta.len()),
                "modified": modified_rfc3339(&meta),
            }));
            if is_dir {
                subdirs.push(path);
            }
        }
        Ok(subdirs)
    }

    /// Map a real path back to its `/memories/...` virtual form for display.
    fn virtual_path(&self, real: &Path) -> String {
        match real.strip_prefix(&self.root) {
            Ok(rel) if rel.as_os_str().is_empty() => MEMORY_ROOT.to_string(),
            Ok(rel) => format!("{MEMORY_ROOT}/{}", rel.to_string_lossy()),
            Err(_) => real.to_string_lossy().into_owned(),
        }
    }

    async fn create(
        &self,
        path: &str,
        file_text: &str,
    ) -> Result<String, MemoryError> {
        let resolved = self.resolve(path)?;
        if !self.extension_allowed(&resolved) {
            return Err(MemoryError::Extension {
                path: path.to_string(),
                allowed: self.extensions.clone(),
            });
        }
        if tokio::fs::try_exists(&resolved).await.unwrap_or(false) {
            return Err(MemoryError::AlreadyExists(path.to_string()));
        }
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&resolved, file_text).await?;
        Ok(format!("File created successfully at: {path}"))
    }

    async fn str_replace(
        &self,
        path: &str,
        old_str: &str,
        new_str: &str,
    ) -> Result<String, MemoryError> {
        let resolved = self.resolve(path)?;
        if !self.is_file(&resolved).await {
            return Err(MemoryError::NotFound(path.to_string()));
        }
        let content = tokio::fs::read_to_string(&resolved).await?;
        let matches = fs::match_lines(&content, old_str);
        match matches.len() {
            0 => Err(MemoryError::NoMatch {
                old_str: old_str.to_string(),
                path: path.to_string(),
            }),
            1 => {
                let updated = content.replacen(old_str, new_str, 1);
                tokio::fs::write(&resolved, &updated).await?;
                Ok(format!(
                    "The memory file has been edited.\n{}",
                    fs::with_line_numbers(&updated, None)
                ))
            }
            _ => Err(MemoryError::MultipleMatches {
                old_str: old_str.to_string(),
                lines: matches,
            }),
        }
    }

    async fn insert(
        &self,
        path: &str,
        insert_line: u64,
        insert_text: &str,
    ) -> Result<String, MemoryError> {
        let resolved = self.resolve(path)?;
        if !self.is_file(&resolved).await {
            return Err(MemoryError::NotFound(path.to_string()));
        }
        let content = tokio::fs::read_to_string(&resolved).await?;
        let updated = fs::insert_after(&content, insert_line, insert_text)
            .ok_or_else(|| MemoryError::InvalidLine {
                insert_line,
                n_lines: content.lines().count(),
            })?;
        tokio::fs::write(&resolved, &updated).await?;
        Ok(format!("The file {path} has been edited."))
    }

    async fn delete(&self, path: &str) -> Result<String, MemoryError> {
        let resolved = self.resolve(path)?;
        match tokio::fs::metadata(&resolved).await {
            Ok(meta) if meta.is_dir() => {
                tokio::fs::remove_dir_all(&resolved).await?
            }
            Ok(_) => tokio::fs::remove_file(&resolved).await?,
            Err(_) => return Err(MemoryError::NotFound(path.to_string())),
        }
        Ok(format!("Successfully deleted {path}"))
    }

    async fn rename(
        &self,
        old_path: &str,
        new_path: &str,
    ) -> Result<String, MemoryError> {
        let from = self.resolve(old_path)?;
        let to = self.resolve(new_path)?;
        if !tokio::fs::try_exists(&from).await.unwrap_or(false) {
            return Err(MemoryError::NotFound(old_path.to_string()));
        }
        if tokio::fs::try_exists(&to).await.unwrap_or(false) {
            return Err(MemoryError::DestExists(new_path.to_string()));
        }
        if to.extension().is_some() && !self.extension_allowed(&to) {
            return Err(MemoryError::Extension {
                path: new_path.to_string(),
                allowed: self.extensions.clone(),
            });
        }
        if let Some(parent) = to.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::rename(&from, &to).await?;
        Ok(format!("Successfully renamed {old_path} to {new_path}"))
    }
}

#[cfg(feature = "memory-fs")]
#[async_trait::async_trait]
impl Tool for FsMemoryBackend {
    fn name(&self) -> &str {
        "memory"
    }

    /// The predefined [`memory`](ServerMethodDef::Memory) tool def. Contributing it
    /// here (rather than expecting the caller to
    /// [`add_tool`](crate::Prompt::add_tool) it separately) is what lets the
    /// backend drop into a [`ToolBox`](crate::tool::ToolBox): the box installs
    /// this def and routes the resulting bare `"memory"` `tool_use` straight
    /// back to [`call`](Self::call).
    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Server(ServerMethodDef::memory())]
    }

    async fn call(&mut self, call: Use) -> crate::tool::Result {
        let id = call.id;
        let command = match Command::try_from(call.input) {
            Ok(command) => command,
            Err(e) => {
                return crate::tool::Result::new(
                    id,
                    format!("Error: could not parse memory command: {e}"),
                )
                .error();
            }
        };
        match self.run(command).await {
            Ok(content) => crate::tool::Result::new(id, content),
            Err(e) => crate::tool::Result::new(id, e.to_string()).error(),
        }
    }
}

/// Format `bytes` like the memory tool does (`4.0K`, `2.0M`); whole bytes under
/// 1 KiB.
#[cfg(feature = "memory-fs")]
fn human_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "K", "M", "G", "T"];
    let mut size = bytes as f64;
    let mut unit = 0;
    while size >= 1024.0 && unit < UNITS.len() - 1 {
        size /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes}B")
    } else {
        format!("{size:.1}{}", UNITS[unit])
    }
}

/// An RFC3339 modified time for `meta`, or `null` if unavailable.
#[cfg(feature = "memory-fs")]
fn modified_rfc3339(meta: &std::fs::Metadata) -> serde_json::Value {
    meta.modified()
        .ok()
        .map(|t| {
            chrono::DateTime::<chrono::Utc>::from(t)
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        })
        .map(serde_json::Value::String)
        .unwrap_or(serde_json::Value::Null)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_known_roundtrips() {
        for raw in [
            r#"{"command":"view","path":"/memories"}"#,
            r#"{"command":"create","path":"/memories/a.md","file_text":"hi\n"}"#,
            r#"{"command":"str_replace","path":"/memories/a.md","old_str":"x","new_str":"y"}"#,
            r#"{"command":"insert","path":"/memories/a.md","insert_line":2,"insert_text":"z\n"}"#,
            r#"{"command":"delete","path":"/memories/a.md"}"#,
            r#"{"command":"rename","old_path":"/memories/a.md","new_path":"/memories/b.md"}"#,
        ] {
            let cmd: Command = serde_json::from_str(raw).unwrap();
            assert!(matches!(cmd, Command::Known(_)), "{raw}");
        }
    }

    #[test]
    fn command_unknown_is_caught_not_dropped() {
        let raw = r#"{"command":"append","path":"/memories/a.md","text":"x"}"#;
        let cmd: Command = serde_json::from_str(raw).unwrap();
        match cmd {
            Command::Unknown { command, rest } => {
                assert_eq!(command, "append");
                assert_eq!(rest["path"], "/memories/a.md");
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsMemoryBackend::new(dir.path()).await.unwrap();
        for evil in ["/memories/../escape.md", "/etc/passwd", "../../x"] {
            assert!(matches!(
                backend.resolve(evil),
                Err(MemoryError::Traversal(_))
            ));
        }
    }

    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn create_view_replace_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsMemoryBackend::new(dir.path()).await.unwrap();

        let created = backend
            .run(Command::Known(Known::Create {
                path: "/memories/notes.md".into(),
                file_text: "alpha\nbeta\n".into(),
            }))
            .await
            .unwrap();
        assert!(created.contains("created successfully"));

        // Non-markdown create is refused.
        assert!(matches!(
            backend
                .run(Command::Known(Known::Create {
                    path: "/memories/notes.txt".into(),
                    file_text: "x".into(),
                }))
                .await,
            Err(MemoryError::Extension { .. })
        ));

        let viewed = backend
            .run(Command::Known(Known::View {
                path: "/memories/notes.md".into(),
                view_range: None,
            }))
            .await
            .unwrap();
        assert!(viewed.contains("     1\talpha"));
        assert!(viewed.contains("     2\tbeta"));

        backend
            .run(Command::Known(Known::StrReplace {
                path: "/memories/notes.md".into(),
                old_str: "beta".into(),
                new_str: "gamma".into(),
            }))
            .await
            .unwrap();
        let body =
            std::fs::read_to_string(dir.path().join("notes.md")).unwrap();
        assert_eq!(body, "alpha\ngamma\n");
    }

    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn str_replace_reports_ambiguity() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsMemoryBackend::new(dir.path()).await.unwrap();
        backend
            .run(Command::Known(Known::Create {
                path: "/memories/d.md".into(),
                file_text: "dup\ndup\n".into(),
            }))
            .await
            .unwrap();
        match backend
            .run(Command::Known(Known::StrReplace {
                path: "/memories/d.md".into(),
                old_str: "dup".into(),
                new_str: "x".into(),
            }))
            .await
        {
            Err(MemoryError::MultipleMatches { lines, .. }) => {
                assert_eq!(lines, vec![1, 2]);
            }
            other => panic!("expected MultipleMatches, got {other:?}"),
        }
    }

    #[test]
    fn memory_definition_roundtrips() {
        use crate::tool::{Memory, MethodDef, ServerMethodDef};
        // Request-side wire shape: a bare versioned `type` + `name`, no schema.
        let server: ServerMethodDef = crate::utils::roundtrip(
            r#"{"type":"memory_20250818","name":"memory"}"#,
        );
        assert!(matches!(server, ServerMethodDef::Memory(_)));
        // `add_tool(Memory::latest())` wraps it as a `MethodDef::Server`.
        let def: MethodDef = Memory::latest().into();
        assert_eq!(
            serde_json::to_value(&def).unwrap(),
            serde_json::json!({ "type": "memory_20250818", "name": "memory" }),
        );
    }

    /// The whole point of #83: dropping the backend into a [`ToolBox`] must
    /// surface *exactly* the def you'd otherwise hand to
    /// `add_tool(Memory::latest())` — bare-named and un-namespaced — so the
    /// wire bytes don't change, only who supplies them.
    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn toolbox_memory_def_matches_hand_added() {
        use crate::tool::{Memory, MethodDef, Tool, ToolBox};

        let dir = tempfile::tempdir().unwrap();
        let tools =
            ToolBox::new().add(FsMemoryBackend::new(dir.path()).await.unwrap());

        let hand: MethodDef = Memory::latest().into();
        assert_eq!(tools.definitions(), vec![hand]);
    }

    /// A bare `"memory"` `tool_use` — exactly what the model emits — routes
    /// through the box to the backend with no namespacing.
    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn toolbox_routes_bare_memory_name() {
        use crate::tool::{Tool, ToolBox, Use};

        let dir = tempfile::tempdir().unwrap();
        let mut tools =
            ToolBox::new().add(FsMemoryBackend::new(dir.path()).await.unwrap());

        let result = tools
            .call(
                Use::new(
                    "memory",
                    serde_json::json!({
                        "command": "create",
                        "path": "/memories/note.md",
                        "file_text": "remember this\n",
                    }),
                )
                .with_id("id"),
            )
            .await;

        assert!(!result.is_error, "{}", result.content);
        assert!(dir.path().join("note.md").exists());
    }

    /// The bare name must survive nesting: an outer box discovers `"memory"` in
    /// a child's subtree (it rides up through `definitions()` un-prefixed) and
    /// routes a call straight down without namespacing.
    #[cfg(feature = "memory-fs")]
    #[tokio::test]
    async fn toolbox_routes_bare_memory_through_nested_box() {
        use crate::tool::{Tool, ToolBox, Use};

        let dir = tempfile::tempdir().unwrap();
        let inner = ToolBox::named("inner")
            .unwrap()
            .add(FsMemoryBackend::new(dir.path()).await.unwrap());
        let mut outer = ToolBox::new().add(inner);

        // The advertised def is the still-bare `"memory"`, not `outer__…`.
        let advertised = outer.definitions();
        assert_eq!(advertised.len(), 1);
        assert_eq!(advertised[0].name(), "memory");

        let result = outer
            .call(
                Use::new(
                    "memory",
                    serde_json::json!({
                        "command": "create",
                        "path": "/memories/nested.md",
                        "file_text": "deep\n",
                    }),
                )
                .with_id("id"),
            )
            .await;

        assert!(!result.is_error, "{}", result.content);
        assert!(dir.path().join("nested.md").exists());
    }

    #[cfg(feature = "memory-fs")]
    #[test]
    fn human_size_matches_memory_format() {
        assert_eq!(human_size(0), "0B");
        assert_eq!(human_size(512), "512B");
        assert_eq!(human_size(4096), "4.0K");
        assert_eq!(human_size(1536), "1.5K");
        assert_eq!(human_size(2 * 1024 * 1024), "2.0M");
    }

    #[tokio::test]
    #[cfg(all(feature = "client", feature = "memory-fs"))]
    #[ignore = "This test requires a real API key."]
    async fn live_memory_tool_writes_to_disk() {
        // The live counterpart to `examples/memory.rs`: give the model the
        // memory tool, ask it to save a fact, execute each `tool_use` locally
        // through `FsMemoryBackend` (rooted at a tempdir), and assert a markdown
        // file actually lands on disk carrying that fact. Memory is supported on
        // Haiku (the cheapest model), unlike PTC.
        use crate::{Client, Id, Prompt, prompt::message::Role};

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let mut memory = FsMemoryBackend::new(dir.path()).await.unwrap();

        let mut chat = Prompt::default()
            .model(Id::Haiku45)
            .add_tool(crate::tool::Memory::latest())
            .add_message((
                Role::User,
                "Please save this to your memory for next time: my favorite \
                 color is green. Then confirm you saved it.",
            ))
            .unwrap();

        // Drive the tool loop exactly as the example does.
        let mut turns = 0;
        loop {
            let message = client.message(&chat).await.unwrap();
            turns += 1;
            assert!(turns <= 12, "runaway memory loop ({turns} turns)");
            let Some(call) = message.tool_use() else {
                break;
            };
            let call = call.clone();
            assert_eq!(call.name, "memory");
            chat.push_message(message).unwrap();
            let result = memory.call(call).await;
            chat.push_message(result).unwrap();
        }

        // The model should have created at least one markdown file mentioning
        // the fact (the backend only permits `.md`, so any create is markdown).
        let body: String = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "md"))
            .filter_map(|p| std::fs::read_to_string(p).ok())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!body.is_empty(), "the model wrote no .md memory file");
        assert!(
            body.to_lowercase().contains("green"),
            "saved memory should mention 'green': {body}"
        );
    }
}
