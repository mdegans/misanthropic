//! Client-side execution of the [text editor tool]
//! ([`ServerMethodDef::TextEditor`]).
//!
//! The text editor is *predefined* (you add it by versioned name via
//! [`TextEditor::latest`], no schema of your own) but *client-executed*: the
//! model emits an ordinary [`Use`] (`name: "str_replace_based_edit_tool"`)
//! whose [`input`](Use::input) is one of a small set of file operations
//! (`view` / `create` / `str_replace` / `insert`), and you run it against a
//! working tree you control. This module provides the typed
//! [`Command`](crate::tool::text_editor::Command) those
//! inputs deserialize into and [`FsEditorBackend`], a filesystem-backed
//! reference executor jailed to a single directory.
//!
//! Like [`memory`](crate::tool::memory), it *defines* like a server tool and
//! *executes* like a custom one — the two are siblings and share the path/text
//! plumbing in `fs`.
//!
//! ```no_run
//! # #[cfg(feature = "text-editor-fs")] // backend is feature-gated; doc isn't
//! # async fn f() -> Result<(), Box<dyn std::error::Error>> {
//! use misanthropic::{Prompt, tool::{TextEditor, Tool, text_editor::FsEditorBackend}};
//!
//! let mut backend = FsEditorBackend::new("./workspace").await?;
//! let mut prompt = Prompt::default().add_tool(TextEditor::latest());
//! // ... when an assistant `tool_use` named "str_replace_based_edit_tool"
//! // arrives as `call`:
//! # let call: misanthropic::tool::Use = todo!();
//! let result = backend.call(call).await; // typed dispatch + canonical reply
//! # let _ = (result, &mut prompt); Ok(())
//! # }
//! ```
//!
//! [text editor tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>
//! [`ServerMethodDef::TextEditor`]: crate::tool::ServerMethodDef::TextEditor
//! [`TextEditor::latest`]: crate::tool::TextEditor::latest
//! [`Use`]: crate::tool::Use

use std::borrow::Cow;
#[cfg(feature = "text-editor-fs")]
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[cfg(feature = "text-editor-fs")]
use super::{MethodDef, ServerMethodDef, Tool, Use, fs};

/// A typed text-editor command, deserialized from an editor [`Use`]'s
/// [`input`](Use::input).
///
/// A known/unknown union (à la [`model::Model`]/[`Caller`]): commands this crate
/// has typed support for land in [`Known`]; anything else (e.g. the `undo_edit`
/// of an older tool version, dropped from `text_editor_20250728`) round-trips
/// through [`Unknown`](Command::Unknown) rather than failing to deserialize a
/// live response.
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

/// The text-editor commands this crate models, internally tagged on `command`.
/// See the [text editor tool] reference for the full semantics of each.
///
/// [text editor tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>
#[derive(Clone, Debug, Deserialize, Serialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "command", rename_all = "snake_case")]
pub enum Known {
    /// Show a file's contents (with line numbers) or list a directory.
    View {
        /// Path to the file or directory to view.
        path: Cow<'static, str>,
        /// Optional inclusive 1-indexed `[start, end]` line range for files
        /// (`-1` as the end reads to EOF).
        #[serde(skip_serializing_if = "Option::is_none")]
        view_range: Option<[i64; 2]>,
    },
    /// Create (or overwrite) a file with the given contents.
    Create {
        /// Destination path.
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
}

impl TryFrom<serde_json::Value> for Command {
    type Error = serde_json::Error;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

/// Why a text-editor operation failed. Its [`Display`](std::fmt::Display) is
/// written for the model — it says what went wrong and (where useful) how to
/// recover, matching the strings the text editor tool is trained on.
#[cfg(feature = "text-editor-fs")]
#[derive(Debug)]
pub enum EditorError {
    /// A path escaped the working directory (traversal / absolute).
    Traversal(String),
    /// The path does not exist.
    NotFound(String),
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
    /// A command this crate does not model (e.g. `undo_edit`).
    UnknownCommand(String),
    /// An underlying I/O error.
    Io(std::io::Error),
}

#[cfg(feature = "text-editor-fs")]
impl std::fmt::Display for EditorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Traversal(p) => write!(
                f,
                "Error: the path {p} is outside the working directory."
            ),
            Self::NotFound(p) => write!(f, "Error: File {p} not found"),
            Self::NoMatch { old_str, path } => write!(
                f,
                "Error: No match found for `{old_str}` in {path}. Please check \
                 your text and try again."
            ),
            Self::MultipleMatches { old_str, lines } => {
                let count = lines.len();
                let lines = lines
                    .iter()
                    .map(|l| l.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                write!(
                    f,
                    "Error: Found {count} matches for `{old_str}` (lines: \
                     {lines}). Please provide more context to make a unique \
                     match."
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
            Self::UnknownCommand(c) => {
                write!(f, "Error: unsupported text editor command `{c}`")
            }
            Self::Io(e) => write!(f, "Error: {e}"),
        }
    }
}

#[cfg(feature = "text-editor-fs")]
impl std::error::Error for EditorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "text-editor-fs")]
impl From<std::io::Error> for EditorError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// A filesystem-backed [text editor tool] executor, jailed to a single working
/// directory. Implements [`Tool`] so an editor `tool_use` dispatches through
/// [`Tool::call`] exactly like a custom tool — it just contributes no
/// [`definitions`](Tool::definitions) schema of its own, since the definition
/// is added with [`TextEditor::latest`](crate::tool::TextEditor::latest).
///
/// Drop it into a [`ToolBox`](crate::tool::ToolBox) like any other tool, or
/// **use it directly** by calling [`Tool::call`] with the editor `tool_use`.
/// Like [`FsMemoryBackend`], it contributes a [`Server`](crate::tool::MethodDef)
/// def (via [`definitions`](Tool::definitions)) routed by its fixed bare wire
/// name `"str_replace_based_edit_tool"` rather than namespaced — so a `ToolBox`
/// installs the def *and* dispatches the resulting `tool_use` back here, with no
/// per-tool special-casing.
///
/// Every path the model sends is taken as root-relative and validated to stay
/// within [`root`] (no `..` or absolute escapes). Unlike the memory backend
/// there is no extension allowlist — the editor edits files of any type.
///
/// [text editor tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>
/// [`FsMemoryBackend`]: crate::tool::memory::FsMemoryBackend
/// [`root`]: FsEditorBackend::root
#[cfg(feature = "text-editor-fs")]
#[derive(Clone, Debug)]
pub struct FsEditorBackend {
    /// The working directory all paths are jailed to.
    root: PathBuf,
    /// Truncate a `view`'s file contents to roughly this many characters, if
    /// set. Mirrored onto the advertised [`definitions`](Tool::definitions).
    max_characters: Option<u32>,
}

#[cfg(feature = "text-editor-fs")]
impl FsEditorBackend {
    /// A backend rooted at `root` (created if missing). Every path the model
    /// sends is resolved under it.
    pub async fn new(root: impl Into<PathBuf>) -> std::io::Result<Self> {
        let root = root.into();
        tokio::fs::create_dir_all(&root).await?;
        Ok(Self {
            root,
            max_characters: None,
        })
    }

    /// Cap `view` output at roughly `max` characters (advertised on the tool
    /// definition as `max_characters`, and enforced locally).
    pub fn with_max_characters(mut self, max: u32) -> Self {
        self.max_characters = Some(max);
        self
    }

    /// The working directory all paths are jailed to.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Map a model-supplied (root-relative) path onto [`root`](Self::root),
    /// rejecting anything that would escape it. See [`fs::resolve_jailed`].
    fn resolve(&self, path: &str) -> Result<PathBuf, EditorError> {
        fs::resolve_jailed(&self.root, path, None)
            .ok_or_else(|| EditorError::Traversal(path.to_string()))
    }

    /// Run one [`Command`], returning the canonical model-facing string.
    async fn run(&self, command: Command) -> Result<String, EditorError> {
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
            Command::Unknown { command, .. } => {
                Err(EditorError::UnknownCommand(command.into_owned()))
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
        range: Option<[i64; 2]>,
    ) -> Result<String, EditorError> {
        let resolved = self.resolve(path)?;
        match tokio::fs::metadata(&resolved).await {
            Ok(meta) if meta.is_dir() => self.list_dir(&resolved).await,
            Ok(meta) if meta.is_file() => {
                let mut content = tokio::fs::read_to_string(&resolved).await?;
                if let Some(max) = self.max_characters
                    && content.chars().count() > max as usize
                {
                    content = content.chars().take(max as usize).collect();
                }
                Ok(fs::with_line_numbers(&content, normalize_range(range)))
            }
            _ => Err(EditorError::NotFound(path.to_string())),
        }
    }

    /// A plain, sorted directory listing (one level), directories suffixed with
    /// `/` — enough for the model to navigate without the memory tool's richer
    /// JSON-with-mtime shape.
    async fn list_dir(&self, real: &Path) -> Result<String, EditorError> {
        let mut read_dir = tokio::fs::read_dir(real).await?;
        let mut names = Vec::new();
        while let Some(entry) = read_dir.next_entry().await? {
            let is_dir =
                entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            let mut name = entry.file_name().to_string_lossy().into_owned();
            if is_dir {
                name.push('/');
            }
            names.push(name);
        }
        names.sort();
        Ok(names.join("\n"))
    }

    async fn create(
        &self,
        path: &str,
        file_text: &str,
    ) -> Result<String, EditorError> {
        let resolved = self.resolve(path)?;
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
    ) -> Result<String, EditorError> {
        let resolved = self.resolve(path)?;
        if !self.is_file(&resolved).await {
            return Err(EditorError::NotFound(path.to_string()));
        }
        let content = tokio::fs::read_to_string(&resolved).await?;
        let matches = fs::match_lines(&content, old_str);
        match matches.len() {
            0 => Err(EditorError::NoMatch {
                old_str: old_str.to_string(),
                path: path.to_string(),
            }),
            1 => {
                let updated = content.replacen(old_str, new_str, 1);
                tokio::fs::write(&resolved, &updated).await?;
                Ok("Successfully replaced text at exactly one location."
                    .to_string())
            }
            _ => Err(EditorError::MultipleMatches {
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
    ) -> Result<String, EditorError> {
        let resolved = self.resolve(path)?;
        if !self.is_file(&resolved).await {
            return Err(EditorError::NotFound(path.to_string()));
        }
        let content = tokio::fs::read_to_string(&resolved).await?;
        let updated = fs::insert_after(&content, insert_line, insert_text)
            .ok_or_else(|| EditorError::InvalidLine {
                insert_line,
                n_lines: content.lines().count(),
            })?;
        tokio::fs::write(&resolved, &updated).await?;
        Ok(format!("The file {path} has been edited."))
    }
}

/// Normalize a text-editor `view_range` (`[i64; 2]`, where `-1` means EOF) into
/// the `[u64; 2]` inclusive range [`fs::with_line_numbers`] expects. A negative
/// or zero end becomes "read to end".
#[cfg(feature = "text-editor-fs")]
fn normalize_range(range: Option<[i64; 2]>) -> Option<[u64; 2]> {
    range.map(|[start, end]| {
        let start = start.max(1) as u64;
        let end = if end < 0 { u64::MAX } else { end as u64 };
        [start, end]
    })
}

#[cfg(feature = "text-editor-fs")]
#[async_trait::async_trait]
impl Tool for FsEditorBackend {
    fn name(&self) -> &str {
        "str_replace_based_edit_tool"
    }

    /// The predefined [`text_editor`](ServerMethodDef::TextEditor) tool def,
    /// carrying this backend's [`max_characters`](Self::with_max_characters).
    /// Contributing it here (rather than expecting the caller to
    /// [`add_tool`](crate::Prompt::add_tool) it separately) is what lets the
    /// backend drop into a [`ToolBox`](crate::tool::ToolBox): the box installs
    /// this def and routes the resulting bare `"str_replace_based_edit_tool"`
    /// `tool_use` straight back to [`call`](Self::call).
    fn definitions(&self) -> Vec<MethodDef> {
        vec![MethodDef::Server(ServerMethodDef::TextEditor(
            crate::tool::TextEditor {
                max_characters: self.max_characters,
                ..Default::default()
            },
        ))]
    }

    async fn call(&mut self, call: Use) -> crate::tool::Result {
        let id = call.id;
        let command = match Command::try_from(call.input) {
            Ok(command) => command,
            Err(e) => {
                return crate::tool::Result::new(
                    id,
                    format!("Error: could not parse editor command: {e}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_known_roundtrips() {
        for raw in [
            r#"{"command":"view","path":"primes.py"}"#,
            r#"{"command":"view","path":"primes.py","view_range":[1,10]}"#,
            r#"{"command":"create","path":"a.py","file_text":"x = 1\n"}"#,
            r#"{"command":"str_replace","path":"a.py","old_str":"x","new_str":"y"}"#,
            r##"{"command":"insert","path":"a.py","insert_line":0,"insert_text":"# top\n"}"##,
        ] {
            let cmd: Command = serde_json::from_str(raw).unwrap();
            assert!(matches!(cmd, Command::Known(_)), "{raw}");
        }
    }

    #[test]
    fn command_unknown_is_caught_not_dropped() {
        // `undo_edit` was removed in text_editor_20250728; it must round-trip
        // through `Unknown` rather than fail to deserialize.
        let raw = r#"{"command":"undo_edit","path":"a.py"}"#;
        let cmd: Command = serde_json::from_str(raw).unwrap();
        match cmd {
            Command::Unknown { command, rest } => {
                assert_eq!(command, "undo_edit");
                assert_eq!(rest["path"], "a.py");
            }
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsEditorBackend::new(dir.path()).await.unwrap();
        for evil in ["../escape.py", "/etc/passwd", "a/../../b"] {
            assert!(matches!(
                backend.resolve(evil),
                Err(EditorError::Traversal(_))
            ));
        }
    }

    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn create_view_replace_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsEditorBackend::new(dir.path()).await.unwrap();

        backend
            .run(Command::Known(Known::Create {
                path: "primes.py".into(),
                file_text: "alpha\nbeta\n".into(),
            }))
            .await
            .unwrap();

        // Any extension is allowed (no memory-style allowlist).
        backend
            .run(Command::Known(Known::Create {
                path: "notes.txt".into(),
                file_text: "x".into(),
            }))
            .await
            .unwrap();

        let viewed = backend
            .run(Command::Known(Known::View {
                path: "primes.py".into(),
                view_range: None,
            }))
            .await
            .unwrap();
        assert!(viewed.contains("     1\talpha"));
        assert!(viewed.contains("     2\tbeta"));

        let ok = backend
            .run(Command::Known(Known::StrReplace {
                path: "primes.py".into(),
                old_str: "beta".into(),
                new_str: "gamma".into(),
            }))
            .await
            .unwrap();
        assert!(ok.contains("exactly one location"));
        let body =
            std::fs::read_to_string(dir.path().join("primes.py")).unwrap();
        assert_eq!(body, "alpha\ngamma\n");
    }

    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn str_replace_reports_ambiguity() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsEditorBackend::new(dir.path()).await.unwrap();
        backend
            .run(Command::Known(Known::Create {
                path: "d.py".into(),
                file_text: "dup\ndup\n".into(),
            }))
            .await
            .unwrap();
        match backend
            .run(Command::Known(Known::StrReplace {
                path: "d.py".into(),
                old_str: "dup".into(),
                new_str: "x".into(),
            }))
            .await
        {
            Err(EditorError::MultipleMatches { lines, .. }) => {
                assert_eq!(lines, vec![1, 2]);
            }
            other => panic!("expected MultipleMatches, got {other:?}"),
        }
    }

    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn view_range_and_max_characters() {
        let dir = tempfile::tempdir().unwrap();
        let backend = FsEditorBackend::new(dir.path())
            .await
            .unwrap()
            .with_max_characters(3);
        backend
            .run(Command::Known(Known::Create {
                path: "f.txt".into(),
                file_text: "abcdefgh\n".into(),
            }))
            .await
            .unwrap();
        // max_characters truncates the raw content before numbering.
        let viewed = backend
            .run(Command::Known(Known::View {
                path: "f.txt".into(),
                view_range: None,
            }))
            .await
            .unwrap();
        assert_eq!(viewed, "     1\tabc\n");
        // The def advertises the cap.
        let defs = backend.definitions();
        let json = serde_json::to_value(&defs[0]).unwrap();
        assert_eq!(json["max_characters"], 3);
    }

    #[test]
    fn text_editor_definition_roundtrips() {
        use crate::tool::{MethodDef, ServerMethodDef, TextEditor};
        // Request-side wire shape: a bare versioned `type` + `name`, no schema.
        let server: ServerMethodDef = crate::utils::roundtrip(
            r#"{"type":"text_editor_20250728","name":"str_replace_based_edit_tool"}"#,
        );
        assert!(matches!(server, ServerMethodDef::TextEditor(_)));
        // `add_tool(TextEditor::latest())` wraps it as a `MethodDef::Server`.
        let def: MethodDef = TextEditor::latest().into();
        assert_eq!(
            serde_json::to_value(&def).unwrap(),
            serde_json::json!({
                "type": "text_editor_20250728",
                "name": "str_replace_based_edit_tool",
            }),
        );
    }

    /// Dropping the backend into a [`ToolBox`] must surface *exactly* the def
    /// you'd otherwise hand to `add_tool(TextEditor::latest())` — bare-named and
    /// un-namespaced — so the wire bytes don't change, only who supplies them.
    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn toolbox_editor_def_matches_hand_added() {
        use crate::tool::{MethodDef, TextEditor, Tool, ToolBox};

        let dir = tempfile::tempdir().unwrap();
        let tools =
            ToolBox::new().add(FsEditorBackend::new(dir.path()).await.unwrap());

        let hand: MethodDef = TextEditor::latest().into();
        assert_eq!(tools.definitions(), vec![hand]);
    }

    /// A bare `"str_replace_based_edit_tool"` `tool_use` — exactly what the
    /// model emits — routes through the box to the backend with no namespacing.
    #[cfg(feature = "text-editor-fs")]
    #[tokio::test]
    async fn toolbox_routes_bare_editor_name() {
        use crate::tool::{Tool, ToolBox, Use};

        let dir = tempfile::tempdir().unwrap();
        let mut tools =
            ToolBox::new().add(FsEditorBackend::new(dir.path()).await.unwrap());

        let result = tools
            .call(
                Use::new(
                    "str_replace_based_edit_tool",
                    serde_json::json!({
                        "command": "create",
                        "path": "note.py",
                        "file_text": "print('hi')\n",
                    }),
                )
                .with_id("id"),
            )
            .await;

        assert!(!result.is_error, "{}", result.content);
        assert!(dir.path().join("note.py").exists());
    }

    #[tokio::test]
    #[cfg(all(feature = "client", feature = "text-editor-fs"))]
    #[ignore = "This test requires a real API key."]
    async fn live_text_editor_fixes_file_on_disk() {
        // The live counterpart to `examples/text_editor.rs`: plant a `primes.py`
        // with a syntax error (a `for` line missing its colon), give the model
        // the text editor tool, execute each `tool_use` locally through
        // `FsEditorBackend` (jailed to a tempdir), and assert the file on disk
        // is actually repaired. The editor is a Claude-4 tool, supported on
        // Haiku 4.5 (the cheapest model).
        use crate::{Client, Id, Prompt, prompt::message::Role};

        const BUGGY: &str = "\
def get_primes(limit):
    primes = []
    for num in range(2, limit + 1)
        if is_prime(num):
            primes.append(num)
    return primes
";

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("primes.py"), BUGGY)
            .await
            .unwrap();
        let mut editor = FsEditorBackend::new(dir.path()).await.unwrap();

        let mut chat = Prompt::default()
            .model(Id::Haiku45)
            .add_tool(crate::tool::TextEditor::latest())
            .add_message((
                Role::User,
                "There's a syntax error in primes.py. Please fix it.",
            ))
            .unwrap();

        // Drive the tool loop exactly as the example does.
        let mut turns = 0;
        loop {
            let message = client.message(&chat).await.unwrap();
            turns += 1;
            assert!(turns <= 12, "runaway editor loop ({turns} turns)");
            let Some(call) = message.tool_use() else {
                break;
            };
            let call = call.clone();
            assert_eq!(call.name, "str_replace_based_edit_tool");
            chat.push_message(message).unwrap();
            let result = editor.call(call).await;
            chat.push_message(result).unwrap();
        }

        // The buggy `for` line must be gone, and a colon-terminated one present.
        let fixed =
            std::fs::read_to_string(dir.path().join("primes.py")).unwrap();
        assert!(
            !fixed.contains("for num in range(2, limit + 1)\n"),
            "the broken `for` line should be gone:\n{fixed}"
        );
        assert!(
            fixed.contains("for num in range(2, limit + 1):"),
            "the `for` line should end with a colon:\n{fixed}"
        );
    }
}
