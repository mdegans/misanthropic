//! [`Tool`] [`Use`] and friends.
use std::{borrow::Cow, hash::Hash};

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::Prompt;
use crate::prompt::message::Content;

mod toolbox;
pub use toolbox::ToolBox;

mod typed;
pub use typed::{ErasedMethod, Method, Methods, ToolArgs, Typed};

/// Shared `impl Tool` body for [`Typed`] and `#[tool]`-generated tools. Not a
/// stable API; named by generated code only.
#[doc(hidden)]
pub use typed::{dispatch_methods, methods_definitions};

/// `#[derive(ToolArgs)]` — the front door for **hand-written** [`Method`]
/// impls: it generates the [`NAME`](ToolArgs::NAME)/[`DESCRIPTION`](ToolArgs::DESCRIPTION)
/// consts (from the struct ident + doc comment, overridable with
/// `#[tool(name = "…", description = "…")]`) so you don't write them by hand.
/// Most tools instead want the all-in-one [`macro@tool`] attribute, which
/// derives this for you. Co-located with the [`ToolArgs`] trait (same path,
/// different namespaces) so one `use misanthropic::tool::ToolArgs;` brings in
/// both, as with `serde`'s `Serialize`.
#[cfg(feature = "derive")]
pub use misanthropic_derive::ToolArgs;

/// `#[tool]` — the all-in-one path: an attribute on an `impl` block that
/// generates the [`Method`] / [`ToolArgs`] / [`Methods`] wiring from
/// `#[method]`-tagged async fns. Wrap the result in [`Typed`] (or
/// [`ToolBox::add_typed`]) to use as a [`Tool`]. For finer control, hand-write
/// [`Method`] impls and reach for [`macro@ToolArgs`] instead.
#[cfg(feature = "derive")]
pub use misanthropic_derive::tool;

#[cfg(feature = "notepad")]
mod notepad;
#[cfg(feature = "notepad")]
pub use notepad::Notepad;

/// Client-side execution of the [`memory`](ServerMethodDef::Memory) tool: the typed
/// [`Command`](memory::Command) vocabulary and the [`FsMemoryBackend`]
/// reference implementation.
///
/// [`FsMemoryBackend`]: memory::FsMemoryBackend
#[cfg(feature = "memory")]
pub mod memory;

/// Client-side execution of the [`text_editor`](ServerMethodDef::TextEditor)
/// tool: the typed [`Command`](text_editor::Command) vocabulary and the
/// [`FsEditorBackend`] reference implementation.
///
/// [`FsEditorBackend`]: text_editor::FsEditorBackend
#[cfg(feature = "text-editor")]
pub mod text_editor;

/// Client-side execution of the [`bash`](ServerMethodDef::Bash) tool: the typed
/// [`Command`](bash::Command) vocabulary, the [`bashd`] daemon wire protocol,
/// the [`BashSandbox`](bash::BashSandbox) trait, and the [`BashTool`](bash::BashTool)
/// adapter. Bring your own sandbox, or enable `bash-container` for `DockerSandbox`.
///
/// [`bashd`]: bash::Request
#[cfg(feature = "bash")]
pub mod bash;

/// Pure path/text helpers shared by the file-oriented client-executed tools'
/// filesystem backends (`memory::FsMemoryBackend`, `text_editor::FsEditorBackend`).
#[cfg(any(feature = "memory-fs", feature = "text-editor-fs"))]
mod fs;

/// Constrain the [`Assistant`]'s choice of [`CustomMethodDef`]s.
///
/// # Note:
/// - Anthropic calls this a "tool" in the API, but since [`Tool`]s can have
///   multiple [`CustomMethodDef`] in this crate, we use "method" instead.
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Choice {
    /// [`Model`] chooses whether and which [`CustomMethodDef`] of a [`Tool`] to use.
    ///
    /// [`Model`]: crate::model::ModelInfo
    Auto {
        /// If `true`, the model uses at most one tool (no parallel calls).
        #[serde(default, skip_serializing_if = "is_false")]
        disable_parallel_tool_use: bool,
    },
    /// Model must use at least one of the provided [`CustomMethodDef`]s.
    Any {
        /// If `true`, the model uses at most one tool (no parallel calls).
        #[serde(default, skip_serializing_if = "is_false")]
        disable_parallel_tool_use: bool,
    },
    /// Model must use a specific [`CustomMethodDef`].
    #[serde(rename = "tool")]
    Method {
        /// The [`MethodDef::name`] to use.
        name: String,
        /// If `true`, the model uses at most one tool (no parallel calls).
        #[serde(default, skip_serializing_if = "is_false")]
        disable_parallel_tool_use: bool,
    },
    /// The model must not use any tool, even if tools are provided.
    None,
}

/// Serde helper: skip a `bool` field when it is `false`.
fn is_false(b: &bool) -> bool {
    !*b
}

impl Default for Choice {
    /// [`Choice::Auto`] with parallel tool use enabled.
    fn default() -> Self {
        Self::Auto {
            disable_parallel_tool_use: false,
        }
    }
}

impl Choice {
    /// [`Model`] chooses whether and which tool to use (the default).
    ///
    /// [`Model`]: crate::model::ModelInfo
    pub fn auto() -> Self {
        Self::default()
    }

    /// Model must use at least one of the provided tools.
    pub fn any() -> Self {
        Self::Any {
            disable_parallel_tool_use: false,
        }
    }

    /// Model must use the [`CustomMethodDef`] with this name.
    pub fn method(name: impl Into<String>) -> Self {
        Self::Method {
            name: name.into(),
            disable_parallel_tool_use: false,
        }
    }

    /// Model must not use any tool.
    pub fn none() -> Self {
        Self::None
    }

    /// Constrain the model to at most one tool call (no parallel use). A no-op
    /// for [`Choice::None`].
    pub fn disable_parallel_tool_use(mut self) -> Self {
        match &mut self {
            Self::Auto {
                disable_parallel_tool_use,
            }
            | Self::Any {
                disable_parallel_tool_use,
            }
            | Self::Method {
                disable_parallel_tool_use,
                ..
            } => *disable_parallel_tool_use = true,
            Self::None => {}
        }
        self
    }
}

/// A **predefined** tool, identified on the wire by a versioned `type` rather
/// than a schema you supply — as opposed to a custom [`CustomMethodDef`]. Add one
/// with [`Prompt::add_tool`] (it takes anything [`Into<MethodDef>`], so these
/// drop in next to custom tools).
///
/// Most are **server-executed**: the API runs them internally and returns a
/// [`Block::ServerToolUse`] plus the tool's result block *in the response* —
/// you never handle execution and never return a [`tool::Result`]. Long-running
/// ones may pause the turn; see [`StopReason::PauseTurn`]. The exception is the
/// memory tool (the `Memory` variant, behind the `memory` feature), which is
/// **client-executed**: the API defines it but the model emits an ordinary
/// [`Use`] you run yourself, exactly like a custom tool.
///
/// Each variant's wire `type` is **versioned** (e.g. `web_search_20250305`);
/// new versions become new variants.
///
/// [`Prompt::add_tool`]: crate::Prompt::add_tool
/// [`Into<MethodDef>`]: MethodDef
/// [`Block::ServerToolUse`]: crate::prompt::message::Block::ServerToolUse
/// [`tool::Result`]: Result
/// [`StopReason::PauseTurn`]: crate::response::StopReason::PauseTurn
#[derive(Clone, Debug, Serialize, Deserialize, derive_more::From)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum ServerMethodDef {
    /// Anthropic's web search tool (`web_search_20250305`). The model issues
    /// queries and receives results it can cite via
    /// [`Citation::WebSearchResultLocation`].
    ///
    /// [`Citation::WebSearchResultLocation`]: crate::prompt::Citation::WebSearchResultLocation
    #[serde(rename = "web_search_20250305")]
    WebSearch(WebSearch),
    /// Anthropic's web fetch tool (`web_fetch_20250910`). The model retrieves
    /// the full text (or, for PDFs, the base64 bytes) of a URL that already
    /// appeared in the conversation and, when [`citations`] is enabled, cites
    /// passages on its response [`Text`] blocks via the document
    /// [`Citation`] locations. The result arrives as a
    /// [`Block::WebFetchToolResult`].
    ///
    /// [`citations`]: WebFetch::citations
    /// [`Citation`]: crate::prompt::Citation
    /// [`Text`]: crate::prompt::message::Block::Text
    /// [`Block::WebFetchToolResult`]: crate::prompt::message::Block::WebFetchToolResult
    #[serde(rename = "web_fetch_20250910")]
    WebFetch(WebFetch),
    /// The [tool-search tool], regex variant (`tool_search_tool_regex_20251119`).
    /// The model writes Python-`re`-style patterns to discover tools marked
    /// [`defer_loading`](CustomMethodDef::defer_loading); the matching definitions are
    /// expanded into the conversation on demand. See [`tool_search_regex`].
    ///
    /// [tool-search tool]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool>
    /// [`tool_search_regex`]: ServerMethodDef::tool_search_regex
    #[serde(rename = "tool_search_tool_regex_20251119")]
    ToolSearchRegex(ToolSearch<ToolSearchRegexName>),
    /// The [tool-search tool], BM25 variant (`tool_search_tool_bm25_20251119`).
    /// Like [`ToolSearchRegex`](Self::ToolSearchRegex) but the model searches
    /// with natural-language queries instead of regex. See
    /// [`tool_search_bm25`].
    ///
    /// [tool-search tool]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/tool-search-tool>
    /// [`tool_search_bm25`]: ServerMethodDef::tool_search_bm25
    #[serde(rename = "tool_search_tool_bm25_20251119")]
    ToolSearchBm25(ToolSearch<ToolSearchBm25Name>),
    /// Anthropic's [code execution] tool (`code_execution_20260120`). The model
    /// writes Python and runs it in a sandboxed container; the run's output
    /// arrives as a [`Block::CodeExecutionToolResult`]. Enabling it also unlocks
    /// [programmatic tool calling]: any custom [`CustomMethodDef`] whose
    /// [`allowed_callers`](CustomMethodDef::allowed_callers) includes
    /// [`code_execution_20260120`] may be invoked from that code, pausing the
    /// turn with a `tool_use` you fulfill normally.
    ///
    /// [code execution]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool>
    /// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    /// [`Block::CodeExecutionToolResult`]: crate::prompt::message::Block::CodeExecutionToolResult
    /// [`code_execution_20260120`]: AllowedCaller::code_execution_20260120
    #[serde(rename = "code_execution_20260120")]
    CodeExecution(CodeExecution),
    /// Anthropic's [memory tool] (`memory_20250818`) — a *client-side*
    /// predefined tool. The API defines it but does **not** run it: the model
    /// emits an ordinary [`Use`] (`name: "memory"`) whose input is a
    /// [`memory::Command`], and *you* execute it (e.g. with
    /// [`memory::FsMemoryBackend`]) and answer with a [`tool::Result`] — just
    /// like a custom tool. So it *defines* like a server tool and *executes*
    /// like a custom one. See [`Memory`].
    ///
    /// [memory tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
    /// [`memory::Command`]: crate::tool::memory::Command
    /// [`memory::FsMemoryBackend`]: crate::tool::memory::FsMemoryBackend
    /// [`tool::Result`]: Result
    #[cfg(feature = "memory")]
    #[serde(rename = "memory_20250818")]
    Memory(Memory),
    /// Anthropic's [text editor tool] (`text_editor_20250728`, the Claude-4
    /// line) — a *client-side* predefined tool, like [`Memory`]. The API
    /// defines it but does **not** run it: the model emits an ordinary [`Use`]
    /// (`name: "str_replace_based_edit_tool"`) whose input is a
    /// [`text_editor::Command`], and *you* execute it (e.g. with
    /// [`text_editor::FsEditorBackend`]) and answer with a [`tool::Result`].
    /// See [`TextEditor`].
    ///
    /// [text editor tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>
    /// [`text_editor::Command`]: crate::tool::text_editor::Command
    /// [`text_editor::FsEditorBackend`]: crate::tool::text_editor::FsEditorBackend
    /// [`tool::Result`]: Result
    #[cfg(feature = "text-editor")]
    #[serde(rename = "text_editor_20250728")]
    TextEditor(TextEditor),
    /// Anthropic's [bash tool] (`bash_20250124`) — a *client-side* predefined
    /// tool, like [`Memory`]. The API defines it but does **not** run it: the
    /// model emits an ordinary [`Use`] (`name: "bash"`) whose input is a
    /// [`bash::Command`], and *you* execute it in a sandbox (e.g. with the
    /// `bash-container` `DockerSandbox`) and answer with a [`tool::Result`]. See
    /// [`Bash`].
    ///
    /// [bash tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool>
    /// [`bash::Command`]: crate::tool::bash::Command
    /// [`tool::Result`]: Result
    #[cfg(feature = "bash")]
    #[serde(rename = "bash_20250124")]
    Bash(Bash),
}

impl ServerMethodDef {
    /// A [`WebSearch`] server tool with default configuration. Configure it
    /// with struct-update syntax, e.g.
    /// `ServerMethodDef::web_search(WebSearch { max_uses: Some(5), ..Default::default() })`.
    pub fn web_search(config: WebSearch) -> Self {
        Self::WebSearch(config)
    }

    /// A [`WebFetch`] server tool with default configuration. Configure it
    /// with struct-update syntax, e.g.
    /// `ServerMethodDef::web_fetch(WebFetch { max_uses: Some(5), ..Default::default() })`.
    pub fn web_fetch(config: WebFetch) -> Self {
        Self::WebFetch(config)
    }

    /// The regex [tool-search tool](Self::ToolSearchRegex). Add it alongside a
    /// catalog of [`defer_loading`](CustomMethodDef::defer_loading) tools (see
    /// [`Prompt::defer_tools`]) so the model can find them on demand without
    /// paying for every schema up front.
    ///
    /// [`Prompt::defer_tools`]: crate::Prompt::defer_tools
    pub fn tool_search_regex() -> Self {
        Self::ToolSearchRegex(ToolSearch::default())
    }

    /// The BM25 [tool-search tool](Self::ToolSearchBm25), the
    /// natural-language counterpart to [`tool_search_regex`](Self::tool_search_regex).
    pub fn tool_search_bm25() -> Self {
        Self::ToolSearchBm25(ToolSearch::default())
    }

    /// The [code execution](Self::CodeExecution) tool with default
    /// configuration. Required for [programmatic tool calling].
    ///
    /// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    pub fn code_execution() -> Self {
        Self::CodeExecution(CodeExecution::default())
    }

    /// The [memory](Self::Memory) tool with default configuration. A
    /// *client-side* tool: you execute its [`Command`](memory::Command)s
    /// yourself (the API does not run it). See [`Memory`].
    #[cfg(feature = "memory")]
    pub fn memory() -> Self {
        Self::Memory(Memory::default())
    }

    /// The [text editor](Self::TextEditor) tool with default configuration. A
    /// *client-side* tool: you execute its [`Command`](text_editor::Command)s
    /// yourself (the API does not run it). See [`TextEditor`].
    #[cfg(feature = "text-editor")]
    pub fn text_editor() -> Self {
        Self::TextEditor(TextEditor::default())
    }

    /// The [bash](Self::Bash) tool with default configuration. A *client-side*
    /// tool: you execute its [`Command`](bash::Command)s in a sandbox yourself
    /// (the API does not run it). See [`Bash`].
    #[cfg(feature = "bash")]
    pub fn bash() -> Self {
        Self::Bash(Bash::default())
    }

    /// Whether this server tool carries a cache breakpoint.
    pub fn is_cached(&self) -> bool {
        self.cache_control().is_some()
    }

    /// The bare wire `name` the model emits for this tool (e.g. `"memory"`,
    /// `"web_search"`) — *not* the versioned `type` tag. Used by
    /// [`ToolBox`](crate::tool::ToolBox) to route client-executed predefined
    /// tools (like [`memory`](Self::Memory)) by their fixed, un-namespaced name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::WebSearch(_) => "web_search",
            Self::WebFetch(_) => "web_fetch",
            Self::ToolSearchRegex(_) => "tool_search_tool_regex",
            Self::ToolSearchBm25(_) => "tool_search_tool_bm25",
            Self::CodeExecution(_) => "code_execution",
            #[cfg(feature = "memory")]
            Self::Memory(_) => "memory",
            #[cfg(feature = "text-editor")]
            Self::TextEditor(_) => "str_replace_based_edit_tool",
            #[cfg(feature = "bash")]
            Self::Bash(_) => "bash",
        }
    }

    /// This tool's cache breakpoint, if any.
    fn cache_control(&self) -> Option<&crate::prompt::message::CacheControl> {
        match self {
            Self::WebSearch(c) => c.cache_control.as_ref(),
            Self::WebFetch(c) => c.cache_control.as_ref(),
            Self::ToolSearchRegex(c) => c.cache_control.as_ref(),
            Self::ToolSearchBm25(c) => c.cache_control.as_ref(),
            Self::CodeExecution(c) => c.cache_control.as_ref(),
            #[cfg(feature = "memory")]
            Self::Memory(c) => c.cache_control.as_ref(),
            #[cfg(feature = "text-editor")]
            Self::TextEditor(c) => c.cache_control.as_ref(),
            #[cfg(feature = "bash")]
            Self::Bash(c) => c.cache_control.as_ref(),
        }
    }

    /// Set this tool's cache breakpoint.
    fn set_cache_control(
        &mut self,
        cache_control: crate::prompt::message::CacheControl,
    ) {
        match self {
            Self::WebSearch(c) => c.cache_control = Some(cache_control),
            Self::WebFetch(c) => c.cache_control = Some(cache_control),
            Self::ToolSearchRegex(c) => c.cache_control = Some(cache_control),
            Self::ToolSearchBm25(c) => c.cache_control = Some(cache_control),
            Self::CodeExecution(c) => c.cache_control = Some(cache_control),
            #[cfg(feature = "memory")]
            Self::Memory(c) => c.cache_control = Some(cache_control),
            #[cfg(feature = "text-editor")]
            Self::TextEditor(c) => c.cache_control = Some(cache_control),
            #[cfg(feature = "bash")]
            Self::Bash(c) => c.cache_control = Some(cache_control),
        }
    }
}

/// Configuration for the tool-search server tools
/// ([`ServerMethodDef::ToolSearchRegex`] / [`ServerMethodDef::ToolSearchBm25`]). The wire
/// `name` (`tool_search_tool_regex` / `tool_search_tool_bm25`) is fixed by the
/// marker type `N` and supplied automatically; the only knob is an optional
/// cache breakpoint. Construct via [`ServerMethodDef::tool_search_regex`] /
/// [`ServerMethodDef::tool_search_bm25`].
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct ToolSearch<N> {
    /// The fixed tool `name`, supplied automatically by [`Default`]. Not meant
    /// to be set by hand.
    #[doc(hidden)]
    #[serde(default)]
    pub name: N,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

/// Configuration for the [`ServerMethodDef::WebSearch`] tool.
///
/// All fields are optional; the wire `name` (`"web_search"`) is fixed and
/// supplied automatically. Use either [`allowed_domains`] or [`blocked_domains`],
/// not both.
///
/// [`allowed_domains`]: WebSearch::allowed_domains
/// [`blocked_domains`]: WebSearch::blocked_domains
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct WebSearch {
    /// The fixed tool `name` (`"web_search"`), supplied automatically by
    /// [`Default`]. Not meant to be set by hand; use `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: WebSearchName,
    /// Maximum number of searches the model may run per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Only return results from these domains (bare host, no scheme). Mutually
    /// exclusive with [`blocked_domains`](WebSearch::blocked_domains).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<Cow<'static, str>>>,
    /// Never return results from these domains (bare host, no scheme). Mutually
    /// exclusive with [`allowed_domains`](WebSearch::allowed_domains).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<Cow<'static, str>>>,
    /// Approximate user location used to bias results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_location: Option<UserLocation>,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl WebSearch {}

/// Configuration for the [`ServerMethodDef::WebFetch`] tool.
///
/// All fields are optional; the wire `name` (`"web_fetch"`) is fixed and
/// supplied automatically. Use either [`allowed_domains`] or [`blocked_domains`],
/// not both. Unlike [`WebSearch`], citations are *off* by default — set
/// [`citations`] to have the model cite passages from the fetched document.
///
/// The model may only fetch a URL that already appeared in the conversation
/// (a user message, a client tool result, or a prior search/fetch result); it
/// cannot fetch URLs it invents.
///
/// [`allowed_domains`]: WebFetch::allowed_domains
/// [`blocked_domains`]: WebFetch::blocked_domains
/// [`citations`]: WebFetch::citations
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct WebFetch {
    /// The fixed tool `name` (`"web_fetch"`), supplied automatically by
    /// [`Default`]. Not meant to be set by hand; use `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: WebFetchName,
    /// Maximum number of fetches the model may run per request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_uses: Option<u32>,
    /// Only fetch from these domains (bare host, no scheme). Mutually exclusive
    /// with [`blocked_domains`](WebFetch::blocked_domains).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_domains: Option<Vec<Cow<'static, str>>>,
    /// Never fetch from these domains (bare host, no scheme). Mutually
    /// exclusive with [`allowed_domains`](WebFetch::allowed_domains).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_domains: Option<Vec<Cow<'static, str>>>,
    /// Enable citations on the fetched document, so the model cites passages on
    /// its response [`Text`](crate::prompt::message::Block::Text) blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citations: Option<crate::prompt::message::CitationsConfig>,
    /// Truncate fetched content to roughly this many tokens, to cap the token
    /// cost of large pages and PDFs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_content_tokens: Option<u32>,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl WebFetch {}

/// Configuration for the [code execution] server tool
/// ([`ServerMethodDef::CodeExecution`]). Construct via
/// [`ServerMethodDef::code_execution`]; the only knob is an optional cache
/// breakpoint.
///
/// [code execution]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool>
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct CodeExecution {
    /// The fixed tool `name` (`"code_execution"`), supplied automatically by
    /// [`Default`]. Not meant to be set by hand; use `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: CodeExecutionName,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

/// Configuration for Anthropic's [memory tool] ([`ServerMethodDef::Memory`]) — a
/// *client-side* predefined tool. Added by versioned name with no schema of
/// your own (construct via [`Memory::latest`]); the model then emits ordinary
/// [`Use`] blocks (`name: "memory"`) carrying a [`memory::Command`] that you
/// execute and answer with a [`tool::Result`]. See [`memory::FsMemoryBackend`]
/// for a ready-made executor.
///
/// The only knob is an optional cache breakpoint. The wire `name` (`"memory"`)
/// is fixed and supplied automatically.
///
/// [memory tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool>
/// [`memory::Command`]: crate::tool::memory::Command
/// [`memory::FsMemoryBackend`]: crate::tool::memory::FsMemoryBackend
/// [`tool::Result`]: Result
#[cfg(feature = "memory")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Memory {
    /// The fixed tool `name` (`"memory"`), supplied automatically by
    /// [`Default`]. Not meant to be set by hand; use `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: MemoryName,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

#[cfg(feature = "memory")]
impl Memory {
    /// The newest memory-tool version this crate supports (`memory_20250818`),
    /// with default configuration. Named `latest` because Anthropic versions
    /// the tool: when a newer one ships it becomes a new [`ServerMethodDef`] variant
    /// and this points at it.
    pub fn latest() -> Self {
        Self::default()
    }
}

/// Bridges the [`Memory`] front-door type straight into a [`MethodDef`] so
/// [`Prompt::add_tool`] accepts it (`Into` is not transitive, so
/// `Memory: Into<ServerMethodDef>` alone would not give `Memory: Into<MethodDef>`).
#[cfg(feature = "memory")]
impl From<Memory> for MethodDef {
    fn from(memory: Memory) -> Self {
        MethodDef::Server(ServerMethodDef::Memory(memory))
    }
}

/// Configuration for Anthropic's [text editor tool]
/// ([`ServerMethodDef::TextEditor`]) — a *client-side* predefined tool, like
/// [`Memory`]. Added by versioned name with no schema of your own (construct
/// via [`TextEditor::latest`]); the model then emits ordinary [`Use`] blocks
/// (`name: "str_replace_based_edit_tool"`) carrying a [`text_editor::Command`]
/// that you execute and answer with a [`tool::Result`]. See
/// [`text_editor::FsEditorBackend`] for a ready-made executor.
///
/// The wire `name` (`"str_replace_based_edit_tool"`) is fixed and supplied
/// automatically. The only knobs are an optional [`max_characters`] truncation
/// cap on `view` and a cache breakpoint.
///
/// [text editor tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool>
/// [`text_editor::Command`]: crate::tool::text_editor::Command
/// [`text_editor::FsEditorBackend`]: crate::tool::text_editor::FsEditorBackend
/// [`tool::Result`]: Result
/// [`max_characters`]: TextEditor::max_characters
#[cfg(feature = "text-editor")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct TextEditor {
    /// The fixed tool `name` (`"str_replace_based_edit_tool"`), supplied
    /// automatically by [`Default`]. Not meant to be set by hand; use
    /// `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: TextEditorName,
    /// Truncate a `view`'s file contents to roughly this many characters, to
    /// cap the token cost of large files. (`text_editor_20250728`+ only.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_characters: Option<u32>,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

#[cfg(feature = "text-editor")]
impl TextEditor {
    /// The newest text-editor version this crate supports
    /// (`text_editor_20250728`, the Claude-4 line), with default configuration.
    /// Named `latest` because Anthropic versions the tool: when a newer one
    /// ships it becomes a new [`ServerMethodDef`] variant and this points at it.
    pub fn latest() -> Self {
        Self::default()
    }
}

/// Bridges the [`TextEditor`] front-door type straight into a [`MethodDef`] so
/// [`Prompt::add_tool`] accepts it (`Into` is not transitive, so
/// `TextEditor: Into<ServerMethodDef>` alone would not give
/// `TextEditor: Into<MethodDef>`).
#[cfg(feature = "text-editor")]
impl From<TextEditor> for MethodDef {
    fn from(editor: TextEditor) -> Self {
        MethodDef::Server(ServerMethodDef::TextEditor(editor))
    }
}

/// Front-door for the [bash tool] ([`ServerMethodDef::Bash`]) — a *client-side*
/// predefined tool, like [`Memory`]/[`TextEditor`]. [`latest`](Self::latest)
/// adds it by versioned name (`bash_20250124`, the model-trained narrow schema
/// that only elicits `command`/`restart`); [`rich`](Self::rich) instead yields a
/// [`Custom`](MethodDef::Custom) def whose schema is *derived* from
/// [`bash::Known`](crate::tool::bash::Known), advertising the full
/// run/restart/poll/kill vocabulary (background jobs, timeouts) to the model.
/// Execute its [`Command`](crate::tool::bash::Command)s with a
/// [`BashTool`](crate::tool::bash::BashTool) over some
/// [`BashSandbox`](crate::tool::bash::BashSandbox).
///
/// [bash tool]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool>
#[cfg(feature = "bash")]
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Bash {
    /// The fixed tool `name` (`"bash"`), supplied automatically by [`Default`].
    /// Not meant to be set by hand; use `..Default::default()`.
    #[doc(hidden)]
    #[serde(default)]
    pub name: BashName,
    /// Set a cache breakpoint on this tool. See [`Prompt::cache`].
    ///
    /// [`Prompt::cache`]: crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

#[cfg(feature = "bash")]
impl Bash {
    /// The newest bash-tool version this crate supports (`bash_20250124`), with
    /// default configuration — the predefined, model-trained schema (only
    /// `command`/`restart`). Named `latest` because Anthropic versions the tool:
    /// when a newer one ships it becomes a new [`ServerMethodDef`] variant and
    /// this points at it.
    pub fn latest() -> Self {
        Self::default()
    }

    /// A [`Custom`](MethodDef::Custom) bash def whose input schema is *derived*
    /// from [`bash::Known`](crate::tool::bash::Known) (via [`schemars`] +
    /// [`sanitize_for_anthropic`], the same path the typed-tool layer uses), so
    /// the model sees the full run/restart/poll/kill vocabulary the predefined
    /// `bash_20250124` schema omits. Use this when you want the model to drive
    /// background jobs (`background`/`timeout_secs`) and `poll`/`kill` them.
    ///
    /// [`sanitize_for_anthropic`]: crate::prompt::output::sanitize_for_anthropic
    pub fn rich() -> CustomMethodDef {
        let mut schema = serde_json::to_value(schemars::schema_for!(
            crate::tool::bash::Known
        ))
        .expect("schemars Schema always serializes");
        crate::prompt::output::sanitize_for_anthropic(&mut schema);
        CustomMethodDef::builder("bash")
            .description(
                "Run shell commands in a persistent sandbox session. Provide \
                 `command` to run (optionally `background: true` and \
                 `timeout_secs`), `restart: true` to reset the session, \
                 `poll: <job>` to check a background job, or `kill: <job>` to \
                 stop one.",
            )
            .schema(schema)
            .build_unchecked()
    }
}

/// Bridges the [`Bash`] front-door type straight into a [`MethodDef`] so
/// [`Prompt::add_tool`] accepts it (`Into` is not transitive — see
/// [`From<TextEditor>`](MethodDef)).
#[cfg(feature = "bash")]
impl From<Bash> for MethodDef {
    fn from(bash: Bash) -> Self {
        MethodDef::Server(ServerMethodDef::Bash(bash))
    }
}

/// Define a zero-sized server-tool `name` marker that always (de)serializes as
/// one constant string, so the wire `name` can never be set to anything else.
/// Each is public-but-`#[doc(hidden)]` so the owning config struct supports
/// `..Default::default()` (a *private* field would hit E0451 under FRU); none
/// are part of the stable API.
macro_rules! tool_name_marker {
    ($(#[$meta:meta])* $name:ident => $wire:literal) => {
        $(#[$meta])*
        #[doc(hidden)]
        #[derive(Clone, Copy, Debug, Default)]
        #[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
        pub struct $name;

        impl Serialize for $name {
            fn serialize<S: serde::Serializer>(
                &self,
                serializer: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                serializer.serialize_str($wire)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: serde::Deserializer<'de>>(
                deserializer: D,
            ) -> std::result::Result<Self, D::Error> {
                let name = Cow::<str>::deserialize(deserializer)?;
                if name == $wire {
                    Ok($name)
                } else {
                    Err(serde::de::Error::custom(concat!(
                        "expected tool name \"",
                        $wire,
                        "\"",
                    )))
                }
            }
        }
    };
}

tool_name_marker!(
    /// The fixed `"web_search"` name for [`WebSearch`].
    WebSearchName => "web_search"
);
tool_name_marker!(
    /// The fixed `"web_fetch"` name for [`WebFetch`].
    WebFetchName => "web_fetch"
);
tool_name_marker!(
    /// The fixed `"tool_search_tool_regex"` name for
    /// [`ServerMethodDef::ToolSearchRegex`].
    ToolSearchRegexName => "tool_search_tool_regex"
);
tool_name_marker!(
    /// The fixed `"tool_search_tool_bm25"` name for
    /// [`ServerMethodDef::ToolSearchBm25`].
    ToolSearchBm25Name => "tool_search_tool_bm25"
);
tool_name_marker!(
    /// The fixed `"code_execution"` name for [`CodeExecution`].
    CodeExecutionName => "code_execution"
);
#[cfg(feature = "memory")]
tool_name_marker!(
    /// The fixed `"memory"` name for [`Memory`].
    MemoryName => "memory"
);
#[cfg(feature = "text-editor")]
tool_name_marker!(
    /// The fixed `"str_replace_based_edit_tool"` name for [`TextEditor`].
    TextEditorName => "str_replace_based_edit_tool"
);
#[cfg(feature = "bash")]
tool_name_marker!(
    /// The fixed `"bash"` name for [`Bash`].
    BashName => "bash"
);

/// Approximate user location used to bias [`WebSearch`] results. Serializes
/// with `type: "approximate"`.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename = "approximate")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct UserLocation {
    /// City name, e.g. `"San Francisco"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<Cow<'static, str>>,
    /// Region or state, e.g. `"California"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub region: Option<Cow<'static, str>>,
    /// ISO 3166-1 alpha-2 country code, e.g. `"US"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<Cow<'static, str>>,
    /// IANA timezone, e.g. `"America/Los_Angeles"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timezone: Option<Cow<'static, str>>,
}

impl UserLocation {}

/// An entry in a [`Prompt`]'s tools array: either a custom [`CustomMethodDef`] you
/// execute yourself via [`Tool::call`], or a [`ServerMethodDef`] the API runs
/// internally.
///
/// Distinguished on the wire by the presence of a `type` field — predefined
/// (server) tools carry a versioned one, custom tools do not. Most users never
/// name this type: [`Prompt::add_tool`] takes anything [`Into`] a `MethodDef`
/// (a [`CustomMethodDef`], a [`ServerMethodDef`], or — with the `memory` feature — a
/// `Memory`) and wraps the right variant.
///
/// [`Prompt::add_tool`]: crate::Prompt::add_tool
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    derive_more::From,
    derive_more::IsVariant,
)]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum MethodDef {
    /// A server-side tool the API executes (carries a `type`).
    Server(ServerMethodDef),
    /// A custom tool you execute via [`Tool::call`].
    Custom(CustomMethodDef),
}

impl MethodDef {
    /// The bare wire `name` this definition advertises — a custom tool's
    /// [`CustomMethodDef::name`] or a [`ServerMethodDef`]'s fixed name. This is the key a
    /// [`ToolBox`](crate::tool::ToolBox) routes on (namespaced for custom
    /// tools, left bare for server-declared ones).
    pub fn name(&self) -> &str {
        match self {
            Self::Custom(method) => &method.name,
            Self::Server(server) => server.name(),
        }
    }

    /// The custom [`CustomMethodDef`], if this is a [`MethodDef::Custom`].
    pub fn as_method(&self) -> Option<&CustomMethodDef> {
        match self {
            Self::Custom(method) => Some(method),
            Self::Server(_) => None,
        }
    }

    /// The custom [`CustomMethodDef`] mutably, if this is a [`MethodDef::Custom`].
    pub fn as_method_mut(&mut self) -> Option<&mut CustomMethodDef> {
        match self {
            Self::Custom(method) => Some(method),
            Self::Server(_) => None,
        }
    }

    /// Returns true if this tool has a cache breakpoint set.
    pub fn is_cached(&self) -> bool {
        match self {
            Self::Custom(method) => method.is_cached(),
            Self::Server(server) => server.is_cached(),
        }
    }

    /// Set a cache breakpoint on this tool (custom or server), with a
    /// caller-provided [`CacheControl`](crate::prompt::message::CacheControl).
    pub fn cache_with(
        &mut self,
        cache_control: crate::prompt::message::CacheControl,
    ) -> &mut Self {
        match self {
            Self::Custom(method) => {
                method.cache_with(cache_control);
            }
            Self::Server(server) => server.set_cache_control(cache_control),
        }
        self
    }
}

/// A `Tool` that the [`Assistant`] can [`Use`]. Tools can have multiple
/// [`CustomMethodDef`]s. Tools should generally go in the [`ToolBox`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
#[async_trait::async_trait]
pub trait Tool: Send {
    /// [`Tool`] name.
    fn name(&self) -> &str;
    /// The [`MethodDef`]s this [`Tool`] contributes to a [`Prompt`]'s tools
    /// array. Usually [`Custom`](MethodDef::Custom) method schemas you execute,
    /// but a client-executed *predefined* tool (e.g. the
    /// [`memory`](crate::tool::memory) backend) instead contributes a
    /// [`Server`](MethodDef::Server) def — added by versioned name, routed by its
    /// fixed bare name rather than namespaced. See
    /// [`ToolBox`](crate::tool::ToolBox).
    fn definitions(&self) -> Vec<MethodDef>;
    /// [`Use`] the [`Tool`], returning a [`tool::Result`].
    ///
    /// [`tool::Result`]: Result
    async fn call(&mut self, call: Use) -> Result;
    /// Serialize tool state to json [`Value`]. [`Null`] if not possible.
    ///
    /// # Note:
    ///
    /// Takes &mut self to allow tools to update internal state during
    /// serialization if needed and because of lifetime issues with `&self`.
    ///
    /// For tools with external persistence (like databases), this should
    /// only serialize configuration/connection info, not the full state.
    ///
    /// [`Value`]: serde_json::Value
    /// [`Null`]: serde_json::Value::Null
    async fn save_json(&mut self) -> serde_json::Value {
        serde_json::Value::Null
    }
    /// Deserialize state from json [`Value`] if possible.
    ///
    /// For tools with external persistence, this should restore configuration
    /// and ensure the external state is accessible/initialized.
    // String is used for the message because a boxed error is not Send.
    async fn load_json(
        &mut self,
        _json: serde_json::Value,
    ) -> std::result::Result<(), String> {
        Ok(())
    }

    /// Called once when the tool is first added to a prompt or toolbox.
    /// Use this to set up initial context, instructions, or static content.
    ///
    /// # Note:
    /// - This is called only once per tool lifetime in a conversation
    /// - Use for setting up tool instructions, initial context blocks
    /// - Should be idempotent in case called multiple times
    async fn on_init(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Called before each turn/message exchange.
    /// Use this to update dynamic context, recent state, or per-turn information.
    ///
    /// # Note:
    /// - This is called before each user message or assistant response
    /// - Use for updating dynamic content like recent memories, current state
    /// - Should efficiently update existing content rather than appending
    async fn on_turn(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Called once when the tool is being torn down (e.g. at the end of a
    /// conversation, via [`ToolBox::teardown`](crate::tool::ToolBox::teardown)
    /// or [`Prompt::teardown_tool`]). Use this to release external resources a
    /// tool acquired in [`on_init`](Self::on_init) — close a connection, stop a
    /// subprocess, tear down a sandbox container.
    ///
    /// # Note:
    /// - Unlike [`on_init`](Self::on_init), teardown is **best-effort**: a
    ///   container's owner runs *every* tool's teardown even if an earlier one
    ///   errors, so a single failure can't leak the rest. Return an error to
    ///   report a problem, but don't rely on it halting other teardowns.
    /// - There is no async [`Drop`]; an explicit teardown is the contract.
    ///   Resource-owning tools may *also* keep a blocking `Drop` as a leak guard.
    ///
    /// [`Prompt::teardown_tool`]: crate::Prompt::teardown_tool
    async fn on_teardown(
        &mut self,
        _prompt: &mut Prompt,
    ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }
}

static_assertions::assert_obj_safe!(Tool);
// Ensure Tool is Send (but not Sync) for use in async contexts and ToolBox
static_assertions::assert_impl_all!(dyn Tool: Send);

/// `CustomMethodDef` definition for a [`Tool`] a [`Model`] can [`Use`] while
/// completing a [`prompt::Message`].
///
/// [`prompt::Message`]: crate::prompt::Message
/// [`Model`]: crate::model::ModelInfo
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(try_from = "MethodBuilder")]
#[serde(rename = "tool")]
pub struct CustomMethodDef {
    /// Name of the function. This should be in a `Tool::function` format.
    pub name: Cow<'static, str>,
    /// Description of the tool. The model will use this as documentation.
    pub description: Cow<'static, str>,
    /// Input schema for the tool. See [tool use guide] for more information.
    /// The schema is not validated by this crate but should conform to the
    /// [JSON Schema] specification.
    ///
    /// [tool use guide]: <https://docs.anthropic.com/en/docs/build-with-claude/tool-use>
    /// [JSON Schema]: <https://json-schema.org/>
    #[serde(rename = "input_schema")]
    pub schema: serde_json::Value,
    /// Set a cache breakpoint. See [`Prompt::cache`] for more information.
    ///
    /// [`Prompt::cache`] crate::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
    /// When `Some(true)`, enables [strict tool use] — the API uses
    /// grammar-constrained decoding so [`Use::input`] is guaranteed to
    /// validate against [`schema`]. Defaults to `None` (best-effort
    /// adherence only).
    ///
    /// Strict mode is compatible with [`Prompt::output_config`] — the API
    /// accepts both in the same request, but any given response turn
    /// emits either a `tool_use` block or the constrained output text,
    /// not both.
    ///
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    /// [`Use::input`]: crate::tool::Use::input
    /// [`schema`]: CustomMethodDef::schema
    /// [`Prompt::output_config`]: crate::Prompt::output_config
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// When `Some(true)`, the API may defer loading this tool's full definition
    /// until the model selects it (an optimization for large tool sets used
    /// with the tool-search tool). Defaults to `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
    /// Which contexts may invoke this tool — the
    /// [`allowed_callers`](https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling)
    /// field. [`None`] (omitted) means the API's default of `["direct"]`. List
    /// [`AllowedCaller::code_execution_20260120`] to let the model call this
    /// tool from a code-execution container ([programmatic tool calling]).
    ///
    /// The docs advise choosing *one* caller per tool rather than enabling
    /// both; a `tool_choice` naming a tool whose callers omit `direct` is a
    /// `400`.
    ///
    /// Programmatic calling is incompatible with a few other knobs: a
    /// code-execution-callable tool cannot also set [`strict`](Self::strict)
    /// (`true`), `tool_choice` cannot *force* it, and
    /// `disable_parallel_tool_use` is unsupported alongside it.
    ///
    /// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<AllowedCaller>>,
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for CustomMethodDef {
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        // Can't panic because derived Serialize
        let mut payload = serde_json::to_value(self).unwrap();
        // Can't panic because we know it's an object
        payload.as_object_mut().unwrap().remove("cache_control");
        payload.as_object_mut().unwrap().remove("strict");
        payload.as_object_mut().unwrap().remove("defer_loading");
        payload.as_object_mut().unwrap().remove("allowed_callers");

        if options.tool_use {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(&payload).unwrap().into(),
                    ),
                    Event::End(TagEnd::CodeBlock),
                ]
                .into_iter(),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

impl TryFrom<MethodBuilder> for CustomMethodDef {
    type Error = ToolBuildError;

    fn try_from(
        builder: MethodBuilder,
    ) -> std::result::Result<Self, Self::Error> {
        builder.build()
    }
}

/// A builder for creating a [`CustomMethodDef`] with some basic validation. See
/// [`CustomMethodDef::builder`] to create one.
pub struct MethodBuilder {
    tool: CustomMethodDef,
}

// `CustomMethodDef` is annotated with `#[serde(try_from = "MethodBuilder")]`, so
// deserializing a `CustomMethodDef` routes through `MethodBuilder::deserialize` and
// then `MethodBuilder::build`. If we derived `Deserialize` on
// `MethodBuilder`, serde would generate an impl that defers to
// `CustomMethodDef::deserialize`, which in turn calls back into
// `MethodBuilder::deserialize` — an infinite loop. So we hand-roll it via
// a private `Foreign` helper struct that owns the actual field mapping.
// Every public field on `CustomMethodDef` must have a matching entry here.
impl<'de> Deserialize<'de> for MethodBuilder {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Foreign {
            name: Cow<'static, str>,
            description: Cow<'static, str>,
            input_schema: serde_json::Value,
            #[serde(default)]
            cache_control: Option<crate::prompt::message::CacheControl>,
            #[serde(default)]
            strict: Option<bool>,
            #[serde(default)]
            defer_loading: Option<bool>,
            #[serde(default)]
            allowed_callers: Option<Vec<AllowedCaller>>,
        }

        let foreign = Foreign::deserialize(deserializer)?;

        let Foreign {
            name,
            description,
            input_schema,
            cache_control,
            strict,
            defer_loading,
            allowed_callers,
        } = foreign;

        Ok(MethodBuilder {
            tool: CustomMethodDef {
                name,
                description,
                schema: input_schema,
                cache_control,
                strict,
                defer_loading,
                allowed_callers,
            },
        })
    }
}

impl MethodBuilder {
    /// Set the description for the tool.
    pub fn description(
        mut self,
        description: impl Into<Cow<'static, str>>,
    ) -> Self {
        self.tool.description = description.into();
        self
    }

    /// Set the [`strict`] flag on the [`CustomMethodDef`], enabling [strict tool
    /// use] (grammar-constrained decoding of tool inputs).
    ///
    /// [`strict`]: CustomMethodDef::strict
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn strict(mut self, strict: bool) -> Self {
        self.tool.strict = Some(strict);
        self
    }

    /// Set the [`defer_loading`] flag on the [`CustomMethodDef`], allowing the API to
    /// defer loading this tool's definition until the model selects it.
    ///
    /// [`defer_loading`]: CustomMethodDef::defer_loading
    pub fn defer_loading(mut self, defer_loading: bool) -> Self {
        self.tool.defer_loading = Some(defer_loading);
        self
    }

    /// Set [`allowed_callers`] — the contexts that may invoke this tool — from
    /// any iterator of [`AllowedCaller`]. An empty iterator clears it back to
    /// the API default (`["direct"]`). For the common "callable only from a
    /// code-execution container" case, see [`programmatic`](Self::programmatic).
    ///
    /// [`allowed_callers`]: CustomMethodDef::allowed_callers
    pub fn allowed_callers(
        mut self,
        callers: impl IntoIterator<Item = AllowedCaller>,
    ) -> Self {
        let callers: Vec<_> = callers.into_iter().collect();
        self.tool.allowed_callers = (!callers.is_empty()).then_some(callers);
        self
    }

    /// Mark this tool callable only from a `code_execution_20260120` container
    /// ([programmatic tool calling]) — shorthand for
    /// [`allowed_callers`]\([`[AllowedCaller::code_execution_20260120()]`]).
    ///
    /// [`allowed_callers`]: Self::allowed_callers
    /// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    pub fn programmatic(self) -> Self {
        self.allowed_callers([AllowedCaller::code_execution_20260120()])
    }

    /// Set a cache breakpoint at this [`CustomMethodDef`] by setting [`cache_control`] to
    /// [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// [`cache_control`]: Spec::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    pub fn cache(mut self) -> Self {
        self.tool.cache_control =
            Some(crate::prompt::message::CacheControl::ephemeral());
        self
    }

    /// Set the [`CustomMethodDef::schema`]. The schema should be a JSON Schema
    /// object conforming to the [JSON Schema] specification like the following
    /// example:
    ///
    /// ```json
    /// {
    ///     "type": "object",
    ///     "properties": {
    ///         "letter": {
    ///             "type": "string",
    ///             "description": "The letter to count",
    ///         },
    ///         "string": {
    ///             "type": "string",
    ///             "description": "The string to count letters in",
    ///         },
    ///     },
    ///     "required": ["letter", "string"],
    /// },
    /// ```
    ///
    /// NOTE: On [`build`], There is some very basic validation done on the
    /// schema to ensure that it is an object with properties and required
    /// fields. This is not exhaustive and does not guarantee that the schema
    /// will be accepted by the API or that the agent will be able to use the
    /// tool.
    ///
    /// [JSON Schema]: <https://json-schema.org/>
    /// [`build`]: MethodBuilder::build
    // TODO: This could be improved by using a JSON Schema library.
    pub fn schema(mut self, schema: serde_json::Value) -> Self {
        self.tool.schema = schema;
        self
    }

    /// Add a string parameter to the schema.
    pub fn string_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "string", required)
    }

    /// Add a number parameter to the schema.
    pub fn number_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "number", required)
    }

    /// Add a boolean parameter to the schema.
    pub fn boolean_param(
        self,
        name: &str,
        description: &str,
        required: bool,
    ) -> Self {
        self.add_param(name, description, "boolean", required)
    }

    /// Helper method to add a parameter to the schema.
    fn add_param(
        mut self,
        name: &str,
        description: &str,
        param_type: &str,
        required: bool,
    ) -> Self {
        // Initialize schema if it's null
        if self.tool.schema.is_null() {
            self.tool.schema = serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            });
        }

        // Add the property
        if let Some(properties) = self
            .tool
            .schema
            .get_mut("properties")
            .and_then(|p| p.as_object_mut())
        {
            properties.insert(
                name.to_string(),
                serde_json::json!({
                    "type": param_type,
                    "description": description
                }),
            );
        }

        // Add to required array if needed
        if required
            && let Some(required_array) = self
                .tool
                .schema
                .get_mut("required")
                .and_then(|r| r.as_array_mut())
        {
            required_array.push(serde_json::Value::String(name.to_string()));
        }

        self
    }

    /// This will build the [`CustomMethodDef`] without checking any of the fields. This is
    /// recommended only with static strings.
    pub fn build_unchecked(self) -> CustomMethodDef {
        self.tool
    }

    /// Build the tool, validating name, description, and the tool schema.
    fn is_valid_input_schema(
        schema: &serde_json::Value,
    ) -> std::result::Result<(), Cow<'static, str>> {
        let obj = if let Some(obj) = schema.as_object() {
            if obj.is_empty() {
                return Err("Input `schema` is an empty object.".into());
            }

            obj
        } else {
            return Err(format!(
                "Input `schema` not an object: `{}`",
                serde_json::to_string_pretty(schema).unwrap(),
            )
            .into());
        };

        // `properties` is optional: a no-arg method (e.g. a `clear` with no
        // fields) has none. Validate its shape only when present, treating
        // absence as an empty property set.
        let empty = serde_json::Map::new();
        let properties = match obj.get("properties") {
            Some(serde_json::Value::Object(o)) => o,
            Some(_) => return Err("`properties` must be an object.".into()),
            None => &empty,
        };

        // `required` is optional per JSON Schema. Validate only when present;
        // every listed key must exist in `properties`.
        if let Some(required) = obj.get("required") {
            let required = required.as_array().ok_or_else(|| {
                format!(
                    "Input `schema` `required` not an array: `{}`",
                    serde_json::to_string(required).unwrap()
                )
            })?;

            for key in required {
                match key.as_str() {
                    Some(key) if properties.contains_key(key) => {}
                    Some(key) => {
                        return Err(format!(
                            "`required` key `{key}` not found in `properties`.",
                        )
                        .into());
                    }
                    None => {
                        return Err(format!(
                            "`required` key not a string: `{}`",
                            serde_json::to_string(key).unwrap()
                        )
                        .into());
                    }
                }
            }
        }

        Ok(())
    }

    /// This will build the [`CustomMethodDef`] and do some basic validation on the fields.
    /// This does not guarantee that the tool will be accepted by the API.
    pub fn build(self) -> std::result::Result<CustomMethodDef, ToolBuildError> {
        if self.tool.name.is_empty() {
            return Err(ToolBuildError::EmptyName);
        }

        if self.tool.description.is_empty() {
            return Err(ToolBuildError::EmptyDescription);
        }

        if self.tool.schema.is_null() {
            return Err(ToolBuildError::EmptyInputSchema);
        }

        if let Err(err_msg) = Self::is_valid_input_schema(&self.tool.schema) {
            return Err(ToolBuildError::InvalidInputSchema {
                message: err_msg,
                schema: self.tool.schema,
            });
        }

        Ok(self.tool)
    }
}

/// Errors that can occur when building a [`CustomMethodDef`] with a [`MethodBuilder`].
#[derive(Debug, thiserror::Error)]
#[allow(missing_docs)]
pub enum ToolBuildError {
    #[error("Name unset.")]
    EmptyName,
    #[error("Description unset.")]
    EmptyDescription,
    #[error("Input schema unset.")]
    EmptyInputSchema,
    #[error("Invalid input schema becuase: {message}")]
    InvalidInputSchema {
        schema: serde_json::Value,
        message: Cow<'static, str>,
    },
}

impl CustomMethodDef {
    /// Use a builder to create a new tool with some very basic validation.
    pub fn builder(name: impl Into<Cow<'static, str>>) -> MethodBuilder {
        MethodBuilder {
            tool: CustomMethodDef {
                name: name.into(),
                description: Cow::Owned(String::new()),
                schema: serde_json::Value::Null,
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            },
        }
    }

    /// Create a simple method with just a name and description.
    /// Uses an empty object schema with no required fields.
    pub fn simple(
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
    ) -> Self {
        CustomMethodDef {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
            cache_control: None,
            strict: None,
            defer_loading: None,
            allowed_callers: None,
        }
    }

    /// Create a method that takes a single string parameter.
    pub fn with_string_param(
        name: impl Into<Cow<'static, str>>,
        description: impl Into<Cow<'static, str>>,
        param_name: &str,
        param_description: &str,
        required: bool,
    ) -> Self {
        let required_array = if required { vec![param_name] } else { vec![] };

        CustomMethodDef {
            name: name.into(),
            description: description.into(),
            schema: serde_json::json!({
                "type": "object",
                "properties": {
                    param_name: {
                        "type": "string",
                        "description": param_description
                    }
                },
                "required": required_array
            }),
            cache_control: None,
            strict: None,
            defer_loading: None,
            allowed_callers: None,
        }
    }

    /// Create a cache breakpoint at this [`CustomMethodDef`] by setting [`cache_control`]
    /// to [`Ephemeral`] See [`Prompt::cache`] for more information.
    ///
    /// Uses the default 5-minute TTL. For a 1-hour TTL, use
    /// [`cache_1h`](Self::cache_1h).
    ///
    /// [`cache_control`]: Self::cache_control
    /// [`Ephemeral`]: crate::prompt::message::CacheControl::Ephemeral
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    pub fn cache(&mut self) -> &mut Self {
        self.cache_with(crate::prompt::message::CacheControl::ephemeral())
    }

    /// Create a 1-hour cache breakpoint at this [`CustomMethodDef`]. Behaves
    /// identically to [`cache`](Self::cache) but uses
    /// [`CacheControl::one_hour`](crate::prompt::message::CacheControl::one_hour).
    pub fn cache_1h(&mut self) -> &mut Self {
        self.cache_with(crate::prompt::message::CacheControl::one_hour())
    }

    /// Create a cache breakpoint at this [`CustomMethodDef`] with a caller-provided
    /// [`CacheControl`](crate::prompt::message::CacheControl).
    pub fn cache_with(
        &mut self,
        cache_control: crate::prompt::message::CacheControl,
    ) -> &mut Self {
        self.cache_control = Some(cache_control);
        self
    }

    /// Returns true if the [`CustomMethodDef`] has a cache breakpoint set (if
    /// `cache_control` is [`Some`]).
    pub fn is_cached(&self) -> bool {
        self.cache_control.is_some()
    }

    /// Set the [`strict`] flag on the [`CustomMethodDef`], enabling [strict tool
    /// use]. See [`MethodBuilder::strict`] for the builder variant.
    ///
    /// [`strict`]: CustomMethodDef::strict
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn strict(&mut self, strict: bool) -> &mut Self {
        self.strict = Some(strict);
        self
    }

    /// Returns `true` if [strict tool use] is enabled on this [`CustomMethodDef`].
    ///
    /// [strict tool use]: <https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/strict-tool-use>
    pub fn is_strict(&self) -> bool {
        self.strict == Some(true)
    }

    /// Try to convert from a serializable value to a [`CustomMethodDef`].
    // A blanket impl for TryFrom<T> where T: Serialize would be nice but it
    // would conflict with the blanket impl for TryFrom<Value> where Value:
    // Serialize. This is a bit of a hack but it works.
    pub fn from_serializable<T>(
        value: T,
    ) -> std::result::Result<CustomMethodDef, serde_json::Error>
    where
        T: Serialize,
    {
        let value = serde_json::to_value(value)?;
        value.try_into()
    }
}

impl TryFrom<serde_json::Value> for CustomMethodDef {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        let builder: MethodBuilder = serde_json::from_value(value)?;
        builder
            .build()
            .map_err(|e| serde::de::Error::custom(e.to_string()))
    }
}

/// Who invoked a tool: the `caller` field surfaced by [programmatic tool
/// calling] on [`Use`] and the server-tool `*_tool_result`
/// [`Block`](crate::prompt::message::Block)s. [`Direct`] is the model calling
/// the tool itself; a code-execution variant means a code-execution container
/// called it on the model's behalf, carrying the `srvtoolu_` id of that call.
///
/// Modeled like [`model::Model`](crate::model::Model): recognized shapes are typed in
/// [`KnownCaller`], and anything else is preserved verbatim in [`Self::Other`]
/// so an unrecognized future caller type still round-trips rather than failing
/// to deserialize a real response.
///
/// [`Direct`]: KnownCaller::Direct
/// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Caller {
    /// A caller shape this crate recognizes.
    Known(KnownCaller),
    /// A caller type not yet modeled, preserved verbatim so it round-trips.
    Other(serde_json::Value),
}

/// The recognized [`Caller`] shapes, distinguished by the wire `type`.
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(tag = "type")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum KnownCaller {
    /// The model called the tool directly (traditional tool use).
    #[serde(rename = "direct")]
    Direct,
    /// A `code_execution_20250825` container called the tool programmatically.
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825 {
        /// The `server_tool_use` id of the code-execution call.
        tool_id: Cow<'static, str>,
    },
    /// A `code_execution_20260120` container called the tool programmatically.
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120 {
        /// The `server_tool_use` id of the code-execution call.
        tool_id: Cow<'static, str>,
    },
}

impl Caller {
    /// The model called the tool directly (traditional tool use). Equivalent
    /// to the API omitting the `caller` field; see [`KnownCaller::Direct`].
    pub fn direct() -> Self {
        Self::Known(KnownCaller::Direct)
    }

    /// A `code_execution_20260120` container called the tool programmatically,
    /// carrying the `srvtoolu_` id of the code-execution call. See
    /// [`KnownCaller::CodeExecution20260120`].
    pub fn code_execution_20260120(
        tool_id: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::Known(KnownCaller::CodeExecution20260120 {
            tool_id: tool_id.into(),
        })
    }

    /// A `code_execution_20250825` container called the tool programmatically,
    /// carrying the `srvtoolu_` id of the code-execution call. See
    /// [`KnownCaller::CodeExecution20250825`].
    pub fn code_execution_20250825(
        tool_id: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self::Known(KnownCaller::CodeExecution20250825 {
            tool_id: tool_id.into(),
        })
    }
}

/// A *context* that may invoke a tool, named in a [`CustomMethodDef`]'s
/// [`allowed_callers`](CustomMethodDef::allowed_callers) to opt that tool into
/// [programmatic tool calling]. Unlike [`Caller`] (which reports who *did*
/// call a tool and carries the `srvtoolu_` id), this is the bare *kind* a tool
/// definition permits — no id.
///
/// Modeled like [`Caller`]/[`model::Model`](crate::model::Model): recognized kinds
/// are typed in [`KnownAllowedCaller`], and anything else is preserved verbatim
/// in [`Self::Other`] so a future, API-versioned caller token still round-trips.
///
/// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
#[serde(untagged)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
pub enum AllowedCaller {
    /// A caller kind this crate recognizes.
    Known(KnownAllowedCaller),
    /// A caller token not yet modeled, preserved verbatim so it round-trips.
    Other(String),
}

/// The recognized [`AllowedCaller`] kinds, each serialized as its bare wire
/// string (`"direct"`, `"code_execution_20260120"`, …).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, Hash)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq, Eq))]
pub enum KnownAllowedCaller {
    /// The model may call the tool directly. The default the API assumes when
    /// `allowed_callers` is omitted.
    #[serde(rename = "direct")]
    Direct,
    /// The model may call the tool from a `code_execution_20260120` container.
    #[serde(rename = "code_execution_20260120")]
    CodeExecution20260120,
    /// The model may call the tool from a `code_execution_20250825` container.
    #[serde(rename = "code_execution_20250825")]
    CodeExecution20250825,
}

impl AllowedCaller {
    /// The model may call the tool directly. See [`KnownAllowedCaller::Direct`].
    ///
    /// `const` so it composes in the [`ALLOWED_CALLERS`] slice the `#[tool]`
    /// macro emits.
    ///
    /// [`ALLOWED_CALLERS`]: crate::tool::ToolArgs::ALLOWED_CALLERS
    pub const fn direct() -> Self {
        Self::Known(KnownAllowedCaller::Direct)
    }

    /// The model may call the tool from a `code_execution_20260120` container.
    /// See [`KnownAllowedCaller::CodeExecution20260120`].
    pub const fn code_execution_20260120() -> Self {
        Self::Known(KnownAllowedCaller::CodeExecution20260120)
    }

    /// The model may call the tool from a `code_execution_20250825` container.
    /// See [`KnownAllowedCaller::CodeExecution20250825`].
    pub const fn code_execution_20250825() -> Self {
        Self::Known(KnownAllowedCaller::CodeExecution20250825)
    }
}

/// `CustomMethodDef` [`Use`] of the model. This should be handled and a response sent
/// back in a [`Block::ToolResult`].
///
/// [`Block::ToolResult`]: crate::prompt::message::Block::ToolResult
#[cfg_attr(
    not(feature = "markdown"),
    derive(derive_more::Display),
    display("\n````json\n{}\n````\n", serde_json::to_string_pretty(self).unwrap())
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[derive(Clone, Debug, Serialize, Deserialize, Hash)]
pub struct Use {
    /// Unique Id for this tool call.
    ///
    /// ## Notes
    /// - This does not have to be a real id. In your examples you can use any
    ///   string so long as it matches a [`tool::Result::tool_use_id`].
    ///
    /// [`tool::Result::tool_use_id`]: crate::tool::Result::tool_use_id
    pub id: Cow<'static, str>,
    /// Name of the tool.
    pub name: Cow<'static, str>,
    /// Input for the tool.
    pub input: serde_json::Value,
    /// Use prompt caching. See [`Prompt::cache`] for more information.
    ///
    /// [`Prompt::cache`]: crate::prompt::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
    /// How this tool was invoked, surfaced by [programmatic tool calling].
    /// [`None`] on a tool the model called directly (equivalent to
    /// [`Caller::Known`]\([`KnownCaller::Direct`]); the API omits the field
    /// in that case). A code-execution variant means a container called the
    /// tool on the model's behalf, carrying the `srvtoolu_` id of that call.
    ///
    /// ## Answering a programmatic call
    ///
    /// You fulfill a code-execution call exactly like a direct one — run the
    /// tool, send a [`tool::Result`](crate::tool::Result) — with two caveats:
    /// the user turn answering it must contain **only** `tool_result` blocks
    /// (no trailing text; the API rejects a mix), and you must resume the same
    /// container via [`Prompt::container`](crate::Prompt::container) before it
    /// idles out. See [`ServerMethodDef::code_execution`].
    ///
    /// [programmatic tool calling]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller: Option<Caller>,
}

impl Use {
    /// Construct a tool [`Use`] for tool `name` with `input` — the semantic
    /// half of a tool call (*which* tool, with *what* arguments). The two
    /// fields that actually drive the model.
    ///
    /// The [`id`](Self::id) defaults to empty; set it with
    /// [`with_id`](Self::with_id) before echoing a constructed call back as
    /// assistant history, so a [`tool::Result`](crate::tool::Result) can
    /// reference it. A `Use` *received* from the API always carries its id.
    /// [`caller`](Self::caller) and [`cache_control`](Self::cache_control)
    /// likewise default to [`None`]; set the caller with
    /// [`with_caller`](Self::with_caller).
    pub fn new(
        name: impl Into<Cow<'static, str>>,
        input: serde_json::Value,
    ) -> Self {
        Self {
            id: Cow::Borrowed(""),
            name: name.into(),
            input,
            cache_control: None,
            caller: None,
        }
    }

    /// Set the [`id`](Self::id). Required before echoing a constructed call
    /// back as assistant history; see [`new`](Self::new).
    #[must_use]
    pub fn with_id(mut self, id: impl Into<Cow<'static, str>>) -> Self {
        self.id = id.into();
        self
    }

    /// Set the [`caller`](Self::caller), marking *how* this tool was invoked
    /// (e.g. [programmatically] from a code-execution container). See
    /// [`Caller`].
    ///
    /// [programmatically]: <https://platform.claude.com/docs/en/agents-and-tools/tool-use/programmatic-tool-calling>
    #[must_use]
    pub fn with_caller(mut self, caller: Caller) -> Self {
        self.caller = Some(caller);
        self
    }
}

impl TryFrom<serde_json::Value> for Use {
    type Error = serde_json::Error;

    fn try_from(
        value: serde_json::Value,
    ) -> std::result::Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Use {
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        if options.tool_use {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(self).unwrap().into(),
                    ),
                    Event::End(TagEnd::CodeBlock),
                ]
                .into_iter(),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

#[cfg(feature = "markdown")]
impl std::fmt::Display for Use {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use crate::markdown::ToMarkdown;

        self.write_markdown(f)
    }
}

/// Result of [`CustomMethodDef`] [`Use`] sent back to the [`Assistant`] as a [`User`]
/// [`Message`].
///
/// [`Assistant`]: crate::prompt::message::Role::Assistant
/// [`User`]: crate::prompt::message::Role::User
/// [`Message`]: crate::prompt::message
#[derive(Clone, Debug, Serialize, Deserialize, Hash, derive_more::Display)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
// FIXME: On the one hand this can clash with the `Result` type from the
// standard library, but on the other hand it's what the API uses. We should
// probably rename this to avoid confusion, since it is confusing.
#[display("{}", self.content)]
pub struct Result {
    /// Unique Id for this tool call.
    pub tool_use_id: Cow<'static, str>,
    /// Output of the tool. If this is an error message it should be written
    /// with the [`Assistant`]'s perspective in mind. It should tell the
    /// [`Assistant`] what went wrong and how they can try to fix it.
    pub content: Content,
    /// Is the result an error message?
    pub is_error: bool,
    /// Use prompt caching. See [`Prompt::cache`] for more information.
    ///
    /// crate::prompt::Prompt::cache
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<crate::prompt::message::CacheControl>,
}

impl Result {
    /// Answer the call with [`tool_use_id`](Self::tool_use_id) `id`, returning
    /// `content`. A success by default ([`is_error`](Self::is_error) is
    /// `false`); chain [`error`](Self::error) to mark it a failure.
    ///
    /// `id` comes first to match the wire order (`tool_use_id`, then
    /// `content`). To set a cache breakpoint, convert into a
    /// [`Block`](crate::prompt::message::Block) / [`Content`] / `Message` and
    /// use their `cache()` helpers — [`cache_control`](Self::cache_control) is
    /// deliberately not exposed on this builder.
    pub fn new(
        id: impl Into<Cow<'static, str>>,
        content: impl Into<Content>,
    ) -> Self {
        Self {
            tool_use_id: id.into(),
            content: content.into(),
            is_error: false,
            cache_control: None,
        }
    }

    /// An error [`Result`] for call `id`, taking its
    /// [`content`](Self::content) from `err`'s [`Display`] and marking
    /// [`is_error`](Self::is_error) `true`. The shorthand for the common
    /// "a tool call failed, hand the error back to the model" path.
    ///
    /// The error text reaches the [`Assistant`], so prefer errors whose
    /// `Display` says what went wrong and how to recover.
    ///
    /// [`Display`]: std::fmt::Display
    pub fn from_error(
        id: impl Into<Cow<'static, str>>,
        err: impl std::error::Error,
    ) -> Self {
        Self::new(id, err.to_string()).error()
    }

    /// Mark this result a failure ([`is_error`](Self::is_error) = `true`). The
    /// [`content`](Self::content) should be written from the [`Assistant`]'s
    /// perspective: what went wrong and how to fix it.
    ///
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    #[must_use]
    pub fn error(mut self) -> Self {
        self.is_error = true;
        self
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Result {
    fn markdown_events_custom(
        &self,
        options: crate::markdown::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{CodeBlockKind, Event, Tag, TagEnd};

        if options.tool_results {
            Box::new(
                [
                    Event::Start(Tag::CodeBlock(CodeBlockKind::Fenced(
                        "json".into(),
                    ))),
                    Event::Text(
                        serde_json::to_string_pretty(self).unwrap().into(),
                    ),
                    Event::End(TagEnd::CodeBlock),
                ]
                .into_iter(),
            )
        } else {
            Box::new(std::iter::empty())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn code_execution_server_tool_wire_shape() {
        // `{"type":"code_execution_20260120","name":"code_execution"}` — the
        // shape sent during the live PTC capture. `From` (derive_more) builds
        // the variant from its config.
        let tool: ServerMethodDef = CodeExecution::default().into();
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "code_execution_20260120");
        assert_eq!(json["name"], "code_execution");
        assert!(json.get("cache_control").is_none());

        let back: ServerMethodDef = serde_json::from_value(json).unwrap();
        assert!(matches!(back, ServerMethodDef::CodeExecution(_)));
    }

    #[test]
    fn web_search_server_tool_wire_shape() {
        let tool = ServerMethodDef::web_search(WebSearch {
            max_uses: Some(5),
            allowed_domains: Some(vec!["anthropic.com".into()]),
            ..Default::default()
        });
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_search_20250305");
        assert_eq!(json["name"], "web_search");
        assert_eq!(json["max_uses"], 5);
        assert_eq!(json["allowed_domains"][0], "anthropic.com");
        // blocked_domains/user_location/cache_control are skipped when None.
        assert!(json.get("blocked_domains").is_none());

        let back: ServerMethodDef = serde_json::from_value(json).unwrap();
        let ServerMethodDef::WebSearch(config) = back else {
            panic!("expected WebSearch");
        };
        assert_eq!(config.max_uses, Some(5));
    }

    #[test]
    fn web_fetch_server_tool_wire_shape() {
        use crate::prompt::message::CitationsConfig;

        let tool = ServerMethodDef::web_fetch(WebFetch {
            max_uses: Some(5),
            allowed_domains: Some(vec!["docs.rs".into()]),
            citations: Some(CitationsConfig { enabled: true }),
            max_content_tokens: Some(50_000),
            ..Default::default()
        });
        let json = serde_json::to_value(&tool).unwrap();
        assert_eq!(json["type"], "web_fetch_20250910");
        assert_eq!(json["name"], "web_fetch");
        assert_eq!(json["max_uses"], 5);
        assert_eq!(json["allowed_domains"][0], "docs.rs");
        assert_eq!(json["citations"]["enabled"], true);
        assert_eq!(json["max_content_tokens"], 50_000);
        // blocked_domains/cache_control are skipped when None.
        assert!(json.get("blocked_domains").is_none());

        let back: ServerMethodDef = serde_json::from_value(json).unwrap();
        let ServerMethodDef::WebFetch(config) = back else {
            panic!("expected WebFetch");
        };
        assert_eq!(config.max_uses, Some(5));
        assert_eq!(config.max_content_tokens, Some(50_000));
    }

    #[test]
    fn tool_search_variants_wire_shape() {
        for (tool, ty, name) in [
            (
                ServerMethodDef::tool_search_regex(),
                "tool_search_tool_regex_20251119",
                "tool_search_tool_regex",
            ),
            (
                ServerMethodDef::tool_search_bm25(),
                "tool_search_tool_bm25_20251119",
                "tool_search_tool_bm25",
            ),
        ] {
            let json = serde_json::to_value(&tool).unwrap();
            assert_eq!(json["type"], ty);
            assert_eq!(json["name"], name);
            // No config beyond type/name when uncached.
            assert!(json.get("cache_control").is_none());

            // Round-trips back to the same variant.
            let back: ServerMethodDef = serde_json::from_value(json).unwrap();
            assert_eq!(back, tool);
        }
    }

    #[test]
    fn tool_search_is_cacheable() {
        let mut def: MethodDef = ServerMethodDef::tool_search_regex().into();
        assert!(!def.is_cached());
        def.cache_with(crate::prompt::message::CacheControl::ephemeral());
        assert!(def.is_cached());
        // The breakpoint survives serialization.
        let json = serde_json::to_value(&def).unwrap();
        assert!(json.get("cache_control").is_some());
    }

    #[test]
    fn tool_search_name_is_fixed() {
        let bad = serde_json::json!({
            "type": "tool_search_tool_regex_20251119",
            "name": "not_the_right_name",
        });
        assert!(serde_json::from_value::<ServerMethodDef>(bad).is_err());
    }

    #[test]
    fn web_search_name_must_be_web_search() {
        // A wrong `name` is rejected on deserialize.
        let bad = serde_json::json!({
            "type": "web_search_20250305",
            "name": "not_web_search",
        });
        assert!(serde_json::from_value::<ServerMethodDef>(bad).is_err());
    }

    #[test]
    fn tooldef_untagged_discriminates_by_type() {
        // A custom tool (no `type`) round-trips as Custom; a server tool (has a
        // versioned `type`) round-trips as Server.
        let custom: MethodDef =
            CustomMethodDef::simple("ping", "Ping a server.").into();
        let server: MethodDef =
            ServerMethodDef::web_search(WebSearch::default()).into();

        let custom_json = serde_json::to_value(&custom).unwrap();
        assert!(custom_json.get("type").is_none());
        assert!(custom_json.get("input_schema").is_some());
        assert!(matches!(
            serde_json::from_value::<MethodDef>(custom_json).unwrap(),
            MethodDef::Custom(_)
        ));

        let server_json = serde_json::to_value(&server).unwrap();
        assert_eq!(server_json["type"], "web_search_20250305");
        assert!(matches!(
            serde_json::from_value::<MethodDef>(server_json).unwrap(),
            MethodDef::Server(_)
        ));
    }

    #[test]
    fn prompt_mixes_custom_and_server_tools_in_tools_array() {
        let prompt = crate::Prompt::default()
            .add_tool(CustomMethodDef::simple("ping", "Ping a server."))
            .add_tool(ServerMethodDef::web_search(WebSearch::default()));

        let json = serde_json::to_value(&prompt).unwrap();
        let tools = json["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 2);
        // Custom tool: no `type`, has `input_schema`.
        assert!(tools[0].get("type").is_none());
        assert_eq!(tools[0]["name"], "ping");
        // Server tool: versioned `type`.
        assert_eq!(tools[1]["type"], "web_search_20250305");

        // The whole prompt round-trips, preserving both tool kinds.
        let back: crate::Prompt = serde_json::from_value(json).unwrap();
        let methods = back.methods.unwrap();
        assert!(matches!(methods[0], MethodDef::Custom(_)));
        assert!(matches!(methods[1], MethodDef::Server(_)));
    }

    #[test]
    fn test_method_simple() {
        let method =
            CustomMethodDef::simple("test_method", "A simple test method");

        assert_eq!(method.name, "test_method");
        assert_eq!(method.description, "A simple test method");
        assert_eq!(
            method.schema,
            serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            })
        );
    }

    #[test]
    fn test_method_with_string_param() {
        let method = CustomMethodDef::with_string_param(
            "get_weather",
            "Get weather for a location",
            "location",
            "The city name",
            true,
        );

        assert_eq!(method.name, "get_weather");
        assert_eq!(method.description, "Get weather for a location");
        assert_eq!(
            method.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city name"
                    }
                },
                "required": ["location"]
            })
        );
    }

    #[test]
    fn test_method_builder_param_helpers() {
        let method = CustomMethodDef::builder("test_method")
            .description("Test method with multiple params")
            .string_param("name", "A person's name", true)
            .number_param("age", "A person's age", false)
            .boolean_param("active", "Whether the person is active", true)
            .build()
            .unwrap();

        assert_eq!(method.name, "test_method");
        assert_eq!(method.description, "Test method with multiple params");

        let expected_schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "A person's name"
                },
                "age": {
                    "type": "number",
                    "description": "A person's age"
                },
                "active": {
                    "type": "boolean",
                    "description": "Whether the person is active"
                }
            },
            "required": ["name", "active"]
        });

        assert_eq!(method.schema, expected_schema);
    }

    #[test]
    fn test_method_builder_param_helpers_with_existing_schema() {
        // Start with an existing schema and add to it
        let method = CustomMethodDef::builder("test_method")
            .description("Test method")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "existing": {
                        "type": "string",
                        "description": "An existing property"
                    }
                },
                "required": ["existing"]
            }))
            .string_param("new_param", "A new parameter", true)
            .build()
            .unwrap();

        let properties = method.schema["properties"].as_object().unwrap();
        assert!(properties.contains_key("existing"));
        assert!(properties.contains_key("new_param"));

        let required = method.schema["required"].as_array().unwrap();
        assert!(
            required
                .contains(&serde_json::Value::String("existing".to_string()))
        );
        assert!(
            required
                .contains(&serde_json::Value::String("new_param".to_string()))
        );
    }

    #[test]
    fn use_try_from_value() {
        let value = serde_json::json!({
            "id": "test_id",
            "name": "test_name",
            "input": {
                "test_key": "test_value"
            }
        });

        let use_ = Use::try_from(value).unwrap();

        assert_eq!(use_.id, "test_id");
        assert_eq!(use_.name, "test_name");
        assert_eq!(
            use_.input,
            serde_json::json!({
                "test_key": "test_value"
            })
        );
    }

    #[test]
    #[cfg(feature = "markdown")]
    fn test_use_markdown() {
        use crate::markdown::ToMarkdown;

        let use_ = Use::new(
            "test_name",
            serde_json::json!({
                "test_key": "test_value"
            }),
        )
        .with_id("test_id");

        let markdown = use_.markdown_verbose();

        assert_eq!(
            markdown.as_ref(),
            "\n````json\n{\n  \"id\": \"test_id\",\n  \"name\": \"test_name\",\n  \"input\": {\n    \"test_key\": \"test_value\"\n  }\n}\n````"
        );

        // By default the tool use is not included in the markdown, however this
        // might change in the future. Really, our Display impl could just
        // return an empty &str but this is more consistent with the rest of the
        // crate.
        assert_eq!(use_.to_string(), "");
    }

    #[test]
    fn test_tool_schema_validation() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "The letter to count",
                },
                "string": {
                    "type": "string",
                    "description": "The string to count letters in",
                },
            },
            "required": ["letter", "string"],
        });

        assert!(MethodBuilder::is_valid_input_schema(&schema).is_ok());

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "letter": {
                    "type": "string",
                    "description": "The letter to count",
                },
                "string": {
                    "type": "string",
                    "description": "The string to count letters in",
                },
            },
            "required": "letter",
        });

        assert!(MethodBuilder::is_valid_input_schema(&schema).is_err());
    }

    #[test]
    fn test_build() {
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build()
            .unwrap();

        assert_eq!(tool.name, "test_name");
        assert_eq!(tool.description, "test_description");
        assert_eq!(
            tool.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            })
        );

        // Test error cases
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": "letter",
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // input schema not an object
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::Value::String("blah".into()))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // Properties not an object
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": "blah",
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // `required` lists keys absent from (here, missing) `properties`
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // No `required` array is valid (all-optional / no-arg methods). It is
        // optional per JSON Schema and treated as empty when absent.
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
            }))
            .build();

        assert!(tool.is_ok());

        // required keys not found in properties
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string", "foo"],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // required keys not strings
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": [1, 2],
            }))
            .build();

        assert!(matches!(
            tool,
            Err(ToolBuildError::InvalidInputSchema { .. })
        ));

        // missing schema
        let tool = CustomMethodDef::builder("test_name")
            .description("test_description")
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyInputSchema)));

        // with missing names and descriptions
        let tool = CustomMethodDef::builder("")
            .description("foo")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyName)));

        let tool = CustomMethodDef::builder("foo")
            .description("")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            }))
            .build();

        assert!(matches!(tool, Err(ToolBuildError::EmptyDescription)));
    }

    #[test]
    fn test_choice_serde() {
        for choice in [
            Choice::auto(),
            Choice::any(),
            Choice::method("test_name"),
            Choice::none(),
            Choice::any().disable_parallel_tool_use(),
        ] {
            let json = serde_json::to_string(&choice).unwrap();
            let choice2: Choice = serde_json::from_str(&json).unwrap();
            assert_eq!(choice, choice2);
        }

        // `disable_parallel_tool_use` is omitted when false.
        assert_eq!(
            serde_json::to_value(Choice::auto()).unwrap(),
            serde_json::json!({ "type": "auto" })
        );
        assert_eq!(
            serde_json::to_value(Choice::none()).unwrap(),
            serde_json::json!({ "type": "none" })
        );
        assert_eq!(
            serde_json::to_value(
                Choice::method("t").disable_parallel_tool_use()
            )
            .unwrap(),
            serde_json::json!({
                "type": "tool",
                "name": "t",
                "disable_parallel_tool_use": true,
            })
        );
    }

    #[test]
    fn test_defer_loading_serde() {
        let mut method = CustomMethodDef::simple("ping", "Ping a server.");
        // Omitted when unset.
        assert!(
            serde_json::to_value(&method)
                .unwrap()
                .get("defer_loading")
                .is_none()
        );

        method.defer_loading = Some(true);
        let json = serde_json::to_value(&method).unwrap();
        assert_eq!(json["defer_loading"], true);

        // Round-trips through the builder-based `Deserialize`.
        let back: CustomMethodDef = serde_json::from_value(json).unwrap();
        assert_eq!(back.defer_loading, Some(true));
    }

    #[test]
    fn allowed_caller_serializes_to_bare_string() {
        // Known kinds are their bare wire token; an unknown one round-trips
        // verbatim through `Other`.
        assert_eq!(
            serde_json::to_value(AllowedCaller::code_execution_20260120())
                .unwrap(),
            serde_json::json!("code_execution_20260120")
        );
        assert_eq!(
            serde_json::to_value(AllowedCaller::direct()).unwrap(),
            serde_json::json!("direct")
        );
        let future = serde_json::json!("code_execution_29991231");
        let parsed: AllowedCaller =
            serde_json::from_value(future.clone()).unwrap();
        assert!(matches!(parsed, AllowedCaller::Other(_)));
        assert_eq!(serde_json::to_value(&parsed).unwrap(), future);
    }

    #[test]
    fn allowed_callers_builder_and_serde() {
        // `.programmatic()` is sugar for the code-execution caller.
        let method = CustomMethodDef::builder("query_sales")
            .description("Query sales.")
            .schema(serde_json::json!({"type": "object"}))
            .programmatic()
            .build()
            .unwrap();
        assert_eq!(
            method.allowed_callers,
            Some(vec![AllowedCaller::code_execution_20260120()])
        );
        let json = serde_json::to_value(&method).unwrap();
        assert_eq!(
            json["allowed_callers"],
            serde_json::json!(["code_execution_20260120"])
        );

        // Round-trips through the builder-based `Deserialize`.
        let back: CustomMethodDef = serde_json::from_value(json).unwrap();
        assert_eq!(back.allowed_callers, method.allowed_callers);

        // Omitted when unset; an empty list clears back to the default.
        let bare = CustomMethodDef::simple("ping", "Ping.");
        assert!(
            serde_json::to_value(&bare)
                .unwrap()
                .get("allowed_callers")
                .is_none()
        );
        let cleared = CustomMethodDef::builder("ping")
            .description("Ping.")
            .schema(serde_json::json!({"type": "object"}))
            .allowed_callers([])
            .build()
            .unwrap();
        assert_eq!(cleared.allowed_callers, None);
    }

    #[test]
    fn test_result_serde() {
        let result = Result::new("test_id", "test_content");

        let json = serde_json::to_string(&result).unwrap();
        let result2: Result = serde_json::from_str(&json).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn test_result_construction() {
        let result = Result::new("test_id", "test_content");

        assert_eq!(result.tool_use_id, "test_id");
        assert_eq!(result.content.to_string(), "test_content");
        assert!(!result.is_error);

        // `from_error` lifts any `std::error::Error` into an error result.
        let err = "boom".parse::<u32>().unwrap_err();
        let result = Result::from_error("test_id", err);
        assert!(result.is_error);
        assert_eq!(result.tool_use_id, "test_id");
    }

    #[test]
    fn test_tool_from_serializable() {
        let tool = CustomMethodDef::from_serializable(serde_json::json!({
            "name": "test_name",
            "description": "test_description",
            "input_schema": {
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            },
        }))
        .unwrap();

        assert_eq!(tool.name, "test_name");
        assert_eq!(tool.description, "test_description");
        assert_eq!(
            tool.schema,
            serde_json::json!({
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                "required": ["letter", "string"],
            })
        );

        // Test invalid schema. Comprehensive testing of this is in the builder
        // tests. This just makes sure that the error is propagated.
        let tool = CustomMethodDef::from_serializable(serde_json::json!({
            "name": "test_name",
            "description": "test_description",
            "input_schema": {
                "type": "object",
                "properties": {
                    "letter": {
                        "type": "string",
                        "description": "The letter to count",
                    },
                    "string": {
                        "type": "string",
                        "description": "The string to count letters in",
                    },
                },
                // should be an array
                "required": "letter",
            },
        }));

        assert!(tool.is_err());
    }

    #[test]
    fn test_method_strict_defaults_none_and_elides() {
        let tool = CustomMethodDef::simple("ping", "Ping a server.");
        assert_eq!(tool.strict, None);
        assert!(!tool.is_strict());

        let wire = serde_json::to_value(&tool).unwrap();
        assert!(
            wire.as_object().unwrap().get("strict").is_none(),
            "strict must be elided when None, got {wire:#}",
        );
    }

    #[test]
    fn test_method_builder_strict_flag() {
        let tool = CustomMethodDef::builder("ping")
            .description("Ping a server.")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            }))
            .strict(true)
            .build()
            .unwrap();

        assert_eq!(tool.strict, Some(true));
        assert!(tool.is_strict());
        let wire = serde_json::to_value(&tool).unwrap();
        assert_eq!(wire["strict"], serde_json::Value::Bool(true));
    }

    #[test]
    fn test_method_strict_mut_setter() {
        let mut tool = CustomMethodDef::simple("ping", "Ping a server.");
        tool.strict(true);
        assert_eq!(tool.strict, Some(true));
    }

    #[test]
    fn test_method_strict_roundtrips_through_deserialize() {
        let wire = serde_json::json!({
            "name": "ping",
            "description": "Ping a server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            },
            "strict": true,
        });
        let tool = CustomMethodDef::from_serializable(wire).unwrap();
        assert_eq!(tool.strict, Some(true));
    }

    #[test]
    fn test_method_builder_preserves_strict() {
        let tool = CustomMethodDef::builder("ping")
            .description("Ping a server.")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "host": { "type": "string", "description": "hostname" }
                },
                "required": ["host"],
            }))
            .strict(true)
            .build()
            .unwrap();
        assert_eq!(tool.strict, Some(true));
    }
}
