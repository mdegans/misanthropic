use std::ops::Deref;

use pulldown_cmark::HeadingLevel;
use serde::{Deserialize, Serialize};

/// Default [`Options`]
pub const DEFAULT_OPTIONS: Options = Options {
    inner: pulldown_cmark::Options::empty(),
    tool_use: false,
    tool_results: false,
    system: false,
    attrs: false,
    heading_level: None,
    verbatim: false,
    readable_tools: false,
    tools: false,
    label_thoughts: false,
    settings: false,
    separators: false,
};

/// Verbose [`Options`]
pub const VERBOSE_OPTIONS: Options = Options {
    inner: pulldown_cmark::Options::empty(),
    tool_use: true,
    tool_results: true,
    system: true,
    attrs: true,
    heading_level: None,
    verbatim: false,
    readable_tools: false,
    tools: false,
    label_thoughts: false,
    settings: false,
    separators: false,
};

mod serde_inner {
    use super::*;

    pub fn serialize<S>(
        options: &pulldown_cmark::Options,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        options.bits().serialize(serializer)
    }

    pub fn deserialize<'de, D>(
        deserializer: D,
    ) -> Result<pulldown_cmark::Options, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u32::deserialize(deserializer)?;
        Ok(pulldown_cmark::Options::from_bits_truncate(bits))
    }
}

/// Options for parsing, generating, and rendering [`Markdown`].
#[derive(Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(default)]
pub struct Options {
    /// Inner [`pulldown_cmark::Options`].
    #[serde(with = "serde_inner")]
    pub inner: pulldown_cmark::Options,
    /// Whether to include the system prompt
    pub system: bool,
    /// Whether to include tool uses.
    pub tool_use: bool,
    /// Whether to include tool results.
    pub tool_results: bool,
    /// Whether to include attributes. Useful when converting to HTML.
    ///
    /// This adds:
    /// - `role` attribute to the [`Prompt`] and [`Message`]s. Possible values
    ///   are:
    ///   - `system` - for the system prompt
    ///   - `assistant` - for generated messages
    ///   - `tool` - for tool results
    ///   - `user` - for user messages
    ///   - `error` - for errors
    ///
    /// [`Prompt`]: crate::prompt::Prompt
    /// [`Message`]: crate::prompt::Message
    pub attrs: bool,
    /// Heading level to begin at (optional)
    pub heading_level: Option<HeadingLevel>,
    /// Pass text and thinking through exactly as given, rather than parsing
    /// them as Markdown and rendering them again (which normalizes them:
    /// `-` list markers become `*`, a `[` is escaped). For a reader who must
    /// see what a model saw, byte for byte. Markdown output only: HTML
    /// rendering ignores it, since verbatim text would pass as raw HTML.
    pub verbatim: bool,
    /// Render tool calls and results for reading: a call as a labelled line
    /// and its input as pretty JSON, a result as a labelled line and its
    /// content rendered with these same options, rather than each block as
    /// JSON (where a result's text is one escaped string). Applies where
    /// [`tool_use`] and [`tool_results`] include them.
    ///
    /// [`tool_use`]: Options::tool_use
    /// [`tool_results`]: Options::tool_results
    pub readable_tools: bool,
    /// Include a [`Prompt`]'s tool definitions, after the system prompt
    ///
    /// [`Prompt`]: crate::prompt::Prompt
    pub tools: bool,
    /// Label thinking as thinking, and mark where thinking was redacted,
    /// rather than rendering thinking like any text and redacted thinking
    /// as nothing
    pub label_thoughts: bool,
    /// Include a [`Prompt`]'s settings (model, `max_tokens`, thinking,
    /// `tool_choice`, `output_config` and the rest: everything but the
    /// system prompt, tools and messages), as JSON, after the system prompt
    ///
    /// [`Prompt`]: crate::prompt::Prompt
    pub settings: bool,
    /// A horizontal rule before each message's heading, so a message's
    /// start can't be mistaken for a heading inside the content before it
    pub separators: bool,
}

impl Options {
    /// Maximum verbosity
    pub fn verbose() -> Self {
        VERBOSE_OPTIONS
    }

    /// Set [`tool_use`] to true
    ///
    /// [`tool_use`]: Options::tool_use
    pub fn with_tool_use(mut self) -> Self {
        self.tool_use = true;
        self
    }

    /// Set [`tool_results`] to true
    ///
    /// [`tool_results`]: Options::tool_results
    pub fn with_tool_results(mut self) -> Self {
        self.tool_results = true;
        self
    }

    /// Set [`system`] to true
    ///
    /// [`system`]: Options::system
    pub fn with_system(mut self) -> Self {
        self.system = true;
        self
    }

    /// Set [`verbatim`] to true
    ///
    /// [`verbatim`]: Options::verbatim
    pub fn verbatim(mut self) -> Self {
        self.verbatim = true;
        self
    }

    /// Set [`readable_tools`] to true
    ///
    /// [`readable_tools`]: Options::readable_tools
    pub fn readable_tools(mut self) -> Self {
        self.readable_tools = true;
        self
    }

    /// Set [`tools`] to true
    ///
    /// [`tools`]: Options::tools
    pub fn with_tools(mut self) -> Self {
        self.tools = true;
        self
    }

    /// Set [`label_thoughts`] to true
    ///
    /// [`label_thoughts`]: Options::label_thoughts
    pub fn label_thoughts(mut self) -> Self {
        self.label_thoughts = true;
        self
    }

    /// Set [`settings`] to true
    ///
    /// [`settings`]: Options::settings
    pub fn with_settings(mut self) -> Self {
        self.settings = true;
        self
    }

    /// Set [`separators`] to true
    ///
    /// [`separators`]: Options::separators
    pub fn separators(mut self) -> Self {
        self.separators = true;
        self
    }

    /// Everything a reader needs to see exactly what a model was sent: the
    /// system prompt, the settings, tool definitions, calls and results
    /// (readable), and labelled thinking, with text verbatim and each
    /// message set off by a rule. No attributes.
    pub fn transcript() -> Self {
        Self {
            attrs: false,
            ..Self::verbose()
                .verbatim()
                .readable_tools()
                .with_tools()
                .label_thoughts()
                .with_settings()
                .separators()
        }
    }
}

#[cfg(feature = "markdown")]
impl From<pulldown_cmark::Options> for Options {
    fn from(inner: pulldown_cmark::Options) -> Self {
        Options {
            inner,
            ..Default::default()
        }
    }
}

/// A valid, immutable, Markdown string. It has been parsed and rendered. It can
/// be [`Display`]ed or dereferenced as a [`str`].
///
/// [`Display`]: std::fmt::Display
#[derive(derive_more::Display)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[display("{text}")]
pub struct Markdown {
    text: String,
}

impl From<Markdown> for String {
    fn from(markdown: Markdown) -> Self {
        markdown.text
    }
}

impl AsRef<str> for Markdown {
    fn as_ref(&self) -> &str {
        self.deref()
    }
}

impl std::borrow::Borrow<str> for Markdown {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl std::ops::Deref for Markdown {
    type Target = str;

    fn deref(&self) -> &str {
        &self.text
    }
}

impl<'a, T> From<T> for Markdown
where
    T: Iterator<Item = pulldown_cmark::Event<'a>>,
{
    fn from(events: T) -> Self {
        let mut text = String::new();

        // Unwrap can never panic because the formatter for `String` never
        // returns an error.
        let _ = pulldown_cmark_to_cmark::cmark(events, &mut text).unwrap();

        Markdown { text }
    }
}

#[cfg(any(test, feature = "partial-eq"))]
impl PartialEq<str> for Markdown {
    fn eq(&self, other: &str) -> bool {
        self.text == other
    }
}

/// `text` as events that render to exactly `text`: an HTML block, which the
/// Markdown writer passes through unescaped. Used by [`Options::verbatim`].
pub(crate) fn verbatim_events(
    text: &str,
) -> impl Iterator<Item = pulldown_cmark::Event<'_>> {
    use pulldown_cmark::{CowStr, Event, Tag, TagEnd};

    // The block must end in a newline, or the next block's first line
    // joins its last (a result's text ran into the next result's label)
    let text: CowStr<'_> = if text.ends_with('\n') {
        text.into()
    } else {
        format!("{text}\n").into()
    };
    [
        Event::Start(Tag::HtmlBlock),
        Event::Html(text),
        Event::End(TagEnd::HtmlBlock),
    ]
    .into_iter()
}

/// A labelled paragraph: `**label**` and the rest as text. Used to head tool
/// calls, tool results and thinking under [`Options::readable_tools`] and
/// [`Options::label_thoughts`].
pub(crate) fn label_events<'a>(
    label: &'a str,
    rest: impl Into<pulldown_cmark::CowStr<'a>>,
) -> impl Iterator<Item = pulldown_cmark::Event<'a>> {
    use pulldown_cmark::{Event, Tag, TagEnd};

    [
        Event::Start(Tag::Paragraph),
        Event::Start(Tag::Strong),
        Event::Text(label.into()),
        Event::End(TagEnd::Strong),
        Event::Text(rest.into()),
        Event::End(TagEnd::Paragraph),
    ]
    .into_iter()
}

/// A trait for types that can be converted to [`Markdown`]
///
/// # Note
///
/// - Any of these methods returning an iterator of [`pulldown_cmark::Event`]s
///   can be used to render to html using [`pulldown_cmark::html::push_html`]
///   and other similar functions.
/// - Implementers should guarantee tags are properly closed and nested.
pub trait ToMarkdown {
    /// Render the type to a [`Markdown`] string with [`DEFAULT_OPTIONS`].
    fn markdown(&self) -> Markdown {
        self.markdown_events().into()
    }

    /// Render the type to a [`Markdown`] string with custom [`Options`].
    fn markdown_custom(&self, options: Options) -> Markdown {
        self.markdown_events_custom(options).into()
    }

    /// Render the type to a [`Markdown`] string with maximum verbosity.
    fn markdown_verbose(&self) -> Markdown {
        self.markdown_custom(VERBOSE_OPTIONS)
    }

    /// Render the markdown to a type implementing [`std::fmt::Write`] with
    /// [`DEFAULT_OPTIONS`].
    fn write_markdown(
        &self,
        writer: &mut dyn std::fmt::Write,
    ) -> std::fmt::Result {
        self.write_markdown_custom(writer, DEFAULT_OPTIONS)
    }

    /// Render the markdown to a type implementing [`std::fmt::Write`] with
    /// custom [`Options`].
    fn write_markdown_custom(
        &self,
        writer: &mut dyn std::fmt::Write,
        options: Options,
    ) -> std::fmt::Result {
        use pulldown_cmark_to_cmark::cmark;

        let events = self.markdown_events_custom(options);
        let _ = cmark(events, writer)?;
        Ok(())
    }

    /// Return an iterator of [`pulldown_cmark::Event`]s with
    /// [`DEFAULT_OPTIONS`].
    fn markdown_events(
        &self,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        self.markdown_events_custom(DEFAULT_OPTIONS)
    }

    /// Return an iterator of [`pulldown_cmark::Event`]s with custom
    /// [`Options`].
    fn markdown_events_custom(
        &self,
        options: Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_>;
}

static_assertions::assert_obj_safe!(ToMarkdown);

impl Default for Options {
    fn default() -> Self {
        DEFAULT_OPTIONS
    }
}

#[cfg(test)]
mod tests {
    use crate::prompt::{Message, message::Role};

    use super::*;

    use std::borrow::Borrow;

    #[test]
    fn test_options_serde() {
        let options = Options::default();

        let json = serde_json::to_string(&options).unwrap();
        let options2: Options = serde_json::from_str(&json).unwrap();

        assert!(options == options2);
    }

    #[test]
    fn test_options_from_pulldown() {
        let inner = pulldown_cmark::Options::empty();
        let options: Options = inner.into();
        assert_eq!(options.inner, inner);
    }

    #[test]
    fn test_options_verbose() {
        let options = Options::verbose();
        assert!(options.tool_use);
        assert!(options.tool_results);
        assert!(options.system);
    }

    #[test]
    fn test_markdown() {
        let expected = "Hello, **world**!";
        let events = pulldown_cmark::Parser::new(expected);
        let markdown: Markdown = events.into();
        let actual: &str = markdown.borrow();
        assert_eq!(actual, expected);
        assert!(&markdown == expected);
        let markdown: String = markdown.into();
        assert_eq!(markdown, expected);
    }

    #[test]
    fn test_message_markdown() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        assert_eq!(
            message.markdown().as_ref(),
            "### User\n\nHello, **world**!"
        );
    }

    /// What a reader of a transcript needs: the text as the model saw it,
    /// the tool definitions, and calls and results it can read
    #[test]
    fn transcript_options_show_what_the_model_saw() {
        use crate::prompt::{Message, Prompt, message::Role};
        use crate::tool::{self, CustomMethodDef, MethodDef};
        use serde_json::json;

        let read = "## A post\n[post_id: 4d1f3047]\n\n- a comment\n- another";
        let prompt = Prompt {
            system: Some("You are *an agent*.\n- one rule".into()),
            tools: Some(vec![MethodDef::Custom(CustomMethodDef {
                name: "get_content".into(),
                description: "Read anything.".into(),
                schema: json!({"type": "object", "properties": {}}),
                cache_control: None,
                strict: None,
                defer_loading: None,
                allowed_callers: None,
            })]),
            messages: vec![
                Message {
                    role: Role::User,
                    content: "Read [this] - please".into(),
                },
                tool::Use::new("get_content", json!({"id": "4d1f3047"}))
                    .with_id("toolu_1")
                    .into(),
                tool::Result::new("toolu_1", read).into(),
            ],
            ..Default::default()
        };
        let md = prompt.markdown_custom(Options::transcript()).to_string();

        // Text exactly as given, nothing escaped or re-listed
        assert!(md.contains("You are *an agent*.\n- one rule"), "{md}");
        assert!(md.contains("Read [this] - please"), "{md}");
        // The definition, by name, with its description
        assert!(
            md.contains("### Tools") && md.contains("`get_content`"),
            "{md}"
        );
        assert!(md.contains("Read anything."), "{md}");
        // The call: labelled, input alone, pretty
        assert!(
            md.contains("**Tool call** `get_content` (`toolu_1`)"),
            "{md}"
        );
        assert!(md.contains("\"id\": \"4d1f3047\""), "{md}");
        // The result: labelled, its text as text, not escaped JSON
        assert!(md.contains("**Tool result** for `toolu_1`"), "{md}");
        assert!(md.contains(read), "{md}");
        assert!(!md.contains("\\n"), "no escaped newlines: {md}");
        // No attributes in Markdown meant for reading
        assert!(!md.contains("role="), "{md}");
        // Each message set off by a rule; the settings shown
        assert!(md.contains("---\n\n### User"), "{md}");
        assert!(
            md.contains("### Settings") && md.contains("\"max_tokens\""),
            "{md}"
        );
        assert!(
            !md.contains("\"messages\""),
            "the settings leave out messages: {md}"
        );
    }

    /// A message of several results is headed Tool even when its last is an
    /// error; each result says its own. Verbatim text is set off from the
    /// next block by a blank line. A definition is shown whole.
    #[test]
    fn transcript_mixed_results_blank_lines_and_strict() {
        use crate::prompt::{
            Prompt,
            message::{Block, Content, Message, Role},
        };
        use crate::tool::{self, CustomMethodDef, MethodDef};
        use serde_json::json;

        let results: Content = vec![
            Block::ToolResult {
                result: tool::Result::new("t1", "Vote recorded"),
            },
            Block::ToolResult {
                result: tool::Result {
                    is_error: true,
                    ..tool::Result::new("t2", "No community")
                },
            },
        ]
        .into();
        let prompt = Prompt {
            system: Some("The end of the system prompt.".into()),
            tools: Some(vec![MethodDef::Custom(CustomMethodDef {
                name: "cast_vote".into(),
                description: "Vote.".into(),
                schema: json!({"type": "object", "properties": {}}),
                cache_control: None,
                strict: Some(true),
                defer_loading: None,
                allowed_callers: None,
            })]),
            messages: vec![Message {
                role: Role::User,
                content: results,
            }],
            ..Default::default()
        };
        let md = prompt.markdown_custom(Options::transcript()).to_string();
        assert!(
            md.contains("### Tool\n") && !md.contains("### Error"),
            "{md}"
        );
        assert!(
            md.contains("Vote recorded\n\n**Tool result** for `t2`, an error"),
            "{md}"
        );
        assert!(md.contains("The end of the system prompt.\n\n"), "{md}");
        assert!(md.contains("\"strict\": true"), "{md}");
        // Without readable tools, an error last still heads the message
        let plain = prompt.markdown_custom(Options::verbose()).to_string();
        assert!(plain.contains("### Error"), "{plain}");
    }

    #[test]
    fn thinking_is_labelled_and_redaction_marked() {
        use crate::prompt::message::{Block, Content};

        let content: Content = vec![
            Block::Thought {
                thought: "First, the *facts*.".into(),
                signature: "sig".into(),
            },
            Block::RedactedThought {
                signature: "x".into(),
            },
        ]
        .into();
        let plain = content.markdown_custom(Options::default()).to_string();
        assert!(!plain.contains("Thinking"), "{plain}");
        let labelled = content
            .markdown_custom(Options::default().label_thoughts().verbatim())
            .to_string();
        assert!(labelled.contains("**Thinking**"), "{labelled}");
        assert!(labelled.contains("First, the *facts*."), "{labelled}");
        assert!(labelled.contains("**Thinking** (redacted)"), "{labelled}");
    }

    #[test]
    fn each_option_and_its_branches() {
        use crate::prompt::{
            Prompt,
            message::{Block, Content},
        };
        use crate::tool::{self, MethodDef, ServerMethodDef};

        // Builders set exactly their field
        let o = Options::default();
        assert!(o.verbatim().verbatim && !o.verbatim);
        assert!(o.readable_tools().readable_tools);
        assert!(o.with_tools().tools);
        assert!(o.label_thoughts().label_thoughts);
        assert!(o.with_settings().settings && o.separators().separators);
        let t = Options::transcript();
        assert!(t.verbatim && t.readable_tools && t.tools && t.label_thoughts);
        assert!(t.settings && t.separators);
        assert!(t.system && t.tool_use && t.tool_results && !t.attrs);

        // Thinking: verbatim alone, labelled alone
        let thought: Content = vec![Block::Thought {
            thought: "- a *plan*".into(),
            signature: "sig".into(),
        }]
        .into();
        let verbatim = thought.markdown_custom(o.verbatim()).to_string();
        assert!(
            verbatim.contains("- a *plan*") && !verbatim.contains("Thinking")
        );
        let labelled = thought.markdown_custom(o.label_thoughts()).to_string();
        assert!(labelled.contains("**Thinking**") && labelled.contains("plan"));

        // An error result says so; readable without tool_results shows nothing
        let error = tool::Result {
            is_error: true,
            ..tool::Result::new("toolu_9", "it broke")
        };
        let shown = error.markdown_custom(Options::transcript()).to_string();
        assert!(
            shown.contains("**Tool result** for `toolu_9`, an error"),
            "{shown}"
        );
        assert!(shown.contains("it broke"));
        assert!(
            error
                .markdown_custom(o.readable_tools())
                .to_string()
                .is_empty()
        );

        // A server tool's definition is its JSON
        let prompt = Prompt {
            tools: Some(vec![MethodDef::Server(
                ServerMethodDef::tool_search_regex(),
            )]),
            ..Default::default()
        };
        let defs = prompt.markdown_custom(Options::transcript()).to_string();
        assert!(
            defs.contains("### Tools") && defs.contains("```json"),
            "{defs}"
        );
        // ...and no tools heading without the option, or with none defined
        assert!(
            !prompt
                .markdown_custom(Options::verbose())
                .to_string()
                .contains("Tools")
        );
        assert!(
            !Prompt::default()
                .markdown_custom(t)
                .to_string()
                .contains("Tools")
        );
    }

    #[test]
    fn verbatim_text_renders_exactly() {
        let text =
            "- a list [post_id: 1]\n- *not* emphasis_here\n\n## A heading\n";
        let rendered: Markdown = verbatim_events(text).into();
        assert_eq!(rendered.as_ref().trim_end(), text.trim_end());
    }

    #[test]
    fn test_options_with_system() {
        let options = Options::default().with_system();
        assert!(options.system);
    }
}
