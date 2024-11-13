use std::ops::Deref;

use pulldown_cmark::html::push_html;

use crate::markdown::ToMarkdown;

pub use crate::markdown::{Options, DEFAULT_OPTIONS, VERBOSE_OPTIONS};


/// Immutable wrapper around a [`String`]. Guaranteed to be valid HTML.
#[derive(derive_more::Display)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[display("{inner}")]
pub struct Html {
    inner: String,
}

impl Html {
    /// Create a new `Html` from a stream of markdown events.
    pub fn from_events<'a>(
        events: impl Iterator<Item = pulldown_cmark::Event<'a>>,
    ) -> Self {
        events.collect::<Html>()
    }

    /// Extend the HTML with a stream of markdown events.
    pub fn extend<'a, It>(
        &mut self,
        events: impl IntoIterator<Item = pulldown_cmark::Event<'a>, IntoIter = It>,
    ) where
        It: Iterator<Item = pulldown_cmark::Event<'a>>,
    {
        use std::borrow::Cow;
        use pulldown_cmark::{Event, Tag, TagEnd, CowStr};
        use xml::escape::escape_str_pcdata;

        let escape_pcdata = |cow_str: CowStr<'a>| -> CowStr<'a> {
            // This is necessary because `escape_str_pcdata` does not have
            // lifetime annotations, although it could since it doesn't copy the
            // string and this is documented.
            match escape_str_pcdata(cow_str.as_ref()) {
                Cow::Borrowed(_) => cow_str.into(),
                Cow::Owned(s) => s.into(),
            }
        };

        let raw: It = events.into_iter();
        let escaped = raw.map(|e| {
            match e {
                // We must escape the HTML to prevent XSS attacks. A frontend should
                // take other measures as well, but we can at least provide some
                // protection.
                Event::Text(cow_str) => {
                    Event::Text(escape_pcdata(cow_str))
                },
                // Without this the escaping test fails because the paragraph
                // tags are missing because of how the markdown is parsed. We
                // always want message content to be in paragraphs.
                Event::Start(Tag::HtmlBlock) => {
                    Event::Start(Tag::CodeBlock(pulldown_cmark::CodeBlockKind::Fenced("html".into())))
                }
                Event::End(TagEnd::HtmlBlock) => {
                    Event::End(TagEnd::CodeBlock)
                }
                Event::Code(cow_str) => {
                    Event::Code(escape_pcdata(cow_str))
                },
                Event::InlineMath(cow_str) => {
                    Event::InlineMath(escape_pcdata(cow_str))
                },
                Event::DisplayMath(cow_str) => {
                    Event::DisplayMath(escape_pcdata(cow_str))
                },
                Event::Html(cow_str) => {
                    Event::Html(escape_pcdata(cow_str))
                },
                Event::InlineHtml(cow_str) => {
                    Event::InlineHtml(escape_pcdata(cow_str))
                },
                Event::FootnoteReference(cow_str) => {
                    Event::FootnoteReference(escape_pcdata(cow_str))
                },
                // No other events have text or attributes that need to be
                // escaped. Heading attributes are already escaped by
                // pulldown-cmark when rendering to HTML, so we don't need to
                // escape them here or we double escape them.
                e => e
            }
        });
        push_html(&mut self.inner, escaped);
    }
}

impl From<Html> for String {
    fn from(html: Html) -> Self {
        html.inner
    }
}

impl AsRef<str> for Html {
    fn as_ref(&self) -> &str {
        self.deref()
    }
}

impl std::borrow::Borrow<str> for Html {
    fn borrow(&self) -> &str {
        self.as_ref()
    }
}

impl std::ops::Deref for Html {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> FromIterator<pulldown_cmark::Event<'a>> for Html {
    fn from_iter<T: IntoIterator<Item = pulldown_cmark::Event<'a>>>(
        iter: T,
    ) -> Self {
        let mut html = Html {
            inner: String::new(),
        };
        html.extend(iter);
        html
    }
}

/// A trait for types that can be converted to HTML. This generally does not
/// need to be implemented directly, as it is already implemented for types
/// that implement [`ToMarkdown`].
///
/// # Note
/// - `attrs` are always enabled for HTML rendering so this does not have to be
///   set on the [`MarkdownOptions`].
///
/// [`MarkdownOptions`]: struct.MarkdownOptions.html
pub trait ToHtml: ToMarkdown {
    /// Render the type to an HTML string.
    fn html(&self) -> Html {
        let mut opts = DEFAULT_OPTIONS;
        opts.attrs = true;
        self.html_custom(DEFAULT_OPTIONS)
    }

    /// Render the type to an HTML string with maximum verbosity.
    fn html_verbose(&self) -> Html {
        self.html_custom(VERBOSE_OPTIONS)
    }

    /// Render the type to an HTML string with custom [`Options`].
    fn html_custom(&self, mut options: Options) -> Html {
        options.attrs = true;
        self.markdown_events_custom(options).collect()
    }
}

impl<T> ToHtml for T where T: ToMarkdown {}

#[cfg(test)]
mod tests {
    use std::borrow::Borrow;

    use serde_json::json;

    use crate::{
        prompt::{message::Role, Message},
        tool, Tool,
    };

    use super::*;

    #[test]
    fn test_message_html() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        assert_eq!(
            message.html().as_ref(),
            "<h3 role=\"user\">User</h3>\n<p>Hello, <strong>world</strong>!</p>\n",
        );
    }

    #[test]
    fn test_prompt_html() {
        let prompt = crate::prompt::Prompt {
            system: Some("Do stuff the user says.".into()),
            tools: Some(vec![Tool {
                name: "python".into(),
                description: "Run a Python script.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Python script to run.",
                        },
                    },
                    "required": ["script"],
                }),
                #[cfg(feature = "prompt-caching")]
                cache_control: None,
            }]),
            messages: vec![
                Message {
                    role: Role::User,
                    content: "Run a hello world python program.".into(),
                },
                tool::Use {
                    id: "id".into(),
                    name: "python".into(),
                    input: json!({
                        "script": "print('Hello, world!')",
                    }),
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                }
                .into(),
                tool::Result {
                    tool_use_id: "id".into(),
                    content: json!({
                        "stdout": "Hello, world!\n",
                    })
                    .to_string()
                    .into(),
                    is_error: false,
                    #[cfg(feature = "prompt-caching")]
                    cache_control: None,
                }
                .into(),
                Message {
                    role: Role::Assistant,
                    content: "It is done!".into(),
                },
            ],
            ..Default::default()
        };

        assert_eq!(
            prompt.html().as_ref(),
            "<h3 role=\"user\">User</h3>\n<p>Run a hello world python program.</p>\n<h3 role=\"assistant\">Assistant</h3>\n<p>It is done!</p>\n",
        );

        assert_eq!(
            prompt.html_verbose().as_ref(),
            "<h3 role=\"system\">System</h3>\n<p>Do stuff the user says.</p>\n<h3 role=\"user\">User</h3>\n<p>Run a hello world python program.</p>\n<h3 role=\"assistant\">Assistant</h3>\n<pre><code class=\"language-json\">{\"type\":\"tool_use\",\"id\":\"id\",\"name\":\"python\",\"input\":{\"script\":\"print('Hello, world!')\"}}</code></pre>\n<h3 role=\"tool\">Tool</h3>\n<pre><code class=\"language-json\">{\"type\":\"tool_result\",\"tool_use_id\":\"id\",\"content\":[{\"type\":\"text\",\"text\":\"{\\\"stdout\\\":\\\"Hello, world!\\\\n\\\"}\"}],\"is_error\":false}</code></pre>\n<h3 role=\"assistant\">Assistant</h3>\n<p>It is done!</p>\n",
        )
    }

    #[test]
    fn test_html_from_events() {
        let events = vec![
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Paragraph),
            pulldown_cmark::Event::Text("Hello, world!".into()),
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph),
        ];

        let html = Html::from_events(events.into_iter());
        assert_eq!(html.as_ref(), "<p>Hello, world!</p>\n");
    }

    #[test]
    fn test_html_extend() {
        let mut html = Html {
            inner: String::new(),
        };

        let events = vec![
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Paragraph),
            pulldown_cmark::Event::Text("Hello, world!".into()),
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph),
        ];

        html.extend(events.into_iter());
        assert_eq!(html.as_ref(), "<p>Hello, world!</p>\n");
    }

    #[test]
    fn test_html_from_iter() {
        let events = vec![
            pulldown_cmark::Event::Start(pulldown_cmark::Tag::Paragraph),
            pulldown_cmark::Event::Text("Hello, world!".into()),
            pulldown_cmark::Event::End(pulldown_cmark::TagEnd::Paragraph),
        ];

        let html: Html = events.into_iter().collect();
        assert_eq!(html.as_ref(), "<p>Hello, world!</p>\n");
    }

    #[test]
    fn test_to_html() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        assert_eq!(
            message.html().as_ref(),
            "<h3 role=\"user\">User</h3>\n<p>Hello, <strong>world</strong>!</p>\n",
        );

        assert_eq!(
            message.html_verbose().as_ref(),
            "<h3 role=\"user\">User</h3>\n<p>Hello, <strong>world</strong>!</p>\n",
        );

        assert_eq!(
            message
                .html_custom(Options {
                    attrs: false,
                    ..DEFAULT_OPTIONS
                })
                .as_ref(),
            // `attrs` are always enabled for HTML rendering
            "<h3 role=\"user\">User</h3>\n<p>Hello, <strong>world</strong>!</p>\n",
     
        );
    }

    #[test]
    fn test_borrow() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        let html: Html = message.html();
        let borrowed: &str = html.borrow();
        assert_eq!(borrowed, html.as_ref());
    }

    #[test]
    fn test_into_string() {
        let message = Message {
            role: Role::User,
            content: "Hello, **world**!".into(),
        };

        let html: Html = message.html();
        let string: String = html.into();
        assert_eq!(string, "<h3 role=\"user\">User</h3>\n<p>Hello, <strong>world</strong>!</p>\n");

    }

    #[test]
    fn test_escaping() {
        use pulldown_cmark::{Event, Tag, TagEnd, HeadingLevel::H3};

        let message = Message {
            role: Role::Assistant,
            content: "bla bla<script>alert('XSS')</script>bla bla".into(),
        };

        assert_eq!(
            message.html().as_ref(),
            "<h3 role=\"assistant\">Assistant</h3>\n<p>bla bla&lt;script&gt;alert('XSS')&lt;/script&gt;bla bla</p>\n",
        );

        let message = Message {
            role: Role::Assistant,
            content: "<script>alert('XSS')</script>".into(),
        };

        assert_eq!(
            message.html_verbose().as_ref(),
            // In the case where a content block is entirely code, it is
            // rendered as a code block. This is mostly done because of how
            // markdown is parsed and we're lazy, but also it's nice behavior.
            "<h3 role=\"assistant\">Assistant</h3>\n<pre><code class=\"language-html\">&lt;script&gt;alert('XSS')&lt;/script&gt;</code></pre>\n",
        );

        // Test escaping of attributes
        let bad_attrs = vec![
            Event::Start(Tag::Heading { level: H3, id: None, classes: vec![], attrs: vec![(r#"<p>badkey</p>"#.into(), Some(r#""sneaky"><script>badvalue</script>"#.into()))] }),
            Event::Text("Hello, world!".into()),
            Event::End(TagEnd::Heading(H3)),
        ];

        let html = Html::from_events(bad_attrs.into_iter());
        // FIXME: This is not the correct behavior. pulldown_cmark is escaping
        // the attributes, but not forward slashes in keys leading to a broken
        // key. This is a bug in pulldown_cmark. Fixing this is a low priority
        // since it only applies to cases where a third party is providing the
        // trait and doing very silly things with attributes.
        assert_eq!(html.as_ref(), r#"<h3 &lt;p&gt;badkey&lt;/p&gt;="&quot;sneaky&quot;&gt;&lt;script&gt;badvalue&lt;/script&gt;">Hello, world!</h3>
"#);
    }
}
