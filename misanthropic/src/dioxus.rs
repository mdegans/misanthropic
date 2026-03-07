//! Support for rendering to HTML using [`dioxus`].
use std::borrow::Cow;

use dioxus::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    cot::{ThoughtOrSpeech, ThoughtsAndSpeech},
    prompt::{
        self,
        message::{self, Block},
    },
};

/// Options for converting to a [`dioxus`] [`Element`].
#[derive(Clone, Serialize, Deserialize)]
pub struct Options<'a> {
    /// [`System`] prompt options.
    ///
    /// [`System`]: opts::System
    pub system: opts::System<'a>,
    /// Chain of [`Thought`] options.
    ///
    /// [`Thought`]: opts::Thought
    pub thought: opts::Thought<'a>,
    /// [`tool::Use`] options.
    pub tool_use: opts::ToolUse<'a>,
    /// [`tool::Result`] options.
    pub tool_result: opts::ToolResult<'a>,
    /// [`Image`] view options.
    ///
    /// [`Image`]: crate::prompt::message::Image
    pub image: opts::Image<'a>,
    /// Speech (visible text) options.
    pub speech: opts::Speech<'a>,
}

impl Default for Options<'static> {
    fn default() -> Self {
        Self {
            system: opts::System::Hidden,
            // The user knows that the agent is thinking but not what.
            thought: opts::Thought::Placeholder {
                class: Cow::Borrowed("thought placeholder"),
            },
            tool_use: opts::ToolUse::Hidden,
            tool_result: opts::ToolResult::Hidden,
            image: opts::Image::Show {
                class: Cow::Borrowed("image show"),
            },
            speech: opts::Speech::Show {
                class: Cow::Borrowed("speech show"),
            },
        }
    }
}

/// Types for [`Options`].
pub mod opts {
    use super::*;

    /// A heading level.
    #[derive(Copy, Clone, Serialize, Deserialize)]
    pub enum HeadingLevel {
        /// `<h1>`.
        H1,
        /// `<h2>`.
        H2,
        /// `<h3>`.
        H3,
        /// `<h4>`.
        H4,
        /// `<h5>`.
        H5,
        /// `<h6>`.
        H6,
    }

    impl HeadingLevel {
        /// Convert to a [`HeadingLevel::element`] with the given text `content`.
        #[allow(unused_variables)] // because macros break this lint
        pub fn element<'a>(self, key: u64, content: Cow<'a, str>) -> Element {
            match self {
                HeadingLevel::H1 => rsx!(h1 {
                    key: key,
                    {content}
                }),
                HeadingLevel::H2 => rsx!(h2 {
                    key: key,
                    {content}
                }),
                HeadingLevel::H3 => rsx!(h3 {
                    key: key,
                    {content}
                }),
                HeadingLevel::H4 => rsx!(h4 {
                    key: key,
                    {content}
                }),
                HeadingLevel::H5 => rsx!(h5 {
                    key: key,
                    {content}
                }),
                HeadingLevel::H6 => rsx!(h6 {
                    key: key,
                    {content}
                }),
            }
        }
    }

    // We don't require the markdown feature for dioxus support, but this might
    // be handy for those who do.
    #[cfg(feature = "markdown")]
    impl From<pulldown_cmark::HeadingLevel> for HeadingLevel {
        fn from(level: pulldown_cmark::HeadingLevel) -> Self {
            match level {
                pulldown_cmark::HeadingLevel::H1 => HeadingLevel::H1,
                pulldown_cmark::HeadingLevel::H2 => HeadingLevel::H2,
                pulldown_cmark::HeadingLevel::H3 => HeadingLevel::H3,
                pulldown_cmark::HeadingLevel::H4 => HeadingLevel::H4,
                pulldown_cmark::HeadingLevel::H5 => HeadingLevel::H5,
                pulldown_cmark::HeadingLevel::H6 => HeadingLevel::H6,
            }
        }
    }

    /// System prompt mode.
    #[derive(Clone, Serialize, Deserialize)]
    pub enum System<'a> {
        /// No element at all.
        Hidden,
        /// Placeholder.
        Placeholder {
            /// Classes to add.
            class: Cow<'a, str>,
        },
        /// Show full system prompt.
        Show {
            /// Classes to add.
            class: Cow<'a, str>,
        },
    }

    /// Thought mode.
    #[derive(Clone, Serialize, Deserialize)]
    pub enum Thought<'a> {
        /// Hide thoughts.
        Hidden,
        /// No content.
        Placeholder {
            /// Classes to add.
            class: Cow<'a, str>,
        },
        /// Show thoughts with `thought` class.
        Show {
            /// Classes to add.
            class: Cow<'a, str>,
        },
    }

    /// [`tool::Use`] options.
    ///
    /// [`tool::Use`]: crate::tool::Use
    #[derive(Clone, Serialize, Deserialize)]
    pub enum ToolUse<'a> {
        /// Hide tool use.
        Hidden,
        /// No content.
        Placeholder {
            /// Show name of tool being used as a heading.
            show_name: Option<HeadingLevel>,
            /// Classes to add.
            class: Cow<'a, str>,
        },
        /// Show tool use with JSON in a `<code>` block.
        Show {
            /// Show name of tool being used as a heading.
            show_name: Option<HeadingLevel>,
            /// Classes to add.
            class: Cow<'a, str>,
        },
    }

    /// [`tool::Result`] options.
    #[derive(Clone, Serialize, Deserialize)]
    pub enum ToolResult<'a> {
        /// Hide tool result.
        Hidden,
        /// No content.
        // The tool result does not actually have the name so we can't show it,
        // however it's always paired with a tool use so we can show that name.
        Placeholder {
            /// Classes to add on error.
            error: Cow<'a, str>,
            /// Classes to add on success.
            ok: Cow<'a, str>,
        },
        /// Show tool result with JSON in a `<code>` block.
        Show {
            /// Classes to add on error.
            error: Cow<'a, str>,
            /// Classes to add on success.
            ok: Cow<'a, str>,
        },
    }

    /// [`Image`] options.
    ///
    /// [`Image`]: crate::prompt::message::Image
    #[derive(Clone, Serialize, Deserialize)]
    pub enum Image<'a> {
        /// Hide images.
        Hidden,
        /// No content.
        Placeholder {
            /// Classes to add.
            class: Cow<'a, str>,
        },
        /// Show images.
        Show {
            /// Classes to add.
            class: Cow<'a, str>,
        },
    }

    /// Speech options.
    #[derive(Clone, Serialize, Deserialize)]
    pub enum Speech<'a> {
        /// Hide speech (Why would you do this?).
        Hidden,
        /// No content. (Why would you do this?)
        Placeholder {
            /// Classes to add.
            class: Cow<'a, str>,
        },
        /// Show speech.
        Show {
            /// Classes to add.
            class: Cow<'a, str>,
        },
    }
}

fn hash<T: std::hash::Hash>(t: &[T]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    t.hash(&mut hasher);
    hasher.finish()
}

/// A type that can convert into an [`Element`].
pub trait IntoElement {
    /// Convert into an [`Element`] with custom options. `key` should be unique
    /// for each element. It will be combined with any children's keys.
    fn into_element_custom(self, key: u64, opts: &Options) -> Element;
}

// Start with the smallest possible example.
#[cfg(feature = "cot")]
impl IntoElement for ThoughtOrSpeech<'_> {
    #[allow(unused_variables)] // because macros break this lint
    fn into_element_custom(self, key: u64, opts: &Options) -> Element {
        match self {
            ThoughtOrSpeech::Thought(thought) => match &opts.thought {
                opts::Thought::Hidden => rsx!(),
                opts::Thought::Placeholder { class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                        "Thinking..."
                    })
                }
                opts::Thought::Show { class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                        {thought}
                    })
                }
            },
            ThoughtOrSpeech::Speech(speech) => match &opts.speech {
                opts::Speech::Hidden => rsx!(),
                opts::Speech::Placeholder { class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                    })
                }
                opts::Speech::Show { class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                        {speech}
                    })
                }
            },
        }
    }
}

impl IntoElement for &Block<'_> {
    fn into_element_custom(self, key: u64, opts: &Options) -> Element {
        #[allow(unused_variables)] // because macros break this lint
        match self {
            Block::Text {
                text,
                cache_control,
                ..
            } => {
                {
                    rsx!({
                        ThoughtsAndSpeech::new(text.as_ref()).enumerate().map(
                            |(i, ts)| {
                                ts.into_element_custom(
                                    hash(&[key, i as u64]),
                                    opts,
                                )
                            },
                        )
                    })
                }
                #[cfg(not(feature = "cot"))]
                {
                    rsx!(div {
                        key: key,
                        class: "text",
                        class: if cache_control.is_some() { "cache" } else { "" },
                        {text.as_ref()}
                    })
                }
            }
            Block::Thought {
                thought: thinking,
                signature,
            } => match &opts.thought {
                opts::Thought::Hidden => rsx!(),
                opts::Thought::Placeholder { class } => {
                    rsx!(div {
                        key: key,
                        id: signature.as_ref(),
                        class: class.as_ref(),
                    })
                }
                opts::Thought::Show { class } => {
                    rsx!(div {
                        key: key,
                        id: signature.as_ref(),
                        class: class.as_ref(),
                        {thinking}
                    })
                }
            },
            Block::RedactedThought { signature } => match &opts.thought {
                opts::Thought::Hidden => rsx!(),
                opts::Thought::Placeholder { class } => {
                    rsx!(div {
                        key: key,
                        id: signature.as_ref(),
                        class: class.as_ref(),
                    })
                }
                opts::Thought::Show { class } => {
                    rsx!(div {
                        key: key,
                        id: signature.as_ref(),
                        class: format!("{} redacted", class),
                        "Anthropic redacted a thought."
                    })
                }
            },
            Block::Image {
                image,
                cache_control,
            } => match &opts.image {
                opts::Image::Hidden => rsx!(),
                opts::Image::Placeholder { class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                    })
                }
                opts::Image::Show { class } => {
                    rsx!(img {
                        key: key,
                        class: class.as_ref(),
                        src: {
                            match image {
                                message::Image::Base64 { media_type, data } => {
                                    format!(
                                        "data:{};base64,{}",
                                        media_type, data
                                    )
                                }
                            }
                        }
                    })
                }
            },
            Block::ToolUse { call } => match &opts.tool_use {
                opts::ToolUse::Hidden => rsx!(),
                opts::ToolUse::Placeholder { show_name, class } => {
                    rsx!(div {
                        key: key,
                        class: class.as_ref(),
                        {show_name.map(
                            |level| level.element(key, call.name.as_ref().into()
                        ))}
                    })
                }
                opts::ToolUse::Show { show_name, class } => {
                    rsx!(code {
                        key: key,
                        lang: "json",
                        class: class.as_ref(),
                        {serde_json::to_string_pretty(call).unwrap()}
                    })
                }
            },
            Block::Document { source, .. } => {
                rsx!(div {
                    key: key,
                    class: "document",
                    {source.to_string()}
                })
            }
            Block::ToolResult { result } => match &opts.tool_result {
                opts::ToolResult::Hidden => rsx!(),
                opts::ToolResult::Placeholder { error, ok } => {
                    rsx!(div {
                        key: key,
                        title: if result.is_error { "Error" } else { "Ok" },
                        class: if result.is_error {
                            error.as_ref()
                        } else {
                            ok.as_ref()
                        },
                    })
                }
                opts::ToolResult::Show { error, ok } => {
                    rsx!(code {
                        key: key,
                        title: if result.is_error { "Error" } else { "Ok" },
                        lang: "json",
                        class: if result.is_error {
                            error.as_ref()
                        } else {
                            ok.as_ref()
                        },
                        {serde_json::to_string_pretty(result).unwrap()}
                    })
                }
            },
        }
    }
}

impl IntoElement for &message::Content<'_> {
    fn into_element_custom(self, key: u64, opts: &Options) -> Element {
        match self {
            message::Content::SinglePart(text) => {
                rsx!(div {
                    key: key,
                    class: "single-part",
                    {ThoughtsAndSpeech::new(text.as_ref())
                        .enumerate()
                        .map(|(i, ts)| ts.into_element_custom(hash(&[key, i as u64]), opts))}
                })
            }
            message::Content::MultiPart(blocks) => {
                rsx!(div {
                    key: key,
                    class: "multi-part",
                    {blocks
                        .iter()
                        .enumerate()
                        .map(|(i, block)| block.into_element_custom(hash(&[key, i as u64]), opts))}
                })
            }
        }
    }
}

impl IntoElement for &message::Message<'_> {
    fn into_element_custom(self, key: u64, opts: &Options) -> Element {
        rsx!(div {
            key: key,
            class: if self.tool_result().is_some() {
                // We lie, because it really should be a separate role and it's
                // much easier to format this way.
                "tool"
            } else if self.role.is_user() {
                "user"
            } else {
                "assistant"
            },
            class: "message",
            // This has to be implemented here and not in content because we
            // need to know the role. This is unfortunate because it's a bit
            // ugly.
            {match &self.content {
                    message::Content::SinglePart(text) => {
                        rsx!(div {
                            key: key,
                            class: "single-part",
                            {if self.role.is_user() {
                                match &opts.speech {
                                    opts::Speech::Hidden => rsx!(),
                                    opts::Speech::Placeholder { class } => {
                                        rsx!(div {
                                            key: hash(&[key, 0]),
                                            class: class.as_ref(),
                                        })
                                    }
                                    opts::Speech::Show { class } => {
                                        rsx!(div {
                                            key: hash(&[key, 0]),
                                            class: class.as_ref(),
                                            {text}
                                        })
                                    }
                                }
                            } else {
                                rsx!(
                                    div {
                                        key: hash(&[key, 0]),
                                        class: "assistant",
                                        {ThoughtsAndSpeech::new(text.as_ref())
                                            .enumerate()
                                            .map(|(i, ts)| ts.into_element_custom(hash(&[key, i as u64]), opts))}
                                    }
                                )
                            }}
                        })
                    }
                    message::Content::MultiPart(blocks) => {
                        rsx!(div {
                            key: key,
                            class: "multi-part",
                            {blocks
                                .iter()
                                .enumerate()
                                .map(|(i, block)| {
                                    block.into_element_custom(
                                        hash(&[key, i as u64]),
                                        opts
                                    )
                                })}
                        })
                    }
                }
            }
        })
    }
}

impl IntoElement for &prompt::Prompt<'_> {
    fn into_element_custom(self, key: u64, opts: &Options) -> Element {
        rsx!(
            div {
                key: key,
                class: "prompt",
                if let Some(system) = &self.system {
                    match &opts.system {
                        opts::System::Hidden => rsx!(),
                        opts::System::Placeholder { class } => {
                            rsx!(div {
                                key: hash(&[key, 1]),
                                class: class.as_ref(),
                            })
                        }
                        opts::System::Show { class } => {
                            rsx!(div {
                                key: hash(&[key, 0]),
                                class: class.as_ref(),
                                {system.into_element_custom(hash(&[key, 1]), opts)}
                            })
                        }
                    }
                }
                div {
                    key: hash(&[key, 2]),
                    class: "messages",
                    {(2..).zip(self.messages.iter()).map(|(i, message)| {
                        message.into_element_custom(hash(&[key, i as u64]), opts)
                    })}
                }
            }
        )
    }
}
