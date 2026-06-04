//! Basic support for parsing chain of thought within XML tags. Nested tags are
//! not supported. Bring the [`Thinkable`] trait into scope to use the methods
//! provided by this module.
use std::borrow::Cow;

use derive_more::derive::Deref;

use crate::prompt::{
    Message,
    message::{Block, Content},
};

/// Supported start tags for [`Thought`]s.
pub const DEFAULT_START_TAGS: &[&str] =
    &["<thinking>", "<think>", "<inner-voice>", "<thought>"];
/// Supported end tags for [`Thought`]s.
pub const DEFAULT_END_TAGS: &[&str] =
    &["</thinking>", "</think>", "</inner-voice>", "</thought>"];

/// Contents of a `<thinking>` element.
#[derive(Debug, Clone, Deref, derive_more::Display)]
#[display("{text}")]
pub struct Thought {
    /// The text inside a thinking element.
    pub text: Cow<'static, str>,
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Thought {
    fn markdown_events_custom(
        &self,
        options: crate::html::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        Box::new(pulldown_cmark::Parser::new_ext(
            self.text.as_ref(),
            options.inner,
        ))
    }
}

/// Content outside thinking elements.
#[derive(Debug, Clone, Deref, derive_more::Display)]
#[display("{text}")]
pub struct Speech {
    /// The text outside thinking elements.
    pub text: Cow<'static, str>,
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for Speech {
    fn markdown_events_custom(
        &self,
        options: crate::html::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        Box::new(pulldown_cmark::Parser::new_ext(
            self.text.as_ref(),
            options.inner,
        ))
    }
}

/// Either a [`Thought`] or [`Speech`].
#[derive(Debug, Clone, derive_more::IsVariant)]
pub enum ThoughtOrSpeech {
    /// An Assistant [`Thought`].
    Thought(Thought),
    /// [`Speech`] intended for the user.
    Speech(Speech),
}

impl ThoughtOrSpeech {
    /// Consumes the [`ThoughtOrSpeech`] and returns the [`Thought`] if it is a
    /// [`Thought`].
    pub fn into_thought(self) -> Option<Thought> {
        match self {
            ThoughtOrSpeech::Thought(thought) => Some(thought),
            _ => None,
        }
    }

    /// Consumes the [`ThoughtOrSpeech`] and returns the [`Speech`] if it is a
    /// [`Speech`].
    pub fn into_speech(self) -> Option<Speech> {
        match self {
            ThoughtOrSpeech::Speech(speech) => Some(speech),
            _ => None,
        }
    }
}

/// An iterator over [`ThoughtOrSpeech`]es.
pub struct ThoughtsAndSpeech<'a> {
    /// The text being iterated over.
    text: &'a str,
    /// The start tags for thoughts.
    start_tags: &'static [&'static str],
    /// The end tags for thoughts.
    end_tags: &'static [&'static str],
    /// The current index in the text.
    index: usize,
}

impl<'a> ThoughtsAndSpeech<'a> {
    /// Create a new [`ThoughtsAndSpeech`] iterator.
    pub fn new(text: &'a str) -> Self {
        Self::new_custom(text, DEFAULT_START_TAGS, DEFAULT_END_TAGS)
    }

    /// Create a new [`ThoughtsAndSpeech`] iterator with custom start and end tags.
    ///
    /// # Panics
    /// - In debug builds if the length of `start_tags` and `end_tags` are not
    ///   equal.
    pub fn new_custom(
        text: &'a str,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Self {
        debug_assert_eq!(start_tags.len(), end_tags.len());
        Self {
            text,
            start_tags,
            end_tags,
            index: 0,
        }
    }
}

impl<'a> Iterator for ThoughtsAndSpeech<'a> {
    type Item = ThoughtOrSpeech;

    #[allow(clippy::needless_return)] // becuase it's harder to read without it
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.text.len() {
            return None;
        }

        // Find the next start tag.
        let start_pair = self.start_tags.iter().find_map(|&tag| {
            self.text[self.index..].find(tag).map(|start| (start, tag))
        });

        // Find the next end tag.
        let end_pair = self.end_tags.iter().find_map(|&tag| {
            self.text[self.index..].find(tag).map(|end| (end, tag))
        });

        if let Some((start, _)) = &start_pair
            && *start != 0
        {
            // There is speech before the start tag. Return it.
            let speech = Some(ThoughtOrSpeech::Speech(Speech {
                text: self.text[self.index..self.index + *start]
                    .to_owned()
                    .into(),
            }));
            self.index += start;
            return speech;
        }

        match (start_pair, end_pair) {
            (Some((start, start_tag)), Some((end, end_tag))) => {
                // We have a pair of tags. We need to return everything between
                // the start and end tags as a thought.
                let thought_start = self.index + start + start_tag.len();
                let thought_end = self.index + end;
                let thought = Some(ThoughtOrSpeech::Thought(Thought {
                    text: self.text[thought_start..thought_end]
                        .to_owned()
                        .into(),
                }));

                // And then set the index to the end of the end tag.
                self.index = thought_end + end_tag.len();
                return thought;
            }
            (Some((start, start_tag)), None) => {
                // We have a start tag, but no end tag. The rest of the text is
                // a thought (because if the agent forgot, we don't want to leak
                // thoughts).
                let thought_start = self.index + start + start_tag.len();
                let thought_end = self.text.len();
                let thought = Some(ThoughtOrSpeech::Thought(Thought {
                    text: self.text[thought_start..thought_end]
                        .to_owned()
                        .into(),
                }));

                self.index = thought_end;
                return thought;
            }
            (None, Some((end, end_tag))) => {
                // We have an end tag, but no start tag. Everything up to the
                // end tag is a thought (same rationale as above).
                let thought = Some(ThoughtOrSpeech::Thought(Thought {
                    text: self.text[self.index..self.index + end]
                        .to_owned()
                        .into(),
                }));
                self.index += end + end_tag.len();
                return thought;
            }
            (None, None) => {
                // There are no tags. The entire text is speech.
                let speech = Some(ThoughtOrSpeech::Speech(Speech {
                    text: self.text[self.index..].to_owned().into(),
                }));
                self.index = self.text.len();
                return speech;
            }
        }
    }
}

/// A trait for types containing [`Thought`]s and [`Speech`].
pub trait Thinkable {
    /// Return an iterator of [`ThoughtOrSpeech`] with custom start and end
    /// tags.
    fn thoughts_and_speech_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech> + '_>;

    /// Return an iterator of [`ThoughtOrSpeech`] with default start and end
    /// tags.
    fn thoughts_and_speech(
        &self,
    ) -> impl Iterator<Item = ThoughtOrSpeech> + '_ {
        self.thoughts_and_speech_custom(DEFAULT_START_TAGS, DEFAULT_END_TAGS)
    }

    /// Return an iterator of [`Thought`]s with default start and end tags.
    fn thoughts(&self) -> impl Iterator<Item = Thought> + '_ {
        self.thoughts_and_speech()
            .filter_map(ThoughtOrSpeech::into_thought)
    }

    /// Return an iterator of [`Thought`]s with custom start and end tags.
    fn thoughts_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> impl Iterator<Item = Thought> + '_ {
        self.thoughts_and_speech_custom(start_tags, end_tags)
            .filter_map(ThoughtOrSpeech::into_thought)
    }

    /// Return an iterator of [`Speech`]es with default start and end tags.
    fn speech(&self) -> impl Iterator<Item = Speech> + '_ {
        self.thoughts_and_speech()
            .filter_map(ThoughtOrSpeech::into_speech)
    }

    /// Return an iterator of [`Speech`]es with custom start and end tags.
    fn speech_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> impl Iterator<Item = Speech> + '_ {
        self.thoughts_and_speech_custom(start_tags, end_tags)
            .filter_map(ThoughtOrSpeech::into_speech)
    }
}

impl Thinkable for Block {
    /// Get an iterator over the thoughts and speech in the block with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech> + '_> {
        match self {
            Block::Text { text, .. } => Box::new(Box::new(
                ThoughtsAndSpeech::new_custom(text, start_tags, end_tags),
            )),
            _ => Box::new(std::iter::empty()),
        }
    }
}

impl Thinkable for Content {
    /// Get an iterator over the thoughts and speech in the content with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech> + '_> {
        Box::new(self.iter().flat_map(move |block| {
            block.thoughts_and_speech_custom(start_tags, end_tags)
        }))
    }
}

impl Thinkable for Message {
    /// Get an iterator over the thoughts and speech in the message with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech> + '_> {
        self.content
            .thoughts_and_speech_custom(start_tags, end_tags)
    }
}

#[cfg(feature = "markdown")]
impl crate::markdown::ToMarkdown for ThoughtOrSpeech {
    fn markdown_events_custom(
        &self,
        options: crate::html::Options,
    ) -> Box<dyn Iterator<Item = pulldown_cmark::Event<'_>> + '_> {
        use pulldown_cmark::{Event, HeadingLevel::H5, Tag, TagEnd};

        let h = options.heading_level.unwrap_or(H5);
        let variant_name = match self {
            ThoughtOrSpeech::Thought(_) => "thought",
            ThoughtOrSpeech::Speech(_) => "speech",
        };
        let variant_name_capitalized = match self {
            ThoughtOrSpeech::Thought(_) => stringify!(Thought),
            ThoughtOrSpeech::Speech(_) => stringify!(Speech),
        };
        let markdown_parsed_text: Box<dyn Iterator<Item = Event> + '_> =
            match self {
                ThoughtOrSpeech::Speech(speech) => {
                    speech.markdown_events_custom(options)
                }
                ThoughtOrSpeech::Thought(thought) => {
                    thought.markdown_events_custom(options)
                }
            };
        let header: Box<dyn Iterator<Item = Event> + '_> = Box::new(
            [
                Event::Start(Tag::Heading {
                    level: h,
                    id: None,
                    classes: if options.attrs {
                        vec![variant_name.into()]
                    } else {
                        vec![]
                    },
                    attrs: vec![],
                }),
                Event::Text(variant_name_capitalized.into()),
                Event::End(TagEnd::Heading(h)),
            ]
            .into_iter(),
        );

        Box::new(header.chain(markdown_parsed_text))
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::Write;

    use crate::{
        html::ToHtml,
        prompt::{self, message::Role},
    };

    use super::*;

    #[test]
    fn test_thoughts() {
        let message: prompt::Message = (
            Role::Assistant,
            "<thinking>Oh dear, it's this schmuck again :/</thinking>It's a pleasure to hear from you again, dear user!<thinking>That was sarcasm.</thinking>Such a treat!",
        ).into();

        let thoughts = message.thoughts().collect::<Vec<_>>();

        assert_eq!(thoughts.len(), 2);
        assert_eq!(thoughts[0].text, "Oh dear, it's this schmuck again :/");
        assert_eq!(thoughts[1].text, "That was sarcasm.");

        let speech = message.speech().collect::<Vec<_>>();

        assert_eq!(speech.len(), 2);
        assert_eq!(
            speech[0].text,
            "It's a pleasure to hear from you again, dear user!"
        );
        assert_eq!(speech[1].text, "Such a treat!");

        // Test with no matching end tag (we consider the rest of the text as a
        // thought).
        let message: prompt::Message = (
            Role::Assistant,
            "Welcome to customer support at Amazon!<thinking>Oh dear, it's this schmuck again :/</thniking>It's a pleasure to hear from you again, dear user!<thinking>That was sarcasm."
        ).into();

        let thoughts = message.thoughts().collect::<Vec<_>>();
        assert_eq!(thoughts.len(), 1);
        assert_eq!(
            thoughts[0].text,
            "Oh dear, it's this schmuck again :/</thniking>It's a pleasure to hear from you again, dear user!<thinking>That was sarcasm."
        );

        // Test with A non-text block
        let tool_result = Block::ToolResult {
            result: crate::tool::Result {
                tool_use_id: "blablak".into(),
                content: "blabla".into(),
                is_error: false,
                cache_control: None,
            },
        };

        assert_eq!(tool_result.thoughts().count(), 0);
    }

    #[test]
    fn test_thoughts_custom() {
        let message: prompt::Message = (
            Role::Assistant,
            "<snark>Oh dear, it's this schmuck again :/</snark>It's a pleasure to hear from you again, dear user!<snark>That was sarcasm.</snark>"
        ).into();

        let thoughts = message
            .thoughts_custom(&["<snark>"], &["</snark>"])
            .collect::<Vec<_>>();

        assert_eq!(thoughts.len(), 2);
        assert_eq!(thoughts[0].text, "Oh dear, it's this schmuck again :/");
        assert_eq!(thoughts[1].text, "That was sarcasm.");

        let speech = message
            .speech_custom(&["<snark>"], &["</snark>"])
            .collect::<Vec<_>>();

        assert_eq!(speech.len(), 1);
        assert_eq!(
            speech[0].text,
            "It's a pleasure to hear from you again, dear user!"
        );

        // Test with no matching end tag (we consider the rest of the text as a
        // thought)
        let message: prompt::Message = (
            Role::Assistant,
            "<snark>Oh dear, it's this schmuck again :/</snrak>It's a pleasure to hear from you again, dear user!"
        ).into();

        let thoughts = message
            .thoughts_custom(&["<snark>"], &["</snark>"])
            .collect::<Vec<_>>();
        assert_eq!(thoughts.len(), 1);
        assert_eq!(
            thoughts[0].text,
            "Oh dear, it's this schmuck again :/</snrak>It's a pleasure to hear from you again, dear user!"
        );
    }

    fn test_thoughts_and_speech_to_html_helper(text: &str, expected: &str) {
        let message: prompt::Message = (Role::Assistant, text).into();

        // TODO: Make this work, but it's not a priority.
        // assert_eq!(message.thoughts_and_speech().html().as_ref(), expected);

        let mut actual = String::new();
        for thought_or_speech in message.thoughts_and_speech() {
            actual.write_str(&thought_or_speech.html()).unwrap();
        }

        assert_eq!(actual, expected);
    }

    #[test]
    fn test_thoughts_and_speech_to_html() {
        test_thoughts_and_speech_to_html_helper(
            "<thinking>Oh dear, it's this schmuck again :/</thinking>It's a pleasure to hear from you again, dear user!<thinking>That was sarcasm.</thinking>Such a treat!",
            "<h5 class=\"thought\">Thought</h5>\n<p>Oh dear, it's this schmuck again :/</p>\n<h5 class=\"speech\">Speech</h5>\n<p>It's a pleasure to hear from you again, dear user!</p>\n<h5 class=\"thought\">Thought</h5>\n<p>That was sarcasm.</p>\n<h5 class=\"speech\">Speech</h5>\n<p>Such a treat!</p>\n",
        );
    }
}
