//! Basic support for parsing chain of thought within XML tags. Nested tags are
//! not supported.
use derive_more::derive::Deref;

use crate::prompt::{
    message::{Block, Content},
    Message,
};

/// Supported start tags for [`Thought`]s.
pub const DEFAULT_START_TAGS: &[&str] =
    &["<thinking>", "<inner-voice>", "<thought>"];
/// Supported end tags for [`Thought`]s.
pub const DEFAULT_END_TAGS: &[&str] =
    &["</thinking>", "</inner-voice>", "</thought>"];

/// Contents of a `<thinking>` element.
#[derive(Debug, Clone, Deref)]
pub struct Thought<'a> {
    /// The text inside a thinking element.
    pub text: &'a str,
}

/// Content outside thinking elements.
#[derive(Debug, Clone, Deref)]
pub struct Speech<'a> {
    /// The text outside thinking elements.
    pub text: &'a str,
}

/// Either a [`Thought`] or [`Speech`].
#[derive(Debug, Clone, derive_more::IsVariant)]
pub enum ThoughtOrSpeech<'a> {
    /// An Assistant [`Thought`].
    Thought(Thought<'a>),
    /// [`Speech`] intended for the user.
    Speech(Speech<'a>),
}

impl<'a> ThoughtOrSpeech<'a> {
    /// Consumes the [`ThoughtOrSpeech`] and returns the [`Thought`] if it is a
    /// [`Thought`].
    pub fn into_thought(self) -> Option<Thought<'a>> {
        match self {
            ThoughtOrSpeech::Thought(thought) => Some(thought),
            _ => None,
        }
    }

    /// Consumes the [`ThoughtOrSpeech`] and returns the [`Speech`] if it is a
    /// [`Speech`].
    pub fn get_speech(self) -> Option<Speech<'a>> {
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
    type Item = ThoughtOrSpeech<'a>;

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

        if let Some((start, _)) = &start_pair {
            if *start != 0 {
                // There is speech before the start tag. Return it.
                let speech = Some(ThoughtOrSpeech::Speech(Speech {
                    text: &self.text[self.index..self.index + *start],
                }));
                self.index += start;
                return speech;
            }
        }

        match (start_pair, end_pair) {
            (Some((start, start_tag)), Some((end, end_tag))) => {
                // We have a pair of tags. We need to return everything between
                // the start and end tags as a thought.
                let thought_start = self.index + start + start_tag.len();
                let thought_end = self.index + end;
                let thought = Some(ThoughtOrSpeech::Thought(Thought {
                    text: &self.text[thought_start..thought_end],
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
                    text: &self.text[thought_start..thought_end],
                }));

                self.index = thought_end;
                return thought;
            }
            (None, Some((end, end_tag))) => {
                // We have an end tag, but no start tag. Everything up to the
                // end tag is a thought (same rationale as above).
                let thought = Some(ThoughtOrSpeech::Thought(Thought {
                    text: &self.text[self.index..self.index + end],
                }));
                self.index += end + end_tag.len();
                return thought;
            }
            (None, None) => {
                // There are no tags. The entire text is speech.
                let speech = Some(ThoughtOrSpeech::Speech(Speech {
                    text: &self.text[self.index..],
                }));
                self.index = self.text.len();
                return speech;
            }
        }
    }
}

/// A trait for types containing [`Thought`]s and [`Speech`].
pub trait Thinkable<'a> {
    /// Return an iterator of [`ThoughtOrSpeech`] with custom start and end
    /// tags.
    fn thoughts_and_speech_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech<'a>> + 'a>;

    /// Return an iterator of [`ThoughtOrSpeech`] with default start and end
    /// tags.
    fn thoughts_and_speech(
        &'a self,
    ) -> impl Iterator<Item = ThoughtOrSpeech<'a>> + 'a {
        self.thoughts_and_speech_custom(DEFAULT_START_TAGS, DEFAULT_END_TAGS)
    }

    /// Return an iterator of [`Thought`]s with default start and end tags.
    fn thoughts(&'a self) -> impl Iterator<Item = Thought<'a>> + 'a {
        self.thoughts_and_speech()
            .filter_map(ThoughtOrSpeech::into_thought)
    }

    /// Return an iterator of [`Thought`]s with custom start and end tags.
    fn thoughts_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> impl Iterator<Item = Thought<'a>> + 'a {
        self.thoughts_and_speech_custom(start_tags, end_tags)
            .filter_map(ThoughtOrSpeech::into_thought)
    }

    /// Return an iterator of [`Speech`]es with default start and end tags.
    fn speech(&'a self) -> impl Iterator<Item = Speech<'a>> + 'a {
        self.thoughts_and_speech()
            .filter_map(ThoughtOrSpeech::get_speech)
    }

    /// Return an iterator of [`Speech`]es with custom start and end tags.
    fn speech_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> impl Iterator<Item = Speech<'a>> + 'a {
        self.thoughts_and_speech_custom(start_tags, end_tags)
            .filter_map(ThoughtOrSpeech::get_speech)
    }
}

impl<'a> Thinkable<'a> for Block<'a> {
    /// Get an iterator over the thoughts and speech in the block with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech<'a>> + 'a> {
        Box::new(match self {
            Block::Text { text, .. } => {
                ThoughtsAndSpeech::new_custom(text, start_tags, end_tags)
            }
            _ => ThoughtsAndSpeech::new_custom("", start_tags, end_tags),
        })
    }
}

impl<'a> Thinkable<'a> for Content<'a> {
    /// Get an iterator over the thoughts and speech in the content with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech<'a>> + 'a> {
        match self {
            Content::SinglePart(cow_str) => Box::new(Box::new(
                ThoughtsAndSpeech::new_custom(cow_str, start_tags, end_tags),
            )),
            Content::MultiPart(blocks) => {
                Box::new(blocks.iter().flat_map(move |block| {
                    block.thoughts_and_speech_custom(start_tags, end_tags)
                }))
            }
        }
    }
}

impl<'a> Thinkable<'a> for Message<'a> {
    /// Get an iterator over the thoughts and speech in the message with custom
    /// start and end tags.
    ///
    /// # Panics
    /// - If the length of `start_tags` and `end_tags` are not equal.
    fn thoughts_and_speech_custom(
        &'a self,
        start_tags: &'static [&'static str],
        end_tags: &'static [&'static str],
    ) -> Box<dyn Iterator<Item = ThoughtOrSpeech<'a>> + 'a> {
        self.content
            .thoughts_and_speech_custom(start_tags, end_tags)
    }
}

#[cfg(test)]
mod tests {
    use crate::prompt::{self, message::Role};

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
                #[cfg(feature = "prompt-caching")]
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
}
