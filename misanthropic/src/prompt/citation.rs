//! Citation types returned in response [`Text`] blocks when documents
//! have `citations.enabled = true`.
//!
//! [`Text`]: super::message::Block::Text

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A citation referencing a location in a source document.
#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Citation<'a> {
    /// Citation from a plain text document (character offsets, 0-indexed,
    /// exclusive end).
    CharLocation {
        /// The exact text being cited.
        cited_text: Cow<'a, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document, if provided.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        document_title: Option<Cow<'a, str>>,
        /// Start character index (0-indexed, inclusive).
        start_char_index: u64,
        /// End character index (0-indexed, exclusive).
        end_char_index: u64,
    },
    /// Citation from a PDF document (page numbers, 1-indexed,
    /// exclusive end).
    PageLocation {
        /// The exact text being cited.
        cited_text: Cow<'a, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document, if provided.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        document_title: Option<Cow<'a, str>>,
        /// Start page number (1-indexed, inclusive).
        start_page_number: u64,
        /// End page number (1-indexed, exclusive).
        end_page_number: u64,
    },
    /// Citation from a custom content document (block indices,
    /// 0-indexed, exclusive end).
    ContentBlockLocation {
        /// The exact text being cited.
        cited_text: Cow<'a, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document, if provided.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        document_title: Option<Cow<'a, str>>,
        /// Start block index (0-indexed, inclusive).
        start_block_index: u64,
        /// End block index (0-indexed, exclusive).
        end_block_index: u64,
    },
    /// Citation from a web search result.
    WebSearchResultLocation {
        /// The exact text being cited.
        cited_text: Cow<'a, str>,
        /// Title of the search result.
        title: Cow<'a, str>,
        /// URL of the search result.
        url: Cow<'a, str>,
        /// Encrypted index for the search result.
        encrypted_index: Cow<'a, str>,
    },
}

impl Citation<'_> {
    /// Convert to a `'static` lifetime by taking ownership of all
    /// [`Cow`] fields.
    pub fn into_static(self) -> Citation<'static> {
        match self {
            Citation::CharLocation {
                cited_text,
                document_index,
                document_title,
                start_char_index,
                end_char_index,
            } => Citation::CharLocation {
                cited_text: Cow::Owned(cited_text.into_owned()),
                document_index,
                document_title: document_title
                    .map(|s| Cow::Owned(s.into_owned())),
                start_char_index,
                end_char_index,
            },
            Citation::PageLocation {
                cited_text,
                document_index,
                document_title,
                start_page_number,
                end_page_number,
            } => Citation::PageLocation {
                cited_text: Cow::Owned(cited_text.into_owned()),
                document_index,
                document_title: document_title
                    .map(|s| Cow::Owned(s.into_owned())),
                start_page_number,
                end_page_number,
            },
            Citation::ContentBlockLocation {
                cited_text,
                document_index,
                document_title,
                start_block_index,
                end_block_index,
            } => Citation::ContentBlockLocation {
                cited_text: Cow::Owned(cited_text.into_owned()),
                document_index,
                document_title: document_title
                    .map(|s| Cow::Owned(s.into_owned())),
                start_block_index,
                end_block_index,
            },
            Citation::WebSearchResultLocation {
                cited_text,
                title,
                url,
                encrypted_index,
            } => Citation::WebSearchResultLocation {
                cited_text: Cow::Owned(cited_text.into_owned()),
                title: Cow::Owned(title.into_owned()),
                url: Cow::Owned(url.into_owned()),
                encrypted_index: Cow::Owned(encrypted_index.into_owned()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serde_char_location() {
        let json = r#"{
            "type": "char_location",
            "cited_text": "The grass is green.",
            "document_index": 0,
            "document_title": "My Document",
            "start_char_index": 0,
            "end_char_index": 20
        }"#;

        let citation: Citation = serde_json::from_str(json).unwrap();
        assert!(matches!(
            citation,
            Citation::CharLocation {
                start_char_index: 0,
                end_char_index: 20,
                ..
            }
        ));

        // Round-trip.
        let serialized = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(citation, deserialized);
    }

    #[test]
    fn serde_page_location() {
        let json = r#"{
            "type": "page_location",
            "cited_text": "Water is essential for life.",
            "document_index": 1,
            "document_title": "PDF Document",
            "start_page_number": 5,
            "end_page_number": 6
        }"#;

        let citation: Citation = serde_json::from_str(json).unwrap();
        assert!(matches!(
            citation,
            Citation::PageLocation {
                start_page_number: 5,
                end_page_number: 6,
                ..
            }
        ));

        let serialized = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(citation, deserialized);
    }

    #[test]
    fn serde_content_block_location() {
        let json = r#"{
            "type": "content_block_location",
            "cited_text": "These are important findings.",
            "document_index": 2,
            "document_title": "Custom Content Document",
            "start_block_index": 0,
            "end_block_index": 1
        }"#;

        let citation: Citation = serde_json::from_str(json).unwrap();
        assert!(matches!(
            citation,
            Citation::ContentBlockLocation {
                start_block_index: 0,
                end_block_index: 1,
                ..
            }
        ));

        let serialized = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(citation, deserialized);
    }

    #[test]
    fn serde_web_search_result_location() {
        let json = r#"{
            "type": "web_search_result_location",
            "cited_text": "Some web content.",
            "title": "Web Page",
            "url": "https://example.com",
            "encrypted_index": "abc123"
        }"#;

        let citation: Citation = serde_json::from_str(json).unwrap();
        assert!(matches!(citation, Citation::WebSearchResultLocation { .. }));

        let serialized = serde_json::to_string(&citation).unwrap();
        let deserialized: Citation = serde_json::from_str(&serialized).unwrap();
        assert_eq!(citation, deserialized);
    }

    #[test]
    fn into_static() {
        let citation = Citation::CharLocation {
            cited_text: "hello".into(),
            document_index: 0,
            document_title: Some("doc".into()),
            start_char_index: 0,
            end_char_index: 5,
        };
        let _: Citation<'static> = citation.into_static();
    }

    #[test]
    fn no_title_omitted() {
        let citation = Citation::CharLocation {
            cited_text: "text".into(),
            document_index: 0,
            document_title: None,
            start_char_index: 0,
            end_char_index: 4,
        };
        let json = serde_json::to_string(&citation).unwrap();
        assert!(!json.contains("document_title"));
    }
}
