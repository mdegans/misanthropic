//! Citation types returned in response [`Text`] blocks when documents
//! have `citations.enabled = true`.
//!
//! [`Text`]: super::message::Block::Text

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

/// A citation referencing a location in a source document.
#[derive(Clone, Debug, Serialize, Deserialize, Hash, PartialEq)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum Citation {
    /// Citation from a plain text document (character offsets, 0-indexed,
    /// exclusive end).
    CharLocation {
        /// The exact text being cited.
        cited_text: Cow<'static, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document. `None` serializes as `null` rather
        /// than being omitted: the API requires this field to be *present*
        /// (`string | null`) on citations sent in a request, so dropping it
        /// breaks multi-turn conversations that echo a prior cited response.
        #[serde(default)]
        document_title: Option<Cow<'static, str>>,
        /// Start character index (0-indexed, inclusive).
        start_char_index: u64,
        /// End character index (0-indexed, exclusive).
        end_char_index: u64,
    },
    /// Citation from a PDF document (page numbers, 1-indexed,
    /// exclusive end).
    PageLocation {
        /// The exact text being cited.
        cited_text: Cow<'static, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document. `None` serializes as `null` rather
        /// than being omitted: the API requires this field to be *present*
        /// (`string | null`) on citations sent in a request, so dropping it
        /// breaks multi-turn conversations that echo a prior cited response.
        #[serde(default)]
        document_title: Option<Cow<'static, str>>,
        /// Start page number (1-indexed, inclusive).
        start_page_number: u64,
        /// End page number (1-indexed, exclusive).
        end_page_number: u64,
    },
    /// Citation from a custom content document (block indices,
    /// 0-indexed, exclusive end).
    ContentBlockLocation {
        /// The exact text being cited.
        cited_text: Cow<'static, str>,
        /// 0-indexed document position in the request.
        document_index: u64,
        /// Title of the cited document. `None` serializes as `null` rather
        /// than being omitted: the API requires this field to be *present*
        /// (`string | null`) on citations sent in a request, so dropping it
        /// breaks multi-turn conversations that echo a prior cited response.
        #[serde(default)]
        document_title: Option<Cow<'static, str>>,
        /// Start block index (0-indexed, inclusive).
        start_block_index: u64,
        /// End block index (0-indexed, exclusive).
        end_block_index: u64,
    },
    /// Citation from a web search result.
    WebSearchResultLocation {
        /// The exact text being cited.
        cited_text: Cow<'static, str>,
        /// Title of the search result.
        title: Cow<'static, str>,
        /// URL of the search result.
        url: Cow<'static, str>,
        /// Encrypted index for the search result.
        encrypted_index: Cow<'static, str>,
    },
}

impl Citation {}

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
    fn construct_char_location() {
        let citation = Citation::CharLocation {
            cited_text: "hello".into(),
            document_index: 0,
            document_title: Some("doc".into()),
            start_char_index: 0,
            end_char_index: 5,
        };
        let _: Citation = citation;
    }

    #[test]
    fn absent_title_serializes_as_null() {
        // The API requires `document_title` to be present on request-side
        // citations (it's `string | null`), so a `None` must serialize as
        // `null`, not be omitted — otherwise echoing a prior cited response
        // back in a multi-turn conversation 400s with "Field required".
        let citation = Citation::CharLocation {
            cited_text: "text".into(),
            document_index: 0,
            document_title: None,
            start_char_index: 0,
            end_char_index: 4,
        };
        let value = serde_json::to_value(&citation).unwrap();
        assert!(value.get("document_title").is_some());
        assert!(value["document_title"].is_null());

        // And it round-trips back to `None`.
        let back: Citation = serde_json::from_value(value).unwrap();
        assert!(matches!(
            back,
            Citation::CharLocation {
                document_title: None,
                ..
            }
        ));
    }

    /// End-to-end citations check against the live API.
    ///
    /// The document states counterfactual "facts" (a purple sky, orange
    /// ground) so the *only* way for the model to answer correctly is to read
    /// and cite the document — a grass-is-green example could be answered from
    /// training data, telling us nothing about whether citations actually
    /// round-tripped.
    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn live_text_document_returns_citation() {
        use crate::{
            Client, Prompt,
            prompt::message::{Block, DocumentSource, Role},
        };

        const DOC: &str = "The sky on planet Zorblax is purple. \
                           The ground on planet Zorblax is orange.";

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let prompt = Prompt::default()
            .add_message((
                Role::User,
                vec![
                    Block::document_with_citations(DocumentSource::from_text(
                        DOC,
                    )),
                    Block::text(
                        "What color is the sky on planet Zorblax? \
                         Answer in one short sentence.",
                    ),
                ],
            ))
            .unwrap();

        let message = client.message(prompt).await.unwrap();

        // The answer must be grounded in the document, not world knowledge.
        assert!(
            message.to_string().to_lowercase().contains("purple"),
            "expected 'purple' in response: {message}"
        );

        // At least one response text block should carry a `CharLocation`
        // citation quoting the purple sentence from our plain-text document.
        let cited = message.inner.content.iter().any(|block| {
            matches!(
                block,
                Block::Text {
                    citations: Some(cs),
                    ..
                } if cs.iter().any(|c| matches!(
                    c,
                    Citation::CharLocation { cited_text, .. }
                        if cited_text.to_lowercase().contains("purple")
                ))
            )
        });
        assert!(
            cited,
            "expected a CharLocation citation quoting the document: \
             {message:#?}"
        );
    }

    /// Same as [`live_text_document_returns_citation`] but for the PDF path: a
    /// base64-encoded [`DocumentSource`] should produce a [`PageLocation`]
    /// citation. The fixture (`test/data/zorblax.pdf`) carries the same
    /// counterfactual "facts" so a correct answer can only come from the doc.
    ///
    /// [`PageLocation`]: Citation::PageLocation
    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn live_pdf_document_returns_page_citation() {
        use crate::{
            Client, Prompt,
            prompt::message::{Block, DocumentSource, Role},
        };

        const PDF: &str =
            concat!(env!("CARGO_MANIFEST_DIR"), "/test/data/zorblax.pdf");

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let source = DocumentSource::from_file(PDF).unwrap();
        let prompt = Prompt::default()
            .add_message((
                Role::User,
                vec![
                    Block::document_with_citations(source),
                    Block::text(
                        "What are the two moons of planet Zorblax named? \
                         Answer in one short sentence.",
                    ),
                ],
            ))
            .unwrap();

        let message = client.message(prompt).await.unwrap();

        // Grounded in the PDF, not world knowledge.
        let answer = message.to_string().to_lowercase();
        assert!(
            answer.contains("pim") && answer.contains("wassel"),
            "expected the moon names in response: {message}"
        );

        // A PDF cites by page, so expect a `PageLocation`.
        let cited = message.inner.content.iter().any(|block| {
            matches!(
                block,
                Block::Text {
                    citations: Some(cs),
                    ..
                } if cs.iter().any(|c| matches!(
                    c,
                    Citation::PageLocation { cited_text, .. }
                        if cited_text.to_lowercase().contains("pim")
                ))
            )
        });
        assert!(
            cited,
            "expected a PageLocation citation quoting the PDF: {message:#?}"
        );
    }

    /// Regression for a multi-turn round-trip: echoing a prior assistant turn
    /// that carries an untitled citation back into the next request must not
    /// drop `document_title`. Previously this 400'd with
    /// `citations.0.page_location.document_title: Field required`.
    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn live_multi_turn_echoes_untitled_citation() {
        use crate::{
            Client, Prompt,
            prompt::message::{Block, DocumentSource, Message, Role},
        };

        const PDF: &str =
            concat!(env!("CARGO_MANIFEST_DIR"), "/test/data/zorblax.pdf");

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let source = DocumentSource::from_file(PDF).unwrap();
        let prompt = Prompt::default()
            .add_message((
                Role::User,
                vec![
                    Block::document_with_citations(source),
                    Block::text("What color is the sky on planet Zorblax?"),
                ],
            ))
            .unwrap();

        // First turn: the model cites the PDF. Our fixture has no title, so
        // the returned citation's `document_title` is `None` — the exact shape
        // that broke the round-trip.
        let first = client.message(&prompt).await.unwrap();
        let assistant: Message = Message::from(first);

        // Second turn: echo the cited assistant turn back, then follow up.
        let prompt = prompt
            .add_message(assistant)
            .unwrap()
            .add_message((Role::User, "And what are its two moons named?"))
            .unwrap();

        let second = client.message(&prompt).await.unwrap();
        let answer = second.to_string().to_lowercase();
        assert!(
            answer.contains("pim") && answer.contains("wassel"),
            "expected the moon names in the follow-up answer: {second}"
        );
    }
}
