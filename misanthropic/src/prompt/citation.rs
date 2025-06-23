use crate::tool::memory_palace::{MemoryId, PromptId, RoomId};
use serde::{Deserialize, Serialize};
use url::Url;

#[derive(Clone, Debug, Deserialize, Serialize, Hash)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub enum Citation {
    /// Character level cite
    #[serde(rename = "char_location")]
    Char {
        /// Cited text
        #[serde(rename = "cited_text")]
        text: String,
        /// Document index
        document_index: usize,
        /// Document title
        #[serde(
            rename = "document_title",
            skip_serializing_if = "Option::is_none"
        )]
        title: Option<String>,
        /// Start character index
        #[serde(rename = "start_char_index")]
        start_char: usize,
        /// End character index
        #[serde(rename = "end_char_index")]
        end_char: usize,
    },
    /// Page cite
    #[serde(rename = "page_location")]
    Page {
        /// Cited text
        #[serde(rename = "cited_text")]
        text: String,
        /// Document index
        document_index: usize,
        /// Document title
        #[serde(
            rename = "document_title",
            skip_serializing_if = "Option::is_none"
        )]
        title: Option<String>,
        /// Start page number
        #[serde(rename = "start_page_number")]
        start_page: usize,
        /// End page number
        #[serde(rename = "end_page_number")]
        end_page: usize,
    },
    /// [`Content`] [`Block`] cite
    ///
    /// [`Content`]: crate::prompt::message::Content
    /// [`Block`]: crate::prompt::message::Block
    #[serde(rename = "content_block_location")]
    ContentBlock {
        /// Cited text
        #[serde(rename = "cited_text")]
        text: String,
        /// Document index
        document_index: usize,
        /// Document title
        #[serde(
            rename = "document_title",
            skip_serializing_if = "Option::is_none"
        )]
        document_title: Option<String>,
        /// Start block index
        start_block_index: usize,
        /// End block index
        end_block_index: usize,
    },
    /// Web Citation
    #[serde(rename = "web_search_result_location")] // Why?
    Web {
        /// Cited text
        #[serde(rename = "cited_text")]
        text: String,
        /// Encrypted index
        encrypted_index: String,
        /// Optional document title
        // The lack of rename isn't a mistake. The API is just inconsistent.
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Url
        url: Url,
    },
}
