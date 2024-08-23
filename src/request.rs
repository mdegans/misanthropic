//! [Anthropic Messages API] [`Request`] types.
//!
//! [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>

use std::num::NonZeroU16;

use crate::{tool, Model, Tool};
use serde::{Deserialize, Serialize};

pub mod message;
pub use message::Message;

/// Request for the [Anthropic Messages API].
///
/// [Anthropic Messages API]: <https://docs.anthropic.com/en/api/messages>
#[derive(Serialize, Deserialize)]
pub struct Request {
    /// [`Model`] to use for inference.
    pub model: Model,
    /// Input [`request::Message`]s. If this ends with an [`Assistant`]
    /// [`Message`], the completion will be constrained by that last message.
    /// Otherwise a new [`Assistant`] [`Message`] will be generated.
    ///
    /// See [Anthropic docs] for more information.
    ///
    /// [`Assistant`]: crate::request::message::Role::Assistant
    /// [`request::Message`]: crate::request::Message
    /// [Anthropic docs]: <https://docs.anthropic.com/en/api/messages>
    pub messages: Vec<Message>,
    /// Max tokens to generate. See Anthropic [docs] for the maximum number of
    /// tokens for each model.
    ///
    /// [docs]: <https://docs.anthropic.com/en/docs/about-claude/models>
    pub max_tokens: NonZeroU16,
    /// Optional info about the request, for example, `user_id` to help
    /// Anthropic detect and prevent abuse. Do not use PII here (email, phone).
    /// Use the [`json!`] macro to create this easily.
    ///
    /// [`json!`]: serde_json::json
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
    /// Optional stop sequences. If the model generates any of these sequences,
    /// the completion will stop with [`StopReason::StopSequence`].
    ///
    /// [`StopReason::StopSequence`]: crate::response::StopReason::StopSequence
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    /// If `true`, the response will be a stream of [`Event`]s. If `false`, the
    /// response will be a single [`response::Message`].
    ///
    /// [`Event`]: crate::stream::Event
    /// [`response::Message`]: crate::response::Message
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// System prompt as [`SinglePart`] or [`MultiPart`] [`Content`].
    ///
    /// [`SinglePart`]: message::Content::SinglePart
    /// [`MultiPart`]: message::Content::MultiPart
    /// [`Content`]: message::Content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<message::Content>,
    /// Temperature for sampling. Must be between 0 and 1. Higher values mean
    /// more randomness. Note that 0.0 is not fully deterministic.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// [`tool::Choice`] for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<tool::Choice>,
    /// Tool definitions for the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Top K tokens to consider for each token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<NonZeroU16>,
    /// Top P nucleus sampling. The probabilities of each token are added in
    /// order from most to least likely until the probability mass exceeds
    /// `top_p`. A token is then sampled from this reduced distribution.
    ///
    /// This is a float between 0 and 1 where higher values mean more
    /// randomness.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
}
