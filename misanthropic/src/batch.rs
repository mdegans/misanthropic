//! Batch [`Requests`] and [`Response`]s.
//!
//! Generic over the prompt type `P: Serialize` so callers can submit
//! [`Prompt`], [`CachedPrompt`], or any other serializable prompt wrapper
//! without conversion.
//!
//! [`Prompt`]: crate::Prompt
//! [`CachedPrompt`]: crate::CachedPrompt
use std::{collections::HashMap, num::NonZeroU16, str::FromStr};

use chrono::{DateTime, Utc};
use reqwest::Url;
use serde::{Deserialize, Serialize, ser::SerializeSeq};

use crate::{client, response};

/// An immutable map of [`Id`] to prompts. Part of a [`Batch`] returned by
/// [`Client::batch`]. Unordered. O(1) lookup by [`Id`].
///
/// The type parameter `P` is the prompt type — typically
/// [`Prompt`](crate::Prompt) or [`CachedPrompt`](crate::CachedPrompt).
///
/// # Note:
/// - The user cannot mutate this, but the API can. For example, [`Ready`] has
///   methods to remove prompts with successful, errored, canceled, or
///   expired results.
///
/// [`Client::batch`]: crate::Client::batch
/// [`Client`]: crate::Client
#[derive(derive_more::Deref)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Prompts<P> {
    pub(crate) prompts: HashMap<Id, P>,
}

impl<P: Clone> Clone for Prompts<P> {
    fn clone(&self) -> Self {
        Self {
            prompts: self.prompts.clone(),
        }
    }
}

impl<P> std::fmt::Debug for Prompts<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(concat!(stringify!(Prompts), " { ... }"))
    }
}

impl<P> Prompts<P> {
    /// Get a prompt by its [`Id`]. Use [`Prompts::keys`] to get all IDs or
    /// [`Prompts::values`] to get all prompts.
    ///
    /// # Note:
    /// - A [`uuid::Uuid`] will also work here.
    pub fn get_id<I>(&self, id: I) -> Option<&P>
    where
        I: Into<Id>,
    {
        self.prompts.get(&id.into())
    }

    /// Try to get a prompt by string ID. Returns an error if the string is
    /// not a valid UUID. Prefer using [`Prompts::get_id`] if possible.
    pub fn try_get<I>(&self, id: I) -> Result<Option<&P>, uuid::Error>
    where
        I: AsRef<str>,
    {
        let s = id.as_ref();
        Ok(self.get(&Id::from_str(s)?))
    }

    /// Convert into the inner [`HashMap`] of prompts. There is no going
    /// back from this. A [`Prompts`] may only be created by the [`Client`].
    ///
    /// [`Client`]: crate::Client
    pub fn into_inner(self) -> HashMap<Id, P> {
        self.prompts
    }
}

impl<P> FromIterator<P> for Prompts<P> {
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = P>,
    {
        Prompts {
            prompts: iter
                .into_iter()
                .map(|prompt| (Id::default(), prompt))
                .collect(),
        }
    }
}

impl<P, I> FromIterator<(I, P)> for Prompts<P>
where
    I: Into<Id>,
{
    fn from_iter<I2>(iter: I2) -> Self
    where
        I2: IntoIterator<Item = (I, P)>,
    {
        Prompts {
            prompts: iter
                .into_iter()
                .map(|(id, prompt)| (id.into(), prompt))
                .collect(),
        }
    }
}

impl<P> IntoIterator for Prompts<P> {
    type Item = (Id, P);
    type IntoIter = std::collections::hash_map::IntoIter<Id, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.prompts.into_iter()
    }
}

impl<'a, P> IntoIterator for &'a Prompts<P> {
    type Item = (&'a Id, &'a P);
    type IntoIter = std::collections::hash_map::Iter<'a, Id, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.prompts.iter()
    }
}

impl<P: Serialize> Serialize for Prompts<P> {
    /// Serialize the [`Prompts`] as a list of [`Request`]s, which is what the
    /// API expects. The data will not be [`Clone`]d.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Outer wrapper to put the { "requests": [ ... ] }
        #[derive(Serialize)]
        struct Outer<'r, P: Serialize> {
            requests: Inner<'r, P>,
        }

        // Inner struct to serialize the actual sequence.
        struct Inner<'r, P> {
            prompts: &'r Prompts<P>,
        }

        // We serialize Prompts as a sequence of Requests. No allocations.
        impl<P: Serialize> Serialize for Inner<'_, P> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: serde::Serializer,
            {
                let mut seq =
                    serializer.serialize_seq(Some(self.prompts.len()))?;
                for (id, prompt) in self.prompts.iter() {
                    seq.serialize_element(&Request { id, prompt })?;
                }
                seq.end()
            }
        }

        // Serialize the inner struct as the structure Anthropic API expects.
        let wrapper = Outer {
            requests: Inner { prompts: self },
        };

        wrapper.serialize(serializer)
    }
}

/// [`Id`] of a prompt in [`Prompts`]. Wrapper around a [`uuid::Uuid`].
#[derive(
    Clone,
    Copy,
    Debug,
    Serialize,
    Deserialize,
    Hash,
    Eq,
    PartialEq,
    PartialOrd,
    Ord,
    derive_more::Display,
    derive_more::Into,
    derive_more::From,
)]
#[serde(transparent)]
#[repr(transparent)] // might as well, in case someone needs this guarantee.
#[display("{uuid}")]
pub struct Id {
    uuid: uuid::Uuid,
}

impl Default for Id {
    fn default() -> Self {
        Id {
            // If this becomes a bottleneck, we can use a thread-local RNG, but
            // it's unlikely to be a problem. Batch creation is not a hot path.
            uuid: uuid::Uuid::new_v4(),
        }
    }
}

impl FromStr for Id {
    type Err = uuid::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Id {
            uuid: uuid::Uuid::parse_str(s)?,
        })
    }
}

/// An individual `Request` in a batch. A prompt with a custom [`Id`]
/// (UUID). The only way to create a [`Request`] is through the [`Client`].
///
/// [`Client`]: crate::Client
#[derive(Serialize)]
struct Request<'r, P: Serialize> {
    #[serde(rename = "custom_id")]
    id: &'r Id,
    #[serde(rename = "params")]
    prompt: &'r P,
}

/// An Anthropic `message_batch` response with [`Batch`] metadata.
#[derive(Clone, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type")]
#[serde(rename = "message_batch")]
pub struct Meta {
    /// Anthropic-assigned Response ID. Format may change.
    pub id: String,
    /// Anthropic `processing_status` of the batch.
    #[serde(rename = "processing_status")]
    pub status: Status,
    /// Statistics for the batch (`request_counts`).
    #[serde(rename = "request_counts")]
    pub stats: Stats,
    /// Time the batch was created.
    pub created_at: DateTime<Utc>,
    /// Time the batch expires.
    pub expires_at: DateTime<Utc>,
    /// Time the batch ended.
    pub ended_at: Option<DateTime<Utc>>,
    /// Time the batch was canceled.
    pub cancel_initiated_at: Option<DateTime<Utc>>,
    /// Time the batch was archived.
    pub archived_at: Option<DateTime<Utc>>,
    /// Results URL. Available after processing ends. A `.jsonl` file with
    /// responses.
    pub results_url: Option<Url>,
}

/// Anthropic `processing_status` for [`Prompts`]. Member of [`Meta`]data.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(rename_all = "snake_case")]
pub enum Status {
    /// Processing is in progress.
    InProgress,
    /// Batch is canceling.
    Canceling,
    /// Processing has ended.
    Ended,
}

/// Request statistics for a batch of [`Prompts`].
#[derive(Clone, Copy, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Stats {
    /// Number of processing requests.
    pub processing: u32,
    /// Number of sucessful requests.
    pub succeeded: u32,
    /// Number of failed requests.
    pub errored: u32,
    /// Number of canceled requests.
    pub canceled: u32,
    /// Number of expired requests.
    pub expired: u32,
}

/// A `Batch` of prompts that can be in one of two states: [`Pending`] or
/// [`Ready`]. Because Anthropic has no way to call back to the client, the user
/// must poll the API using [`Client::batch_poll`] until the Batch is [`Ready`].
///
/// [`Client::batch_poll`]: crate::Client::batch_poll
/// [`Batch::is_ready`].
// Anthropic should really add a webhook or something.
#[derive(derive_more::IsVariant)]
pub enum Batch<P> {
    /// Needs more [`Client::batch_poll`]ing.
    Pending(Pending<P>),
    /// Results are [`Ready`]. See [`Ready::get_result`] for getting individual
    /// prompt completions.
    ///
    /// # Note:
    /// - They are not guaranteed to be complete. Some requests might have been
    ///   canceled or expired. You will have to re-submit those. See [`Ready`]
    ///   for methods to remove them for resubmission.
    Ready(Ready<P>),
}

// Manual Serialize: only Pending needs to be serializable (for polling).
// Ready contains results which we don't need to re-serialize.
impl<P: Serialize> Serialize for Batch<P> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Batch::Pending(p) => p.serialize(serializer),
            Batch::Ready(r) => r.serialize(serializer),
        }
    }
}

/// A pending [`Batch`] containing prompts and [`Meta`]data with
/// processing details. Can only be created by [`Client::batch`]. Can only be
/// mutated by polling the API with [`Client::batch_poll`].
///
/// [`Client::batch`]: crate::Client::batch
/// [`Client::batch_poll`]: crate::Client::batch_poll
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
pub struct Pending<P> {
    pub(crate) prompts: Prompts<P>,
    pub(crate) meta: Meta,
}

impl<P: Clone> Clone for Pending<P> {
    fn clone(&self) -> Self {
        Self {
            prompts: self.prompts.clone(),
            meta: self.meta.clone(),
        }
    }
}

// Manual Serialize: only needed when P: Serialize (for submitting).
impl<P: Serialize> Serialize for Pending<P> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize just the prompts (the API submission format).
        // Meta is handled separately by the client.
        self.prompts.serialize(serializer)
    }
}

impl<P> Pending<P> {
    /// Returns the `message_batch` [`Meta`]data of the batch.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Returns the [`Status`] of the batch, last polled.
    pub fn status(&self) -> Status {
        self.meta.status
    }

    /// Returns true if the batch is done processing.
    pub fn is_done(&self) -> bool {
        matches!(self.status(), Status::Ended)
    }

    /// Returns the [`Url`] where the results of the batch can be found if the
    /// processing status is `Ended`. This points to a `.jsonl` file.
    pub fn results_url(&self) -> Option<&Url> {
        self.meta.results_url.as_ref()
    }

    /// Get the [`Prompts`] from the response. This is a map-like interface.
    /// The [`Prompts`] are immutable.
    pub fn prompts(&self) -> &Prompts<P> {
        &self.prompts
    }

    /// Decompose the batch into its parts.
    pub fn decompose(self) -> (Prompts<P>, Meta) {
        (self.prompts, self.meta)
    }
}

/// A completed batch of prompts with [`BatchResult`]s.
pub struct Ready<P> {
    /// The prompts that were processed. It is guaranteed that every [`Id`] in
    /// `results` is in `pending.prompts`.
    pub(crate) pending: Pending<P>,
    /// The results of processing the prompts. It is guaranteed that every
    /// [`Id`] in `results` is in `pending.prompts`.
    pub(crate) results: HashMap<Id, BatchResult>,
}

// Manual Serialize for Ready (P may or may not be Serialize at this point).
impl<P: Serialize> Serialize for Ready<P> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut s = serializer.serialize_struct("Ready", 2)?;
        s.serialize_field("pending", &self.pending)?;
        s.serialize_field("results", &self.results)?;
        s.end()
    }
}

impl<P> Ready<P> {
    /// Get the result for a specific prompt by its [`Id`].
    pub fn get_result(&self, id: Id) -> Option<&BatchResult> {
        self.results.get(&id)
    }

    /// Decompose the batch into its parts. This is a one-way operation.
    pub fn decompose(self) -> (Pending<P>, HashMap<Id, BatchResult>) {
        (self.pending, self.results)
    }

    /// Iterate over successful prompts and their [`response::Message`]s.
    pub fn iter_ok(&self) -> impl Iterator<Item = (&P, &response::Message)> {
        self.results.iter().filter_map(|(id, result)| {
            if let BatchResult::Ok(msg) = result {
                Some((self.pending.prompts.get(id)?, msg))
            } else {
                None
            }
        })
    }

    /// Remove all prompts and [`response::Message`]s from the batch with
    /// successful results. This is a one-way operation.
    pub fn remove_ok(&mut self) -> Vec<(P, response::Message)> {
        let ids: Vec<_> = self
            .results
            .iter()
            .filter_map(
                |(id, result)| {
                    if result.is_ok() { Some(*id) } else { None }
                },
            )
            .collect();

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let succeeded =
                match self.pending.meta.stats.succeeded.checked_sub(1) {
                    Some(s) => s,
                    None => {
                        #[cfg(feature = "log")]
                        log::error!(
                            "meta.stats.succeeded underflowed for batch: {}",
                            self.pending.meta.id
                        );
                        0
                    }
                };
            self.pending.meta.stats.succeeded = succeeded;

            let prompt =
                self.pending.prompts.prompts.remove(&id).expect(
                    "Class invariant violated: Ready prompts missing ID",
                );
            let result = self.results.remove(&id);

            if let Some(BatchResult::Ok(msg)) = result {
                results.push((prompt, msg));
            } else {
                panic!("Code above does not check result.is_ok()");
            }
        }

        results
    }

    /// Iterate over errored prompts and their [`client::AnthropicError`]s.
    pub fn iter_errors(
        &self,
    ) -> impl Iterator<Item = (&P, &client::AnthropicError)> {
        self.results.iter().filter_map(|(id, result)| {
            if let BatchResult::Error(e) = result {
                Some((&self.pending.prompts[id], e))
            } else {
                None
            }
        })
    }

    /// Remove all prompts and [`response::Message`]s from the batch with
    /// errors. This is a one-way operation.
    pub fn remove_errors(&mut self) -> Vec<(P, client::AnthropicError)> {
        let ids: Vec<_> = self
            .results
            .iter()
            .filter_map(
                |(id, result)| {
                    if result.is_error() { Some(*id) } else { None }
                },
            )
            .collect();

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let errored = match self.pending.meta.stats.errored.checked_sub(1) {
                Some(s) => s,
                None => {
                    #[cfg(feature = "log")]
                    log::error!(
                        "meta.stats.errored underflowed for batch: {}",
                        self.pending.meta.id
                    );
                    0
                }
            };
            self.pending.meta.stats.errored = errored;

            let prompt =
                self.pending.prompts.prompts.remove(&id).expect(
                    "Class invariant violated: Ready prompts missing ID",
                );
            let result = self.results.remove(&id);

            if let Some(BatchResult::Error(e)) = result {
                results.push((prompt, e));
            } else {
                panic!("Code above does not check result.is_error()");
            }
        }
        results
    }

    /// Iterate over canceled prompts.
    pub fn iter_canceled(&self) -> impl Iterator<Item = &P> {
        self.results.iter().filter_map(|(id, result)| {
            if result.is_canceled() {
                Some(&self.pending.prompts[id])
            } else {
                None
            }
        })
    }

    /// Remove all prompts from the batch that were canceled.
    pub fn remove_canceled(&mut self) -> Vec<P> {
        let ids: Vec<_> = self
            .results
            .iter()
            .filter_map(|(id, result)| {
                if result.is_canceled() {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let canceled = match self.pending.meta.stats.canceled.checked_sub(1)
            {
                Some(s) => s,
                None => {
                    #[cfg(feature = "log")]
                    log::error!(
                        "meta.stats.canceled underflowed for batch: {}",
                        self.pending.meta.id
                    );
                    0
                }
            };
            self.pending.meta.stats.canceled = canceled;

            let prompt =
                self.pending.prompts.prompts.remove(&id).expect(
                    "Class invariant violated: Ready prompts missing ID",
                );
            self.results.remove(&id);

            results.push(prompt);
        }
        results
    }

    /// Iterate over expired prompts.
    pub fn iter_expired(&self) -> impl Iterator<Item = &P> {
        self.results.iter().filter_map(|(id, result)| {
            if result.is_expired() {
                Some(&self.pending.prompts[id])
            } else {
                None
            }
        })
    }

    /// Remove all prompts from the batch that expired.
    pub fn remove_expired(&mut self) -> Vec<P> {
        let ids: Vec<_> = self
            .results
            .iter()
            .filter_map(
                |(id, result)| {
                    if result.is_expired() { Some(*id) } else { None }
                },
            )
            .collect();

        let mut results = Vec::with_capacity(ids.len());
        for id in ids {
            let expired = match self.pending.meta.stats.expired.checked_sub(1) {
                Some(s) => s,
                None => {
                    #[cfg(feature = "log")]
                    log::error!(
                        "meta.stats.expired underflowed for batch: {}",
                        self.pending.meta.id
                    );
                    0
                }
            };
            self.pending.meta.stats.expired = expired;

            let prompt =
                self.pending.prompts.prompts.remove(&id).expect(
                    "Class invariant violated: Ready prompts missing ID",
                );
            self.results.remove(&id);

            results.push(prompt);
        }
        results
    }

    /// Iterate over all prompts and their [`BatchResult`]s by reference.
    ///
    /// To consume the batch into owned `(Id, P, BatchResult)` triples, use the
    /// [`IntoIterator`] impl directly (`for … in ready`, `.into_iter()`, or
    /// `.collect()`).
    pub fn iter(&self) -> Iter<'_, P> {
        Iter {
            prompts: &self.pending.prompts,
            results: self.results.iter(),
        }
    }
}

/// Owning iterator over a [`Ready`] batch, yielding `(Id, prompt, BatchResult)`
/// triples. Created by [`Ready`]'s [`IntoIterator`] impl.
pub struct IntoIter<P> {
    /// Drained as `results` is consumed; every `results` [`Id`] is present here.
    prompts: HashMap<Id, P>,
    results: std::collections::hash_map::IntoIter<Id, BatchResult>,
}

impl<P> Iterator for IntoIter<P> {
    type Item = (Id, P, BatchResult);

    fn next(&mut self) -> Option<Self::Item> {
        let (id, result) = self.results.next()?;
        // Class invariant: every `results` Id is in `prompts`.
        let prompt = self.prompts.remove(&id).unwrap();
        Some((id, prompt, result))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.results.size_hint()
    }
}

impl<P> ExactSizeIterator for IntoIter<P> {}

impl<P> IntoIterator for Ready<P> {
    type Item = (Id, P, BatchResult);
    type IntoIter = IntoIter<P>;

    fn into_iter(self) -> Self::IntoIter {
        let (pending, results) = self.decompose();
        IntoIter {
            prompts: pending.prompts.into_inner(),
            results: results.into_iter(),
        }
    }
}

/// Borrowing iterator over a [`Ready`] batch, yielding `(Id, &prompt,
/// &BatchResult)` triples. Created by [`Ready::iter`] or `&Ready`'s
/// [`IntoIterator`].
pub struct Iter<'a, P> {
    prompts: &'a Prompts<P>,
    results: std::collections::hash_map::Iter<'a, Id, BatchResult>,
}

impl<'a, P> Iterator for Iter<'a, P> {
    type Item = (Id, &'a P, &'a BatchResult);

    fn next(&mut self) -> Option<Self::Item> {
        let (id, result) = self.results.next()?;
        Some((*id, &self.prompts[id], result))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.results.size_hint()
    }
}

impl<P> ExactSizeIterator for Iter<'_, P> {}

impl<'a, P> IntoIterator for &'a Ready<P> {
    type Item = (Id, &'a P, &'a BatchResult);
    type IntoIter = Iter<'a, P>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Helper for deserializing batch results from the API's JSONL format.
///
/// Uses `'static` lifetime because batch results are deserialized from
/// owned `serde_json::Value` data (the JSONL response is downloaded as a
/// String and parsed via Value intermediate to avoid borrow issues).
#[derive(Deserialize)]
pub(crate) struct IdentifiedBatchResult {
    #[serde(rename = "custom_id")]
    pub(crate) id: Id,
    pub(crate) result: BatchResult,
}

/// A [`BatchResult`] is the result of processing a prompt in a batch.
///
/// The API returns different content keys per variant:
/// - `succeeded` → `{ "type": "succeeded", "message": { ... } }`
/// - `errored` → `{ "type": "errored", "error": { ... } }`
/// - `canceled` / `expired` → `{ "type": "canceled" }` (no content)
///
/// Response data is always owned (`'static`) since it is deserialized from
/// the batch results JSONL download.
#[derive(Serialize, derive_more::IsVariant)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum BatchResult {
    /// The batch was canceled and this prompt was not processed.
    Canceled,
    /// The batch expired and this prompt was not processed.
    Expired,
    /// Sucessful response to a prompt.
    #[serde(rename = "succeeded")]
    Ok(response::Message),
    /// Error response to a prompt.
    #[serde(rename = "errored")]
    Error(client::AnthropicError),
}

impl<'de> Deserialize<'de> for BatchResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Use Value as intermediate to avoid lifetime issues with borrowed data
        let value = serde_json::Value::deserialize(deserializer)?;
        let type_str = value
            .get("type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| serde::de::Error::missing_field("type"))?;

        match type_str {
            "succeeded" => {
                let message = value.get("message").ok_or_else(|| {
                    serde::de::Error::missing_field("message")
                })?;
                let msg: response::Message =
                    serde_json::from_value(message.clone())
                        .map_err(serde::de::Error::custom)?;
                Ok(BatchResult::Ok(msg))
            }
            "errored" => {
                let error = value
                    .get("error")
                    .ok_or_else(|| serde::de::Error::missing_field("error"))?;
                let err: client::AnthropicError =
                    serde_json::from_value(error.clone())
                        .map_err(serde::de::Error::custom)?;
                Ok(BatchResult::Error(err))
            }
            "canceled" => Ok(BatchResult::Canceled),
            "expired" => Ok(BatchResult::Expired),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["succeeded", "errored", "canceled", "expired"],
            )),
        }
    }
}

impl From<BatchResult> for Result<response::Message, client::AnthropicError> {
    fn from(result: BatchResult) -> Self {
        match result {
            BatchResult::Ok(msg) => Ok(msg),
            BatchResult::Error(e) => Err(e),
            BatchResult::Canceled => Err(client::AnthropicError::Unknown {
                code: Some(NonZeroU16::new(204).unwrap()),
                message: "Batch result was cancelled.".into(),
            }),
            BatchResult::Expired => Err(client::AnthropicError::Unknown {
                code: Some(NonZeroU16::new(408).unwrap()),
                message: "Batch result expired".into(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use uuid::Uuid;

    use crate::{
        AnthropicModel, Prompt, model, prompt::message::Content,
        response::Usage,
    };

    use super::*;
    #[test]
    fn test_meta_serde() {
        const JSON: &str = r#"{
  "id": "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {
    "processing": 100,
    "succeeded": 50,
    "errored": 30,
    "canceled": 10,
    "expired": 10
  },
  "ended_at": "2024-08-20T18:37:24.100435Z",
  "created_at": "2024-08-20T18:37:24.100435Z",
  "expires_at": "2024-08-20T18:37:24.100435Z",
  "archived_at": "2024-08-20T18:37:24.100435Z",
  "cancel_initiated_at": "2024-08-20T18:37:24.100435Z",
  "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_013Zva2CMHLNnXjNJJKqJ2EF/results"
}"#;

        let meta: Meta = serde_json::from_str(JSON).unwrap();
        let json = serde_json::to_string(&meta).unwrap();
        let meta2: Meta = serde_json::from_str(&json).unwrap();
        assert!(meta == meta2);
    }

    #[test]
    fn test_prompts_serialize() {
        let prompts: Prompts<Prompt> = Prompts {
            prompts: HashMap::new(),
        };
        let json = serde_json::to_string(&prompts).unwrap();
        assert_eq!(json, r#"{"requests":[]}"#);

        let id: Id = Uuid::nil().into();
        let prompts: Prompts<Prompt> = Prompts {
            prompts: [(id, Prompt::default())].into_iter().collect(),
        };

        let json = serde_json::to_string_pretty(&prompts).unwrap();
        assert_eq!(
            json,
            r#"{
  "requests": [
    {
      "custom_id": "00000000-0000-0000-0000-000000000000",
      "params": {
        "model": "claude-haiku-4-5",
        "messages": [],
        "max_tokens": 4096
      }
    }
  ]
}"#
        );
    }

    #[test]
    fn test_prompts_get_id() {
        let id: Id = Uuid::max().into();
        let prompts: Prompts<Prompt> = Prompts {
            prompts: [(id, Prompt::default())].into_iter().collect(),
        };

        assert_eq!(prompts.get_id(id), Some(&Prompt::default()));
    }

    #[test]
    fn test_prompts_try_get() {
        let id = Uuid::max();
        let prompts: Prompts<Prompt> = Prompts {
            prompts: [(id.into(), Prompt::default())].into_iter().collect(),
        };

        assert_eq!(
            prompts.try_get(id.to_string()),
            Ok(Some(&Prompt::default()))
        );
        assert!(prompts.try_get("not a UUID").is_err());
    }

    #[test]
    fn test_prompts_into_inner() {
        let id: Id = Uuid::max().into();
        let prompts: Prompts<Prompt> = Prompts {
            prompts: [(id, Prompt::default())].into_iter().collect(),
        };

        let inner = prompts.into_inner();
        assert_eq!(inner.len(), 1);
        assert_eq!(inner.get(&id), Some(&Prompt::default()));
    }

    #[test]
    fn test_prompt_from_iter() {
        let prompts: Prompts<Prompt> =
            [Prompt::default()].into_iter().collect();
        assert_eq!(prompts.len(), 1);
        let id = Id::default();
        let prompts: Prompts<Prompt> =
            [(id, Prompt::default())].into_iter().collect();
        assert_eq!(prompts.len(), 1);
        assert_eq!(prompts.get_id(id), Some(&Prompt::default()));
    }

    #[test]
    fn test_prompt_into_iter() {
        // value
        let id = Id::default();
        let prompts: Prompts<Prompt> =
            [(id, Prompt::default())].into_iter().collect();
        let mut iter = prompts.into_iter();
        let (id2, prompt) = iter.next().unwrap();
        assert_eq!(id, id2);
        assert_eq!(prompt, Prompt::default());
        assert!(iter.next().is_none());

        // reference
        let id = Id::default();
        let prompts: Prompts<Prompt> =
            [(id, Prompt::default())].into_iter().collect();
        let mut iter = (&prompts).into_iter();
        let (id2, prompt) = iter.next().unwrap();
        assert_eq!(id, *id2);
        assert_eq!(prompt, &Prompt::default());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_prompts_format() {
        let id = Id::default();
        let prompts: Prompts<Prompt> = Prompts {
            prompts: [(id, Prompt::default())].into_iter().collect(),
        };

        assert_eq!(format!("{:?}", prompts), "Prompts { ... }");
    }

    #[test]
    fn test_id_deserialize() {
        let id: Id = Uuid::max().into();
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""ffffffff-ffff-ffff-ffff-ffffffffffff""#);

        let id2: Id = serde_json::from_str(&json).unwrap();
        assert_eq!(id, id2);
    }

    const PENDING_ID: &str = "msgbatch_013Zva2CMHLNnXjNJJKqJ2EF";
    const PENDING_RESULTS_URL: &str = "https://api.anthropic.com/v1/messages/batches/msgbatch_013Zva2CMHLNnXjNJJKqJ2EF/results";
    const PENDING_STATS: Stats = Stats {
        processing: 100,
        succeeded: 50,
        errored: 30,
        canceled: 10,
        expired: 10,
    };

    const ERROR_ID: Id = Id { uuid: Uuid::nil() };
    const CANCELED_ID: Id = Id {
        uuid: Uuid::from_u128(1),
    };
    const EXPIRED_ID: Id = Id {
        uuid: Uuid::from_u128(2),
    };

    // Generate a Pending instance and an Id guaranteed to be in it.
    fn gen_pending() -> (Id, Pending<Prompt>) {
        let id = Id::default();

        let prompts: Prompts<Prompt> = Prompts {
            prompts: [
                (id, Prompt::default()),
                (ERROR_ID, Prompt::default()),
                (CANCELED_ID, Prompt::default()),
                (EXPIRED_ID, Prompt::default()),
            ]
            .into_iter()
            .collect(),
        };

        let meta = Meta {
            id: PENDING_ID.into(),
            status: Status::InProgress,
            stats: PENDING_STATS,
            created_at: Utc::now(),
            expires_at: Utc::now(),
            ended_at: Some(Utc::now()),
            cancel_initiated_at: Some(Utc::now()),
            archived_at: Some(Utc::now()),
            results_url: Some(Url::parse(PENDING_RESULTS_URL).unwrap()),
        };

        (id, Pending { prompts, meta })
    }

    #[test]
    fn test_pending_meta() {
        let (_, pending) = gen_pending();
        let meta = pending.meta();
        assert_eq!(meta.id, PENDING_ID);
    }

    #[test]
    fn test_pending_status() {
        let (_, pending) = gen_pending();
        assert!(pending.status() == Status::InProgress);
    }

    #[test]
    fn test_pending_is_done() {
        let (_, pending) = gen_pending();
        assert!(!pending.is_done());
    }

    #[test]
    fn test_pending_results_url() {
        let (_, pending) = gen_pending();
        assert_eq!(
            pending.results_url(),
            Some(&Url::parse(PENDING_RESULTS_URL).unwrap())
        );
    }

    #[test]
    fn test_pending_prompts() {
        let (id, pending) = gen_pending();
        let prompts = pending.prompts();
        assert_eq!(prompts.len(), 4);
        assert_eq!(prompts.get_id(id), Some(&Prompt::default()));
    }

    #[test]
    fn test_pending_decompose() {
        let (id, pending) = gen_pending();
        let (prompts, meta) = pending.decompose();
        assert_eq!(prompts.len(), 4);
        assert_eq!(prompts.get_id(id), Some(&Prompt::default()));
        assert_eq!(meta.id, PENDING_ID);
    }

    fn gen_ready() -> (Id, Ready<Prompt>) {
        let (id, pending) = gen_pending();
        let mut results = HashMap::new();
        results.insert(
            id,
            BatchResult::Ok(response::Message {
                id: PENDING_ID.into(),
                inner: Content::from("Hello roboto!").into(),
                model: model::Id::Anthropic(AnthropicModel::Haiku30),
                stop_reason: None,
                stop_sequence: Some("potato".into()),
                usage: Usage::default(),
            }),
        );
        results.insert(
            ERROR_ID,
            BatchResult::Error(client::AnthropicError::Billing {
                message: "you are too poor".into(),
            }),
        );
        results.insert(CANCELED_ID, BatchResult::Canceled);
        results.insert(EXPIRED_ID, BatchResult::Expired);

        let ready = Ready { pending, results };
        (id, ready)
    }

    #[test]
    fn test_ready_get_result() {
        let (id, ready) = gen_ready();
        assert!(ready.get_result(id).is_some());
        let result = ready.get_result(id).unwrap();
        if let BatchResult::Ok(msg) = result {
            assert_eq!(msg.id, PENDING_ID);
        } else {
            panic!("Expected Ok result");
        }
    }

    #[test]
    fn test_ready_decompose() {
        let (id, ready) = gen_ready();
        let (pending, results) = ready.decompose();
        assert!(results.contains_key(&id));
        assert_eq!(pending.meta.id, PENDING_ID);
    }

    #[test]
    fn test_ready_iter_ok() {
        let (_, ready) = gen_ready();
        let mut iter = ready.iter_ok();
        let (prompt, msg) = iter.next().unwrap();
        assert_eq!(prompt, &Prompt::default());
        assert_eq!(msg.id, PENDING_ID);
        assert_eq!(msg.inner, Content::from("Hello roboto!").into());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ready_into_iter_and_collect() {
        let (id, ready) = gen_ready();

        // By reference: borrows without consuming, and reports an exact size.
        assert_eq!(ready.iter().len(), 4);
        let ids: Vec<Id> = (&ready).into_iter().map(|(id, ..)| id).collect();
        assert_eq!(ids.len(), 4);
        assert!(ids.contains(&id));

        // By value: consumes into owned triples.
        let owned: Vec<_> = ready.into_iter().collect();
        assert_eq!(owned.len(), 4);
        assert!(owned.iter().any(|(i, _p, _r)| *i == id));

        // Plays well with `FromIterator` — the point of the trait impl.
        let (id, ready) = gen_ready();
        let map: HashMap<Id, BatchResult> =
            ready.into_iter().map(|(id, _p, r)| (id, r)).collect();
        assert_eq!(map.len(), 4);
        assert!(map.contains_key(&id));
    }

    #[test]
    fn test_ready_remove_ok() {
        let (_, mut ready) = gen_ready();
        assert!(ready.results.len() == 4);
        let (prompt, msg) = ready.remove_ok().pop().unwrap();
        assert_eq!(prompt, Prompt::default());
        assert_eq!(msg.id, PENDING_ID);
        assert_eq!(msg.inner, Content::from("Hello roboto!").into());
        assert_eq!(ready.pending.meta.stats.succeeded, 49);
        assert_eq!(ready.results.len(), 3);
    }

    #[test]
    fn test_ready_iter_errors() {
        let (_, ready) = gen_ready();
        let mut iter = ready.iter_errors();
        let (prompt, err) = iter.next().unwrap();
        assert_eq!(prompt, &Prompt::default());
        assert_eq!(
            err,
            &client::AnthropicError::Billing {
                message: "you are too poor".into()
            }
        );
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ready_remove_errors() {
        let (_, mut ready) = gen_ready();
        assert!(ready.results.len() == 4);
        let (prompt, err) = ready.remove_errors().pop().unwrap();
        assert_eq!(prompt, Prompt::default());
        assert_eq!(
            err,
            client::AnthropicError::Billing {
                message: "you are too poor".into()
            }
        );
        assert_eq!(ready.pending.meta.stats.errored, 29);
        assert_eq!(ready.results.len(), 3);
    }

    #[test]
    fn test_ready_iter_canceled() {
        let (_, ready) = gen_ready();
        let mut iter = ready.iter_canceled();
        let prompt = iter.next().unwrap();
        assert_eq!(prompt, &Prompt::default());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ready_remove_canceled() {
        let (_, mut ready) = gen_ready();
        assert!(ready.results.len() == 4);
        let prompt = ready.remove_canceled().pop().unwrap();
        assert_eq!(prompt, Prompt::default());
        assert_eq!(ready.pending.meta.stats.canceled, 9);
        assert_eq!(ready.results.len(), 3);
    }

    #[test]
    fn test_ready_iter_expired() {
        let (_, ready) = gen_ready();
        let mut iter = ready.iter_expired();
        let prompt = iter.next().unwrap();
        assert_eq!(prompt, &Prompt::default());
        assert!(iter.next().is_none());
    }

    #[test]
    fn test_ready_remove_expired() {
        let (_, mut ready) = gen_ready();
        assert!(ready.results.len() == 4);
        let prompt = ready.remove_expired().pop().unwrap();
        assert_eq!(prompt, Prompt::default());
        assert_eq!(ready.pending.meta.stats.expired, 9);
        assert_eq!(ready.results.len(), 3);
    }

    #[test]
    fn test_ready_iter() {
        let (id, ready) = gen_ready();
        let results: HashMap<Id, (&Prompt, &BatchResult)> = ready
            .iter()
            .map(|(id, prompt, result)| (id, (prompt, result)))
            .collect();
        assert_eq!(results.len(), 4);
        assert!(results.contains_key(&id));
        assert!(results.contains_key(&ERROR_ID));
        assert!(results.contains_key(&CANCELED_ID));
        assert!(results.contains_key(&EXPIRED_ID));
    }

    #[test]
    fn test_ready_into_iter() {
        let (id, ready) = gen_ready();
        let results: HashMap<Id, (Prompt, BatchResult)> = ready
            .into_iter()
            .map(|(id, prompt, result)| (id, (prompt, result)))
            .collect();
        assert_eq!(results.len(), 4);
        assert!(results.contains_key(&id));
        assert!(results.contains_key(&ERROR_ID));
        assert!(results.contains_key(&CANCELED_ID));
        assert!(results.contains_key(&EXPIRED_ID));
    }

    #[test]
    fn test_batch_result_into_result() {
        let ok = Result::from(BatchResult::Ok(response::Message {
            id: PENDING_ID.into(),
            inner: Content::from("Hello roboto!").into(),
            model: model::Id::Anthropic(AnthropicModel::Haiku30),
            stop_reason: None,
            stop_sequence: Some("potato".into()),
            usage: Usage::default(),
        }));
        assert!(ok.is_ok());

        let err =
            Result::from(BatchResult::Error(client::AnthropicError::Billing {
                message: "you are too poor".into(),
            }));
        assert!(err.is_err());

        let canceled = Result::from(BatchResult::Canceled);
        assert!(canceled.is_err());

        let expired = Result::from(BatchResult::Expired);
        assert!(expired.is_err());
    }
}
