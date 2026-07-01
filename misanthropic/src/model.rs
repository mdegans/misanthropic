//! [`Model`] to use for inference.
use std::borrow::Cow;
use std::collections::BTreeMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::prompt::Effort;

/// All available models, as returned by
/// [`Client::models`](crate::Client::models) — a thin wrapper over a `Vec` of
/// [`ModelInfo`] that [`Deref`](std::ops::Deref)s (and
/// [`DerefMut`](std::ops::DerefMut)s) to it.
///
/// The wire response nests the list under a `data` key, so this is a struct
/// rather than a bare `Vec`. [`data`](Self::data) is public and the type is
/// [`FromIterator`] / [`From<Vec>`](From) so tests can mock a model list
/// without round-tripping through JSON — e.g.
/// `[info(Id::Opus48, …)].into_iter().collect::<Models>()`.
#[derive(
    Debug,
    Clone,
    Default,
    Serialize,
    Deserialize,
    derive_more::Deref,
    derive_more::DerefMut,
)]
#[serde(rename_all = "snake_case")]
pub struct Models {
    /// List of available models.
    pub data: Vec<ModelInfo>,
}

impl FromIterator<ModelInfo> for Models {
    fn from_iter<I: IntoIterator<Item = ModelInfo>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

impl From<Vec<ModelInfo>> for Models {
    fn from(data: Vec<ModelInfo>) -> Self {
        Self { data }
    }
}

impl IntoIterator for Models {
    type Item = ModelInfo;
    type IntoIter = std::vec::IntoIter<ModelInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a> IntoIterator for &'a Models {
    type Item = &'a ModelInfo;
    type IntoIter = std::slice::Iter<'a, ModelInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a> IntoIterator for &'a mut Models {
    type Item = &'a mut ModelInfo;
    type IntoIter = std::slice::IterMut<'a, ModelInfo>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

/// Model information, as returned by [`Client::models`](crate::Client::models).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ModelInfo {
    /// Model ID.
    pub id: Model,
    /// Human-readable display name, e.g. `"Claude Opus 4.6"`.
    pub display_name: Cow<'static, str>,
    /// What the model supports — see [`Capabilities`].
    #[serde(default)]
    pub capabilities: Capabilities,
    /// Maximum number of input tokens the model accepts.
    #[serde(default)]
    pub max_input_tokens: u32,
    /// Maximum number of tokens the model can generate in a response.
    #[serde(default)]
    pub max_tokens: u32,
    /// Object-type discriminator. Always [`Kind::Model`] here.
    #[serde(default, rename = "type")]
    pub kind: Kind,
    /// Created at.
    pub created_at: DateTime<Utc>,
}

impl ModelInfo {
    /// Whether this *offered* model (as returned by
    /// [`Client::models`](crate::Client::models)) meets a `requested` spec:
    /// same [`id`](Self::id) and [`kind`](Self::kind), token ceilings at least
    /// as high, and [`Capabilities::satisfies`] for the rest.
    /// [`display_name`](Self::display_name) and [`created_at`](Self::created_at)
    /// are ignored.
    ///
    /// A `requested` token ceiling of `0` means "no requirement"; an *offered*
    /// `0` (e.g. an older response that omitted the field) meets only a `0`
    /// request — the conservative call.
    pub fn satisfies(&self, requested: &ModelInfo) -> bool {
        self.id == requested.id
            && self.kind == requested.kind
            && requested.max_input_tokens <= self.max_input_tokens
            && requested.max_tokens <= self.max_tokens
            && self.capabilities.satisfies(&requested.capabilities)
    }
}

/// Object-type discriminator on a [`ModelInfo`]. Always [`Kind::Model`] for the
/// `/v1/models` endpoint.
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq,
)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Kind {
    /// A model.
    #[default]
    Model,
}

/// Whether a single model [`Capability`] is supported — the leaf node of the
/// [`Capabilities`] tree, a bare `{ "supported": bool }`.
///
/// Compares against `bool` for sugar, so `caps.pdf_input == true` reads
/// straight through to [`supported`](Self::supported).
#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq,
)]
pub struct Capability {
    /// Whether the capability is supported.
    pub supported: bool,
}

impl From<bool> for Capability {
    fn from(supported: bool) -> Self {
        Self { supported }
    }
}

impl From<Capability> for bool {
    fn from(c: Capability) -> Self {
        c.supported
    }
}

impl PartialEq<bool> for Capability {
    fn eq(&self, other: &bool) -> bool {
        self.supported == *other
    }
}

impl PartialEq<Capability> for bool {
    fn eq(&self, other: &Capability) -> bool {
        *self == other.supported
    }
}

impl Capability {
    /// Whether this *offered* capability meets `requested` — the boolean
    /// implication `requested ⟹ self`. A capability the requester didn't ask
    /// for (`requested` unsupported) imposes no constraint.
    pub fn satisfies(&self, requested: &Capability) -> bool {
        self.supported || !requested.supported
    }
}

/// Whether `offered` meets every entry `requested` marks supported: each such
/// key must be present and supported in `offered`. Keys the requester didn't
/// ask for impose nothing. The leaf rule behind the map-bearing capabilities
/// ([`ContextManagement`], [`EffortSupport`], [`ThinkingSupport`]).
fn map_satisfies<K: Ord>(
    offered: &BTreeMap<K, Capability>,
    requested: &BTreeMap<K, Capability>,
) -> bool {
    requested.iter().all(|(key, want)| {
        !want.supported || offered.get(key).is_some_and(|have| have.supported)
    })
}

/// What a [`ModelInfo`] supports, from the `capabilities` field of the
/// `/v1/models` response.
///
/// Every field defaults to unsupported when absent, and unknown future
/// capabilities are ignored on deserialization — mirroring the forward-compat
/// stance of [`Model::Custom`].
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Capabilities {
    /// [Message Batches](crate::Client::batch) support.
    #[serde(default)]
    pub batch: Capability,
    /// Citations support.
    #[serde(default)]
    pub citations: Capability,
    /// Server-side code-execution tool support.
    #[serde(default)]
    pub code_execution: Capability,
    /// Context-management (context editing) support and its strategies.
    #[serde(default)]
    pub context_management: ContextManagement,
    /// Reasoning-[`effort`](crate::prompt::Effort) support, per level.
    #[serde(default)]
    pub effort: EffortSupport,
    /// Image input support.
    #[serde(default)]
    pub image_input: Capability,
    /// PDF input support.
    #[serde(default)]
    pub pdf_input: Capability,
    /// Structured-output support.
    #[serde(default)]
    pub structured_outputs: Capability,
    /// Extended-[`thinking`](crate::prompt::Thinking) support and its types.
    #[serde(default)]
    pub thinking: ThinkingSupport,
}

impl Capabilities {
    /// Whether this *offered* set meets `requested` — every capability the
    /// requester asked for is offered, per [`Capability::satisfies`] (and the
    /// per-strategy/level/type subset checks for the map-bearing ones).
    /// Capabilities the requester didn't ask for impose nothing.
    pub fn satisfies(&self, requested: &Capabilities) -> bool {
        self.batch.satisfies(&requested.batch)
            && self.citations.satisfies(&requested.citations)
            && self.code_execution.satisfies(&requested.code_execution)
            && self
                .context_management
                .satisfies(&requested.context_management)
            && self.effort.satisfies(&requested.effort)
            && self.image_input.satisfies(&requested.image_input)
            && self.pdf_input.satisfies(&requested.pdf_input)
            && self
                .structured_outputs
                .satisfies(&requested.structured_outputs)
            && self.thinking.satisfies(&requested.thinking)
    }
}

/// Context-management support — the `context_management` capability.
///
/// Beyond the top-level [`supported`](Self::supported) flag, the API reports a
/// flag per strategy (e.g. `clear_tool_uses_20250919`, `compact_20260112`).
/// These are date-versioned and open-ended, so they are kept as an untyped map
/// rather than an enum.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ContextManagement {
    /// Whether context management is supported at all.
    #[serde(default)]
    pub supported: bool,
    /// Supported strategies, keyed by their API name.
    #[serde(flatten)]
    pub strategies: BTreeMap<String, Capability>,
}

impl ContextManagement {
    /// Whether this *offered* support meets `requested`: the top-level flag
    /// follows `requested ⟹ self`, and every [strategy](Self::strategies) the
    /// requester asked for must be offered (see `map_satisfies`).
    pub fn satisfies(&self, requested: &ContextManagement) -> bool {
        (self.supported || !requested.supported)
            && map_satisfies(&self.strategies, &requested.strategies)
    }
}

/// Reasoning-[`effort`](crate::prompt::Effort) support — the `effort`
/// capability.
///
/// The API reports a flag per level (`low`, `medium`, `high`, `xhigh`,
/// `max`), kept as an untyped map so new levels don't break parsing.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EffortSupport {
    /// Whether configurable effort is supported at all.
    #[serde(default)]
    pub supported: bool,
    /// Supported levels, keyed by [`Effort`]. Levels this crate doesn't know
    /// land in [`Effort::Custom`] rather than breaking the parse.
    #[serde(flatten)]
    pub levels: BTreeMap<Effort, Capability>,
}

impl EffortSupport {
    /// Whether this *offered* support meets `requested`: the top-level flag
    /// follows `requested ⟹ self`, and every [level](Self::levels) the
    /// requester asked for must be offered (see `map_satisfies`).
    pub fn satisfies(&self, requested: &EffortSupport) -> bool {
        (self.supported || !requested.supported)
            && map_satisfies(&self.levels, &requested.levels)
    }
}

/// Extended-[`thinking`](crate::prompt::Thinking) support — the `thinking`
/// capability.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ThinkingSupport {
    /// Whether extended thinking is supported.
    #[serde(default)]
    pub supported: bool,
    /// Supported thinking types (e.g. `adaptive`, `enabled`), keyed by name.
    #[serde(default)]
    pub types: BTreeMap<String, Capability>,
}

impl ThinkingSupport {
    /// Whether this *offered* support meets `requested`: the top-level flag
    /// follows `requested ⟹ self`, and every [type](Self::types) the requester
    /// asked for must be offered (see `map_satisfies`).
    pub fn satisfies(&self, requested: &ThinkingSupport) -> bool {
        (self.supported || !requested.supported)
            && map_satisfies(&self.types, &requested.types)
    }
}

/// The model to use for inference — a known Anthropic [`Id`], or a custom id
/// string for a model this crate doesn't enumerate.
#[derive(
    Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
#[serde(rename_all = "snake_case", untagged)]
pub enum Model {
    /// Anthropic model.
    Anthropic(Id),
    /// Custom model id.
    Custom(Cow<'static, str>),
}

impl Model {
    /// The model's wire id — the canonical string the API expects (e.g.
    /// `claude-opus-4-8`). For a known [`Anthropic`](Self::Anthropic) model this
    /// delegates to [`Id::name`]; for a [`Custom`](Self::Custom) model it is the
    /// caller-provided string verbatim.
    pub fn name(&self) -> &str {
        match self {
            Model::Anthropic(id) => id.name(),
            Model::Custom(name) => name,
        }
    }

    /// Whether this model accepts a mid-conversation [`System`] turn — a
    /// [`Role::System`] message *within* the `messages` array, distinct from the
    /// top-level [`Prompt::system`] field. Hard-gated to [`Opus48`](Id::Opus48)
    /// and later; a [`Custom`](Self::Custom) model is treated conservatively as
    /// unsupported.
    ///
    /// Used by [`Prompt::resolve_role`](crate::Prompt::resolve_role) to seat a
    /// pushed [`Notification`](crate::tool::Notification) at a role the model
    /// understands.
    ///
    /// [`System`]: crate::prompt::message::Role::System
    /// [`Role::System`]: crate::prompt::message::Role::System
    /// [`Prompt::system`]: crate::Prompt::system
    pub fn supports_system_role(&self) -> bool {
        // Fable 5 verified live 2026-06-11 (placement grammar enforced, turn
        // honored); Mythos 5 is the same underlying model.
        matches!(
            self,
            Model::Anthropic(Id::Opus48 | Id::Fable5 | Id::Mythos5)
        )
    }

    /// Pick the first of `preferred` [`Role`]s this model supports, for seating
    /// a pushed [`Notification`](crate::tool::Notification). Only
    /// [`Role::System`] is capability-gated (see
    /// [`supports_system_role`](Self::supports_system_role)); [`User`] and
    /// [`Assistant`] are always available. An empty list (or one whose every
    /// entry is unsupported) falls back to [`User`].
    ///
    /// [`Prompt::resolve_role`](crate::Prompt::resolve_role) delegates here.
    ///
    /// [`Role`]: crate::prompt::message::Role
    /// [`Role::System`]: crate::prompt::message::Role::System
    /// [`User`]: crate::prompt::message::Role::User
    /// [`Assistant`]: crate::prompt::message::Role::Assistant
    pub fn resolve_role(
        &self,
        preferred: &[crate::prompt::message::Role],
    ) -> crate::prompt::message::Role {
        use crate::prompt::message::Role;
        preferred
            .iter()
            .copied()
            .find(|role| match role {
                Role::User | Role::Assistant => true,
                Role::System => self.supports_system_role(),
            })
            .unwrap_or(Role::User)
    }
}

impl std::fmt::Display for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl<T> From<T> for Model
where
    T: Into<Cow<'static, str>>,
{
    fn from(s: T) -> Self {
        // Unwrap can't panic because we have a catch-all variant.
        serde_json::from_str(&format!("\"{}\"", s.into())).unwrap()
    }
}

impl From<Id> for Model {
    fn from(value: Id) -> Self {
        Model::Anthropic(value)
    }
}

impl PartialEq<Id> for Model {
    fn eq(&self, other: &Id) -> bool {
        match self {
            Model::Anthropic(model) => model == other,
            Model::Custom(s) => s.as_ref() == other.name(),
        }
    }
}

impl<S> PartialEq<S> for Model
where
    S: AsRef<str>,
{
    fn eq(&self, other: &S) -> bool {
        match self {
            Model::Anthropic(model) => model.name() == other.as_ref(),
            Model::Custom(s) => s.as_ref() == other.as_ref(),
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Model::Anthropic(Id::default())
    }
}

/// A known Anthropic model id — the canonical wire id strings (e.g.
/// `claude-opus-4-8`). The [`Anthropic`](Model::Anthropic) arm of [`Model`].
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Deserialize,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    Serialize,
)]
#[cfg_attr(test, derive(strum::EnumIter))]
#[serde(rename_all = "snake_case")]
pub enum Id {
    // ── Claude 3.x ───────────────────────────────────────────────────────
    /// Sonnet 3.7 (latest)
    #[serde(rename = "claude-3-7-sonnet-latest")]
    Sonnet37,
    /// Sonnet 3.7 2025-02-19
    #[serde(rename = "claude-3-7-sonnet-20250219")]
    Sonnet37_20250219,
    /// Sonnet 3.5 (latest)
    #[serde(rename = "claude-3-5-sonnet-latest")]
    Sonnet35,
    /// Sonnet 3.5 2024-06-20
    #[serde(rename = "claude-3-5-sonnet-20240620")]
    Sonnet35_20240620,
    /// Sonnet 3.5 2024-10-22
    #[serde(rename = "claude-3-5-sonnet-20241022")]
    Sonnet35_20241022,
    /// Opus 3.0 (latest)
    #[serde(rename = "claude-3-opus-latest")]
    Opus30,
    /// Opus 3.0 2024-02-29
    #[serde(rename = "claude-3-opus-20240229")]
    Opus30_20240229,
    /// Haiku 3.5 (latest)
    #[serde(rename = "claude-3-5-haiku-latest")]
    Haiku35,
    /// Haiku 3.5 2024-10-22
    #[serde(rename = "claude-3-5-haiku-20241022")]
    Haiku35_20241022,
    /// Haiku 3.0 2024-03-07
    #[serde(
        rename = "claude-3-haiku-20240307",
        alias = "claude-3-haiku-latest"
    )]
    Haiku30,

    // ── Claude 4.x ───────────────────────────────────────────────────────
    /// Opus 4.0 2025-05-14
    #[serde(rename = "claude-opus-4-20250514")]
    Opus40_20250514,
    /// Opus 4.0 (latest)
    #[serde(rename = "claude-opus-4-0")]
    Opus40,
    /// Sonnet 4.0 2025-05-14
    #[serde(rename = "claude-sonnet-4-20250514")]
    Sonnet40_20250514,
    /// Sonnet 4.0 (latest)
    #[serde(rename = "claude-sonnet-4-0")]
    Sonnet40,
    /// Opus 4.1 2025-08-05
    #[serde(rename = "claude-opus-4-1-20250805")]
    Opus41_20250805,
    /// Opus 4.1 (latest)
    #[serde(rename = "claude-opus-4-1")]
    Opus41,
    /// Haiku 4.5 2025-10-01
    #[serde(rename = "claude-haiku-4-5-20251001")]
    Haiku45_20251001,
    /// Haiku 4.5 (latest). This is the default model.
    #[default]
    #[serde(rename = "claude-haiku-4-5")]
    Haiku45,
    /// Sonnet 4.5 2025-09-29
    #[serde(rename = "claude-sonnet-4-5-20250929")]
    Sonnet45_20250929,
    /// Sonnet 4.5 (latest)
    #[serde(rename = "claude-sonnet-4-5")]
    Sonnet45,
    /// Opus 4.5 2025-11-01
    #[serde(rename = "claude-opus-4-5-20251101")]
    Opus45_20251101,
    /// Opus 4.5 (latest)
    #[serde(rename = "claude-opus-4-5")]
    Opus45,
    /// Sonnet 4.6
    #[serde(rename = "claude-sonnet-4-6")]
    Sonnet46,
    /// Opus 4.6
    #[serde(rename = "claude-opus-4-6")]
    Opus46,
    /// Opus 4.7. Supports the 1M-token context window via the
    /// `context-1m-2025-08-07` beta header — see `Client::beta`; there is
    /// no separate wire id (Claude Code's `[1m]` suffix is UI notation).
    #[serde(rename = "claude-opus-4-7")]
    Opus47,
    /// Opus 4.8 (latest flagship). First model to support
    /// [mid-conversation system messages](crate::prompt::message::Role::System).
    /// 1M context via the `context-1m-2025-08-07` beta header.
    #[serde(rename = "claude-opus-4-8")]
    Opus48,

    // ── Claude 5.x ───────────────────────────────────────────────────────
    /// Fable 5 — the first Mythos-class model (above Opus in capability).
    /// The 1M-token context window is the default (no beta header), with up
    /// to 128k output tokens.
    #[serde(rename = "claude-fable-5")]
    Fable5,
    /// Mythos 5 — Fable 5 without the dual-use safety measures; available
    /// only to approved organizations (account-gated). 1M context default.
    #[serde(rename = "claude-mythos-5")]
    Mythos5,
}

impl Id {
    /// The canonical wire id — the exact string the API expects and echoes
    /// back (e.g. `claude-opus-4-8`), matching this variant's `serde(rename)`.
    ///
    /// This is the single source of truth for a known model's string form;
    /// [`Model::name`] delegates here. There is deliberately no short "display"
    /// form on [`Id`] — a human-readable label is the API's concern and lives on
    /// [`ModelInfo::display_name`], which `/v1/models` returns.
    pub fn name(self) -> &'static str {
        match self {
            Id::Sonnet37 => "claude-3-7-sonnet-latest",
            Id::Sonnet37_20250219 => "claude-3-7-sonnet-20250219",
            Id::Sonnet35 => "claude-3-5-sonnet-latest",
            Id::Sonnet35_20240620 => "claude-3-5-sonnet-20240620",
            Id::Sonnet35_20241022 => "claude-3-5-sonnet-20241022",
            Id::Opus30 => "claude-3-opus-latest",
            Id::Opus30_20240229 => "claude-3-opus-20240229",
            Id::Haiku35 => "claude-3-5-haiku-latest",
            Id::Haiku35_20241022 => "claude-3-5-haiku-20241022",
            Id::Haiku30 => "claude-3-haiku-20240307",
            Id::Opus40_20250514 => "claude-opus-4-20250514",
            Id::Opus40 => "claude-opus-4-0",
            Id::Sonnet40_20250514 => "claude-sonnet-4-20250514",
            Id::Sonnet40 => "claude-sonnet-4-0",
            Id::Opus41_20250805 => "claude-opus-4-1-20250805",
            Id::Opus41 => "claude-opus-4-1",
            Id::Haiku45_20251001 => "claude-haiku-4-5-20251001",
            Id::Haiku45 => "claude-haiku-4-5",
            Id::Sonnet45_20250929 => "claude-sonnet-4-5-20250929",
            Id::Sonnet45 => "claude-sonnet-4-5",
            Id::Opus45_20251101 => "claude-opus-4-5-20251101",
            Id::Opus45 => "claude-opus-4-5",
            Id::Sonnet46 => "claude-sonnet-4-6",
            Id::Opus46 => "claude-opus-4-6",
            Id::Opus47 => "claude-opus-4-7",
            Id::Opus48 => "claude-opus-4-8",
            Id::Fable5 => "claude-fable-5",
            Id::Mythos5 => "claude-mythos-5",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "client")]
    use crate::Client;
    #[cfg(feature = "client")]
    use crate::{Prompt, prompt::message::Role};
    #[cfg(feature = "client")]
    use strum::IntoEnumIterator;

    #[cfg(feature = "client")]
    const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

    #[cfg(feature = "client")]
    fn load_api_key() -> Option<String> {
        use std::fs::File;
        use std::io::Read;
        use std::path::Path;

        let mut file =
            File::open(Path::new(CRATE_ROOT).join("api.key")).ok()?;
        let mut key = String::new();
        file.read_to_string(&mut key).unwrap();
        Some(key.trim().to_string())
    }

    #[test]
    fn test_model_deserialize() {
        const JSON:&[u8] = b"{\"data\":[{\"type\":\"model\",\"id\":\"claude-3-5-sonnet-20241022\",\"display_name\":\"Claude 3.5 Sonnet (New)\",\"created_at\":\"2024-10-22T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-5-haiku-20241022\",\"display_name\":\"Claude 3.5 Haiku\",\"created_at\":\"2024-10-22T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-5-sonnet-20240620\",\"display_name\":\"Claude 3.5 Sonnet (Old)\",\"created_at\":\"2024-06-20T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-haiku-20240307\",\"display_name\":\"Claude 3 Haiku\",\"created_at\":\"2024-03-07T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-opus-20240229\",\"display_name\":\"Claude 3 Opus\",\"created_at\":\"2024-02-29T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-3-sonnet-20240229\",\"display_name\":\"Claude 3 Sonnet\",\"created_at\":\"2024-02-29T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-2.1\",\"display_name\":\"Claude 2.1\",\"created_at\":\"2023-11-21T00:00:00Z\"},{\"type\":\"model\",\"id\":\"claude-2.0\",\"display_name\":\"Claude 2.0\",\"created_at\":\"2023-07-11T00:00:00Z\"}],\"has_more\":false,\"first_id\":\"claude-3-5-sonnet-20241022\",\"last_id\":\"claude-2.0\"}";
        let models = serde_json::from_slice::<Models>(JSON).unwrap();
        assert_eq!(models.len(), 8);
    }

    /// #119: a `Models` can be built without deserializing — from an iterator,
    /// from a `Vec`, or mutated in place through `DerefMut` — so tests can mock
    /// the model list. `data` is public too.
    #[test]
    fn test_models_construct_for_mocking() {
        let offered = [
            info(Id::Opus48, 200_000, 64_000),
            info(Id::Haiku45, 200_000, 64_000),
        ];

        // FromIterator / .collect()
        let mut models: Models = offered.iter().cloned().collect();
        assert_eq!(models.len(), 2); // via Deref<Target = Vec<_>>
        assert_eq!(models[0].id, Id::Opus48);

        // DerefMut lets it be mutated in place.
        models.push(info(Id::Sonnet45, 200_000, 64_000));
        assert_eq!(models.len(), 3);
        models.pop();

        // From<Vec> and the public `data` field.
        let from_vec = Models::from(offered.to_vec());
        assert_eq!(from_vec.data.len(), 2);

        // IntoIterator by value.
        let ids: Vec<Model> = from_vec.into_iter().map(|m| m.id).collect();
        assert_eq!(ids, [Id::Opus48, Id::Haiku45]); // Model: PartialEq<Id>

        // Round-trips through the wire shape it deserializes from.
        let json = serde_json::to_string(&models).unwrap();
        assert_eq!(serde_json::from_str::<Models>(&json).unwrap().len(), 2);
    }

    #[test]
    fn test_model_capabilities_deserialize() {
        // A current-shape `/v1/models` entry, with the full `capabilities`
        // tree, token limits, and the `type` discriminator.
        const JSON: &str = r#"{
          "id": "claude-opus-4-6",
          "capabilities": {
            "batch": { "supported": true },
            "citations": { "supported": true },
            "code_execution": { "supported": true },
            "context_management": {
              "clear_thinking_20251015": { "supported": true },
              "clear_tool_uses_20250919": { "supported": true },
              "compact_20260112": { "supported": true },
              "supported": true
            },
            "effort": {
              "high": { "supported": true },
              "low": { "supported": true },
              "max": { "supported": true },
              "medium": { "supported": true },
              "supported": true,
              "xhigh": { "supported": true }
            },
            "image_input": { "supported": true },
            "pdf_input": { "supported": true },
            "structured_outputs": { "supported": true },
            "thinking": {
              "supported": true,
              "types": {
                "adaptive": { "supported": true },
                "enabled": { "supported": true }
              }
            }
          },
          "created_at": "2026-02-04T00:00:00Z",
          "display_name": "Claude Opus 4.6",
          "max_input_tokens": 200000,
          "max_tokens": 64000,
          "type": "model"
        }"#;

        let model: ModelInfo = serde_json::from_str(JSON).unwrap();
        assert_eq!(model.id, Id::Opus46);
        assert_eq!(model.display_name, "Claude Opus 4.6");
        assert_eq!(model.max_input_tokens, 200000);
        assert_eq!(model.max_tokens, 64000);
        assert_eq!(model.kind, Kind::Model);

        let caps = &model.capabilities;
        assert!(caps.batch.supported);
        assert!(caps.citations.supported);
        assert!(caps.code_execution.supported);
        assert!(caps.image_input.supported);
        assert!(caps.pdf_input.supported);
        assert!(caps.structured_outputs.supported);

        // Sugar: a `Capability` compares straight against `bool`, both ways.
        assert!(caps.pdf_input == true);
        assert!(true == caps.image_input);

        // Open-ended strategy / level maps land in their sub-maps, with
        // `supported` pulled out of the flattened object.
        assert!(caps.context_management.supported);
        assert!(
            caps.context_management.strategies["compact_20260112"].supported
        );
        assert!(!caps.context_management.strategies.contains_key("supported"));

        // Effort levels are keyed by the typed `Effort`; a known level is a
        // unit variant, and `supported` does not leak into the map.
        assert!(caps.effort.supported);
        assert!(caps.effort.levels[&Effort::XHigh].supported);
        assert_eq!(caps.effort.levels.len(), 5);
        assert!(!caps.effort.levels.contains_key(&Effort::from("supported")));

        assert!(caps.thinking.supported);
        assert!(caps.thinking.types["adaptive"].supported);

        // Round-trips: re-serializing and parsing yields the same value.
        let json = serde_json::to_string(&model).unwrap();
        let round: ModelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(round.capabilities, model.capabilities);
        assert_eq!(round.kind, Kind::Model);
    }

    #[test]
    fn test_model_capabilities_default_when_absent() {
        // An older-shape entry with no `capabilities` / token limits still
        // parses, defaulting to "unsupported" / zero.
        const JSON: &str = r#"{
          "type": "model",
          "id": "claude-3-5-haiku-20241022",
          "display_name": "Claude 3.5 Haiku",
          "created_at": "2024-10-22T00:00:00Z"
        }"#;

        let model: ModelInfo = serde_json::from_str(JSON).unwrap();
        assert_eq!(model.capabilities, Capabilities::default());
        assert_eq!(model.max_tokens, 0);
        assert!(!model.capabilities.thinking.supported);
        assert!(model.capabilities.thinking.types.is_empty());
    }

    /// A minimal [`ModelInfo`] for negotiation tests: given `id`, default caps,
    /// and the given token ceilings.
    fn info(id: Id, max_input_tokens: u32, max_tokens: u32) -> ModelInfo {
        ModelInfo {
            id: id.into(),
            display_name: "test".into(),
            capabilities: Capabilities::default(),
            max_input_tokens,
            max_tokens,
            kind: Kind::Model,
            created_at: "2026-01-01T00:00:00Z".parse().unwrap(),
        }
    }

    #[test]
    fn test_capability_satisfies() {
        let yes = Capability::from(true);
        let no = Capability::from(false);

        // requested ⟹ offered: only "asked but not offered" fails.
        assert!(yes.satisfies(&yes));
        assert!(yes.satisfies(&no)); // offered extra, ignored
        assert!(no.satisfies(&no)); // didn't ask
        assert!(!no.satisfies(&yes)); // asked, not offered
    }

    #[test]
    fn test_capabilities_satisfies_flat() {
        let mut offered = Capabilities::default();
        let mut requested = Capabilities::default();

        // Empty request is met by anything, including an empty offer.
        assert!(offered.satisfies(&requested));

        // Ask for pdf the offer lacks -> rejected; grant it -> satisfied.
        requested.pdf_input = true.into();
        assert!(!offered.satisfies(&requested));
        offered.pdf_input = true.into();
        assert!(offered.satisfies(&requested));

        // An unrequested capability the offer happens to provide is ignored.
        offered.citations = true.into();
        assert!(offered.satisfies(&requested));
    }

    #[test]
    fn test_capabilities_satisfies_nested_maps() {
        let mut offered = Capabilities::default();
        let mut requested = Capabilities::default();

        // Request a thinking type and the top-level flag.
        requested.thinking.supported = true;
        requested
            .thinking
            .types
            .insert("adaptive".into(), true.into());

        // Offer the flag but not the type -> rejected.
        offered.thinking.supported = true;
        assert!(!offered.satisfies(&requested));

        // Offer the type as well -> satisfied.
        offered
            .thinking
            .types
            .insert("adaptive".into(), true.into());
        assert!(offered.satisfies(&requested));

        // A strategy present but explicitly unsupported does not satisfy.
        requested
            .context_management
            .strategies
            .insert("compact_20260112".into(), true.into());
        offered
            .context_management
            .strategies
            .insert("compact_20260112".into(), false.into());
        assert!(!offered.satisfies(&requested));
        offered
            .context_management
            .strategies
            .insert("compact_20260112".into(), true.into());
        assert!(offered.satisfies(&requested));

        // Effort levels are keyed by the typed `Effort`.
        requested.effort.levels.insert(Effort::High, true.into());
        assert!(!offered.satisfies(&requested));
        offered.effort.levels.insert(Effort::High, true.into());
        assert!(offered.satisfies(&requested));
    }

    #[test]
    fn test_model_info_satisfies_id_and_tokens() {
        let offered = info(Id::Opus48, 200_000, 64_000);

        // Identical -> satisfies.
        assert!(offered.satisfies(&info(Id::Opus48, 200_000, 64_000)));

        // Different id -> never.
        assert!(!offered.satisfies(&info(Id::Haiku45, 200_000, 64_000)));

        // requested ceilings <= offered pass; exceeding either fails.
        assert!(offered.satisfies(&info(Id::Opus48, 100_000, 32_000)));
        assert!(offered.satisfies(&info(Id::Opus48, 0, 0))); // no requirement
        assert!(!offered.satisfies(&info(Id::Opus48, 200_001, 64_000)));
        assert!(!offered.satisfies(&info(Id::Opus48, 200_000, 64_001)));

        // An offer with unknown (0) ceilings meets only a 0 request.
        let unknown = info(Id::Opus48, 0, 0);
        assert!(unknown.satisfies(&info(Id::Opus48, 0, 0)));
        assert!(!unknown.satisfies(&info(Id::Opus48, 1, 0)));
        assert!(!unknown.satisfies(&info(Id::Opus48, 0, 1)));
    }

    #[test]
    fn test_model_info_satisfies_capabilities() {
        let mut offered = info(Id::Opus48, 200_000, 64_000);
        let mut requested = info(Id::Opus48, 200_000, 64_000);

        requested.capabilities.batch = true.into();
        assert!(!offered.satisfies(&requested));
        offered.capabilities.batch = true.into();
        assert!(offered.satisfies(&requested));
    }

    #[test]
    fn test_id_name() {
        // `Id::name` is the canonical wire id — identical to `Model::name` and
        // to the variant's `serde(rename)`. There is no short display form.
        assert_eq!(Id::Sonnet35.name(), "claude-3-5-sonnet-latest");
        assert_eq!(Id::Opus30_20240229.name(), "claude-3-opus-20240229");
        assert_eq!(Id::Haiku30.name(), "claude-3-haiku-20240307");
        assert_eq!(Id::Opus48.name(), "claude-opus-4-8");
        assert_eq!(Id::Fable5.name(), "claude-fable-5");

        // `Model::name` delegates to `Id::name` for known models.
        assert_eq!(Model::Anthropic(Id::Opus48).name(), Id::Opus48.name());

        let model: Model = "custom_model".into();
        assert_eq!(model.name(), "custom_model");
        assert_eq!(model, "custom_model");
    }

    /// Regression for #109: `Model`'s hand-written `PartialEq` impls compare
    /// against the **wire id**, not a short display form. Before the fix they
    /// reached for `Id::name`'s display string and these were all `false`.
    #[test]
    fn test_model_eq_uses_wire_id() {
        use strum::IntoEnumIterator;
        // Every known id equals its own wire string, in both directions, and
        // serde round-trips a bare wire string to the same `Anthropic` variant.
        for id in Id::iter() {
            let wire = id.name();
            let anthropic = Model::Anthropic(id);

            // PartialEq<S: AsRef<str>> for Model
            assert_eq!(anthropic, wire, "Anthropic({id:?}) == {wire:?}");
            // PartialEq<Id> for Model, exercised via a Custom carrying the wire
            // id — `Model::from` would normalize it, so build Custom directly.
            assert_eq!(
                Model::Custom(wire.into()),
                id,
                "Custom({wire:?}) == {id:?}"
            );
            // The wire string parses back to the known variant (not Custom).
            assert_eq!(Model::from(wire), anthropic, "from({wire:?})");
        }
    }

    // Some of these overlap, but it's fine.

    #[test]
    fn test_id_from_str() {
        let model: Model = "custom_model".into();
        assert_eq!(model, "custom_model");
    }

    #[test]
    fn test_id_conversion_from_anthropic_model() {
        let model: Model = Id::Sonnet35.into();
        assert_eq!(model, Id::Sonnet35);
    }

    #[test]
    fn test_id_conversion_from_str() {
        // custom model
        let model: Model = "custom_model".into();
        assert_eq!(model, "custom_model");

        // known model
        let model: Model = "claude-3-5-sonnet-latest".into();
        assert_eq!(model, Id::Sonnet35);

        // Claude 4
        let model: Model = "claude-opus-4-6".into();
        assert_eq!(model, Id::Opus46);
        let model: Model = "claude-haiku-4-5".into();
        assert_eq!(model, Id::Haiku45);
    }

    #[test]
    fn test_default_model() {
        assert_eq!(Id::default(), Id::Haiku45);
        assert_eq!(Model::default(), Id::Haiku45);
    }

    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "This test requires a real API key."]
    async fn test_ids_are_valid() {
        // Not probed live: RETIRED ids 404 for everyone *on the API* (the
        // whole Claude 3 family, verified 2026-06-11 — retirement differs
        // by surface: claude.ai un-retired Opus 3 by popular demand);
        // GATED ids exist but 404 on accounts without the entitlement
        // (Mythos is org-approved). A *typo'd* new variant still fails:
        // it's in neither list. When a model retires, move it here —
        // consciously.
        const RETIRED: &[Id] = &[
            Id::Haiku30,
            Id::Haiku35,
            Id::Haiku35_20241022,
            Id::Opus30,
            Id::Opus30_20240229,
            // Claude 4.0 dated snapshots retired 2026-06-15 (#118): the id is
            // still recognized (404 "model: …", not an unknown-model error) but
            // no longer served. The undated `-4-0` aliases resolve server-side
            // and stay live.
            Id::Opus40_20250514,
            Id::Sonnet40_20250514,
            Id::Sonnet35,
            Id::Sonnet35_20240620,
            Id::Sonnet35_20241022,
            Id::Sonnet37,
            Id::Sonnet37_20250219,
        ];
        const GATED: &[Id] = &[Id::Mythos5];

        let key = load_api_key().expect("API key not found");
        let client = Client::new(key).unwrap();

        let mut prompt = Prompt::default()
            .add_message((Role::User, "Emit just the \"🙏\" emoji, please."))
            .unwrap();

        for model in Id::iter() {
            if RETIRED.contains(&model) || GATED.contains(&model) {
                continue;
            }
            prompt.model = model.into();

            // If this fails (because a new model was added), it should be:
            // * added to the list of models above and
            // * the `latest` aliases should be updated
            // * the `name` method updated
            //
            // 15 sequential live calls — concurrently with the rest of the
            // `--ignored` suite — *will* see transient 429/529s, so retry
            // those rather than flake, backing off on `retry_after()` (see
            // `crate::utils::retry_transient`). A non-retryable error (e.g. a
            // genuine 404 for a typo'd/retired id) still surfaces here.
            let response =
                crate::utils::retry_transient(&format!("{model:?}"), || {
                    client.message(&prompt)
                })
                .await
                .unwrap_or_else(|e| panic!("{model:?}: {e}"));

            // Only date-pinned ids echo back verbatim; aliases (3.x
            // `-latest`, 4.x+ undated like `claude-opus-4-0`) resolve
            // server-side to whatever dated id is current.
            let pinned = Model::from(model)
                .name()
                .rsplit('-')
                .next()
                .is_some_and(|tail| {
                    tail.len() == 8 && tail.bytes().all(|b| b.is_ascii_digit())
                });
            if pinned {
                assert_eq!(response.model, model);
            }
        }
    }
}
