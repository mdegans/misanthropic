//! Structured output configuration for [`Prompt`](crate::Prompt).
//!
//! When [`Prompt::output_config`](crate::Prompt::output_config) is set, the
//! model's response is constrained by grammar-based decoding to a single
//! [`Text`](crate::prompt::message::Block::Text) [`Block`](crate::prompt::message::Block)
//! whose body conforms to the supplied JSON Schema. See the [Anthropic
//! structured outputs guide] for supported models, schema limitations, and
//! the [`Refusal`](crate::response::StopReason::Refusal) stop reason.
//!
//! [Anthropic structured outputs guide]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs>

use std::borrow::Cow;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// Structured output configuration for a [`Prompt`].
///
/// Constrains the model to emit a single [`Text`] [`Block`] matching the
/// configured [`OutputFormat`]. Changing this field invalidates the
/// [prompt cache] for the conversation thread — keep schemas stable across a
/// session when caching matters.
///
/// See the [Anthropic structured outputs guide] for supported models,
/// schema limitations, and the [`Refusal`] [`StopReason`] that can occur
/// when the model declines to produce structured output.
///
/// [`Prompt`]: crate::Prompt
/// [`Text`]: crate::prompt::message::Block::Text
/// [`Block`]: crate::prompt::message::Block
/// [`Refusal`]: crate::response::StopReason::Refusal
/// [`StopReason`]: crate::response::StopReason
/// [prompt cache]: <https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching>
/// [Anthropic structured outputs guide]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs>
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[non_exhaustive]
pub struct OutputConfig {
    /// Desired [`OutputFormat`] for the response. `None` leaves the response
    /// unconstrained — useful for an [`effort`](Self::effort)-only config.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<OutputFormat>,
    /// How eagerly the model spends tokens. `None` uses the API default
    /// ([`Effort::High`]). Orthogonal to [`format`](Self::format); see
    /// [`Effort`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub effort: Option<Effort>,
}

/// How eagerly the model spends tokens, set on [`OutputConfig::effort`].
///
/// Affects *all* output tokens — text, tool calls, and extended thinking — so
/// it works with or without [`Thinking`] enabled. On Claude 4 it is the
/// recommended way to control thinking depth, paired with
/// [`Thinking::adaptive`]. No beta header is required.
///
/// [`Thinking`]: crate::prompt::Thinking
/// [`Thinking::adaptive`]: crate::prompt::Thinking::adaptive
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Effort {
    /// Most efficient — significant token savings with some capability
    /// reduction. Good for simple or latency-sensitive tasks.
    Low,
    /// Balanced token savings. A solid default for agentic work.
    Medium,
    /// High capability. The API default — identical to omitting effort.
    High,
    /// Extended capability for long-horizon agentic and coding work. Only
    /// Opus 4.7 and newer. Pair with a large `max_tokens`.
    XHigh,
    /// Absolute maximum capability, no constraint on token spend.
    Max,
    /// A level this crate doesn't know — e.g. one Anthropic adds after this
    /// release. Like [`Id::Custom`](crate::model::Id::Custom), it round-trips
    /// over the wire, so a level read from a model's
    /// [`capabilities`](crate::model::Capabilities) can be sent right back on
    /// a request.
    Custom(Cow<'static, str>),
}

impl Effort {
    /// The wire string for this level, e.g. `"xhigh"`.
    pub fn as_str(&self) -> &str {
        match self {
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
            Effort::Custom(s) => s,
        }
    }
}

impl std::fmt::Display for Effort {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<Cow<'static, str>> for Effort {
    fn from(s: Cow<'static, str>) -> Self {
        match s.as_ref() {
            "low" => Effort::Low,
            "medium" => Effort::Medium,
            "high" => Effort::High,
            "xhigh" => Effort::XHigh,
            "max" => Effort::Max,
            _ => Effort::Custom(s),
        }
    }
}

impl From<&str> for Effort {
    fn from(s: &str) -> Self {
        Effort::from(Cow::Owned(s.to_owned()))
    }
}

impl From<String> for Effort {
    fn from(s: String) -> Self {
        Effort::from(Cow::Owned(s))
    }
}

impl Serialize for Effort {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Effort {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        // Owned: borrowing from the deserializer would tie `Effort`'s lifetime
        // to the input. Custom levels are rare, so the allocation is cheap;
        // borrow explicitly via [`Effort::from`] a `&str` when it matters.
        Ok(Effort::from(String::deserialize(deserializer)?))
    }
}

/// Format the response must conform to.
///
/// Currently only [`JsonSchema`] is supported upstream; the enum is
/// `#[non_exhaustive]` and tagged so new format variants can be added
/// without a major bump.
///
/// [`JsonSchema`]: OutputFormat::JsonSchema
#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    derive_more::IsVariant,
    derive_more::From,
)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum OutputFormat {
    /// Constrain output to a [JSON Schema].
    ///
    /// [JSON Schema]: <https://json-schema.org/>
    JsonSchema(JsonSchemaFormat),
}

/// Payload of [`OutputFormat::JsonSchema`].
///
/// See [Anthropic docs] for the supported schema subset (no recursive
/// schemas, no numeric range constraints, objects must set
/// `additionalProperties: false`, etc.).
///
/// [Anthropic docs]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[non_exhaustive]
pub struct JsonSchemaFormat {
    /// The JSON Schema to enforce.
    pub schema: serde_json::Value,
}

impl OutputConfig {
    /// Construct from a raw [JSON Schema] value. The schema is not
    /// validated by this crate — the caller is responsible for conformance
    /// with Anthropic's [supported subset].
    ///
    /// [JSON Schema]: <https://json-schema.org/>
    /// [supported subset]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
    pub fn json_schema(schema: serde_json::Value) -> Self {
        Self {
            format: Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema })),
            effort: None,
        }
    }

    /// An [`effort`]-only config, leaving the response [`format`]
    /// unconstrained.
    ///
    /// [`effort`]: Self::effort
    /// [`format`]: Self::format
    pub fn effort(effort: Effort) -> Self {
        Self {
            format: None,
            effort: Some(effort),
        }
    }

    /// Set the [`effort`](Self::effort), preserving the [`format`](Self::format).
    pub fn with_effort(mut self, effort: Effort) -> Self {
        self.effort = Some(effort);
        self
    }

    /// Overlay the set (`Some`) fields of `other` onto `self`, leaving
    /// `self`'s untouched. Lets the granular [`Prompt`] builders compose
    /// [`format`](Self::format) and [`effort`](Self::effort) in any order.
    ///
    /// [`Prompt`]: crate::Prompt
    pub(crate) fn overlay(&mut self, other: OutputConfig) {
        let OutputConfig { format, effort } = other;
        if format.is_some() {
            self.format = format;
        }
        if effort.is_some() {
            self.effort = effort;
        }
    }

    /// Construct from any type implementing [`JsonSchema`]. The generated
    /// schema is post-processed to match Anthropic's [supported subset]:
    /// objects get `additionalProperties: false`, and keywords that
    /// Anthropic rejects (numeric ranges, string lengths, etc.) are
    /// stripped. See [`sanitize_for_anthropic`] for the full list.
    ///
    /// [`JsonSchema`]: schemars::JsonSchema
    /// [supported subset]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
    /// [`sanitize_for_anthropic`]: self::sanitize_for_anthropic
    pub fn for_type<T: schemars::JsonSchema>() -> Self {
        let mut schema = serde_json::to_value(schemars::schema_for!(T))
            .expect("schemars Schema always serializes");
        sanitize_for_anthropic(&mut schema);
        Self::json_schema(schema)
    }
}

impl From<JsonSchemaFormat> for OutputConfig {
    fn from(format: JsonSchemaFormat) -> Self {
        Self {
            format: Some(OutputFormat::JsonSchema(format)),
            effort: None,
        }
    }
}

impl From<Effort> for OutputConfig {
    fn from(effort: Effort) -> Self {
        Self::effort(effort)
    }
}

impl From<serde_json::Value> for OutputConfig {
    /// Treats the value as a raw JSON Schema.
    fn from(schema: serde_json::Value) -> Self {
        Self::json_schema(schema)
    }
}

impl From<serde_json::Value> for JsonSchemaFormat {
    fn from(schema: serde_json::Value) -> Self {
        Self { schema }
    }
}

impl From<OutputFormat> for OutputConfig {
    fn from(format: OutputFormat) -> Self {
        Self {
            format: Some(format),
            effort: None,
        }
    }
}

/// Recursively transform a JSON Schema produced by [`schemars`] into the
/// subset Anthropic's structured output accepts. Exposed publicly so
/// callers who construct schemas manually can apply the same fixups; the
/// transform is idempotent and safe to re-apply.
///
/// Mutations, per [Anthropic's limits]:
///
/// * Adds `additionalProperties: false` to every object schema that
///   doesn't already set it — Anthropic requires it to be explicitly
///   `false` on all objects.
/// * Renames `oneOf` → `anyOf`. Anthropic supports `anyOf` but not
///   `oneOf`; for schemars-emitted enum schemas the variants are
///   mutually exclusive by construction so the two are semantically
///   equivalent on a per-value basis (any value that matches exactly
///   one subschema also matches at least one).
/// * Removes numeric constraints: `minimum`, `maximum`,
///   `exclusiveMinimum`, `exclusiveMaximum`, `multipleOf`.
/// * Removes string constraints: `minLength`, `maxLength`.
/// * Removes array constraints other than `minItems: 0 | 1`: `maxItems`,
///   `uniqueItems`, and `minItems` when it is outside `{0, 1}`.
///
/// Leaves supported keywords (`required`, `properties`, `items`, `enum`,
/// `const`, `anyOf`, `allOf`, `$ref`, `$defs`, `description`, `title`,
/// string `format`, `pattern`, `default`) untouched.
///
/// [`schemars`]: https://docs.rs/schemars
/// [Anthropic's limits]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
pub fn sanitize_for_anthropic(value: &mut serde_json::Value) {
    /// Keywords Anthropic rejects on any subschema.
    const UNSUPPORTED: &[&str] = &[
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
        "uniqueItems",
    ];

    match value {
        serde_json::Value::Object(map) => {
            for key in UNSUPPORTED {
                map.remove(*key);
            }
            // `minItems` is only supported for values 0 or 1; drop it
            // otherwise.
            let drop_min_items = map
                .get("minItems")
                .and_then(|v| v.as_u64())
                .is_some_and(|n| n > 1);
            if drop_min_items {
                map.remove("minItems");
            }
            // schemars emits `oneOf` for enum variants; Anthropic
            // accepts `anyOf` only. For mutually-exclusive subschemas
            // (the only shape schemars produces here) the two are
            // equivalent. If `anyOf` is already present we keep it and
            // drop the `oneOf`.
            if let Some(one_of) = map.remove("oneOf") {
                map.entry("anyOf").or_insert(one_of);
            }
            let is_object_schema = map
                .get("type")
                .and_then(|t| t.as_str())
                .is_some_and(|t| t == "object")
                || map.contains_key("properties");
            if is_object_schema && !map.contains_key("additionalProperties") {
                map.insert(
                    "additionalProperties".to_string(),
                    serde_json::Value::Bool(false),
                );
            }
            for v in map.values_mut() {
                sanitize_for_anthropic(v);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                sanitize_for_anthropic(item);
            }
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn serde_roundtrip() {
        let cfg = OutputConfig::json_schema(json!({
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"],
            "additionalProperties": false,
        }));
        let wire = serde_json::to_value(&cfg).unwrap();
        assert_eq!(
            wire,
            json!({
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": { "name": { "type": "string" } },
                        "required": ["name"],
                        "additionalProperties": false,
                    }
                }
            })
        );
        let back: OutputConfig = serde_json::from_value(wire).unwrap();
        assert_eq!(back, cfg);
    }

    #[test]
    fn effort_only_config_omits_format() {
        let cfg = OutputConfig::effort(Effort::Medium);
        assert_eq!(
            serde_json::to_value(&cfg).unwrap(),
            json!({ "effort": "medium" }),
            "effort-only config must not emit a format key"
        );
        assert_eq!(
            serde_json::from_value::<OutputConfig>(json!({"effort": "medium"}))
                .unwrap(),
            cfg
        );
    }

    #[test]
    fn effort_levels_serialize_lowercase() {
        for (effort, wire) in [
            (Effort::Low, "low"),
            (Effort::Medium, "medium"),
            (Effort::High, "high"),
            (Effort::XHigh, "xhigh"),
            (Effort::Max, "max"),
        ] {
            assert_eq!(
                serde_json::to_value(&effort).unwrap(),
                json!(wire),
                "{effort:?} should serialize as {wire:?}"
            );
        }
    }

    #[test]
    fn effort_custom_round_trips() {
        // An unknown level deserializes to `Custom`, serializes back verbatim,
        // and a known string still resolves to its unit variant.
        let custom: Effort = serde_json::from_value(json!("ultra")).unwrap();
        assert_eq!(custom, Effort::Custom("ultra".into()));
        assert_eq!(serde_json::to_value(&custom).unwrap(), json!("ultra"));
        assert_eq!(custom.as_str(), "ultra");

        let known: Effort = serde_json::from_value(json!("xhigh")).unwrap();
        assert_eq!(known, Effort::XHigh);

        // `Custom` is usable on a request, not just readable from a model.
        let cfg = OutputConfig::effort(Effort::from("ultra"));
        assert_eq!(
            serde_json::to_value(&cfg).unwrap(),
            json!({ "effort": "ultra" })
        );
    }

    #[test]
    fn overlay_composes_format_and_effort() {
        // format set first, effort overlaid: both survive.
        let mut cfg = OutputConfig::json_schema(json!({"type": "object"}));
        cfg.overlay(OutputConfig::effort(Effort::Low));
        assert!(cfg.format.is_some());
        assert_eq!(cfg.effort, Some(Effort::Low));

        // effort set first, format overlaid: both survive.
        let mut cfg = OutputConfig::effort(Effort::Max);
        cfg.overlay(OutputConfig::json_schema(json!({"type": "object"})));
        assert!(cfg.format.is_some());
        assert_eq!(cfg.effort, Some(Effort::Max));
    }

    #[test]
    fn from_value_treats_input_as_schema() {
        let schema = json!({"type": "object", "properties": {}});
        let cfg: OutputConfig = schema.clone().into();
        let Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema: inner })) =
            &cfg.format
        else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        assert_eq!(inner, &schema);
    }

    #[test]
    fn from_format() {
        let fmt = JsonSchemaFormat {
            schema: json!({"type": "object"}),
        };
        let cfg: OutputConfig = fmt.clone().into();
        let Some(OutputFormat::JsonSchema(got)) = &cfg.format else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        assert_eq!(got, &fmt);
    }

    #[test]
    fn is_variant_helper() {
        let cfg = OutputConfig::json_schema(json!({}));
        assert!(cfg.format.unwrap().is_json_schema());
    }

    #[test]
    fn for_type_emits_additional_properties_false() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Sample {
            name: String,
            count: u32,
        }

        let cfg = OutputConfig::for_type::<Sample>();
        let Some(OutputFormat::JsonSchema(JsonSchemaFormat { schema })) =
            &cfg.format
        else {
            panic!("expected json_schema format, got {:?}", cfg.format);
        };
        // The top-level object must have additionalProperties: false.
        assert_eq!(
            schema.get("additionalProperties"),
            Some(&serde_json::Value::Bool(false)),
            "schema missing additionalProperties: false — got {schema:#}"
        );
        // And the top-level object has the expected properties.
        let props = schema.get("properties").unwrap().as_object().unwrap();
        assert!(props.contains_key("name"));
        assert!(props.contains_key("count"));
    }

    /// Live end-to-end smoke test against the Anthropic API. Requires
    /// `api.key` in the crate root and the `client` feature. Run with
    /// `cargo test -p misanthropic --features client -- --ignored live`.
    ///
    /// Validates that the serialized `output_config` is accepted by the
    /// API and that the response round-trips through
    /// [`Message::json::<T>()`](crate::response::Message::json) into a
    /// typed struct.
    #[cfg(feature = "client")]
    #[tokio::test]
    #[ignore = "live — requires API key at misanthropic/api.key"]
    async fn live_structured_output_roundtrip() {
        use crate::prompt::message::{Content, Role};
        use crate::{AnthropicModel, Client, Prompt};

        #[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
        #[allow(dead_code)]
        struct CapitalFact {
            country: String,
            capital: String,
            population_millions: u32,
        }

        let key = crate::utils::load_api_key().await;
        let client = Client::new(key).unwrap();

        let prompt = Prompt::default()
            .model(AnthropicModel::Haiku45)
            .structured_output::<CapitalFact>()
            .add_message((
                Role::User,
                Content::text("Give me a capital-fact entry for France."),
            ))
            .unwrap();

        let response = client.message(&prompt).await.expect("API call");
        let fact: CapitalFact = response
            .json()
            .expect("json() should parse structured output");
        // We don't hard-code the answer — just assert the fields are
        // populated. The grammar guarantees shape, not content.
        assert_eq!(fact.country.to_lowercase(), "france");
        assert!(
            !fact.capital.is_empty(),
            "capital should be populated: {fact:?}"
        );
    }

    #[test]
    fn for_type_strips_unsupported_numeric_keywords() {
        // u32 makes schemars emit `minimum: 0`, which Anthropic rejects.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct NumericFields {
            count: u32,
            tolerance: f64,
        }

        let cfg = OutputConfig::for_type::<NumericFields>();
        let wire = serde_json::to_string(&cfg).unwrap();
        for kw in [
            "minimum",
            "maximum",
            "exclusiveMinimum",
            "exclusiveMaximum",
            "multipleOf",
            "minLength",
            "maxLength",
        ] {
            assert!(
                !wire.contains(&format!("\"{kw}\"")),
                "expected {kw:?} to be stripped, got {wire}"
            );
        }
    }

    #[test]
    fn for_type_rewrites_one_of_to_any_of_for_enums_with_descriptions() {
        // schemars emits `oneOf` when enum variants carry per-variant
        // metadata (doc comments become descriptions). Anthropic rejects
        // `oneOf` but accepts `anyOf`; `sanitize_for_anthropic` rewrites
        // the key. Plain unit-variant enums without docs emit a flat
        // `{"type": "string", "enum": [...]}` instead — no rewrite
        // needed there.
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        enum Category {
            /// A new feature.
            Feat,
            /// A bug fix.
            Fix,
            /// Internal rework.
            Refactor,
        }

        let cfg = OutputConfig::for_type::<Category>();
        let wire = serde_json::to_string(&cfg).unwrap();
        assert!(
            !wire.contains("\"oneOf\""),
            "oneOf must be rewritten to anyOf, got {wire}"
        );
        assert!(
            wire.contains("\"anyOf\""),
            "expected anyOf to replace oneOf, got {wire}"
        );
    }

    #[test]
    fn sanitize_preserves_any_of_when_both_present() {
        let mut schema = serde_json::json!({
            "anyOf": [{"type": "string"}],
            "oneOf": [{"type": "integer"}],
        });
        sanitize_for_anthropic(&mut schema);
        // `anyOf` wins, `oneOf` is dropped.
        assert_eq!(
            schema.get("anyOf").unwrap(),
            &serde_json::json!([{"type": "string"}]),
        );
        assert!(schema.get("oneOf").is_none());
    }

    #[test]
    fn sanitize_is_idempotent() {
        let mut schema = serde_json::json!({
            "type": "object",
            "properties": {
                "count": { "type": "integer", "minimum": 0 }
            },
            "required": ["count"],
        });
        sanitize_for_anthropic(&mut schema);
        let once = schema.clone();
        sanitize_for_anthropic(&mut schema);
        assert_eq!(schema, once, "sanitize_for_anthropic must be idempotent");
        // And the output should lack `minimum` and set additionalProperties.
        assert!(
            schema
                .to_string()
                .contains("\"additionalProperties\":false")
        );
        assert!(!schema.to_string().contains("\"minimum\""));
    }

    #[test]
    fn for_type_recurses_into_nested_objects() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Inner {
            x: i32,
        }
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Outer {
            inner: Inner,
        }

        let cfg = OutputConfig::for_type::<Outer>();
        let wire = serde_json::to_string(&cfg).unwrap();
        // Every object schema we emitted should carry the flag. Simpler to
        // assert on the string: the substring must appear for each object.
        let count = wire.matches("\"additionalProperties\":false").count();
        assert!(
            count >= 2,
            "expected additionalProperties:false on outer + inner, got {count} in {wire}"
        );
    }
}
