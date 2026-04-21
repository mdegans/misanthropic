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

use serde::{Deserialize, Serialize};

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
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(any(feature = "partial-eq", test), derive(PartialEq))]
#[non_exhaustive]
pub struct OutputConfig {
    /// Desired [`OutputFormat`] for the response.
    pub format: OutputFormat,
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
            format: OutputFormat::JsonSchema(JsonSchemaFormat { schema }),
        }
    }

    /// Construct from any type implementing [`JsonSchema`]. The generated
    /// schema is post-processed to match Anthropic's [supported subset]:
    /// objects get `additionalProperties: false`, and keywords that
    /// Anthropic rejects (numeric ranges, string lengths, etc.) are
    /// stripped. See [`sanitize_for_anthropic`] for the full list.
    ///
    /// Requires the `json-schema` feature.
    ///
    /// [`JsonSchema`]: schemars::JsonSchema
    /// [supported subset]: <https://docs.anthropic.com/en/docs/build-with-claude/structured-outputs#json-schema-limitations>
    /// [`sanitize_for_anthropic`]: self::sanitize_for_anthropic
    #[cfg(feature = "json-schema")]
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
            format: OutputFormat::JsonSchema(format),
        }
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
        Self { format }
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
#[cfg(feature = "json-schema")]
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
    fn from_value_treats_input_as_schema() {
        let schema = json!({"type": "object", "properties": {}});
        let cfg: OutputConfig = schema.clone().into();
        let OutputFormat::JsonSchema(JsonSchemaFormat { schema: inner }) =
            &cfg.format;
        assert_eq!(inner, &schema);
    }

    #[test]
    fn from_format() {
        let fmt = JsonSchemaFormat {
            schema: json!({"type": "object"}),
        };
        let cfg: OutputConfig = fmt.clone().into();
        let OutputFormat::JsonSchema(got) = &cfg.format;
        assert_eq!(got, &fmt);
    }

    #[test]
    fn is_variant_helper() {
        let cfg = OutputConfig::json_schema(json!({}));
        assert!(cfg.format.is_json_schema());
    }

    #[cfg(feature = "json-schema")]
    #[test]
    fn for_type_emits_additional_properties_false() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Sample {
            name: String,
            count: u32,
        }

        let cfg = OutputConfig::for_type::<Sample>();
        let OutputFormat::JsonSchema(JsonSchemaFormat { schema }) = &cfg.format;
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
    /// `api.key` in the crate root and the `json-schema` + `client`
    /// features. Run with `cargo test -p misanthropic --features
    /// json-schema -- --ignored live`.
    ///
    /// Validates that the serialized `output_config` is accepted by the
    /// API and that the response round-trips through
    /// [`Message::json::<T>()`](crate::response::Message::json) into a
    /// typed struct.
    #[cfg(all(feature = "json-schema", feature = "client"))]
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

    #[cfg(feature = "json-schema")]
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

    #[cfg(feature = "json-schema")]
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

    #[cfg(feature = "json-schema")]
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
