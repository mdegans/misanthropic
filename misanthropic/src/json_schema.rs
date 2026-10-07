//! Helpers for the `json-schema` feature: schema-only stand-ins for foreign
//! types (`schemars` implements none of `chrono`, `uuid`, `url` or `langsan`
//! without features we don't enable), each naming the JSON *wire* form serde
//! produces for that type, and [`Tag`] for the one serde representation
//! `schemars` doesn't model. Fields borrow the stand-ins with
//! `#[schemars(with = "…")]`; none is ever constructed.
//!
//! What "truthful" means here, per `schemars` contract: every value we
//! serialize validates against the serialize schema, and every value that
//! validates against the deserialize schema deserializes. The deserialize
//! schema may be *stricter* than serde (see [`Tag`]), never looser.

use std::borrow::Cow;

use schemars::{JsonSchema, Schema, SchemaGenerator, json_schema};

/// A string with a JSON Schema `format`: the shape `chrono`'s `DateTime`
/// (`date-time`), `uuid::Uuid` (`uuid`) and `url::Url` (`uri`) serialize to.
macro_rules! formatted_string {
    ($(#[$meta:meta])* $name:ident => $format:literal) => {
        $(#[$meta])*
        pub(crate) struct $name;

        impl JsonSchema for $name {
            fn inline_schema() -> bool {
                true
            }

            fn schema_name() -> Cow<'static, str> {
                stringify!($name).into()
            }

            fn json_schema(_: &mut SchemaGenerator) -> Schema {
                json_schema!({ "type": "string", "format": $format })
            }
        }
    };
}

formatted_string!(
    /// An RFC 3339 timestamp, as `chrono::DateTime<Utc>` serializes.
    DateTime => "date-time"
);
#[cfg(feature = "batch")]
formatted_string!(
    /// A hyphenated UUID, as `uuid::Uuid` serializes.
    Uuid => "uuid"
);
#[cfg(feature = "batch")]
formatted_string!(
    /// An absolute URL, as `url::Url` serializes.
    Uri => "uri"
);

/// A string that must equal `value` — a marker type that (de)serializes as a
/// fixed string, like a role marker or a server tool's `name`.
pub(crate) fn const_str(value: &str) -> Schema {
    json_schema!({ "type": "string", "const": value })
}

/// Adds the `"type": <tag>` property a `#[serde(tag = "type")]` *struct*
/// carries. `schemars` models internally tagged enums but not tagged structs,
/// so without this their schemas omit the discriminator. It is required, as
/// serialized; serde ignores the tag when deserializing a struct (any value, or
/// none, is accepted), so here the schema is stricter than the deserializer.
///
/// Use as `#[schemars(transform = Tag("…"))]`, with the same string as the
/// serde `rename`.
pub(crate) struct Tag(pub(crate) &'static str);

impl schemars::transform::Transform for Tag {
    fn transform(&mut self, schema: &mut Schema) {
        let tag = const_str(self.0).to_value();
        // First, to read like the wire, and rebuilt rather than inserted so
        // it lands first under either `serde_json::Map` backend.
        let properties = std::iter::once(("type".to_owned(), tag))
            .chain(
                schema
                    .get("properties")
                    .and_then(|p| p.as_object())
                    .into_iter()
                    .flatten()
                    .filter(|(name, _)| name.as_str() != "type")
                    .map(|(name, sub)| (name.clone(), sub.clone())),
            )
            .collect::<serde_json::Map<_, _>>();
        let required = std::iter::once("type".into())
            .chain(
                schema
                    .get("required")
                    .and_then(|r| r.as_array())
                    .into_iter()
                    .flatten()
                    .filter(|name| name.as_str() != Some("type"))
                    .cloned(),
            )
            .collect::<Vec<_>>();
        schema.insert("properties".into(), properties.into());
        schema.insert("required".into(), required.into());
    }
}

/// The `json-schema` feature's truthfulness gate: the generated schemas must
/// accept what serde actually produces and accepts.
///
/// No JSON Schema validator is in the dependency tree, so [`validate`] is a
/// small one covering the draft 2020-12 keywords `schemars` emits. It panics on
/// any keyword it doesn't know, so a schema can't pass by using a construct the
/// checker silently skips. Its own tests ([`validator`]) pin that it rejects.
///
/// Each schema is checked under both `schemars` contracts: the *deserialize*
/// schema against the raw wire JSON we accept (captured fixtures, hand-built
/// inputs), and both schemas against what we serialize.
#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use schemars::{JsonSchema, generate::SchemaSettings};
    use serde::{Serialize, de::DeserializeOwned};
    use serde_json::{Value, json};

    use crate::{
        Prompt,
        prompt::{
            CachedPrompt, Citation, Effort, OutputConfig, Thinking,
            message::{
                Block, CacheControl, CitationsConfig, Content, ContentText,
                DocumentSource, Image, MediaType, Message, Role, UserMessage,
            },
        },
        response, tool,
    };

    // --- The validator ----------------------------------------------------------

    /// Annotation keywords: no effect on validity.
    const ANNOTATIONS: &[&str] = &[
        "$schema",
        "$defs",
        "title",
        "description",
        "default",
        "examples",
        "format",
        "deprecated",
        "readOnly",
        "writeOnly",
    ];

    /// Validate `instance` against the root `schema`, returning every violation as
    /// `"<instance path>: <reason>"`.
    fn validate(schema: &Value, instance: &Value) -> Vec<String> {
        let mut errors = Vec::new();
        check(schema, schema, instance, "$", &mut errors);
        errors
    }

    fn check(
        root: &Value,
        schema: &Value,
        instance: &Value,
        path: &str,
        errors: &mut Vec<String>,
    ) {
        let schema = match schema {
            Value::Bool(true) => return,
            Value::Bool(false) => {
                errors
                    .push(format!("{path}: schema `false` rejects everything"));
                return;
            }
            Value::Object(schema) => schema,
            other => panic!("not a schema: {other}"),
        };

        for (keyword, value) in schema {
            let fail = |errors: &mut Vec<String>, why: String| {
                errors.push(format!("{path}: {why}"));
            };
            match keyword.as_str() {
                k if ANNOTATIONS.contains(&k) => {}
                "$ref" => {
                    let target = resolve(root, value.as_str().unwrap());
                    check(root, target, instance, path, errors);
                }
                "type" => {
                    let types: Vec<&str> = match value {
                        Value::String(t) => vec![t],
                        Value::Array(ts) => {
                            ts.iter().map(|t| t.as_str().unwrap()).collect()
                        }
                        other => panic!("bad `type`: {other}"),
                    };
                    if !types.iter().any(|t| is_type(instance, t)) {
                        fail(
                            errors,
                            format!("{instance} is not of type {types:?}"),
                        );
                    }
                }
                "enum" => {
                    if !value.as_array().unwrap().contains(instance) {
                        fail(errors, format!("{instance} not in enum {value}"));
                    }
                }
                "const" => {
                    if value != instance {
                        fail(errors, format!("{instance} != const {value}"));
                    }
                }
                "properties"
                | "required"
                | "additionalProperties"
                | "propertyNames" => {
                    let Some(object) = instance.as_object() else {
                        continue;
                    };
                    match keyword.as_str() {
                        "properties" => {
                            for (name, sub) in value.as_object().unwrap() {
                                if let Some(v) = object.get(name) {
                                    let p = format!("{path}.{name}");
                                    check(root, sub, v, &p, errors);
                                }
                            }
                        }
                        "required" => {
                            for name in value.as_array().unwrap() {
                                let name = name.as_str().unwrap();
                                if !object.contains_key(name) {
                                    fail(errors, format!("missing `{name}`"));
                                }
                            }
                        }
                        "additionalProperties" => {
                            let known = schema.get("properties");
                            for (name, v) in object {
                                if known.and_then(|k| k.get(name)).is_none() {
                                    let p = format!("{path}.{name}");
                                    check(root, value, v, &p, errors);
                                }
                            }
                        }
                        _ => {
                            for name in object.keys() {
                                let key = Value::String(name.clone());
                                let p = format!("{path}[key {name}]");
                                check(root, value, &key, &p, errors);
                            }
                        }
                    }
                }
                "items" | "minItems" | "maxItems" => {
                    let Some(array) = instance.as_array() else {
                        continue;
                    };
                    match keyword.as_str() {
                        "items" => {
                            for (i, v) in array.iter().enumerate() {
                                let p = format!("{path}[{i}]");
                                check(root, value, v, &p, errors);
                            }
                        }
                        "minItems" if array.len() < bound(value) => {
                            fail(errors, format!("fewer than {value} items"))
                        }
                        "maxItems" if array.len() > bound(value) => {
                            fail(errors, format!("more than {value} items"))
                        }
                        _ => {}
                    }
                }
                "minLength" => {
                    let Some(text) = instance.as_str() else {
                        continue;
                    };
                    if text.chars().count() < bound(value) {
                        fail(errors, format!("shorter than {value}"));
                    }
                }
                "minProperties" => {
                    let Some(object) = instance.as_object() else {
                        continue;
                    };
                    if object.len() < bound(value) {
                        fail(errors, format!("fewer than {value} properties"));
                    }
                }
                "minimum" | "maximum" => {
                    let Some(n) = instance.as_f64() else { continue };
                    let limit = value.as_f64().unwrap();
                    let ok = match keyword.as_str() {
                        "minimum" => n >= limit,
                        _ => n <= limit,
                    };
                    if !ok {
                        fail(errors, format!("{n} violates {keyword} {limit}"));
                    }
                }
                "allOf" | "anyOf" | "oneOf" => {
                    let passing = value
                        .as_array()
                        .unwrap()
                        .iter()
                        .filter(|sub| {
                            let mut sub_errors = Vec::new();
                            check(root, sub, instance, path, &mut sub_errors);
                            sub_errors.is_empty()
                        })
                        .count();
                    let total = value.as_array().unwrap().len();
                    let ok = match keyword.as_str() {
                        "allOf" => passing == total,
                        "anyOf" => passing >= 1,
                        _ => passing == 1,
                    };
                    if !ok {
                        fail(
                            errors,
                            format!(
                                "{passing}/{total} subschemas of `{keyword}` match"
                            ),
                        );
                    }
                }
                "not" => {
                    let mut sub_errors = Vec::new();
                    check(root, value, instance, path, &mut sub_errors);
                    if sub_errors.is_empty() {
                        fail(errors, "matches a `not` schema".into());
                    }
                }
                other => panic!("validator does not support `{other}`"),
            }
        }
    }

    fn bound(value: &Value) -> usize {
        value.as_u64().unwrap() as usize
    }

    fn is_type(instance: &Value, kind: &str) -> bool {
        match kind {
            "null" => instance.is_null(),
            "boolean" => instance.is_boolean(),
            "object" => instance.is_object(),
            "array" => instance.is_array(),
            "string" => instance.is_string(),
            "number" => instance.is_number(),
            "integer" => {
                instance.is_i64()
                    || instance.is_u64()
                    || instance.as_f64().is_some_and(|f| f.fract() == 0.0)
            }
            other => panic!("unknown type `{other}`"),
        }
    }

    /// Resolve a local `#/…` JSON pointer.
    fn resolve<'a>(root: &'a Value, reference: &str) -> &'a Value {
        let pointer = reference
            .strip_prefix('#')
            .unwrap_or_else(|| panic!("non-local $ref {reference}"));
        root.pointer(pointer)
            .unwrap_or_else(|| panic!("unresolved $ref {reference}"))
    }

    // --- Schemas ------------------------------------------------------------------

    fn deserialize_schema<T: JsonSchema>() -> Value {
        let settings = SchemaSettings::draft2020_12().for_deserialize();
        settings
            .into_generator()
            .into_root_schema_for::<T>()
            .to_value()
    }

    fn serialize_schema<T: JsonSchema>() -> Value {
        let settings = SchemaSettings::draft2020_12().for_serialize();
        settings
            .into_generator()
            .into_root_schema_for::<T>()
            .to_value()
    }

    #[track_caller]
    fn assert_valid(schema: &Value, instance: &Value, what: &str) {
        let errors = validate(schema, instance);
        assert!(
            errors.is_empty(),
            "{what} does not validate:\n  {}\ninstance: {instance:#}",
            errors.join("\n  ")
        );
    }

    #[track_caller]
    fn assert_invalid(schema: &Value, instance: &Value, what: &str) {
        assert!(
            !validate(schema, instance).is_empty(),
            "{what} validates but must not: {instance:#}"
        );
    }

    /// Serialize `value` and check the result against both contracts' schemas.
    #[track_caller]
    fn assert_serialized_valid<T: Serialize + JsonSchema>(
        value: &T,
        what: &str,
    ) {
        let json = serde_json::to_value(value).unwrap();
        assert_valid(&serialize_schema::<T>(), &json, what);
        assert_valid(&deserialize_schema::<T>(), &json, what);
    }

    /// Check raw wire JSON against the deserialize schema, then what serde makes of
    /// it, re-serialized, against the serialize schema.
    #[track_caller]
    fn assert_wire_valid<T>(raw: &Value, what: &str)
    where
        T: Serialize + DeserializeOwned + JsonSchema,
    {
        assert_valid(&deserialize_schema::<T>(), raw, what);
        let value: T = serde_json::from_value(raw.clone())
            .unwrap_or_else(|e| panic!("{what} does not deserialize: {e}"));
        let reserialized = serde_json::to_value(&value).unwrap();
        assert_valid(&serialize_schema::<T>(), &reserialized, what);
    }

    // --- Captured wire fixtures ---------------------------------------------------

    const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test/data");

    /// `(name, parsed JSON)` for every file in `dir` whose name passes `want`.
    fn fixtures(
        dir: &str,
        want: impl Fn(&str) -> bool,
    ) -> Vec<(String, Value)> {
        let mut out: Vec<(String, Value)> =
            fs::read_dir(Path::new(DATA).join(dir))
                .unwrap()
                .map(|entry| entry.unwrap().path())
                .filter(|p| p.file_name().unwrap().to_str().is_some_and(&want))
                .map(|p| {
                    let name =
                        p.file_name().unwrap().to_str().unwrap().to_owned();
                    let text = fs::read_to_string(&p).unwrap();
                    (name, serde_json::from_str(&text).unwrap())
                })
                .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// The `Ok` events of every wrapped SSE fixture (`*.sse.stream.jsonl`).
    fn stream_events() -> Vec<(String, Value)> {
        ["", "server_tools", "incremental"]
            .into_iter()
            .flat_map(|dir| {
                let dir_path = Path::new(DATA).join(dir);
                let mut files: Vec<_> = fs::read_dir(&dir_path)
                    .unwrap()
                    .map(|entry| entry.unwrap().path())
                    .filter(|p| {
                        p.to_str()
                            .is_some_and(|s| s.ends_with(".sse.stream.jsonl"))
                    })
                    .collect();
                files.sort();
                files
            })
            .flat_map(|path| {
                let name =
                    path.file_name().unwrap().to_str().unwrap().to_owned();
                fs::read_to_string(&path)
                    .unwrap()
                    .lines()
                    .enumerate()
                    .filter_map(|(i, line)| {
                        let wrapped: Value = serde_json::from_str(line).ok()?;
                        Some((
                            format!("{name}:{}", i + 1),
                            wrapped.get("Ok")?.clone(),
                        ))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[test]
    fn captured_requests_match_prompt_schema() {
        let requests = fixtures("requests", |n| n.ends_with(".json"));
        assert!(!requests.is_empty());
        for (name, raw) in requests {
            assert_wire_valid::<Prompt>(&raw, &name);
            assert_wire_valid::<CachedPrompt>(&raw, &name);
        }
        let raw = fixtures("", |n| n == "interleaved_tool.prompt.json");
        assert_eq!(raw.len(), 1);
        assert_wire_valid::<Prompt>(&raw[0].1, &raw[0].0);
    }

    #[test]
    fn captured_responses_match_message_schema() {
        let mut responses = fixtures("stop", |n| n.ends_with(".response.json"));
        responses.extend(fixtures("", |n| n.ends_with(".response.json")));
        assert!(responses.len() >= 4);
        for (name, raw) in responses {
            assert_wire_valid::<response::Message>(&raw, &name);
        }
    }

    #[test]
    fn captured_blocks_match_block_schema() {
        let blocks = fixtures("server_tools", |n| {
            n.ends_with(".json") && !n.ends_with(".jsonl")
        });
        assert!(blocks.len() >= 15);
        for (name, raw) in blocks {
            assert_wire_valid::<Block>(&raw, &name);
        }
    }

    /// Every event the API streamed, as a whole [`Event`](crate::stream::Event).
    #[test]
    fn streamed_events_match_event_schema() {
        let events = stream_events();
        assert!(events.len() >= 100, "only {} events", events.len());
        for (name, event) in events {
            assert_wire_valid::<crate::stream::Event>(&event, &name);
        }
    }

    /// Every block and message the API streamed: `content_block_start` blocks
    /// (thinking, redacted thinking, server-tool results, …) and the
    /// `message_start` message.
    #[test]
    fn streamed_blocks_and_messages_match_schemas() {
        let (mut blocks, mut messages) = (0, 0);
        for (name, event) in stream_events() {
            match event["type"].as_str() {
                Some("content_block_start") => {
                    assert_wire_valid::<Block>(&event["content_block"], &name);
                    blocks += 1;
                }
                Some("message_start") => {
                    assert_wire_valid::<response::Message>(
                        &event["message"],
                        &name,
                    );
                    messages += 1;
                }
                _ => {}
            }
        }
        assert!(blocks >= 20, "only {blocks} streamed blocks checked");
        assert!(messages >= 5, "only {messages} streamed messages checked");
    }

    // --- Representative values ------------------------------------------------------

    /// One of each block a caller builds (as opposed to the server-tool blocks the
    /// fixtures cover).
    fn sample_blocks() -> Vec<Block> {
        vec![
            Block::text("Hello."),
            Block::Text {
                text: "The grass is green.".into(),
                citations: Some(vec![
                    Citation::CharLocation {
                        cited_text: "The grass is green.".into(),
                        document_index: 0,
                        document_title: Some("Facts".into()),
                        start_char_index: 0,
                        end_char_index: 19,
                    },
                    Citation::PageLocation {
                        cited_text: "Water.".into(),
                        document_index: 1,
                        document_title: None,
                        start_page_number: 2,
                        end_page_number: 3,
                    },
                ]),
                cache_control: Some(CacheControl::one_hour()),
            },
            Block::Thought {
                thought: "Let me think.".into(),
                signature: "c2lnbmF0dXJl".into(),
            },
            Block::RedactedThought {
                signature: "ZW5jcnlwdGVk".into(),
            },
            Block::Image {
                image: Image::from_compressed(MediaType::Png, b"\x89PNG"),
                cache_control: None,
            },
            Block::Image {
                image: Image::from_url("https://example.com/cat.webp"),
                cache_control: Some(CacheControl::ephemeral()),
            },
            Block::document_with_citations(DocumentSource::from_base64(
                "JVBERi0=",
            )),
            Block::Document {
                source: DocumentSource::from_content(vec![ContentText {
                    text: "A chunk.".into(),
                }]),
                title: Some("Chunks".into()),
                context: Some("Context.".into()),
                citations: Some(CitationsConfig { enabled: false }),
                cache_control: None,
            },
            Block::document(DocumentSource::from_text("Plain text.")),
            Block::document(DocumentSource::from_url(
                "https://example.com/a.pdf",
            )),
            Block::document(DocumentSource::from_file_id("file_123")),
            Block::ToolUse {
                call: tool::Use::new("lookup", json!({"q": "rust"}))
                    .with_id("toolu_1")
                    .with_caller(tool::Caller::code_execution_20260120(
                        "srv_1",
                    )),
            },
            Block::ToolResult {
                result: tool::Result::new(
                    "toolu_1",
                    vec![Block::text("found"), Block::tool_reference("lookup")],
                )
                .error(),
            },
            Block::tool_reference("lookup"),
        ]
    }

    #[test]
    fn sample_blocks_match_block_schema() {
        for block in sample_blocks() {
            assert_serialized_valid(&block, &format!("{block:?}"));
        }
    }

    #[test]
    fn content_accepts_a_string_but_serializes_an_array() {
        let text = Value::String("Hi.".into());
        assert_valid(&deserialize_schema::<Content>(), &text, "string content");
        assert_invalid(&serialize_schema::<Content>(), &text, "string content");
        assert_serialized_valid(&Content::text("Hi."), "text content");
        assert_serialized_valid(&Content(sample_blocks()), "every block");
        assert_wire_valid::<Message>(
            &json!({"role": "user", "content": "Hi."}),
            "string-content message",
        );
    }

    /// A prompt exercising most fields: system content, every block kind, client
    /// and server tools, a tool choice, thinking, structured output and effort.
    #[test]
    fn full_prompt_matches_prompt_schema() {
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct Answer {
            answer: String,
        }

        let lookup = tool::CustomMethodDef::try_from(json!({
            "name": "lookup",
            "description": "Look something up.",
            "input_schema": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
            "strict": true,
            "allowed_callers": ["code_execution_20260120", "a_future_caller"],
        }))
        .unwrap();
        let mut prompt = Prompt::default()
            .system("Be brief.")
            .add_tool(lookup)
            .add_tool(tool::ServerMethodDef::web_search(tool::WebSearch {
                max_uses: Some(2),
                user_location: Some(tool::UserLocation {
                    city: Some("Lisbon".into()),
                    ..Default::default()
                }),
                ..Default::default()
            }))
            .add_tool(tool::ServerMethodDef::web_fetch(Default::default()))
            .add_tool(tool::ServerMethodDef::tool_search_regex())
            .add_tool(tool::ServerMethodDef::tool_search_bm25())
            .add_tool(tool::ServerMethodDef::code_execution())
            .tool_choice(
                tool::Choice::method("lookup").disable_parallel_tool_use(),
            )
            .thinking(Thinking::enabled(1024.try_into().unwrap()))
            .structured_output::<Answer>();
        prompt.output_config.as_mut().unwrap().effort = Some(Effort::XHigh);
        // Every block in one turn: not a valid conversation, so set directly
        // rather than through `add_message`'s turn-order checks.
        prompt.messages = vec![(Role::User, Content(sample_blocks())).into()];
        prompt.stop_sequences = Some(vec!["STOP".into()]);
        prompt.temperature = Some(0.5);
        prompt.top_k = Some(5.try_into().unwrap());
        prompt.metadata.insert("user_id".into(), "u1".into());

        assert_serialized_valid(&prompt, "full prompt");
        assert_serialized_valid(&CachedPrompt::from(prompt), "cached prompt");

        let custom_effort = OutputConfig::effort(Effort::from("ludicrous"));
        assert_serialized_valid(&custom_effort, "custom effort");
    }

    /// A `CustomMethodDef` deserializes through `build_structural`, so its
    /// deserialize schema carries those checks; a `MethodBuilder` doesn't.
    #[test]
    fn custom_method_def_schema_has_structural_checks() {
        let ok = json!({
            "name": "f",
            "description": "d",
            "input_schema": {"type": "object", "properties": {}},
        });
        let strict = deserialize_schema::<tool::CustomMethodDef>();
        assert_wire_valid::<tool::CustomMethodDef>(&ok, "custom tool");
        let loose = deserialize_schema::<tool::MethodBuilder>();
        for (field, bad) in [
            ("name", json!("")),
            ("description", json!("")),
            ("input_schema", json!({})),
            ("input_schema", json!("object")),
            ("input_schema", json!({"type": "object", "required": "q"})),
        ] {
            let mut tool = ok.clone();
            tool[field] = bad;
            assert_invalid(&strict, &tool, field);
            assert!(
                serde_json::from_value::<tool::CustomMethodDef>(tool.clone())
                    .is_err()
            );
            assert_valid(&loose, &tool, field);
        }
    }

    /// A `/v1/models` entry: flattened capability maps, a date, the `type`.
    #[test]
    fn model_info_matches_schema() {
        use crate::model::{ModelInfo, Models};

        let info = json!({
            "id": "claude-opus-4-6",
            "capabilities": {
                "batch": {"supported": true},
                "context_management": {
                    "clear_thinking_20251015": {"supported": true},
                    "supported": true,
                },
                "effort": {
                    "high": {"supported": true},
                    "a_future_level": {"supported": false},
                    "supported": true,
                },
                "thinking": {
                    "supported": true,
                    "types": {"adaptive": {"supported": true}},
                },
            },
            "created_at": "2026-02-04T00:00:00Z",
            "display_name": "Claude Opus 4.6",
            "max_input_tokens": 200000,
            "max_tokens": 64000,
            "type": "model",
        });
        assert_wire_valid::<ModelInfo>(&info, "model info");
        let models = json!({"data": [info], "has_more": false});
        assert_wire_valid::<Models>(&models, "models");

        let mut bad = models;
        bad["data"][0]["capabilities"]["effort"]["high"] = true.into();
        assert_invalid(&deserialize_schema::<Models>(), &bad, "bare bool");
    }

    #[test]
    fn user_message_role_is_pinned() {
        let schema = deserialize_schema::<UserMessage>();
        assert_valid(&schema, &json!({"role": "user", "content": []}), "user");
        assert_invalid(
            &schema,
            &json!({"role": "assistant", "content": []}),
            "assistant as a UserMessage",
        );
    }

    #[cfg(feature = "batch")]
    #[test]
    fn batch_request_and_meta_match_schemas() {
        use crate::batch::{Meta, Prompts};

        let prompts: Prompts<Prompt> =
            [Prompt::default().add_message((Role::User, "Hi.")).unwrap()]
                .into_iter()
                .collect();
        assert_serialized_valid(&prompts, "batch requests");
        let schema = serialize_schema::<Prompts<Prompt>>();
        assert_invalid(
            &schema,
            &json!({"requests": [{"params": {}}]}),
            "no id",
        );

        let meta = json!({
            "type": "message_batch",
            "id": "msgbatch_01",
            "processing_status": "ended",
            "request_counts": {
                "processing": 0, "succeeded": 1, "errored": 0,
                "canceled": 0, "expired": 0,
            },
            "created_at": "2024-09-24T18:37:24.100435Z",
            "expires_at": "2024-09-25T18:37:24.100435Z",
            "ended_at": "2024-09-24T18:39:24.100435Z",
            "cancel_initiated_at": null,
            "archived_at": null,
            "results_url": "https://api.anthropic.com/v1/messages/batches/msgbatch_01/results",
        });
        assert_wire_valid::<Meta>(&meta, "batch meta");
    }

    #[test]
    fn anthropic_errors_match_schema() {
        use crate::client::AnthropicError;

        for raw in [
            json!({"type": "invalid_request_error", "message": "bad"}),
            json!({"type": "rate_limit_error", "message": "slow down"}),
            json!({"type": "a_future_error", "message": "new"}),
        ] {
            assert_wire_valid::<AnthropicError>(&raw, "anthropic error");
        }
        let error = AnthropicError::RateLimit {
            message: "slow down".into(),
            retry_after: Some(3),
        };
        assert_serialized_valid(&error, "rate limit with retry_after");
    }

    // --- The schemas aren't vacuous ----------------------------------------------

    #[test]
    fn block_schema_rejects_malformed_blocks() {
        let schema = deserialize_schema::<Block>();
        for (bad, why) in [
            (json!({"type": "thinking", "signature": "x"}), "no thinking"),
            (json!({"type": "nonsense", "text": "x"}), "unknown type"),
            (json!({"type": "text"}), "no text"),
            (json!({"type": "text", "text": 1}), "non-string text"),
            (
                json!({"type": "image", "source": {"type": "base64", "data": "x",
                       "media_type": "image/bmp"}}),
                "unsupported media type",
            ),
            (
                json!({"type": "tool_use", "name": "f", "input": {}}),
                "no id",
            ),
            (json!("text"), "a bare string"),
            (
                json!({"type": "document", "source": {"type": "content",
                       "content": [{"text": "x"}]}}),
                "a content chunk without its `type` tag",
            ),
        ] {
            assert_invalid(&schema, &bad, why);
        }
    }

    #[test]
    fn prompt_schema_rejects_malformed_prompts() {
        let schema = deserialize_schema::<Prompt>();
        let ok = json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hi."}],
        });
        assert_valid(&schema, &ok, "minimal prompt");
        for field in ["model", "max_tokens", "messages"] {
            let mut bad = ok.clone();
            bad.as_object_mut().unwrap().remove(field);
            assert_invalid(&schema, &bad, field);
        }
        let mut bad = ok.clone();
        bad["messages"][0]["role"] = "narrator".into();
        assert_invalid(&schema, &bad, "unknown role");
        let mut bad = ok;
        bad["max_tokens"] = 0.into();
        assert_invalid(&schema, &bad, "zero max_tokens");
    }

    /// The validator itself: it must reject, or the tests above prove nothing.
    mod validator {
        use super::*;

        #[test]
        fn rejects() {
            let schema = json!({
                "$defs": {"Name": {"type": "string", "enum": ["a", "b"]}},
                "type": "object",
                "properties": {
                    "name": {"$ref": "#/$defs/Name"},
                    "n": {"type": "integer", "minimum": 0},
                    "tag": {"const": "t"},
                    "list": {"type": "array", "items": {"type": "boolean"},
                             "minItems": 1},
                    "one": {"oneOf": [{"type": "string"}, {"const": "x"}]},
                },
                "required": ["name"],
                "additionalProperties": false,
            });
            assert_valid(&schema, &json!({"name": "a", "n": 1}), "ok");
            for (bad, why) in [
                (json!({}), "missing"),
                (json!({"name": "c"}), "enum through $ref"),
                (json!({"name": "a", "n": -1}), "minimum"),
                (json!({"name": "a", "n": 1.5}), "integer"),
                (json!({"name": "a", "tag": "u"}), "const"),
                (json!({"name": "a", "list": []}), "minItems"),
                (json!({"name": "a", "list": [1]}), "items"),
                (json!({"name": "a", "one": "x"}), "oneOf matching twice"),
                (json!({"name": "a", "extra": 1}), "additionalProperties"),
                (json!([]), "type"),
            ] {
                assert_invalid(&schema, &bad, why);
            }
        }

        #[test]
        #[should_panic(expected = "does not support `pattern`")]
        fn refuses_unknown_keywords() {
            validate(&json!({"pattern": "^a"}), &json!("a"));
        }
    }
}
