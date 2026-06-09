#[cold]
#[inline(always)]
pub(crate) fn cold_path() {}

/// Assert an exact serde round-trip of a wire fixture, and return the parsed
/// value for further assertions.
///
/// `fixture` is raw JSON captured from the API (see `test/data/README.md` for
/// the capture discipline). This deserializes it into `T`, re-serializes, and
/// asserts the result equals the captured input at the value level.
///
/// No response type uses `#[serde(deny_unknown_fields)]` — an undocumented,
/// renamed, or mis-tagged wire field is therefore *silently dropped* on the way
/// in. This assertion is the offline guard that catches that: the
/// re-serialized value would be missing the field (or carry a wrong tag) and
/// the comparison fails loudly in `just test`, with no API call.
#[cfg(test)]
pub(crate) fn roundtrip<T>(fixture: &str) -> T
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    roundtrip_checked(fixture).unwrap_or_else(|e| {
        panic!(
            "{e}\nNo response type uses deny_unknown_fields, so this \
             assertion is the only offline guard against wire drift."
        )
    })
}

/// The non-panicking core of [`roundtrip`]: returns the mismatch as an `Err`
/// instead of asserting, so a caller iterating many fixtures (the
/// [`wire_coverage`](crate::tests) gate) can collect *every* failure and report
/// them at once rather than dying on the first.
#[cfg(test)]
pub(crate) fn roundtrip_checked<T>(fixture: &str) -> Result<T, String>
where
    T: serde::Serialize + serde::de::DeserializeOwned,
{
    let captured: serde_json::Value = serde_json::from_str(fixture)
        .map_err(|e| format!("invalid JSON: {e}"))?;
    let parsed: T = serde_json::from_value(captured.clone()).map_err(|e| {
        format!("does not deserialize into the target type: {e}")
    })?;
    let reserialized = serde_json::to_value(&parsed)
        .map_err(|e| format!("re-serialize: {e}"))?;
    if !value_equal_modulo_nulls(&reserialized, &captured) {
        return Err("wire round-trip mismatch: a field was dropped, renamed, \
                    added, or mis-tagged"
            .to_string());
    }
    Ok(parsed)
}

/// Value equality that treats an *explicit `null`* and an *absent key* as the
/// same thing (recursively).
///
/// The wire sends some fields as explicit `null` (`stop_details`, `page_age`)
/// while older API versions omit them; modeling that faithfully would force a
/// double-`Option` on every such field for zero information gain — dropping a
/// `null` loses nothing. So fixture comparisons strip null-valued keys from
/// both sides first. The guard still fires the moment the API populates the
/// field with a real value our types drop.
#[cfg(test)]
pub(crate) fn value_equal_modulo_nulls(
    a: &serde_json::Value,
    b: &serde_json::Value,
) -> bool {
    fn strip(v: &serde_json::Value) -> serde_json::Value {
        match v {
            serde_json::Value::Object(map) => serde_json::Value::Object(
                map.iter()
                    .filter(|(_, v)| !v.is_null())
                    .map(|(k, v)| (k.clone(), strip(v)))
                    .collect(),
            ),
            serde_json::Value::Array(arr) => {
                serde_json::Value::Array(arr.iter().map(strip).collect())
            }
            other => other.clone(),
        }
    }
    strip(a) == strip(b)
}

/// One line of a wrapped SSE-JSONL fixture (`{"Ok": <event>}` /
/// `{"Err": <error event>}`, see `test/data/README.md`), round-tripped. The
/// streaming analogue of a single [`roundtrip`] call; collected (never
/// panicked) so [`SseFixture`] can report *every* bad line at once rather than
/// dying on the first.
#[cfg(test)]
pub(crate) struct SseLine {
    /// 1-based line number, for the failure report.
    pub line: usize,
    /// The parsed line — an `Ok` [`Event`](crate::stream::Event) or an `Err`
    /// [`ErrorEvent`](crate::stream::ErrorEvent) (both arms are *typed*, so
    /// captured error frames get real parse coverage) — or the deserialize
    /// error.
    pub parsed:
        Result<Result<crate::stream::Event, crate::stream::ErrorEvent>, String>,
    /// Whether the re-serialized event is value-equal to the captured line.
    /// This is the **gate** — a `false` means a field was dropped, renamed, or
    /// mis-tagged (the same drift [`roundtrip`] guards against, per line).
    pub value_equal: bool,
    /// Whether the re-serialization is *byte*-identical to the captured line
    /// (compact form, same key order). **Informational only — never asserted.**
    /// Usually `false` (the wire's key order rarely matches serde's declaration
    /// order); a `true` just means ordering happened to be stable for that line.
    pub byte_stable: bool,
}

/// A captured SSE-JSONL stream fixture, round-tripped line by line. Returned by
/// [`roundtrip_sse`]; assert it with [`assert_round_trips`](Self::assert_round_trips)
/// and take the parsed events with [`into_events`](Self::into_events).
#[cfg(test)]
pub(crate) struct SseFixture {
    /// Per-line round-trip results, in file order.
    pub lines: Vec<SseLine>,
}

#[cfg(test)]
impl SseFixture {
    /// Panic **once** with every bad line (a parse error or a value mismatch),
    /// not just the first — so one run surfaces the whole picture. Byte-
    /// stability is reported as an aggregate note and never fails. Returns
    /// `self` for chaining.
    pub fn assert_round_trips(&self) -> &Self {
        let problems: Vec<String> = self
            .lines
            .iter()
            .filter_map(|l| match &l.parsed {
                Err(e) => Some(format!("  line {}: parse error: {e}", l.line)),
                Ok(_) if !l.value_equal => Some(format!(
                    "  line {}: wire round-trip mismatch (a field was dropped, \
                     renamed, added, or mis-tagged)",
                    l.line
                )),
                Ok(_) => None,
            })
            .collect();
        assert!(
            problems.is_empty(),
            "SSE-JSONL round-trip failed for {} of {} lines:\n{}\n(no Event \
             type uses deny_unknown_fields, so this value-equality check is the \
             only offline guard against streaming wire drift)",
            problems.len(),
            self.lines.len(),
            problems.join("\n"),
        );
        self
    }

    /// Consume into the parsed `Ok`/`Err` lines, in order, for assembly
    /// assertions (feed them through the `stream` combinators, or assemble
    /// manually). `Event` isn't `Clone`, so this consumes the fixture. Panics
    /// if any line failed to parse — call
    /// [`assert_round_trips`](Self::assert_round_trips) first.
    pub fn into_results(
        self,
    ) -> Vec<Result<crate::stream::Event, crate::stream::ErrorEvent>> {
        self.lines
            .into_iter()
            .map(|l| l.parsed.expect("line parsed (call assert first)"))
            .collect()
    }

    /// [`into_results`](Self::into_results), keeping only the `Ok` events —
    /// the common case for assembly when a fixture has no error frames.
    pub fn into_events(self) -> Vec<crate::stream::Event> {
        self.into_results().into_iter().flatten().collect()
    }

    /// How many lines re-serialized byte-identically — the informational
    /// "ordering is stable" signal. Never gated on.
    pub fn byte_stable_count(&self) -> usize {
        self.lines.iter().filter(|l| l.byte_stable).count()
    }
}

/// Round-trip every line of a captured, **wrapped** SSE-JSONL fixture — one
/// `{"Ok": <event>}` / `{"Err": <error event>}` per line, raw wire bytes
/// inside the wrapper (see `test/data/README.md`) — returning the per-line
/// results. The streaming twin of [`roundtrip`]: same load-bearing
/// re-serialize-and-compare guard, applied per line, but collected rather than
/// asserted so the caller (via [`SseFixture::assert_round_trips`]) can report
/// all failures at once. Blank lines are skipped.
#[cfg(test)]
pub(crate) fn roundtrip_sse(fixture: &str) -> SseFixture {
    type Parsed = Result<crate::stream::Event, crate::stream::ErrorEvent>;

    let lines = fixture
        .lines()
        .enumerate()
        .filter(|(_, raw)| !raw.trim().is_empty())
        .map(|(i, raw)| {
            let line = i + 1;
            let captured: serde_json::Value = serde_json::from_str(raw)
                .expect("SSE-JSONL line is valid JSON");
            match serde_json::from_value::<Parsed>(captured.clone()) {
                Ok(parsed) => {
                    let reser = serde_json::to_value(&parsed)
                        .expect("line re-serializes to JSON");
                    let value_equal =
                        value_equal_modulo_nulls(&reser, &captured);
                    let byte_stable = serde_json::to_string(&parsed)
                        .map(|s| s == raw.trim())
                        .unwrap_or(false);
                    SseLine {
                        line,
                        parsed: Ok(parsed),
                        value_equal,
                        byte_stable,
                    }
                }
                Err(e) => SseLine {
                    line,
                    parsed: Err(e.to_string()),
                    value_equal: false,
                    byte_stable: false,
                },
            }
        })
        .collect();
    SseFixture { lines }
}

#[cfg(all(test, feature = "client"))]
pub(crate) const CRATE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

// Load the API key from the `api.key` file in the crate root.
#[cfg(all(test, feature = "client"))]
pub(crate) async fn load_api_key() -> String {
    use std::path::Path;

    let path = Path::new(CRATE_ROOT).join("api.key");
    tokio::fs::read_to_string(path)
        .await
        .ok()
        .unwrap()
        .trim()
        .to_string()
}
