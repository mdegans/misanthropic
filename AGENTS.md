# AGENTS.md

**Read [`CLAUDE.md`](CLAUDE.md) first** — it is the project guide for every
coding agent here, not only Claude: how we work, the build/test gate (`just
test`), code style, and the wire-fixture discipline.

One rule agents break by reflex, repeated so it can't be missed:

- **No `serde_json::json!` in library code.** Build a typed `Serialize` value
  instead. Untyped JSON lets the wire shape drift without a compile error;
  typed values force every change through the compiler and the tests.
  `misanthropic/tests/no_json_macro.rs` enforces it. See *No `json!`* in
  `CLAUDE.md`.
