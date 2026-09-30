//! Compile-fail tests for `schema-order-check` in `#[derive(ToolArgs)]`.
//! Apart from `ui.rs` so the snapshots only run with the feature on;
//! regenerate with `TRYBUILD=overwrite`.
#![cfg(all(feature = "derive", feature = "schema-order-check"))]

#[test]
fn ui_order() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui-order/*.rs");
}
