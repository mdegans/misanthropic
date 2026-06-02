//! Compile-fail tests for the `derive` proc-macros. The `.stderr` snapshots
//! pin the macro's own diagnostics; regenerate with `TRYBUILD=overwrite`.
#![cfg(feature = "derive")]

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/*.rs");
}
