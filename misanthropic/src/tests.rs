//! Crate-internal, cross-cutting test modules.
//!
//! Grouped under one `#[cfg(test)]` parent (declared in `lib.rs`) so submodules
//! that span several modules don't each need their own `cfg(test)` plumbing.

mod wire_coverage;
