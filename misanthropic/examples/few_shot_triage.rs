//! Example: *few-shot* structured output. Triage a free-text bug report into a
//! structured `Triage` using [`Prompt::with_examples`] for priming.
//!
//! The win over zero-shot [`Prompt::structured_output`]: one or two
//! schema-conformant exemplars in the history teach the model the *depth* of
//! field population you want — here, that `repro_steps` should be concrete and
//! non-empty, and `is_regression` should be inferred from phrasing like
//! "started after the last release". Without exemplars, smaller models tend to
//! return a single vague repro step or leave `is_regression` at its default.
//!
//! [`with_examples`] pulls double duty: each `(input, output)` pair becomes a
//! [`Role::User`] turn followed by a [`Role::Assistant`] turn (the exemplar
//! serialized to JSON), *and* the exemplar type `Triage` seeds the
//! [`output_config`] schema — so there is no separate
//! `structured_output::<Triage>()` call and the constraint can never drift
//! from the examples.
//!
//! # Usage
//!
//! ```sh
//! cargo run --features client --example few_shot_triage -- \
//!     "Search returns no results for any query since this morning."
//! ```
//!
//! With no argument a sample report is triaged. Expects `ANTHROPIC_API_KEY` in
//! the environment, or prompts on stdin.
//!
//! [`Prompt::with_examples`]: misanthropic::Prompt::with_examples
//! [`with_examples`]: misanthropic::Prompt::with_examples
//! [`Prompt::structured_output`]: misanthropic::Prompt::structured_output
//! [`output_config`]: misanthropic::Prompt::output_config

mod utils;

use clap::Parser;
use misanthropic::{Client, Id, Prompt, prompt::message::Role};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Triage severity. Generated first so it anchors the rest of the triage.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[schemars(rename_all = "snake_case")]
enum Severity {
    /// Cosmetic or low-impact.
    Low,
    /// Degraded but workable.
    Medium,
    /// Major feature broken for many users.
    High,
    /// Outage, data loss, or security issue.
    Critical,
}

/// Structured triage of a free-text bug report.
///
/// Derives both [`Serialize`] (to render exemplars into assistant turns) and
/// [`Deserialize`] (to parse the model's response), plus [`JsonSchema`] for the
/// output constraint. Field order is the generation order, so each field is
/// available as context for the next.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Triage {
    /// One-line, imperative summary of the underlying problem.
    summary: String,
    /// How bad it is, chosen after summarizing.
    severity: Severity,
    /// The component most likely at fault (e.g. "auth-ui", "checkout-api").
    component: String,
    /// Concrete, ordered steps to reproduce. Never leave empty when the report
    /// implies them — infer the obvious steps a tester would follow.
    repro_steps: Vec<String>,
    /// True when the report indicates the behavior regressed (e.g. "worked
    /// last week", "started after the update").
    is_regression: bool,
}

/// Triage a free-text bug report into a structured form.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// The bug report to triage. A default report is used if omitted.
    report: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    #[cfg(feature = "log")]
    env_logger::init();

    let cli = Cli::parse();
    let client = Client::new(utils::api_key()?)?;

    // The report to triage: CLI arg, or a default.
    let report = cli.report.unwrap_or_else(|| {
        "Checkout total shows $0.00 even though the cart has items. Only \
         happens on mobile Safari. A few customers reported it today."
            .to_string()
    });

    // Two fully-populated exemplars prime the *depth* of the output: rich
    // `repro_steps` and a correctly-inferred `is_regression`. The output schema
    // is taken from `Triage` by `with_examples`, so no separate
    // `structured_output::<Triage>()` is needed.
    let base =
        cli.common
            .configure(Prompt::default().model(Id::Haiku45).set_system(
                "You triage incoming bug reports into a structured form. \
                 Infer concrete reproduction steps and whether the issue is \
                 a regression from the wording of the report.",
            ));
    let prompt = base
        .with_examples([
            (
                "Safari users say the login button does nothing when clicked. \
                 Started after last week's release.",
                Triage {
                    summary: "Login button unresponsive on Safari".into(),
                    severity: Severity::High,
                    component: "auth-ui".into(),
                    repro_steps: vec![
                        "Open the app in Safari 17".into(),
                        "Click the 'Log in' button".into(),
                        "Observe: nothing happens; no network request fires"
                            .into(),
                    ],
                    is_regression: true,
                },
            ),
            (
                "Uploads over about 5 MB silently fail on mobile, no error \
                 is shown to the user.",
                Triage {
                    summary: "Large uploads fail silently on mobile".into(),
                    severity: Severity::Medium,
                    component: "upload-service".into(),
                    repro_steps: vec![
                        "On a mobile browser, open the upload dialog".into(),
                        "Select a file larger than 5 MB".into(),
                        "Submit and observe: the upload fails with no error"
                            .into(),
                    ],
                    is_regression: false,
                },
            ),
        ])?
        .add_message((Role::User, report))?;

    let triage: Triage = client.message(&prompt).await?.json()?;

    println!("{triage:#?}");
    Ok(())
}
