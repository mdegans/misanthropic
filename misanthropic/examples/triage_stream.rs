//! Example: incremental structured output via [`FilterExt::json_items`].
//! Several bug reports are triaged in one call as an [`Items`]`<Triage>`,
//! and each `Triage` prints the moment its closing brace arrives on the wire
//! — not at end of turn. The non-streaming, few-shot sibling is
//! `few_shot_triage`.
//!
//! ```sh
//! cargo run --features client --example triage_stream -- \
//!     "Search returns no results since this morning." \
//!     "Dark mode resets to light on every page load."
//! ```
//!
//! [`FilterExt::json_items`]: misanthropic::stream::FilterExt::json_items
//! [`Items`]: misanthropic::prompt::Items

mod utils;

use clap::Parser;
use futures::TryStreamExt;
use misanthropic::{
    Client, Id, Prompt, prompt::Items, prompt::message::Role, stream::FilterExt,
};
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
/// Field order is generation order — each field is context for the next.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct Triage {
    /// One-line, imperative summary of the underlying problem.
    summary: String,
    /// How bad it is, chosen after summarizing.
    severity: Severity,
    /// The component most likely at fault (e.g. "auth-ui", "checkout-api").
    component: String,
    /// True when the report indicates the behavior regressed.
    is_regression: bool,
}

/// Triage a queue of bug reports, printing each as it streams in.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// Bug reports to triage. Defaults are used if omitted.
    reports: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let cli = Cli::parse();
    utils::log_init(cli.common.verbose);
    let client = Client::new(utils::api_key()?)?;

    let reports = if cli.reports.is_empty() {
        vec![
            "Checkout total shows $0.00 even though the cart has items. \
             Only happens on mobile Safari."
                .to_string(),
            "Password reset emails arrive after 30+ minutes. Started after \
             the queue migration last Tuesday."
                .to_string(),
            "The settings page footer overlaps the save button on narrow \
             windows."
                .to_string(),
        ]
    } else {
        cli.reports
    };

    let numbered = reports
        .iter()
        .enumerate()
        .map(|(i, report)| format!("{}. {report}", i + 1))
        .collect::<Vec<_>>()
        .join("\n");

    let prompt = cli
        .common
        .configure(
            Prompt::default()
                .model(Id::Haiku45)
                .system(
                    "You triage incoming bug reports into a structured \
                     form, one triage per numbered report, in order.",
                )
                .structured_output::<Items<Triage>>(),
        )
        .add_message((Role::User, numbered))?;

    // `json_items` yields each completed element of the outermost array as
    // soon as its bytes arrive — the `Items` wrapper exists because the API
    // requires a top-level object schema.
    let stream = client.stream(&prompt).await?;
    let triages = stream.json_items::<Triage>();
    futures::pin_mut!(triages);

    while let Some(triage) = triages.try_next().await? {
        println!("{triage:#?}");
    }

    Ok(())
}
