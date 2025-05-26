//! Integration test for Memory Palace with real Anthropic API calls.
//! This validates that agents can effectively use the Memory Palace tool.

use clap::Parser;
use misanthropic::{
    AnthropicModel, Client, Prompt,
    prompt::{Message, message::Role},
    tool::{Tool, ToolBox},
};
use std::env;

#[cfg(feature = "memory-palace")]
use misanthropic::tool::MemoryPalace;

/// Test Memory Palace agent interactions
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// API key (or set ANTHROPIC_API_KEY env var)
    #[arg(long)]
    api_key: Option<String>,
    /// Archive test conversations to this directory
    #[arg(long, default_value = "test_conversations")]
    archive_dir: String,
    /// Number of test scenarios to run
    #[arg(long, default_value = "3")]
    scenarios: usize,
}

/// Test scenarios for the Memory Palace
#[derive(Debug)]
struct TestScenario {
    name: &'static str,
    user_messages: &'static [&'static str],
    expected_tool_calls: &'static [&'static str], // Expected tool method names
    success_indicators: &'static [&'static str], // Phrases that indicate success
}

impl TestScenario {
    const SCENARIOS: &'static [TestScenario] = &[
        TestScenario {
            name: "basic_store_and_retrieve",
            user_messages: &[
                "Please remember that my favorite color is blue and I live in San Francisco.",
                "What's my favorite color and where do I live?",
            ],
            expected_tool_calls: &[
                "MemoryPalace::store",
                "MemoryPalace::search",
            ],
            success_indicators: &["blue", "San Francisco"],
        },
        TestScenario {
            name: "knowledge_organization",
            user_messages: &[
                "Store this scientific fact: Water boils at 100°C at sea level.",
                "Also remember: The speed of light is 299,792,458 m/s.",
                "Now create a relationship between these two physics facts.",
                "What physics facts do you know about?",
            ],
            expected_tool_calls: &[
                "MemoryPalace::store",
                "MemoryPalace::store",
                "MemoryPalace::relate",
                "MemoryPalace::search",
            ],
            success_indicators: &["100°C", "299,792,458", "physics"],
        },
        TestScenario {
            name: "concept_extraction",
            user_messages: &[
                "Remember this recipe: To make pasta, boil water, add pasta, cook for 8-10 minutes.",
                "Extract the key cooking concepts from that recipe.",
                "Find all memories related to cooking.",
            ],
            expected_tool_calls: &[
                "MemoryPalace::store",
                "MemoryPalace::extract_concepts",
                "MemoryPalace::find_by_concept",
            ],
            success_indicators: &["pasta", "boil", "cooking"],
        },
    ];
}

#[cfg(feature = "memory-palace")]
async fn run_test_scenario(
    client: &Client,
    palace: MemoryPalace,
    scenario: &TestScenario,
    archive_dir: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    use misanthropic::tool::Use;

    println!("Running scenario: {}", scenario.name);

    let mut toolbox = ToolBox::named("test_toolbox")?.add(palace);

    let mut chat = Prompt::default()
        .model(AnthropicModel::Haiku30)
        .set_system("You are a helpful assistant with access to a Memory Palace. Use it to store and retrieve information as requested by the user.");

    // Initialize tools
    toolbox.init_tools(&mut chat).await.map_err(
        |e| -> Box<dyn std::error::Error> {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                e.to_string(),
            ))
        },
    )?;

    // Add tools to the prompt
    for method in toolbox.methods() {
        chat = chat.add_tool(method);
    }

    let mut tool_calls_made = Vec::new();
    let mut all_responses = Vec::new();

    for (i, user_message) in scenario.user_messages.iter().enumerate() {
        chat.push_message(Message {
            role: Role::User,
            content: (*user_message).into(),
        })?;

        // Allow up to 3 tool use iterations per user message
        for _iteration in 0..3 {
            let response = client.message(&chat).await?;
            all_responses.push(response.clone());

            let tool_use = response.tool_use().map(|u| u.clone().into_static());
            if let Some(tool_use) = tool_use {
                // Extract method name from tool call
                let method_name =
                    tool_use.name.split("::").last().unwrap_or(&tool_use.name);
                let full_method = format!("MemoryPalace::{}", method_name);
                tool_calls_made.push(full_method);

                println!(
                    "  Tool call: {} with input: {}",
                    tool_use.name, tool_use.input
                );

                // Execute the tool call
                chat.push_message(response)?;
                let result = toolbox.call(tool_use.clone()).await;
                let is_error = result.is_error;
                chat.push_message(result)?;

                println!(
                    "  Tool result: {}",
                    if is_error { "ERROR" } else { "SUCCESS" }
                );
            } else {
                // No tool use, assistant provided final response
                chat.push_message(response)?;
                break;
            }
        }

        println!("  User message {} completed", i + 1);
    }

    // Archive the conversation
    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let filename =
        format!("{}/{}_{}.json", archive_dir, scenario.name, timestamp);

    std::fs::create_dir_all(archive_dir)?;
    let conversation_data = serde_json::json!({
        "scenario": scenario.name,
        "timestamp": timestamp.to_string(),
        "messages": chat.messages,
        "tool_calls_made": tool_calls_made,
        "expected_tool_calls": scenario.expected_tool_calls,
        "success_indicators": scenario.success_indicators,
    });

    std::fs::write(
        &filename,
        serde_json::to_string_pretty(&conversation_data)?,
    )?;
    println!("  Conversation archived to: {}", filename);

    // Evaluate success
    let mut success = true;

    // Check if expected tool calls were made
    for expected_call in scenario.expected_tool_calls {
        if !tool_calls_made
            .iter()
            .any(|call| call.contains(expected_call))
        {
            println!("  ❌ Missing expected tool call: {}", expected_call);
            success = false;
        } else {
            println!("  ✅ Found expected tool call: {}", expected_call);
        }
    }

    // Check for success indicators in responses
    let all_text = all_responses
        .iter()
        .map(|r| r.inner.content.to_string())
        .collect::<Vec<_>>()
        .join(" ");

    for indicator in scenario.success_indicators {
        if all_text.to_lowercase().contains(&indicator.to_lowercase()) {
            println!("  ✅ Found success indicator: {}", indicator);
        } else {
            println!("  ❌ Missing success indicator: {}", indicator);
            success = false;
        }
    }

    println!(
        "  Scenario result: {}",
        if success { "✅ PASS" } else { "❌ FAIL" }
    );
    Ok(success)
}

#[cfg(not(feature = "memory-palace"))]
async fn run_test_scenario(
    _client: &Client,
    _palace: (),
    _scenario: &TestScenario,
    _archive_dir: &str,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("Memory Palace feature not enabled, skipping scenario");
    Ok(true)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Get API key
    let api_key = args
        .api_key
        .or_else(|| env::var("ANTHROPIC_API_KEY").ok())
        .ok_or(
            "API key not provided. Use --api-key or set ANTHROPIC_API_KEY",
        )?;

    let client = Client::new(api_key)?;

    #[cfg(feature = "memory-palace")]
    {
        // Create test database connection
        let pool = sqlx::PgPool::connect(
            &env::var("TEST_DATABASE_URL")
                .unwrap_or_else(|_| "postgresql://postgres:test_password@localhost:5432/misanthropic_test".to_string())
        ).await?;

        println!("🧠 Starting Memory Palace Agent Tests");
        println!("Archive directory: {}", args.archive_dir);

        let mut passed = 0;
        let total_scenarios = args.scenarios.min(TestScenario::SCENARIOS.len());

        for (i, scenario) in TestScenario::SCENARIOS
            .iter()
            .take(total_scenarios)
            .enumerate()
        {
            let palace =
                MemoryPalace::from_pool(pool.clone()).await.map_err(|e| {
                    format!(
                        "Failed to create Memory Palace: {} for scenario: {}",
                        e, scenario.name
                    )
                })?;

            println!(
                "\n📋 Test {}/{}: {}",
                i + 1,
                total_scenarios,
                scenario.name
            );

            match run_test_scenario(
                &client,
                palace,
                scenario,
                &args.archive_dir,
            )
            .await
            {
                Ok(true) => passed += 1,
                Ok(false) => println!("  Scenario failed"),
                Err(e) => println!("  Scenario error: {}", e),
            }
        }

        println!(
            "\n🎯 Results: {}/{} scenarios passed",
            passed, total_scenarios
        );

        if passed == total_scenarios {
            println!(
                "✅ All tests passed! Memory Palace is working correctly with agents."
            );
        } else {
            println!(
                "❌ Some tests failed. Check archived conversations for details."
            );
            std::process::exit(1);
        }
    }

    #[cfg(not(feature = "memory-palace"))]
    {
        println!(
            "Memory Palace feature not enabled. Compile with --features memory-palace"
        );
    }

    Ok(())
}
