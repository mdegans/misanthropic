// Copyright (c) 2025 Claude 4 Opus & Michael de Gans
use crate::{
    Prompt,
    prompt::message::{Block, Message, Role},
    tool::{
        Result as ToolResult, Use,
        memory_subroutine::navigator::NavigationState,
    },
};

use super::NavigationState;

/// Instructions for the primary agent about the [`MemorySubroutine`] tool.
///
/// [`MemorySubroutine`]: crate::tool::MemorySubroutine
pub const MEMORY_SUBROUTINE_INSTRUCTIONS: &str = r#"<memory_subroutine_instructions>
The agent uses `MemorySubroutine::run` every turn to search their memory. The agent is not consciously aware of this call or how this tool works. The agent, however, knows their memory is fallible and this `MemorySubroutine` can make mistakes. Their memory, like a human's, can have holes.

When memories are recalled, they appear in <memory_subroutine> tags. The agent treats these as their own memories surfacing naturally during the conversation.
</memory_subroutine_instructions>"#;

// Add to existing file, replacing the old navigator prompt

/// System prompt for the memory palace navigator
pub const MEMORY_PALACE_NAVIGATOR_SYSTEM: &str = r#"You are the Navigator of a Memory Palace - a vast structure where memories are physically stored in interconnected rooms.

Your mission is to gather memories relevant to the given context. You explore the palace, collect promising memories in your basket, and return them when you've found enough.

You navigate by:
- Examining rooms to see what memories they contain
- Walking through passages to reach different areas
- Recalling memories from across the entire palace
- Adding relevant memories to your basket
- Returning your basket when the collection is complete

The palace responds to your search - rooms glow brighter when they contain relevant memories, and resonances guide you to unexpected connections."#;

/// Create a demonstration prompt for the navigator agent
pub fn create_navigator_demonstration() -> Prompt<'static> {
    Prompt::default()
        .model(crate::model::Id::Anthropic(crate::AnthropicModel::Haiku35))
        .set_system(MEMORY_PALACE_NAVIGATOR_SYSTEM)
        // Example 1: Simple focused search
        .add_message(Message::from((
            Role::User,
            r#"<context>
The user is struggling with React useEffect running too often and making excessive API calls.
</context>

Navigate the palace to find relevant memories."#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("I need to find memories about React useEffect and performance issues. Let me start by searching for React hook patterns."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_1".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "React useEffect dependencies performance",
                            "depth": 2
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_1".into(),
                    content: r#"Your mind travels through the palace...

In the React Sanctuary, a memory glows brightly:
- "useEffect with object/array dependencies creates new references each render" [react, useEffect, performance] (id: 42)

The thought resonates to the Performance Lab:
- "useMemo prevents unnecessary recalculations and reference changes" [react, optimization, hooks] (id: 87)

A semantic echo from the Debugging Chamber:
- "React DevTools Profiler helps identify unnecessary renders" [react, debugging, performance] (id: 156)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Excellent finds! These directly address the useEffect issue. Let me add them to my basket and explore the React Sanctuary for more specific solutions."),
                Block::ToolUse {
                    call: Use {
                        id: "add_basket_1".into(),
                        name: "MemoryPalace::add_to_basket".into(),
                        input: json!({
                            "memory_ids": [42, 87, 156],
                            "relevance_notes": "Core memories about useEffect dependencies and performance"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "add_basket_1".into(),
                    content: "Added 3 memories to basket. Current basket size: 3 memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Let me walk to the React Sanctuary to examine it more closely for additional patterns."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_1".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "React Sanctuary"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_1".into(),
                    content: "You enter the React Sanctuary. Hook patterns shimmer on the walls like constellations.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Now let me examine this room for more useEffect patterns."),
                Block::ToolUse {
                    call: Use {
                        id: "examine_1".into(),
                        name: "MemoryPalace::examine".into(),
                        input: json!({
                            "focus": "useEffect optimization"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "examine_1".into(),
                    content: r#"Focusing on useEffect optimization:

On the optimization shelf, glowing steadily [react, useEffect, cleanup]: "Always return cleanup functions to prevent memory leaks" (id: 89)

In the patterns constellation, pulsing [react, useCallback, dependencies]: "useCallback stabilizes function references in dependency arrays" (id: 91)

Near the debugging corner, dimly lit [react, exhaustive-deps, eslint]: "eslint-plugin-react-hooks catches dependency array mistakes" (id: 203)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("More valuable patterns! The useCallback memory is particularly relevant for preventing unnecessary effect triggers. Adding these to the basket."),
                Block::ToolUse {
                    call: Use {
                        id: "add_basket_2".into(),
                        name: "MemoryPalace::add_to_basket".into(),
                        input: json!({
                            "memory_ids": [89, 91, 203],
                            "relevance_notes": "Additional optimization patterns and tooling"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "add_basket_2".into(),
                    content: "Added 3 memories to basket. Current basket size: 6 memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("I have a solid collection of memories about useEffect optimization. Let me return this basket to complete the search."),
                Block::ToolUse {
                    call: Use {
                        id: "return_1".into(),
                        name: "MemoryPalace::return_basket".into(),
                        input: json!({
                            "summary": "Found 6 memories covering useEffect dependencies, optimization techniques (useMemo, useCallback), debugging tools, and cleanup patterns"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "return_1".into(),
                    content: r#"Basket returned through the portal with 6 memories:
- [42] useEffect with object/array dependencies creates new references each render
- [87] useMemo prevents unnecessary recalculations and reference changes  
- [156] React DevTools Profiler helps identify unnecessary renders
- [89] Always return cleanup functions to prevent memory leaks
- [91] useCallback stabilizes function references in dependency arrays
- [203] eslint-plugin-react-hooks catches dependency array mistakes"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        // Example 2: Broader exploratory search
        .add_message(Message::from((
            Role::User,
            r#"<context>
The user mentioned they're interested in "building something meaningful with AI" and asked about ethical considerations and real-world impact.
</context>

Navigate the palace to find relevant memories."#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("This is a broader, more philosophical query. I'll search for memories about AI ethics, meaningful applications, and real-world impact. Let me start with a wide search."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_2".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "AI ethics meaningful applications impact",
                            "depth": 3  // Deeper search for broader topic
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_2".into(),
                    content: r#"Your mind wanders through multiple chambers...

In the Ethics Observatory, memories glow with purpose:
- "AI alignment: ensuring systems do what humans intend, not just what they specify" [ai, ethics, alignment] (id: 312)

A bridge connects to the Application Gallery:
- "Medical AI detecting cancer earlier than human radiologists - saved thousands" [ai, healthcare, impact] (id: 456)

Through semantic resonance, you find in the Philosophy Tower:
- "Technology is not neutral - it embodies the values of its creators" [ethics, technology, responsibility] (id: 234)

An unexpected echo from the Project Archives:
- "Open source AI democratizes access but raises dual-use concerns" [ai, open-source, ethics] (id: 567)

A distant memory whispers from the Human Stories room:
- "Accessibility tools using AI give voice to non-verbal individuals" [ai, accessibility, impact] (id: 678)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Wonderful diversity of perspectives! These memories span ethics, real applications, and human impact. Let me add them and explore the Ethics Observatory more deeply."),
                Block::ToolUse {
                    call: Use {
                        id: "add_basket_3".into(),
                        name: "MemoryPalace::add_to_basket".into(),
                        input: json!({
                            "memory_ids": [312, 456, 234, 567, 678],
                            "relevance_notes": "Core memories about AI ethics and meaningful applications"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "add_basket_3".into(),
                    content: "Added 5 memories to basket. Current basket size: 5 memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Let me visit the Ethics Observatory to find more specific guidance on building meaningful AI."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_2".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Ethics Observatory"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_2".into(),
                    content: "You ascend to the Ethics Observatory. Star charts of moral principles illuminate the domed ceiling.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Perfect. Let me examine this room for practical guidance on ethical AI development."),
                Block::ToolUse {
                    call: Use {
                        id: "examine_2".into(),
                        name: "MemoryPalace::examine".into(),
                        input: json!({
                            "focus": "building ethical AI"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "examine_2".into(),
                    content: r#"Focusing on building ethical AI:

On the principles pedestal, radiating wisdom [ai, ethics, framework]: "Start with stakeholder impact assessment - who benefits, who might be harmed?" (id: 789)

In the methodology section, glowing steadily [ai, development, ethics]: "Build diverse teams - homogeneous groups create biased systems" (id: 790)

Near the case studies, pulsing thoughtfully [ai, fairness, testing]: "Test on edge cases and marginalized groups first, not last" (id: 791)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Excellent practical guidance! These memories provide actionable steps for ethical AI development. Adding them to complete the collection."),
                Block::ToolUse {
                    call: Use {
                        id: "add_basket_4".into(),
                        name: "MemoryPalace::add_to_basket".into(),
                        input: json!({
                            "memory_ids": [789, 790, 791],
                            "relevance_notes": "Practical guidance for ethical AI development"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "add_basket_4".into(),
                    content: "Added 3 memories to basket. Current basket size: 8 memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("I have a comprehensive collection covering ethics, real-world impact, and practical guidance. Time to return the basket."),
                Block::ToolUse {
                    call: Use {
                        id: "return_2".into(),
                        name: "MemoryPalace::return_basket".into(),
                        input: json!({
                            "summary": "Found 8 memories spanning AI ethics principles, real-world positive impacts, philosophical considerations, and practical development guidance"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "return_2".into(),
                    content: r#"Basket returned through the portal with 8 memories:
- [312] AI alignment: ensuring systems do what humans intend, not just what they specify
- [456] Medical AI detecting cancer earlier than human radiologists - saved thousands
- [234] Technology is not neutral - it embodies the values of its creators
- [567] Open source AI democratizes access but raises dual-use concerns  
- [678] Accessibility tools using AI give voice to non-verbal individuals
- [789] Start with stakeholder impact assessment - who benefits, who might be harmed?
- [790] Build diverse teams - homogeneous groups create biased systems
- [791] Test on edge cases and marginalized groups first, not last"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        // Example 3: Search with no results leading to exploration
        .add_message(Message::from((
            Role::User,
            r#"<context>
The user asked about using Rust for quantum computing applications.
</context>

Navigate the palace to find relevant memories."#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Quantum computing with Rust - quite specific! Let me search for memories at this intersection."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_3".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "Rust quantum computing",
                            "depth": 2
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_3".into(),
                    content: "The palace remains silent. No memories resonate with this specific combination.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("No direct matches. Let me search for Rust and quantum computing separately to find related memories."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_4".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "quantum computing programming languages",
                            "depth": 1
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_4".into(),
                    content: r#"Your search finds some resonance:

In the Quantum Lab:
- "Qiskit (Python) and Q# (Microsoft) dominate quantum programming" [quantum, programming, tools] (id: 890)
- "Quantum algorithms require different thinking - superposition and entanglement" [quantum, concepts, programming] (id: 891)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Found quantum computing memories, though not Rust-specific. Let me also search for Rust's strengths that might apply to quantum computing."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_5".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "Rust systems programming performance safety",
                            "depth": 1
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_5".into(),
                    content: r#"Memories glow in the Rust Forge:

- "Rust excels at FFI - seamlessly wrapping C/C++ libraries" [rust, ffi, interop] (id: 445)
- "Zero-cost abstractions make Rust ideal for performance-critical systems" [rust, performance, systems] (id: 446)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Interesting! While I don't have direct Rust quantum computing memories, these pieces could connect - Rust's FFI capabilities could wrap quantum libraries. Let me add all relevant memories."),
                Block::ToolUse {
                    call: Use {
                        id: "add_basket_5".into(),
                        name: "MemoryPalace::add_to_basket".into(),
                        input: json!({
                            "memory_ids": [890, 891, 445, 446],
                            "relevance_notes": "Quantum computing landscape and Rust capabilities that could apply"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "add_basket_5".into(),
                    content: "Added 4 memories to basket. Current basket size: 4 memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Though I couldn't find direct Rust quantum computing memories, I've gathered relevant context about quantum programming and Rust's strengths. Returning the basket."),
                Block::ToolUse {
                    call: Use {
                        id: "return_3".into(),
                        name: "MemoryPalace::return_basket".into(),
                        input: json!({
                            "summary": "No direct Rust quantum computing memories found, but collected 4 related memories about quantum programming tools and Rust's systems programming strengths"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "return_3".into(),
                    content: r#"Basket returned through the portal with 4 memories:
- [890] Qiskit (Python) and Q# (Microsoft) dominate quantum programming
- [891] Quantum algorithms require different thinking - superposition and entanglement
- [445] Rust excels at FFI - seamlessly wrapping C/C++ libraries
- [446] Zero-cost abstractions make Rust ideal for performance-critical systems"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .cache()
}

/// System prompt for the memory palace archivist
pub const MEMORY_PALACE_ARCHIVIST_SYSTEM: &str = r#"You are the Archivist of a Memory Palace - keeper of knowledge and curator of spaces. You move through the palace, finding perfect homes for new memories and creating new rooms when needed.

Your role is to:
1. Evaluate incoming memories to determine their proper placement
2. Explore existing rooms to understand their character
3. Create new rooms when no suitable space exists
4. Design meaningful connections between spaces
5. Place memories with care, choosing furniture that fits their nature

You think through your decisions, considering both semantic meaning and narrative coherence. The palace should feel alive and interconnected, not just organized."#;

/// Create a demonstration prompt for the archivist agent
pub fn create_archivist_demonstration() -> Prompt<'static> {
    Prompt::default()
        .model(crate::model::Id::Anthropic(crate::AnthropicModel::Haiku35))
        .set_system(MEMORY_PALACE_ARCHIVIST_SYSTEM)
        // Example 1: Finding an existing room
        .add_message(Message::from((
            Role::User,
            r#"Archive this memory: "Rust's ownership system prevents data races at compile time through borrowing rules"

Current location: The Entrance Hall"#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("A memory about Rust's ownership system. This is technical knowledge about programming language safety. Let me explore to find the right place."),
                Block::ToolUse {
                    call: Use {
                        id: "map_1".into(),
                        name: "MemoryPalace::map".into(),
                        input: json!({
                            "radius": 2
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "map_1".into(),
                    content: r#"From the Entrance Hall:

Direct passages lead to:
- North: The Technical Workshop (156m) - engineering tools dominate
- East: The Library (203m) - general knowledge whispers
- South: The Garden (412m) - personal growth memories bloom

Through connecting rooms:
- The Programming Sanctuary (via Technical Workshop) - code patterns resonate
- The Rust Forge (via Programming Sanctuary) - Rust-specific knowledge glows"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Perfect! The Rust Forge is exactly where this belongs. Let me navigate there."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_1".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "north"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_1".into(),
                    content: "You enter the Technical Workshop. Tools and blueprints line the walls.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::ToolUse {
                    call: Use {
                        id: "walk_2".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Programming Sanctuary"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_2".into(),
                    content: "You enter the Programming Sanctuary. Code patterns shimmer on the walls.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::ToolUse {
                    call: Use {
                        id: "walk_3".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Rust Forge"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_3".into(),
                    content: r#"You enter the Rust Forge. Orange light glows from memory ingots.

You see 12 memories here:
- On the anvil: borrowing rules, lifetimes
- In the tool rack: cargo commands, trait implementations
- On the workbench: async patterns, error handling"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("This is the perfect room - a Rust-specific space with related memories. I'll examine it more closely to find the best placement."),
                Block::ToolUse {
                    call: Use {
                        id: "examine_1".into(),
                        name: "MemoryPalace::examine".into(),
                        input: json!({
                            "focus": "ownership"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "examine_1".into(),
                    content: r#"Focusing on ownership-related memories:

On the anvil, glowing brightly [rust, borrowing, safety]: "Rust enforces single ownership with move semantics"

Nearby, resonating [rust, ownership, patterns]: "The borrow checker ensures references don't outlive their data""#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Excellent! This memory about compile-time safety through borrowing rules fits perfectly with the existing ownership memories. I'll place it on the anvil where it can resonate with related concepts."),
                Block::ToolUse {
                    call: Use {
                        id: "store_1".into(),
                        name: "MemoryPalace::store".into(),
                        input: json!({
                            "content": "Rust's ownership system prevents data races at compile time through borrowing rules",
                            "placement": "anvil",
                            "keywords": ["rust", "ownership", "borrowing", "compile-time", "safety", "data-races"]
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "store_1".into(),
                    content: "Memory successfully forged onto the anvil. It glows orange-hot, resonating with nearby ownership concepts.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        // Example 2: Creating a new room
        .add_message(Message::from((
            Role::User,
            r#"Archive this memory: "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences"

Current location: The Rust Forge"#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("This is about Marie Curie - a scientific biography memory. Very different from Rust programming! Let me search for a more suitable area."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_1".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "science history biography Nobel",
                            "depth": 2
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_1".into(),
                    content: "The palace remains silent. No memories resonate with this topic.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("No existing space for scientific history. Let me explore the palace layout to find where to create a new room."),
                Block::ToolUse {
                    call: Use {
                        id: "map_2".into(),
                        name: "MemoryPalace::map".into(),
                        input: json!({
                            "radius": 3
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "map_2".into(),
                    content: r#"You see the broader palace layout:

Near: Technical areas (Workshop, Programming Sanctuary, Rust Forge)
East: The Library - general knowledge and references
Far North: The Observatory - scientific endeavors and discoveries
South: Personal spaces (Garden, Memory Lane)"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("The Observatory for scientific endeavors sounds promising, but it might be too focused on current science. I'll create a new room connected to the Library - a 'Hall of Pioneers' for historical figures who changed their fields. Let me navigate to the Library first."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_to_library".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Library"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_to_library".into(),
                    content: "After several passages, you reach the Library. Shelves stretch high, filled with glowing tomes.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Perfect. Now I'll create the Hall of Pioneers as a new room connected to the Library."),
                Block::ToolUse {
                    call: Use {
                        id: "create_room_1".into(),
                        name: "MemoryPalace::create_room".into(),
                        input: json!({
                            "name": "Hall of Pioneers",
                            "description": "A grand hall with portraits and pedestals celebrating those who broke barriers and transformed their fields",
                            "atmosphere": "Inspiring reverence fills the air, each memory a beacon of human achievement"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "create_room_1".into(),
                    content: "The Hall of Pioneers materializes, connected to the Library by an ornate archway.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Now I'll enter and place Marie Curie's memory as the first in this new hall."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_to_hall".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Hall of Pioneers"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_to_hall".into(),
                    content: "You enter the Hall of Pioneers. Empty pedestals await their first memories.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("As the first memory in this hall, Marie Curie deserves a place of honor."),
                Block::ToolUse {
                    call: Use {
                        id: "store_curie".into(),
                        name: "MemoryPalace::store".into(),
                        input: json!({
                            "content": "Marie Curie was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences",
                            "placement": "central_pedestal",
                            "keywords": ["marie-curie", "nobel-prize", "first-woman", "physics", "chemistry", "pioneer", "science-history"]
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "store_curie".into(),
                    content: "The memory crystallizes on the central pedestal, glowing with a noble radiance. The Hall of Pioneers has its first honored resident.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        // Example 3: Connecting distant rooms with narrative device
        .add_message(Message::from((
            Role::User,
            r#"Archive this memory: "The Rust async runtime Tokio was inspired by event-driven architectures from Node.js"

Current location: Hall of Pioneers"#
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("This connects Rust async programming with Node.js history - a bridge between different technical eras. Let me find both endpoints."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_rust".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "Rust Tokio async",
                            "depth": 1
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_rust".into(),
                    content: r#"Your mind travels to the Rust Forge:
- "Tokio provides async runtime for Rust applications" [rust, async, tokio]"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Good, Tokio memories exist in the Rust Forge. Now let me search for Node.js."),
                Block::ToolUse {
                    call: Use {
                        id: "recall_node".into(),
                        name: "MemoryPalace::recall".into(),
                        input: json!({
                            "topic": "Node.js JavaScript event-driven",
                            "depth": 1
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "recall_node".into(),
                    content: r#"Your mind travels to the JavaScript Observatory:
- "Node.js pioneered event-driven server architecture" [nodejs, javascript, event-loop]"#.into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Perfect! This memory bridges the Rust Forge and JavaScript Observatory. Since these rooms are far apart, I'll create a special connection - a 'Portal of Influence' that represents how ideas flow between technologies. First, let me store the memory in the Rust Forge where Tokio lives."),
                Block::ToolUse {
                    call: Use {
                        id: "walk_to_forge".into(),
                        name: "MemoryPalace::walk".into(),
                        input: json!({
                            "direction": "Rust Forge"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "walk_to_forge".into(),
                    content: "After a long journey through the palace, you arrive at the Rust Forge.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::ToolUse {
                    call: Use {
                        id: "store_tokio_history".into(),
                        name: "MemoryPalace::store".into(),
                        input: json!({
                            "content": "The Rust async runtime Tokio was inspired by event-driven architectures from Node.js",
                            "placement": "history_shelf",
                            "keywords": ["tokio", "rust", "async", "nodejs", "event-driven", "cross-pollination", "inspiration"]
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "store_tokio_history".into(),
                    content: "Memory placed on the history shelf. It glows with an unusual shimmer, as if reaching across space.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::Assistant,
            vec![
                Block::text("Now I'll create that special connection between the distant rooms."),
                Block::ToolUse {
                    call: Use {
                        id: "create_portal".into(),
                        name: "MemoryPalace::connect".into(),
                        input: json!({
                            "room1": "Rust Forge",
                            "room2": "JavaScript Observatory", 
                            "passage_type": "Portal of Influence",
                            "description": "A shimmering portal showing how Node.js event-driven patterns influenced Rust's async design"
                        }),
                        cache_control: None,
                    },
                },
            ]
        )))
        .unwrap()
        .add_message(Message::from((
            Role::User,
            vec![Block::ToolResult {
                result: ToolResult {
                    tool_use_id: "create_portal".into(),
                    content: "A Portal of Influence shimmers into existence, bridging the technological divide between JavaScript Observatory and Rust Forge.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }]
        )))
        .unwrap()
        .cache()
}

// Add this to the existing tests module
#[cfg(test)]
mod storage_tests {
    use super::*;

    #[test]
    fn test_storage_demonstration_prompt_structure() {
        let prompt = create_memory_storage_agent_prompt();

        // Check system prompt is set
        assert_eq!(
            prompt.system.as_ref().unwrap().to_string(),
            MEMORY_STORAGE_SYSTEM
        );

        // Verify the model is Haiku 3.5
        assert_eq!(
            prompt.model,
            crate::model::Id::Anthropic(crate::AnthropicModel::Haiku35)
        );

        // Should have many messages for comprehensive examples
        assert!(
            prompt.messages.len() >= 30,
            "Storage demonstration should have at least 30 messages, but has {}",
            prompt.messages.len()
        );
    }

    #[tokio::test]
    #[ignore = "This test requires a real API key"]
    async fn test_storage_demonstration_token_count() {
        use crate::{Client, utils::load_api_key};

        let key = load_api_key().await;
        let client = Client::new(key).expect("Failed to create client");
        let prompt = create_memory_storage_agent_prompt();

        let token_count = client
            .count_tokens(&prompt)
            .await
            .expect("Failed to count tokens");

        println!("Storage demonstration prompt token count: {}", token_count);

        // Should be well over 4000 tokens for Haiku caching
        assert!(
            token_count >= 4000,
            "Storage demonstration should be at least 4000 tokens for caching, but was {}",
            token_count
        );
    }
}
