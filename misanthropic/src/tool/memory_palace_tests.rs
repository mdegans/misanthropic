#[cfg(test)]
mod tests {
    use crate::{
        Prompt, Tool,
        tool::{MemoryPalace, Use},
    };

    use serde_json::json;
    use sqlx::PgPool;

    async fn create_test_palace(test_id: &str) -> MemoryPalace {
        let database_url =
            std::env::var("TEST_DATABASE_URL").unwrap_or_else(|_| {
                "postgresql://postgres:test_password@localhost:5432/misanthropic_test"
                    .to_string()
            });

        let pool = PgPool::connect(&database_url)
            .await
            .expect("Failed to connect to test database");

        // Create a unique schema for this test to avoid conflicts
        let schema_name = format!("test_{}", test_id.replace("-", "_"));

        // Drop and recreate schema in a transaction
        let mut tx = pool.begin().await.expect("Failed to begin transaction");

        sqlx::query(&format!("DROP SCHEMA IF EXISTS {} CASCADE", schema_name))
            .execute(&mut *tx)
            .await
            .expect("Failed to drop test schema");

        sqlx::query(&format!("CREATE SCHEMA {}", schema_name))
            .execute(&mut *tx)
            .await
            .expect("Failed to create test schema");

        tx.commit().await.expect("Failed to commit schema setup");

        // Create the MemoryPalace and initialize it with the schema
        MemoryPalace::from_pool_with_schema(pool, schema_name)
            .await
            .expect("Failed to create MemoryPalace")
    }

    #[tokio::test]
    async fn test_on_init_adds_instructions() {
        let mut palace = create_test_palace("on_init_adds_instructions").await;
        let mut prompt = Prompt::default();

        palace.on_init(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert!(system_content.contains("Memory Palace"));
        assert!(system_content.contains("MemoryPalace::store"));
        assert!(system_content.contains("MemoryPalace::search"));
    }

    #[tokio::test]
    async fn test_on_init_does_not_duplicate() {
        let mut palace = create_test_palace("on_init_no_duplicate").await;
        let mut prompt = Prompt::default()
            .set_system("<memory_palace_instructions>Already here</memory_palace_instructions>");

        palace.on_init(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert_eq!(
            system_content
                .matches("<memory_palace_instructions>")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn test_store_memory() {
        let mut palace = create_test_palace("store_memory").await;

        let memory_id = palace
            .store_memory(
                "kitchen",
                "A secret ingredient",
                ["cooking", "secret"],
            )
            .await
            .expect("Failed to store memory");

        assert!(memory_id > 0);

        let memories = palace.search("secret ingredient").await.unwrap();
        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].2.content, "A secret ingredient");
    }

    #[tokio::test]
    async fn test_list_rooms() {
        let mut palace = create_test_palace("list_rooms").await;

        // Store some memories in different rooms
        palace
            .store_memory("kitchen", "A memory in the kitchen", ["cooking"])
            .await
            .expect("Failed to store memory");

        palace
            .store_memory("library", "A memory in the library", ["books"])
            .await
            .expect("Failed to store memory");

        // List rooms
        let rooms = palace.list_rooms().await.expect("Failed to list rooms");
        assert_eq!(rooms.len(), 2);

        // Find kitchen room
        let kitchen_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "kitchen")
            .unwrap();
        assert_eq!(kitchen_room.2, 1); // 1 memory in kitchen

        // Find library room
        let library_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "library")
            .unwrap();
        assert_eq!(library_room.2, 1); // 1 memory in library
    }

    #[tokio::test]
    async fn test_connect_rooms() {
        let mut palace = create_test_palace("connect_rooms").await;

        // Create rooms by storing memories
        palace
            .store_memory("library", "A book", [])
            .await
            .expect("Failed to store memory");

        palace
            .store_memory("study", "Study notes", [])
            .await
            .expect("Failed to store memory");

        // Connect the rooms
        palace
            .connect_rooms("library", "study")
            .await
            .expect("Failed to connect rooms");

        // Check connections
        let rooms = palace.list_rooms().await.expect("Failed to list rooms");
        let library_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "library")
            .unwrap();
        assert!(library_room.3.contains(&"study".to_string()));

        let study_room = rooms
            .iter()
            .find(|(name, _, _, _)| name == "study")
            .unwrap();
        assert!(study_room.3.contains(&"library".to_string()));
    }

    #[tokio::test]
    async fn test_memory_relationships() {
        let mut palace = create_test_palace("memory_relationships").await;

        // Store two memories
        let memory_id1 = palace
            .store_memory("science", "E = mc²", ["physics", "einstein"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory(
                "science",
                "Theory of relativity",
                ["physics", "einstein"],
            )
            .await
            .expect("Failed to store memory");

        // Create a relationship
        let result = palace
            .relate_memories(memory_id1, memory_id2, "related_to", 0.9)
            .await
            .expect("Failed to create relationship");

        assert!(result.contains("related_to"));
        assert!(result.contains(&memory_id1.to_string()));
        assert!(result.contains(&memory_id2.to_string()));

        // Find related memories
        let related = palace
            .find_resonating_memories(memory_id1, 2, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 1);
        assert_eq!(related[0].2.id, memory_id2);
        assert_eq!(related[0].3, "related_to");
        assert_eq!(related[0].4, 0.9);
    }

    #[tokio::test]
    async fn test_concepts() {
        let mut palace = create_test_palace("concepts").await;

        // Store a memory
        let memory_id = palace
            .store_memory(
                "science",
                "Photosynthesis converts light to energy",
                ["biology"],
            )
            .await
            .expect("Failed to store memory");

        // Extract concepts
        let result = palace
            .extract_concepts(
                memory_id,
                ["photosynthesis", "energy", "biology"],
            )
            .await
            .expect("Failed to extract concepts");

        assert!(result.contains("3 concepts"));
        assert!(result.contains("photosynthesis"));
        assert!(result.contains("energy"));
        assert!(result.contains("biology"));

        // Find memories by concept
        let memories = palace
            .find_memories_by_concept("photosynthesis")
            .await
            .expect("Failed to find memories by concept");

        assert_eq!(memories.len(), 1);
        assert_eq!(memories[0].2.id, memory_id);
        assert_eq!(memories[0].3, 1.0); // confidence
    }

    #[tokio::test]
    async fn test_graph_stats() {
        let mut palace = create_test_palace("graph_stats").await;

        // Initially empty
        let stats =
            palace.get_graph_stats().await.expect("Failed to get stats");

        assert!(stats.contains("Total Memories: 0"));
        assert!(stats.contains("Total Rooms: 0"));

        // Add some data
        let memory_id1 = palace
            .store_memory("room1", "Memory 1", ["tag1"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("room2", "Memory 2", ["tag2"])
            .await
            .expect("Failed to store memory");

        palace
            .relate_memories(memory_id1, memory_id2, "related", 1.0)
            .await
            .expect("Failed to relate memories");

        palace
            .extract_concepts(memory_id1, ["concept1"])
            .await
            .expect("Failed to extract concepts");

        let stats =
            palace.get_graph_stats().await.expect("Failed to get stats");

        assert!(stats.contains("Total Memories: 2"));
        assert!(stats.contains("Total Rooms: 2"));
        assert!(stats.contains("Total Relationships: 1"));
        assert!(stats.contains("Total Concepts: 1"));
    }

    #[tokio::test]
    async fn test_tool_interface() {
        let palace = create_test_palace("tool_interface").await;

        // Test name
        assert_eq!(palace.name(), "MemoryPalace");

        // Test methods
        let methods: Vec<_> = palace.methods().collect();
        assert!(!methods.is_empty());

        let method_names: Vec<_> =
            methods.iter().map(|m| m.name.as_ref()).collect();
        assert!(method_names.contains(&"MemoryPalace::store"));
        assert!(method_names.contains(&"MemoryPalace::search"));
        assert!(method_names.contains(&"MemoryPalace::list_rooms"));
    }

    #[tokio::test]
    async fn test_tool_call_store() {
        let mut palace = create_test_palace("tool_call_store").await;

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::store".into(),
            input: json!({
                "room": "test_room",
                "content": "test content",
                "tags": ["test_tag"]
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Memory stored"));
        assert!(result.content.to_string().contains("test_room"));
    }

    #[tokio::test]
    async fn test_tool_call_search() {
        let mut palace = create_test_palace("tool_call_search").await;

        // First store something to search for
        palace
            .store_memory(
                "library",
                "A fascinating book about AI",
                ["technology", "AI"],
            )
            .await
            .expect("Failed to store memory");

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::search".into(),
            input: json!({
                "query": "AI"
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Found 1 memories"));
    }

    #[tokio::test]
    async fn test_tool_call_invalid_method() {
        let mut palace = create_test_palace("tool_call_invalid_method").await;

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::invalid_method".into(),
            input: json!({}),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Unknown method"));
        assert!(result.content.to_string().contains("Available methods"));
    }

    #[tokio::test]
    async fn test_tool_call_missing_parameters() {
        let mut palace =
            create_test_palace("tool_call_missing_parameters").await;

        // Test store without required parameters
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::store".into(),
            input: json!({
                "room": "test_room"
                // missing content and tags
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Missing required"));
    }

    #[tokio::test]
    async fn test_save_load_json() {
        let mut palace1 = create_test_palace("save_load_json").await;

        // Add some data
        palace1
            .store_memory("room1", "content1", ["tag1"])
            .await
            .expect("Failed to store memory");

        palace1
            .store_memory("room2", "content2", ["tag2"])
            .await
            .expect("Failed to store memory");

        // Save state (should only contain schema name)
        let json = palace1.save_json().await;
        assert!(json.is_object());
        assert!(json.get("schema_name").is_some());

        // Verify the JSON only contains the schema name, not the full data
        assert!(!json.to_string().contains("content1"));
        assert!(!json.to_string().contains("content2"));

        // Create new palace with different schema initially
        let mut palace2 = create_test_palace("save_load_json_second").await;

        // Load state - this should switch palace2 to use the same schema as palace1
        palace2.load_json(json).await.expect("Failed to load state");

        // Verify data is accessible (since palace2 now uses the same schema)
        let results =
            palace2.search("content1").await.expect("Failed to search");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].2.content, "content1");

        // Verify both memories are accessible
        let all_rooms =
            palace2.list_rooms().await.expect("Failed to list rooms");
        assert_eq!(all_rooms.len(), 2);
    }

    #[tokio::test]
    async fn test_apply_to_prompt() {
        let mut palace = create_test_palace("apply_to_prompt").await;
        let mut prompt = Prompt::default();

        palace.on_init(&mut prompt).await.unwrap();
        palace.on_turn(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert!(system_content.contains("Memory Palace"));
        assert!(system_content.contains("MemoryPalace::store"));
        assert!(system_content.contains("MemoryPalace::search"));
        assert!(system_content.contains("Rooms"));
        assert!(system_content.contains("Relationships"));
    }

    #[tokio::test]
    async fn test_enhanced_prompt_application() {
        let mut palace =
            create_test_palace("enhanced_prompt_application").await;
        let mut prompt = Prompt::default();

        palace.on_init(&mut prompt).await.unwrap();
        palace.on_turn(&mut prompt).await.unwrap();

        let system_content = prompt.system.unwrap().to_string();
        assert!(system_content.contains("Memory Palace"));
        assert!(system_content.contains("MemoryPalace::store"));
        assert!(system_content.contains("MemoryPalace::search"));
        assert!(system_content.contains("Rooms"));
        assert!(system_content.contains("Relationships"));
    }

    #[tokio::test]
    async fn test_bfs_with_distance() {
        let mut palace = create_test_palace("bfs_with_distance").await;

        // Create a chain of related memories
        let memory_id1 = palace
            .store_memory("science", "Quantum mechanics", ["physics"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("science", "Wave-particle duality", ["physics"])
            .await
            .expect("Failed to store memory");

        let memory_id3 = palace
            .store_memory("science", "Double-slit experiment", ["physics"])
            .await
            .expect("Failed to store memory");

        // Create relationships: 1 -> 2 -> 3
        palace
            .relate_memories(memory_id1, memory_id2, "leads_to", 0.9)
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id2, memory_id3, "demonstrates", 0.8)
            .await
            .expect("Failed to create relationship");

        // Test BFS from memory 1
        let results = palace
            .find_memories_bfs(memory_id1, 3, 0.8, 0.1)
            .await
            .expect("Failed to find memories via BFS");

        assert_eq!(results.len(), 2); // Should find memories 2 and 3

        // Memory 2 should have higher path strength (direct connection)
        let memory2_result =
            results.iter().find(|r| r.2.id == memory_id2).unwrap();
        let memory3_result =
            results.iter().find(|r| r.2.id == memory_id3).unwrap();

        assert!(memory2_result.3 > memory3_result.3); // path strength comparison
        assert!(memory2_result.4 < memory3_result.4); // distance comparison (memory2 is closer)
    }

    #[tokio::test]
    async fn test_tool_call_find_bfs() {
        let mut palace = create_test_palace("tool_call_find_bfs").await;

        // Store and relate some memories
        let memory_id1 = palace
            .store_memory("lab", "Start experiment", ["science"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("lab", "Record results", ["science"])
            .await
            .expect("Failed to store memory");

        palace
            .relate_memories(memory_id1, memory_id2, "leads_to", 0.8)
            .await
            .expect("Failed to create relationship");

        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::find_bfs".into(),
            input: json!({
                "memory_id": memory_id1,
                "max_distance": 2,
                "decay_factor": 0.8,
                "min_score": 0.1
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        dbg!(&result);
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Found 1 memories"));
        assert!(result.content.to_string().contains("Record results"));
        assert!(result.content.to_string().contains("Path Score"));
    }

    #[tokio::test]
    async fn test_find_related_memories_comprehensive() {
        let mut palace =
            create_test_palace("find_related_memories_comprehensive").await;

        // Create a complex network of related memories
        let memory_id1 = palace
            .store_memory(
                "physics",
                "Quantum mechanics",
                ["science", "physics"],
            )
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory(
                "physics",
                "Wave function",
                ["science", "physics", "quantum"],
            )
            .await
            .expect("Failed to store memory");

        let memory_id3 = palace
            .store_memory(
                "physics",
                "Superposition",
                ["science", "physics", "quantum"],
            )
            .await
            .expect("Failed to store memory");

        let memory_id4 = palace
            .store_memory(
                "chemistry",
                "Molecular orbitals",
                ["science", "chemistry"],
            )
            .await
            .expect("Failed to store memory");

        let memory_id5 = palace
            .store_memory(
                "history",
                "Einstein biography",
                ["history", "people"],
            )
            .await
            .expect("Failed to store memory");

        // Create relationships with different strengths
        palace
            .relate_memories(memory_id1, memory_id2, "defines", 0.9)
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id2, memory_id3, "exhibits", 0.8)
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id1, memory_id4, "relates_to", 0.3) // weak relationship
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id3, memory_id5, "discovered_by", 0.7)
            .await
            .expect("Failed to create relationship");

        // Test 1: Find direct relationships only (depth 1)
        let related = palace
            .find_resonating_memories(memory_id1, 1, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 2); // Should find memory2 and memory4
        let memory_ids: Vec<i64> = related.iter().map(|r| r.2.id).collect();
        assert!(memory_ids.contains(&memory_id2));
        assert!(memory_ids.contains(&memory_id4));

        // Test 2: Find with higher minimum strength (should filter out weak relationship)
        let related = palace
            .find_resonating_memories(memory_id1, 1, 0.5)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 1); // Should only find memory2 (strength 0.9)
        assert_eq!(related[0].2.id, memory_id2);
        assert_eq!(related[0].3, "defines");
        assert_eq!(related[0].4, 0.9);

        // Test 3: Find with depth 2 (should include indirect relationships)
        let related = palace
            .find_resonating_memories(memory_id1, 2, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 3); // Should find memory2, memory3, memory4
        let memory_ids: Vec<i64> = related.iter().map(|r| r.2.id).collect();
        assert!(memory_ids.contains(&memory_id2));
        assert!(memory_ids.contains(&memory_id3));
        assert!(memory_ids.contains(&memory_id4));

        // Verify ordering of
        assert!(related[0].4 >= related[1].4);
        assert!(related[1].4 >= related[2].4);

        // Test 4: Find with depth 3 (should include Einstein biography)
        let related = palace
            .find_resonating_memories(memory_id1, 3, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 4); // Should find all connected memories
        let memory_ids: Vec<i64> = related.iter().map(|r| r.2.id).collect();
        assert!(memory_ids.contains(&memory_id2));
        assert!(memory_ids.contains(&memory_id3));
        assert!(memory_ids.contains(&memory_id4));
        assert!(memory_ids.contains(&memory_id5));

        // Test 5: Non-existent memory ID
        let related = palace
            .find_resonating_memories(99999, 2, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 0); // Should find nothing

        // Test 6: Very high minimum strength (should find nothing)
        let related = palace
            .find_resonating_memories(memory_id1, 2, 0.99)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 0); // No relationships strong enough

        // Test 7: Zero depth (should find nothing)
        let related = palace
            .find_resonating_memories(memory_id1, 0, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 0); // No traversal allowed
    }

    #[tokio::test]
    async fn test_find_related_memories_edge_cases() {
        let mut palace =
            create_test_palace("find_related_memories_edge_cases").await;

        // Test with circular relationships
        let memory_id1 = palace
            .store_memory("loop", "Memory A", ["test"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("loop", "Memory B", ["test"])
            .await
            .expect("Failed to store memory");

        let memory_id3 = palace
            .store_memory("loop", "Memory C", ["test"])
            .await
            .expect("Failed to store memory");

        // Create circular relationships: A -> B -> C -> A
        palace
            .relate_memories(memory_id1, memory_id2, "leads_to", 0.8)
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id2, memory_id3, "leads_to", 0.8)
            .await
            .expect("Failed to create relationship");

        palace
            .relate_memories(memory_id3, memory_id1, "leads_to", 0.8)
            .await
            .expect("Failed to create relationship");

        // Should handle circular relationships without infinite loops
        let related = palace
            .find_resonating_memories(memory_id1, 5, 0.1) // high depth
            .await
            .expect("Failed to find related memories");

        // Should find each memory only once despite circular relationships
        assert_eq!(related.len(), 2); // Should find memory2 and memory3, not memory1 again
        let memory_ids: Vec<i64> = related.iter().map(|r| r.2.id).collect();
        assert!(memory_ids.contains(&memory_id2));
        assert!(memory_ids.contains(&memory_id3));
        assert!(!memory_ids.contains(&memory_id1)); // Should not include starting memory

        // Test with isolated memory (no relationships)
        let isolated_memory = palace
            .store_memory("isolated", "Lonely memory", ["isolated"])
            .await
            .expect("Failed to store memory");

        let related = palace
            .find_resonating_memories(isolated_memory, 2, 0.1)
            .await
            .expect("Failed to find related memories");

        assert_eq!(related.len(), 0); // Should find nothing

        // Test with self-referential relationship (if allowed by DB constraints)
        // This might fail depending on DB constraints, so we'll test gracefully
        let self_ref_result = palace
            .relate_memories(memory_id1, memory_id1, "self_ref", 1.0)
            .await;

        // If self-referential relationships are allowed, test that they don't break the query
        if self_ref_result.is_ok() {
            let related = palace
                .find_resonating_memories(memory_id1, 1, 0.1)
                .await
                .expect("Failed to find related memories");

            // Should still work and not include the memory itself
            let memory_ids: Vec<i64> = related.iter().map(|r| r.2.id).collect();
            assert!(!memory_ids.contains(&memory_id1));
        }
    }

    #[tokio::test]
    async fn test_find_related_memories_tool_call() {
        let mut palace =
            create_test_palace("find_related_memories_tool_call").await;

        // Set up some test data
        let memory_id1 = palace
            .store_memory("test", "Root memory", ["test"])
            .await
            .expect("Failed to store memory");

        let memory_id2 = palace
            .store_memory("test", "Related memory", ["test"])
            .await
            .expect("Failed to store memory");

        palace
            .relate_memories(memory_id1, memory_id2, "connected", 0.8)
            .await
            .expect("Failed to create relationship");

        // Test valid tool call
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::find_related".into(),
            input: json!({
                "memory_id": memory_id1,
                "max_depth": 2,
                "min_strength": 0.1
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(result.content.to_string().contains("Found 1 related"));
        assert!(result.content.to_string().contains("Related memory"));

        // Test with missing required parameter
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::find_related".into(),
            input: json!({
                "max_depth": 2,
                "min_strength": 0.1
                // missing memory_id
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Missing required"));

        // Test with invalid memory_id type
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::find_related".into(),
            input: json!({
                "memory_id": "not_a_number",
                "max_depth": 2,
                "min_strength": 0.1
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(result.is_error);
        assert!(result.content.to_string().contains("Missing required"));

        // Test with non-existent memory_id
        let call = Use {
            id: "test_id".into(),
            name: "MemoryPalace::find_related".into(),
            input: json!({
                "memory_id": 99999,
                "max_depth": 2,
                "min_strength": 0.1
            }),
            #[cfg(feature = "prompt-caching")]
            cache_control: None,
        };

        let result = palace.call(call).await;
        assert!(!result.is_error);
        assert!(
            result
                .content
                .to_string()
                .contains("No related memories found")
        );
    }
}
