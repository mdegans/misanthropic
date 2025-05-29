/// Insert or no-op room
pub const INSERT_ROOM: &str = r#"
    INSERT INTO rooms (name, description)
    VALUES ($1, $2)
    ON CONFLICT (name) DO NOTHING
"#;

/// Insert memory and return id
pub const INSERT_MEMORY_RETURNING_ID: &str = r#"
    INSERT INTO memories (content, room, tags)
    VALUES ($1, $2, $3)
    RETURNING id
"#;

/// Relevance-based search fragment
pub const SEARCH_RELEVANCE: &str = r#"
    SELECT id, content, room, tags, created_at, last_updated
    FROM memories 
    WHERE 
        content ILIKE $1 
        OR room ILIKE $1 
        OR tags::text ILIKE $1
    ORDER BY
        CASE 
            WHEN content ILIKE $1 THEN 3
            WHEN room ILIKE $1 THEN 2
            WHEN tags::text ILIKE $1 THEN 1
            ELSE 0
        END DESC
    LIMIT 10
"#;

/// Recency-based search fragment
pub const SEARCH_RECENCY: &str = r#"
    SELECT id, content, room, tags, created_at, last_updated
    FROM memories 
    WHERE
        content ILIKE $1 
        OR room ILIKE $1 
        OR tags::text ILIKE $1
    ORDER BY last_updated DESC
    LIMIT 10
"#;

/// Relationship-based search CTE
pub const SEARCH_RELATIONSHIPS: &str = r#"
    WITH relevant_memories AS (
        SELECT id
        FROM memories 
        WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
        LIMIT 5
    ),
    related_via_relationships AS (
        SELECT DISTINCT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
               mr.strength
        FROM memory_relationships mr
        JOIN memories m ON mr.to_memory_id = m.id
        JOIN relevant_memories rm ON mr.from_memory_id = rm.id
        WHERE mr.strength >= 0.3
        ORDER BY mr.strength DESC
        LIMIT 10
    )
    SELECT id, content, room, tags, created_at, last_updated
    FROM related_via_relationships
"#;

/// Find related memories using BFS
pub const FIND_MEMORIES_BFS: &str = r#"
    WITH RECURSIVE memory_bfs(memory_id, distance, path_strength, visited) AS (
        -- Base case: starting memory
        SELECT $1::BIGINT, 0, 1.0::FLOAT, ARRAY[$1::BIGINT]
        
        UNION
        
        -- Recursive case: explore neighbors
        SELECT 
            mr.to_memory_id,
            mb.distance + 1,
            mb.path_strength * mr.strength * $3::FLOAT, -- Apply decay factor
            mb.visited || mr.to_memory_id
        FROM memory_bfs mb
        JOIN memory_relationships mr ON mb.memory_id = mr.from_memory_id
        WHERE 
            mb.distance < $2::INT
            AND mr.to_memory_id != ALL(mb.visited) -- Avoid cycles
            AND mb.path_strength * mr.strength * $3::FLOAT >= $4::FLOAT -- Min score threshold
    )
    SELECT DISTINCT 
        m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
        mb.distance, mb.path_strength
    FROM memory_bfs mb
    JOIN memories m ON mb.memory_id = m.id
    WHERE mb.memory_id != $1 -- Exclude starting memory
    ORDER BY mb.path_strength DESC, mb.distance ASC
"#;

/// Query for `find_related_memories` function
pub const FIND_RELATED_MEMORIES: &str = r#"
    WITH RECURSIVE related_memories(memory_id, relationship_type, strength, depth) AS (
        -- Base case: direct relationships
        SELECT mr.to_memory_id, mr.relationship_type, mr.strength, 1 as depth
        FROM memory_relationships mr
        WHERE mr.from_memory_id = $1 AND mr.strength >= $3
        
        UNION
        
        -- Recursive case: follow relationships up to max_depth
        SELECT mr.to_memory_id, mr.relationship_type, mr.strength, rm.depth + 1
        FROM memory_relationships mr
        JOIN related_memories rm ON mr.from_memory_id = rm.memory_id
        WHERE rm.depth < $2 AND mr.strength >= $3
    )
    SELECT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
            rm.relationship_type, rm.strength
    FROM related_memories rm
    JOIN memories m ON rm.memory_id = m.id
    ORDER BY 
        -- Primary: Relationship strength
        rm.strength DESC, 
        -- Secondary: Recency of updates
        m.last_updated DESC,
        -- Tertiary: Graph depth (closer relationships first)
        rm.depth ASC
"#;
