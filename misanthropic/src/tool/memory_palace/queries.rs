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
    WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
    ORDER BY 
        CASE 
            WHEN content ILIKE $1 THEN 3
            WHEN room ILIKE $1 THEN 2
            WHEN tags::text ILIKE $1 THEN 1
            ELSE 0
        END DESC,
        created_at DESC
    LIMIT 10
"#;

/// Recency-based search fragment
pub const SEARCH_RECENCY: &str = r#"
    SELECT id, content, room, tags, created_at, last_updated
    FROM memories 
    WHERE content ILIKE $1 OR room ILIKE $1 OR tags::text ILIKE $1
    ORDER BY last_updated DESC
    LIMIT 10
"#;

/// Relationship-based search CTE
pub const SEARCH_RELATIONSHIPS: &str = r#"
    WITH related_memories AS (
        SELECT DISTINCT m.id, m.content, m.room, m.tags, m.created_at, m.last_updated
        FROM memories m
        JOIN memory_relationships mr ON (m.id = mr.from_memory_id OR m.id = mr.to_memory_id)
        JOIN memories search_mem ON (
            (mr.from_memory_id = search_mem.id AND search_mem.content ILIKE $1) OR
            (mr.to_memory_id = search_mem.id AND search_mem.content ILIKE $1)
        )
        WHERE m.content ILIKE $1 OR m.room ILIKE $1 OR m.tags::text ILIKE $1
    )
    SELECT id, content, room, tags, created_at, last_updated
    FROM related_memories
    ORDER BY created_at DESC
    LIMIT 10
"#;

/// Find related memories using BFS
pub const FIND_MEMORIES_BFS: &str = r#"
    WITH RECURSIVE memory_bfs AS (
        -- Base case: start with the given memory
        SELECT 
            $1::bigint as memory_id,
            1.0::double precision as path_strength,
            0 as distance,
            ARRAY[$1::bigint] as path
        
        UNION ALL
        
        -- Recursive case: explore relationships
        SELECT 
            CASE 
                WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                ELSE mr.from_memory_id
            END as memory_id,
            mb.path_strength * mr.strength * ($3::double precision) as path_strength,
            mb.distance + 1 as distance,
            mb.path || CASE 
                WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                ELSE mr.from_memory_id
            END as path
        FROM memory_bfs mb
        JOIN memory_relationships mr ON (mr.from_memory_id = mb.memory_id OR mr.to_memory_id = mb.memory_id)
        WHERE 
            mb.distance < $2
            AND mb.path_strength * mr.strength * ($3::double precision) >= ($4::double precision)
            AND NOT (CASE 
                WHEN mr.from_memory_id = mb.memory_id THEN mr.to_memory_id
                ELSE mr.from_memory_id
            END = ANY(mb.path))
    )
    SELECT DISTINCT ON (mb.memory_id)
        m.id, m.content, m.room, m.tags, m.created_at, m.last_updated,
        mb.path_strength, mb.distance
    FROM memory_bfs mb
    JOIN memories m ON mb.memory_id = m.id
    WHERE mb.memory_id != $1
    ORDER BY mb.memory_id, mb.path_strength DESC, mb.distance ASC
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
    ORDER BY rm.strength DESC, rm.depth ASC
"#;
