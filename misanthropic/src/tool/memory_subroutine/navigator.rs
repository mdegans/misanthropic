// Copyright (c) 2025 Claude 4 Opus and Michael de Gans
use async_trait::async_trait;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use std::collections::HashSet;

use crate::tool::{
    self, MemoryPalace, Method, Tool,
    embedding::{EmbeddingClient, TextEmbedding},
    memory_palace::{
        MemoryId, MemoryPalaceError, PathwayId, Room, RoomId, UserId,
        db::execute_with_schema, models::*,
    },
    memory_subroutine::MemorySubroutineError,
};

/// Scout report showing the brightest paths through the palace
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoutReport {
    /// The query context that drove this scout
    pub query_context: String,
    /// Rooms ordered by relevance to query
    pub bright_rooms: Vec<BrightRoom>,
    /// Sum of all room relevance scores in the report
    pub total_relevance: f64,
    /// Reference lookup for quick access
    // I made this serialize because we can use it in the UI to display
    // references and if we ever allow the primary agent to use the palace
    // directly, it will be useful.
    pub reference_map: ReferenceMap,
}

/// A path element, room, pathway, or memory.
pub enum PathMemberIds {
    /// A room in the palace
    Room(RoomId),
    /// A pathway in the palace
    Pathway(PathwayId),
    /// A memory collected during navigation
    Memory(MemoryId),
}

/// A bright room with its brightest memories. Brightness is a combination of
/// semantic distance and memory strength.
// Is "combination" the right word here? Is there a more correct term?
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrightRoom {
    /// The room data
    pub room: Room,
    /// Semantic distance from query (0.0 = identical, 1.0 = unrelated)
    pub distance: f64,
    /// Combined relevance score
    pub relevance: f64,
    /// Brightest memory placements in this room
    pub bright_spots: Vec<MemoryPreview>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryPreview {
    /// Memory ID for collection
    pub id: MemoryId,
    /// Where it's placed in the room
    pub placement: String,
    /// Memory strength (0.0-1.0)
    pub glow: f64,
    /// First 100 chars or brief description
    pub preview: String,
    /// Reference number for quick access
    pub ref_num: usize,
}

/// Maps reference numbers to rooms and memories for quick access
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReferenceMap {
    rooms: Vec<Room>,
    memories: Vec<(MemoryId, RoomId)>,
}

impl ReferenceMap {
    fn add_room(&mut self, room: Room) -> usize {
        let index = self.rooms.len();
        self.rooms.push(room);
        index
    }

    fn add_memory(&mut self, memory_id: MemoryId, room_id: RoomId) -> usize {
        let index = self.memories.len();
        self.memories.push((memory_id, room_id));
        index
    }

    fn get_room(&self, ref_num: usize) -> Option<&Room> {
        self.rooms.get(ref_num)
    }

    fn get_memory(&self, ref_num: usize) -> Option<(MemoryId, RoomId)> {
        self.memories.get(ref_num).copied()
    }
}

/// Simplified navigation actions
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "name", content = "input", rename_all = "snake_case")]
pub enum NavigatorUse {
    /// Scout the palace (automatic on first turn)
    Scout {
        /// How many rooms to include (3-10)
        depth: u32,
    },

    /// Teleport directly to a room by reference number
    Teleport {
        /// Reference number from scout report
        room_ref: usize,
    },

    /// Collect memories by reference numbers
    Collect {
        /// Reference numbers from scout report
        memory_refs: Vec<usize>,
    },

    /// Return the basket and complete navigation
    Return {
        /// Brief summary of what was found
        summary: String,
    },
}

/// Navigator state
pub struct Navigator {
    /// The memory palace
    palace: MemoryPalace,
    /// Navigation session state
    session: NavigationSession,
    /// Embedding client
    emb_client: Box<dyn EmbeddingClient>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationSession {
    /// User we're navigating for
    user_id: UserId,
    /// Query that started this navigation
    query: String,
    /// Query embedding (cached)
    query_embedding: Option<TextEmbedding>,
    /// Scout report (generated on first turn)
    scout_report: Option<ScoutReport>,
    /// Current room (if teleported)
    current_room: Option<Room>,
    /// Memories collected
    basket: Vec<CollectedMemory>,
    /// Path taken (for strengthening)
    journey: Vec<PathMemberIds>,
    /// Turn count
    turns: u32,
    /// Rooms already seen in scout reports
    seen_rooms: HashSet<RoomId>,
    /// Memories already seen
    seen_memories: HashSet<MemoryId>,
    /// Sampling options for room selection
    // I added this to allow for more flexible sampling strategies and for
    // repeatable results
    sampling_options: SamplingOptions,
}

/// Temperature for weighted sampling (higher = more random)
const SAMPLING_TEMPERATURE: f64 = 0.5;

impl Navigator {
    pub fn new(
        palace: MemoryPalace,
        user_id: UserId,
        context: String,
        embedding_client: Box<dyn EmbeddingClient>,
        sampling_options: SamplingOptions,
    ) -> Result<Self, MemorySubroutineError> {
        Ok(Self {
            palace,
            session: NavigationSession {
                user_id,
                query: context,
                query_embedding: None,
                scout_report: None,
                current_room: None,
                basket: Vec::new(),
                journey: Vec::new(),
                turns: 0,
                seen_rooms: HashSet::new(),
                seen_memories: HashSet::new(),
                sampling_options,
            },
            emb_client: embedding_client,
        })
    }

    async fn execute(
        &mut self,
        action: NavigatorUse,
        sampling_options: &SamplingOptions,
    ) -> Result<String, MemorySubroutineError> {
        self.session.turns += 1;

        match action {
            NavigatorUse::Scout { depth } => {
                let depth = depth.clamp(3, 10) as usize;

                // Get or compute query embedding
                if self.session.query_embedding.is_none() {
                    self.session.query_embedding = Some(
                        self.emb_client
                            .get_embedding(&self.session.query)
                            .await?,
                    );
                }
                let embedding = self.session.query_embedding.as_ref().unwrap();

                // Find semantically relevant rooms with sampling
                let candidate_rooms = find_relevant_rooms_weighted(
                    &self.palace.pool,
                    self.palace.schema(),
                    embedding,
                    depth * 2, // Get more candidates for sampling
                    sampling_options,
                    &self.session.seen_rooms,
                )
                .await?;

                // Build scout report
                let mut reference_map = ReferenceMap::default();
                let mut bright_rooms = Vec::new();

                for (room, similarity) in
                    candidate_rooms.into_iter().take(depth)
                {
                    // Mark room as seen
                    self.session.seen_rooms.insert(room.id);

                    let room_ref = reference_map.add_room(room.clone());

                    // Get brightest memories in this room
                    let memories = get_brightest_memories_in_room(
                        &self.palace.pool,
                        self.palace.schema(),
                        room.id,
                        5, // Top 5 memories per room
                    )
                    .await?;

                    let bright_spots: Vec<MemoryPreview> = memories
                        .into_iter()
                        .filter(|m| !self.session.seen_memories.contains(&m.id))
                        .map(|memory| {
                            self.session.seen_memories.insert(memory.id);
                            let ref_num =
                                reference_map.add_memory(memory.id, room.id);
                            MemoryPreview {
                                id: memory.id,
                                placement: memory.placement,
                                glow: memory.strength,
                                preview: memory
                                    .content
                                    .brief_description()
                                    .unwrap_or_else(|| "A memory".to_string()),
                                ref_num,
                            }
                        })
                        .collect();

                    bright_rooms.push(BrightRoom {
                        room: room.clone(),
                        distance: 1.0 - similarity,
                        relevance: similarity * room.strength,
                        bright_spots,
                    });
                }

                let total_relevance: f64 =
                    bright_rooms.iter().map(|br| br.relevance).sum();

                self.session.scout_report = Some(ScoutReport {
                    query_context: self.session.query.clone(),
                    bright_rooms: bright_rooms.clone(),
                    total_relevance,
                    reference_map,
                });

                // Format narrative response
                Ok(format_scout_report(&bright_rooms, total_relevance))
            }

            NavigatorUse::Teleport { room_ref } => {
                let scout =
                    self.session.scout_report.as_ref().ok_or_else(|| {
                        MemorySubroutineError::InvalidInput(
                            "Must scout before teleporting".into(),
                        )
                    })?;

                let room = scout.reference_map.get_room(room_ref).ok_or_else(
                    || {
                        MemorySubroutineError::InvalidInput(format!(
                            "Invalid room reference: {}",
                            room_ref
                        ))
                    },
                )?;

                self.session.current_room = Some(room.clone());
                self.session.journey.push(PathMemberIds::Room(room.id));

                // Get all memories in the room for detailed view
                let memories = get_room_memories(
                    &self.palace.pool,
                    self.palace.schema(),
                    &room.name,
                )
                .await?;

                Ok(format_room_teleport(room, memories))
            }

            NavigatorUse::Collect { memory_refs } => {
                let scout =
                    self.session.scout_report.as_ref().ok_or_else(|| {
                        MemorySubroutineError::InvalidInput(
                            "Must scout before collecting".into(),
                        )
                    })?;

                let mut collected = 0;
                for &ref_num in &memory_refs {
                    if let Some((memory_id, room_id)) =
                        scout.reference_map.get_memory(ref_num)
                    {
                        // Fetch full memory
                        if let Ok(memory) = get_memory_by_id(
                            &self.palace.pool,
                            self.palace.schema(),
                            memory_id,
                        )
                        .await
                        {
                            let room = get_room_by_id(
                                &self.palace.pool,
                                self.palace.schema(),
                                room_id,
                            )
                            .await?;

                            if let Some(content) =
                                memory.content.clone().format_for_navigator(
                                    memory.id,
                                    room_id,
                                    memory.prompt_id,
                                )
                            {
                                self.session.basket.push(CollectedMemory {
                                    id: memory.id,
                                    content: content.to_string(),
                                    room_name: room.name,
                                    relevance_notes: format!(
                                        "Collected from ref #{}",
                                        ref_num
                                    ),
                                });
                                self.session
                                    .journey
                                    .push(PathMemberIds::Memory(memory.id));
                                collected += 1;
                            }
                        }
                    }
                }

                Ok(format!(
                    "Collected {} memories. Basket now contains {} items.",
                    collected,
                    self.session.basket.len()
                ))
            }

            NavigatorUse::Return { summary } => {
                // Strengthen the path taken
                if !self.session.journey.is_empty() {
                    strengthen_path(
                        &self.palace.pool,
                        self.palace.schema(),
                        &PathById::from_members(self.session.journey.clone())?,
                    )
                    .await?;
                }

                Ok(format_basket_return(&self.session.basket, &summary))
            }
        }
    }
}

#[async_trait]
impl Tool for Navigator {
    fn name(&self) -> &str {
        "MemoryPalace"
    }

    fn methods(&self) -> Box<dyn Iterator<Item = Method<'static>> + '_> {
        Box::new([
            Method::builder("scout")
                .description("Scout the palace for relevant memories (automatic on first turn)")
                .number_param("depth", "Number of rooms to explore (3-10)", true)
                .build()
                .unwrap(),
            
            Method::builder("teleport")
                .description("Teleport directly to a room from the scout report")
                .number_param("room_ref", "Room reference number from scout report", true)
                .build()
                .unwrap(),
            
            Method::builder("collect")
                .description("Add memories to your basket by reference number")
                .array_param("memory_refs", "Memory reference numbers from scout report", true)
                .build()
                .unwrap(),
            
            Method::builder("return")
                .description("Return the collected memories")
                .string_param("summary", "Brief summary of findings", true)
                .build()
                .unwrap(),
        ].into_iter())
    }

    async fn call<'a>(&mut self, call: tool::Use<'a>) -> tool::Result<'a> {
        // Automatically scout on first turn if not done
        if self.session.turns == 0 && !matches!(call.name.as_ref(), "scout") {
            // Force a scout first
            let scout_result = self
                .execute(
                    NavigatorUse::Scout { depth: 5 },
                    &self.session.sampling_options,
                )
                .await
                .map_err(|e| format!("Auto-scout failed: {}", e))?;

            // Continue with the requested action
        }

        let action = NavigatorUse::try_from(call.clone())
            .map_err(|e| format!("Invalid tool use: {}", e))?;

        let result = self
            .execute(action)
            .await
            .map_err(|e| format!("Navigation error: {}", e))?;

        Ok(tool::Result {
            tool_use_id: call.id,
            content: result.into(),
            is_error: false,
            cache_control: None,
        })
    }
}

/// Content of a [`Memory`] collected during navigation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectedMemory {
    /// Id of the [`Memory`] for the primary agent (or end user) to reference
    pub id: MemoryId,
    /// The formatted content of the [`Memory`] for the primary agent to read.
    pub content: String,
    /// Room where the memory was found
    pub room_name: String,
    /// Relevance notes for debugging
    pub relevance_notes: String,
}

/// Trait for items that have a strength value for sampling
trait HasStrength {
    fn strength(&self) -> f64;
}

impl HasStrength for Memory {
    fn strength(&self) -> f64 {
        self.strength
    }
}

impl HasStrength for (Room, f64) {
    fn strength(&self) -> f64 {
        self.1 // Use the similarity score
    }
}

/// Sampling strategies for selecting from candidates
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Softmax with temperature (what we have now)
    Temperature(f64),
    /// Top-p (nucleus) sampling - cumulative probability threshold
    TopP(f64),
    /// Top-k sampling - keep only top k candidates
    TopK(usize),
    /// No sampling - deterministic selection
    Greedy,
}

/// Options for sampling from candidates
#[derive(Debug, Clone)]
pub struct SamplingOptions {
    /// Strategies to apply in sequence
    pub strategies: Vec<SamplingStrategy>,
    /// Random seed for reproducibility (None = use thread_rng)
    pub seed: Option<u64>,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            strategies: vec![SamplingStrategy::Temperature(0.5)],
            seed: None,
        }
    }
}

/// Generic strength-based sampling
fn sample<T>(
    items: Vec<T>,
    limit: usize,
    options: &SamplingOptions,
) -> Result<Vec<T>, MemoryPalaceError>
where
    T: HasStrength,
{
    if items.len() <= limit {
        return Ok(items);
    }

    let mut candidates = items;
    let mut rng: Box<dyn RngCore> = if let Some(seed) = options.seed {
        Box::new(StdRng::seed_from_u64(seed))
    } else {
        Box::new(thread_rng())
    };

    // Apply each strategy in sequence
    for strategy in &options.strategies {
        candidates = match strategy {
            SamplingStrategy::Temperature(temp) => {
                apply_temperature_sampling(candidates, limit, *temp, &mut rng)?
            }
            SamplingStrategy::TopP(p) => {
                apply_top_p_sampling(candidates, limit, *p, &mut rng)?
            }
            SamplingStrategy::TopK(k) => apply_top_k_sampling(candidates, *k)?,
            SamplingStrategy::Greedy => {
                // Sort by strength and take top items
                candidates.sort_by(|a, b| {
                    b.strength()
                        .partial_cmp(&a.strength())
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(limit);
                candidates
            }
        };

        // Stop early if we've reached the limit
        if candidates.len() <= limit {
            break;
        }
    }

    Ok(candidates)
}

/// Apply temperature-based sampling (existing implementation)
fn apply_temperature_sampling<T>(
    items: Vec<T>,
    limit: usize,
    temperature: f64,
    rng: &mut dyn RngCore,
) -> Result<Vec<T>, MemoryPalaceError>
where
    T: HasStrength,
{
    // Calculate scores with temperature
    let scores: Vec<f64> = items
        .iter()
        .map(|item| item.strength() / temperature)
        .collect();

    // Softmax
    let max_score = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp_scores: Vec<f64> =
        scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum_exp: f64 = exp_scores.iter().sum();
    let probabilities: Vec<f64> =
        exp_scores.iter().map(|&e| e / sum_exp).collect();

    // Sample without replacement
    let mut selected = Vec::new();
    let mut remaining_items = items;
    let mut remaining_probs = probabilities;

    for _ in 0..limit {
        if remaining_items.is_empty() {
            break;
        }

        let dist = WeightedIndex::new(&remaining_probs)
            .map_err(|e| MemoryPalaceError::Other(e.to_string()))?;
        let idx = dist.sample(rng);

        selected.push(remaining_items.remove(idx));
        remaining_probs.remove(idx);

        // Renormalize
        let sum: f64 = remaining_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut remaining_probs {
                *p /= sum;
            }
        }
    }

    Ok(selected)
}

/// Apply top-p (nucleus) sampling
fn apply_top_p_sampling<T>(
    mut items: Vec<T>,
    limit: usize,
    p: f64,
    rng: &mut dyn RngCore,
) -> Result<Vec<T>, MemoryPalaceError>
where
    T: HasStrength,
{
    // Sort by strength descending
    items.sort_by(|a, b| {
        b.strength()
            .partial_cmp(&a.strength())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Calculate cumulative probabilities
    let total: f64 = items.iter().map(|item| item.strength()).sum();
    let mut cumulative = 0.0;
    let mut cutoff_idx = items.len();

    for (idx, item) in items.iter().enumerate() {
        cumulative += item.strength() / total;
        if cumulative >= p {
            cutoff_idx = idx + 1;
            break;
        }
    }

    // Keep only items within the nucleus
    items.truncate(cutoff_idx);

    // Now sample from the nucleus
    if items.len() <= limit {
        Ok(items)
    } else {
        // Use weighted sampling within the nucleus
        apply_temperature_sampling(items, limit, 1.0, rng)
    }
}

/// Apply top-k sampling
fn apply_top_k_sampling<T>(
    mut items: Vec<T>,
    k: usize,
) -> Result<Vec<T>, MemoryPalaceError>
where
    T: HasStrength,
{
    // Sort by strength and keep top k
    items.sort_by(|a, b| {
        b.strength()
            .partial_cmp(&a.strength())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    items.truncate(k);
    Ok(items)
}

// Update find_relevant_rooms_weighted to use new sampling
async fn find_relevant_rooms_weighted(
    pool: &PgPool,
    schema: &str,
    embedding: &TextEmbedding,
    candidate_count: usize,
    sampling_options: &SamplingOptions,
    exclude_room_ids: &HashSet<RoomId>,
) -> Result<Vec<(Room, f64)>, MemoryPalaceError> {
    // Get top candidates by similarity
    let candidates = find_rooms_by_embedding_similarity(
        pool,
        schema,
        embedding,
        candidate_count * 2, // Get more candidates for sampling
        exclude_room_ids,
    )
    .await?;

    if candidates.is_empty() {
        return Ok(vec![]);
    }

    // Apply sampling strategies
    let sampled = sample(candidates, candidate_count, sampling_options)?;

    Ok(sampled)
}

/// Find rooms by embedding similarity
async fn find_rooms_by_embedding_similarity(
    pool: &PgPool,
    schema: &str,
    embedding: &TextEmbedding,
    limit: usize,
    exclude_room_ids: &HashSet<RoomId>,
) -> Result<Vec<(Room, f64)>, MemoryPalaceError> {
    let model_name = format!("{}_{}", embedding.model.as_str(), "centroid");

    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            // Convert HashSet to Vec for SQL
            let exclude_ids: Vec<Uuid> =
                exclude_room_ids.iter().map(|id| id.0).collect();

            // First try rooms with centroid embeddings
            let query = if exclude_ids.is_empty() {
                sqlx::query_as(
                    r#"
                    SELECT r.*, 1 - (e.embedding <=> $1::vector) as similarity
                    FROM rooms r
                    JOIN room_embeddings re ON r.id = re.room_id
                    JOIN embeddings e ON re.embedding_id = e.id
                    WHERE e.model_name = $2
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#,
                )
                .bind(&embedding.embedding[..])
                .bind(&model_name)
                .bind(limit as i64)
            } else {
                sqlx::query_as(
                    r#"
                    SELECT r.*, 1 - (e.embedding <=> $1::vector) as similarity
                    FROM rooms r
                    JOIN room_embeddings re ON r.id = re.room_id
                    JOIN embeddings e ON re.embedding_id = e.id
                    WHERE e.model_name = $2
                    AND r.id != ALL($4::uuid[])
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#,
                )
                .bind(&embedding.embedding[..])
                .bind(&model_name)
                .bind(limit as i64)
                .bind(&exclude_ids)
            };

            let results: Vec<(Room, f64)> = query.fetch_all(&mut **tx).await?;

            if !results.is_empty() {
                return Ok(results);
            }

            // Fallback: find rooms based on memory similarity
            let fallback_query = if exclude_ids.is_empty() {
                sqlx::query_as(
                    r#"
                    WITH room_similarities AS (
                        SELECT 
                            r.*,
                            AVG(1 - (e.embedding <=> $1::vector)) as similarity
                        FROM memories m
                        JOIN rooms r ON m.room_id = r.id
                        JOIN memory_embeddings me ON m.id = me.memory_id
                        JOIN embeddings e ON me.embedding_id = e.id
                        WHERE e.model_name = $2
                        GROUP BY r.id
                    )
                    SELECT * FROM room_similarities
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#,
                )
                .bind(&embedding.embedding[..])
                .bind(&embedding.model)
                .bind(limit as i64)
            } else {
                sqlx::query_as(
                    r#"
                    WITH room_similarities AS (
                        SELECT 
                            r.*,
                            AVG(1 - (e.embedding <=> $1::vector)) as similarity
                        FROM memories m
                        JOIN rooms r ON m.room_id = r.id
                        JOIN memory_embeddings me ON m.id = me.memory_id
                        JOIN embeddings e ON me.embedding_id = e.id
                        WHERE e.model_name = $2
                        AND r.id != ALL($4::uuid[])
                        GROUP BY r.id
                    )
                    SELECT * FROM room_similarities
                    ORDER BY similarity DESC
                    LIMIT $3
                    "#,
                )
                .bind(&embedding.embedding[..])
                .bind(&embedding.model)
                .bind(limit as i64)
                .bind(&exclude_ids)
            };

            let results: Vec<(Room, f64)> =
                fallback_query.fetch_all(&mut **tx).await?;
            Ok(results)
        })
    })
    .await
}

/// Get all memories in a specific room
async fn get_room_memories(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
) -> Result<Vec<Memory>, MemoryPalaceError> {
    let room_name = room_name.to_string();
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT m.*
                FROM memories m
                JOIN rooms r ON m.room_id = r.id
                WHERE r.name = $1
                ORDER BY m.strength DESC, m.last_accessed DESC
                "#,
            )
            .bind(&room_name)
            .fetch_all(&mut **tx)
            .await?;

            Ok(memories)
        })
    })
    .await
}

/// Truncate content to a maximum length
fn truncate_content(content: &str, max_len: usize) -> &str {
    if content.len() <= max_len {
        content
    } else {
        &content[..content
            .char_indices()
            .take(max_len)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0)]
    }
}

/// Format tags for display
fn format_tags(tags: &[String]) -> String {
    if tags.is_empty() {
        String::new()
    } else {
        format!("[{}]", tags.join(", "))
    }
}

/// Calculate memory glow based on strength and recency
fn calculate_memory_glow(memory: &Memory) -> &'static str {
    let recency_days = (chrono::Utc::now() - memory.last_accessed).num_days();
    let recency_factor = (30.0 - recency_days.min(30) as f64) / 30.0;
    let access_factor = (memory.access_count as f64).ln().max(0.0) / 10.0;
    let combined =
        memory.strength * 0.6 + recency_factor * 0.3 + access_factor * 0.1;

    match combined {
        s if s > 0.8 => "🌟",
        s if s > 0.6 => "✨",
        s if s > 0.4 => "💫",
        _ => "🌑",
    }
}

/// Get brightest memories in a room
async fn get_brightest_memories_in_room(
    pool: &PgPool,
    schema: &str,
    room_id: RoomId,
    limit: usize,
) -> Result<Vec<Memory>, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let memories: Vec<Memory> = sqlx::query_as(
                r#"
                SELECT *
                FROM memories
                WHERE room_id = $1
                ORDER BY strength DESC, last_accessed DESC
                LIMIT $2
                "#,
            )
            .bind(room_id)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?;

            Ok(memories)
        })
    })
    .await
}

/// Format the scout report for narrative display
fn format_scout_report(
    rooms: &[BrightRoom],
    cumulative_relevance: f64,
) -> String {
    let mut report = format!(
        "Scout Report - Total Relevance: {:.1}\n\n",
        cumulative_relevance * 100.0
    );

    report.push_str("The palace responds to your query, rooms glowing with recognition:\n\n");

    for (room_idx, bright_room) in rooms.iter().enumerate() {
        let glow_desc = describe_glow(bright_room.relevance);

        report.push_str(&format!(
            "[Room #{}] {} - {}\n",
            room_idx, bright_room.room.name, glow_desc
        ));

        report.push_str(&format!("  {}\n", bright_room.room.description));

        if !bright_room.bright_spots.is_empty() {
            report.push_str("  Brightest memories:\n");
            for spot in &bright_room.bright_spots {
                let glow = describe_glow(spot.glow);
                report.push_str(&format!(
                    "  - [#{}] {} on {} - {}\n",
                    spot.ref_num,
                    glow,
                    spot.placement,
                    truncate_content(&spot.preview, 50)
                ));
            }
        }

        report.push('\n');
    }

    report
}

/// Convert strength/relevance to narrative description
fn describe_glow(strength: f64) -> &'static str {
    match strength {
        s if s > 0.9 => "🌟 Blazing",
        s if s > 0.7 => "✨ Brilliant",
        s if s > 0.5 => "💫 Glowing",
        s if s > 0.3 => "🌙 Shimmering",
        s if s > 0.1 => "⭐ Glimmering",
        _ => "🌑 Dim",
    }
}

/// Format room teleport narrative
fn format_room_teleport(room: &Room, memories: Vec<Memory>) -> String {
    let mut narrative = format!(
        "You materialize in {}.\n{}\n\n",
        room.name, room.description
    );

    if memories.is_empty() {
        narrative.push_str("The room awaits its first memories.");
    } else {
        narrative
            .push_str(&format!("You see {} memories here:\n", memories.len()));

        // Group by placement
        let mut by_placement = std::collections::HashMap::new();
        for memory in &memories {
            by_placement
                .entry(memory.placement.clone())
                .or_insert_with(Vec::new)
                .push(memory);
        }

        for (placement, mems) in by_placement {
            narrative.push_str(&format!("\nOn the {}:\n", placement));
            for (idx, mem) in mems.iter().enumerate().take(3) {
                let preview = mem
                    .content
                    .brief_description()
                    .unwrap_or_else(|| "A memory".to_string());
                narrative.push_str(&format!(
                    "- {}\n",
                    truncate_content(&preview, 60)
                ));
            }
            if mems.len() > 3 {
                narrative
                    .push_str(&format!("  ...and {} more\n", mems.len() - 3));
            }
        }
    }

    narrative
}

/// Format basket return
fn format_basket_return(basket: &[CollectedMemory], summary: &str) -> String {
    let mut result = format!("Returning with {} memories:\n\n", basket.len());

    for memory in basket {
        result.push_str(&format!(
            "- From {}: {}\n",
            memory.room_name,
            truncate_content(&memory.content, 60)
        ));
    }

    if !summary.is_empty() {
        result.push_str(&format!("\nSummary: {}", summary));
    }

    result
}

/// Follow a passage from current room to get destination
pub async fn follow_passage(
    pool: &PgPool,
    schema: &str,
    from_room: &str,
    direction: &str,
) -> Result<Room, MemoryPalaceError> {
    let from_room = from_room.to_string();
    let direction = direction.to_lowercase();

    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            // Get the current room
            let current_room: Room =
                sqlx::query_as("SELECT * FROM rooms WHERE name = $1")
                    .bind(&from_room)
                    .fetch_one(&mut **tx)
                    .await
                    .map_err(|_| {
                        MemoryPalaceError::RoomNotFound(from_room.clone())
                    })?;

            // Try to find a room by the direction (could be room name or passage type)
            let destination: Option<Room> = sqlx::query_as(
                r#"
                SELECT r.*
                FROM rooms r
                JOIN pathways p ON (
                    (p.room_a = $1 AND p.room_b = r.id) OR
                    (p.room_b = $1 AND p.room_a = r.id)
                )
                WHERE LOWER(r.name) = $2 OR LOWER(p.passage_type) = $2
                LIMIT 1
                "#,
            )
            .bind(current_room.id)
            .bind(&direction)
            .fetch_optional(&mut **tx)
            .await?;

            if let Some(room) = destination {
                // Update traversal tracking
                sqlx::query(
                    r#"UPDATE pathways 
                       SET traversal_count = traversal_count + 1,
                           last_traversed = NOW()
                       WHERE (room_a = $1 AND room_b = $2) 
                          OR (room_b = $1 AND room_a = $2)"#,
                )
                .bind(current_room.id)
                .bind(room.id)
                .execute(&mut **tx)
                .await?;

                return Ok(room);
            }

            Err(MemoryPalaceError::Other(format!(
                "No passage '{}' from {}",
                direction, from_room
            )))
        })
    })
    .await
}

/// Get a rich description of a room
pub async fn get_room_description(
    pool: &PgPool,
    schema: &str,
    room_name: &str,
) -> Result<String, MemoryPalaceError> {
    let room_name = room_name.to_string();

    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let room: Room =
                sqlx::query_as("SELECT * FROM rooms WHERE name = $1")
                    .bind(&room_name)
                    .fetch_one(&mut **tx)
                    .await
                    .map_err(|_| {
                        MemoryPalaceError::RoomNotFound(room_name.clone())
                    })?;

            let connections: Vec<(String, String)> = sqlx::query_as(
                r#"
                SELECT 
                    p.passage_type,
                    r.name as connected_room
                FROM pathways p
                JOIN rooms r ON r.id = CASE
                    WHEN p.room_a = $1 THEN p.room_b
                    ELSE p.room_a
                END
                WHERE p.room_a = $1 OR p.room_b = $1
                ORDER BY p.strength DESC, r.name
                "#,
            )
            .bind(room.id)
            .fetch_all(&mut **tx)
            .await?;

            let mut desc =
                format!("You enter {}. {}", room.name, room.description);

            if room.memory_count > 0 {
                desc.push_str(&format!(
                    "\n\nYou see {} memor{} here.",
                    room.memory_count,
                    if room.memory_count == 1 { "y" } else { "ies" }
                ));
            } else {
                desc.push_str("\n\nThe room is empty of memories.");
            }

            if !connections.is_empty() {
                desc.push_str("\n\nPassages lead:\n");
                for (passage_type, destination) in connections {
                    desc.push_str(&format!(
                        "- {} to {}\n",
                        passage_type, destination
                    ));
                }
            }

            Ok(desc)
        })
    })
    .await
}

/// Get memory by ID
async fn get_memory_by_id(
    pool: &PgPool,
    schema: &str,
    id: MemoryId,
) -> Result<Memory, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let memory = sqlx::query_as("SELECT * FROM memories WHERE id = $1")
                .bind(id)
                .fetch_one(&mut **tx)
                .await?;

            Ok(memory)
        })
    })
    .await
}

/// Get room by ID
async fn get_room_by_id(
    pool: &PgPool,
    schema: &str,
    id: RoomId,
) -> Result<Room, MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let room = sqlx::query_as("SELECT * FROM rooms WHERE id = $1")
                .bind(id)
                .fetch_one(&mut **tx)
                .await?;

            Ok(room)
        })
    })
    .await
}

/// Semantic search for memories
async fn semantic_search(
    pool: &PgPool,
    schema: &str,
    query: &str,
    limit: usize,
    embedding_client: &dyn EmbeddingClient,
) -> Result<Vec<ScoredMemory>, MemoryPalaceError> {
    let query_embedding = embedding_client
        .get_embedding(query)
        .await
        .map_err(|e| MemoryPalaceError::Other(e.to_string()))?;

    let model_name =
        format!("{}_{}", embedding_client.name(), embedding_client.model());

    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            let results: Vec<ScoredMemory> = sqlx::query_as(
                r#"
                SELECT 
                    m.*,
                    1 - (e.embedding <=> $1::vector) as similarity_score
                FROM memories m
                JOIN memory_embeddings me ON m.id = me.memory_id
                JOIN embeddings e ON me.embedding_id = e.id
                WHERE e.model_name = $2
                ORDER BY similarity_score DESC
                LIMIT $3
                "#,
            )
            .bind(&query_embedding.embedding[..])
            .bind(&model_name)
            .bind(limit as i64)
            .fetch_all(&mut **tx)
            .await?
            .into_iter()
            .map(|(memory, score)| ScoredMemory {
                memory,
                journey: PathById::from_members(vec![PathMemberIds::Memory(
                    memory.id,
                )])
                .unwrap(),
                relevance_score: score,
                recency_score: 0.0,
                relationship_score: 0.0,
                final_score: score,
            })
            .collect();

            Ok(results)
        })
    })
    .await
}

/// Strengthen a path through the palace
async fn strengthen_path(
    pool: &PgPool,
    schema: &str,
    path: &PathById,
) -> Result<(), MemoryPalaceError> {
    execute_with_schema(pool, schema, |tx| {
        Box::pin(async move {
            // Strengthen each pathway in the path
            let mut prev_room: Option<RoomId> = None;

            for member in path.iter() {
                match member {
                    PathMemberIds::Room(room_id) => {
                        // Update room strength and visit count
                        sqlx::query(
                            r#"UPDATE rooms 
                               SET strength = LEAST(1.0, strength + 0.05),
                                   visit_count = visit_count + 1,
                                   last_visited = NOW()
                               WHERE id = $1"#,
                        )
                        .bind(room_id)
                        .execute(&mut **tx)
                        .await?;

                        prev_room = Some(*room_id);
                    }
                    PathMemberIds::Pathway(pathway_id) => {
                        // Update pathway strength
                        sqlx::query(
                            r#"UPDATE pathways
                               SET strength = LEAST(1.0, strength + 0.05),
                                   traversal_count = traversal_count + 1,
                                   last_traversed = NOW()
                               WHERE id = $1"#,
                        )
                        .bind(pathway_id)
                        .execute(&mut **tx)
                        .await?;
                    }
                    PathMemberIds::Memory(memory_id) => {
                        // Update memory strength and access count
                        sqlx::query(
                            r#"UPDATE memories
                               SET strength = LEAST(1.0, strength + 0.1),
                                   access_count = access_count + 1,
                                   last_accessed = NOW()
                               WHERE id = $1"#,
                        )
                        .bind(memory_id)
                        .execute(&mut **tx)
                        .await?;
                    }
                }
            }

            Ok(())
        })
    })
    .await
}

/// Format memories found in a room for display
fn format_room_contents(room: &Room, memories: Vec<Memory>) -> String {
    if memories.is_empty() {
        return format!(
            "You examine {}. The room is empty, waiting for memories to be stored.",
            room.name
        );
    }

    let mut content = format!("Examining {}...\n\n", room.name);

    // Group memories by placement
    let mut by_placement: std::collections::HashMap<String, Vec<&Memory>> =
        std::collections::HashMap::new();

    for memory in &memories {
        by_placement
            .entry(memory.placement.clone())
            .or_default()
            .push(memory);
    }

    // Sort placements for consistent output
    let mut placements: Vec<_> = by_placement.keys().cloned().collect();
    placements.sort();

    for placement in placements {
        let placement_memories = &by_placement[&placement];

        content.push_str(&format!("On the {}:\n", placement));

        for mem in placement_memories.iter().take(5) {
            let glow = calculate_memory_glow(mem);
            let brief = mem
                .content
                .brief_description()
                .unwrap_or_else(|| "A memory".to_string());

            content.push_str(&format!(
                "- {} {}: \"{}\" (id: {})\n",
                glow,
                format_tags(&mem.tags),
                truncate_content(&brief, 60),
                mem.id.0
            ));
        }

        if placement_memories.len() > 5 {
            content.push_str(&format!(
                "  ...and {} more\n",
                placement_memories.len() - 5
            ));
        }

        content.push('\n');
    }

    content
}
