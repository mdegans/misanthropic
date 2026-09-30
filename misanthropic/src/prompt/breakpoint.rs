//! Where a request's `cache_control` markers sit, and the rules Anthropic
//! holds them to — see [`Prompt::check_cache`]. The placement methods
//! ([`Prompt::cache`], [`Prompt::cache_windowed`], [`Prompt::auto_cache`], …)
//! plan here before they touch the prompt, so a placement the rules refuse
//! leaves it as it was.

use super::{
    MAX_CACHE_CONTROLS_PER_REQUEST, Prompt,
    index::{BlockIndex, Index, IndexMut},
    message::{CacheControl, CacheTtl},
};

/// Where a `cache_control` marker sits in a [`Prompt`]. The derived [`Ord`]
/// is the order Anthropic reads markers in: `tools`, `system`, `messages`,
/// then the automatic slot. [`Display`](std::fmt::Display) prints the path
/// Anthropic's own errors use (`messages.2.content.0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Breakpoint {
    /// On [`Prompt::tools`]`[i]`, custom or server.
    Tool(usize),
    /// On a [`Prompt::system`] or [`Prompt::messages`] block.
    Block(BlockIndex),
    /// The top-level automatic slot ([`Prompt::cache_control`]), read last.
    Auto,
}

impl std::fmt::Display for Breakpoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tool(i) => write!(f, "tools.{i}"),
            Self::Block(BlockIndex::System(i)) => write!(f, "system.{i}"),
            Self::Block(BlockIndex::Message((m, b))) => {
                write!(f, "messages.{m}.content.{b}")
            }
            Self::Auto => write!(f, "cache_control"),
        }
    }
}

/// `cache_control` markers Anthropic would reject with a 400, from
/// [`Prompt::check_cache`] or a placement that would build them.
///
/// Probed on the free `count_tokens` endpoint (claude-haiku-4-5,
/// 2026-09-30): markers are read in [`Breakpoint`] order, a 1-hour marker
/// may not follow a 5-minute one ("a ttl='1h' cache_control block must not
/// come after a ttl='5m' cache_control block"), and the automatic slot must
/// match the TTL of a marker on the block it lands on ("When both are
/// specified on the same block, they must have matching TTLs").
///
/// ```
/// use misanthropic::{Prompt, prompt::message::Role};
/// use misanthropic::prompt::{BlockIndex, Breakpoint, CacheError};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// // A manual kept for an hour, the conversation after it for five
/// // minutes: 1-hour markers first, as Anthropic reads them.
/// let prompt = Prompt::default()
///     .system("<a long manual>")
///     .cache_1h()?
///     .add_message((Role::User, "Where do I start?"))?
///     .cache();
/// assert!(prompt.check_cache().is_ok());
///
/// // The other way round is a 400, so it is never built.
/// let error = Prompt::default()
///     .system("<a long manual>")
///     .cache()
///     .add_message((Role::User, "Where do I start?"))?
///     .cache_1h()
///     .unwrap_err();
/// assert_eq!(
///     error,
///     CacheError::TtlOrder {
///         earlier: Breakpoint::Block(BlockIndex::System(0)),
///         later: Breakpoint::Block(BlockIndex::Message((0, 0))),
///     }
/// );
/// # Ok(())
/// # }
/// ```
// Deliberately not `Serialize`, like `TurnOrderError`: an un-unwrapped
// placement `Result` must not pass an `impl Serialize` bound to the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    /// More markers than Anthropic accepts — each marked block counts, and
    /// the automatic slot does too, even on a marked block.
    #[error(
        "{found} cache_control markers; Anthropic accepts at most \
         {MAX_CACHE_CONTROLS_PER_REQUEST}"
    )]
    TooMany {
        /// How many the request carries.
        found: usize,
    },
    /// The automatic slot lands on a block with a marker of its own, and
    /// their TTLs differ.
    #[error(
        "the automatic cache_control lands on {at}, whose own marker has a \
         different ttl; they must match"
    )]
    AutoMismatch {
        /// The block both mark.
        at: Breakpoint,
    },
    /// A 1-hour marker after a 5-minute one.
    #[error(
        "{later}: a 1h cache_control must not come after the 5m one at \
         {earlier}"
    )]
    TtlOrder {
        /// The first 5-minute marker.
        earlier: Breakpoint,
        /// The first 1-hour marker after it.
        later: Breakpoint,
    },
}
static_assertions::assert_impl_all!(CacheError: Send, Sync);

/// A marker as the rules see it: where, and whether it lives an hour.
#[derive(Clone, Copy, Debug)]
struct Mark {
    at: Breakpoint,
    hour: bool,
}

impl Mark {
    fn new(at: Breakpoint, cache_control: &CacheControl) -> Self {
        let hour = matches!(cache_control.ttl(), CacheTtl::OneHour);
        Self { at, hour }
    }

    /// The message this marker is in, if it is in one.
    fn message(&self) -> Option<usize> {
        match self.at {
            Breakpoint::Block(BlockIndex::Message((m, _))) => Some(m),
            _ => None,
        }
    }
}

/// A placement worked out on the side: the markers the request would carry,
/// and the edits that get it there. Applying it is all that touches the
/// prompt, so a refused plan changes nothing.
#[derive(Debug)]
pub(super) struct Plan {
    /// Every marker the request would carry, in wire order.
    marks: Vec<Mark>,
    /// Where the automatic slot would land.
    target: Option<Breakpoint>,
    /// Positions to mark with `cache_control`.
    fresh: Vec<Breakpoint>,
    /// Positions to unmark, to fit the budget.
    evicted: Vec<Breakpoint>,
    cache_control: CacheControl,
}

impl Plan {
    /// This plan, if the request it leaves passes [`Prompt::check_cache`].
    pub(super) fn checked(self) -> Result<Self, CacheError> {
        check(&self.marks, self.target)?;
        Ok(self)
    }
}

impl Prompt {
    /// Check this request's `cache_control` markers against the rules
    /// Anthropic rejects a request for (see [`CacheError`]): at most 4,
    /// counting the automatic slot; no 1-hour marker after a 5-minute one,
    /// in `tools` → `system` → `messages` order with the automatic slot
    /// last; and an automatic slot landing on a marked block matches its
    /// TTL.
    ///
    /// The placement methods never build a request that fails this, but
    /// markers set by hand (or [`Block::cache_with`], or a pushed message)
    /// can; [`Chat`](crate::Chat) checks before every request.
    ///
    /// [`Block::cache_with`]: super::message::Block::cache_with
    pub fn check_cache(&self) -> Result<(), CacheError> {
        check(&self.marks(), self.auto_target())
    }

    /// How many `cache_control` markers this request carries, as Anthropic
    /// counts them.
    pub(super) fn cache_markers(&self) -> usize {
        self.marks().len()
    }

    /// Every marker, in wire order.
    fn marks(&self) -> Vec<Mark> {
        let tools = self.tools.iter().flatten().enumerate();
        let tools = tools.filter_map(|(i, tool)| {
            Some(Mark::new(Breakpoint::Tool(i), tool.cache_control()?))
        });
        let system = self.system.iter().flat_map(|system| {
            system.iter().enumerate().filter_map(|(i, block)| {
                let at = Breakpoint::Block(BlockIndex::System(i));
                Some(Mark::new(at, block.cache_control()?))
            })
        });
        let messages = self.messages.iter().enumerate().flat_map(|(m, msg)| {
            msg.content
                .iter()
                .enumerate()
                .filter_map(move |(b, block)| {
                    let at = Breakpoint::Block(BlockIndex::Message((m, b)));
                    Some(Mark::new(at, block.cache_control()?))
                })
        });
        let auto = self.cache_control.iter();
        let auto = auto.map(|auto| Mark::new(Breakpoint::Auto, auto));
        tools.chain(system).chain(messages).chain(auto).collect()
    }

    /// Where the automatic slot lands: the last block that can carry a
    /// marker, the last tool when there is none.
    fn auto_target(&self) -> Option<Breakpoint> {
        let messages = self.messages.iter().enumerate().rev();
        let messages = messages.flat_map(|(m, message)| {
            let blocks = message.content.iter().enumerate().rev();
            blocks.map(move |(b, block)| (BlockIndex::Message((m, b)), block))
        });
        let system = self.system.iter().flat_map(|system| {
            let blocks = system.iter().enumerate().rev();
            blocks.map(|(i, block)| (BlockIndex::System(i), block))
        });
        messages
            .chain(system)
            .find(|(_, block)| block.cache_slot().is_some())
            .map(|(at, _)| Breakpoint::Block(at))
            .or_else(|| {
                let last = self.tools.as_ref()?.len().checked_sub(1)?;
                Some(Breakpoint::Tool(last))
            })
    }

    /// The last block of `messages[m]`, if it can carry a marker.
    pub(super) fn message_end(&self, m: usize) -> Option<Breakpoint> {
        let content = &self.messages.get(m)?.content;
        let b = content.len().checked_sub(1)?;
        content[b]
            .cache_slot()
            .map(|_| Breakpoint::Block(BlockIndex::Message((m, b))))
    }

    /// Plan marking `targets` with `cache_control`, then sliding the message
    /// window to fit the budget: `kept` says which markers stay, the
    /// messages in `keep` ranking first within a TTL. `tools`, `system` and
    /// the automatic slot are never evicted.
    ///
    /// A 5-minute block marker yields to a 1-hour marker at or after its
    /// place that the request keeps, which already caches that prefix, for
    /// longer: placing it would break the TTL order (or the automatic slot's
    /// match), so it is left out. That makes a 5-minute block placement
    /// never the cause of a [`CacheError`].
    pub(super) fn plan(
        &self,
        targets: impl IntoIterator<Item = Breakpoint>,
        cache_control: CacheControl,
        keep: &[usize],
    ) -> Plan {
        let hour = matches!(cache_control.ttl(), CacheTtl::OneHour);
        let yields = |at: Breakpoint, over: &Mark| {
            !hour && at != Breakpoint::Auto && over.hour && over.at >= at
        };
        let mut marks = self.marks();
        // Markers outside `messages` stay, and a 1-hour one is never
        // replaced by a 5-minute one, so both decide before the fit.
        let mut fresh: Vec<Breakpoint> = targets
            .into_iter()
            .filter(|&at| {
                !marks.iter().any(|mark| {
                    yields(at, mark)
                        && (mark.at == at || mark.message().is_none())
                })
            })
            .collect();
        let placed: Vec<Breakpoint> =
            marks.iter().map(|mark| mark.at).collect();
        for &at in &fresh {
            let mark = Mark { at, hour };
            match marks.binary_search_by_key(&at, |mark| mark.at) {
                Ok(i) => marks[i] = mark,
                Err(i) => marks.insert(i, mark),
            }
        }

        // Then against the 1-hour message markers the fit keeps, not every
        // one there is: an evicted one covers nothing. The fit ranks them
        // before any 5-minute marker, so leaving one of those out changes
        // nothing about which.
        let lasting = kept(&marks, keep);
        let skipped: Vec<Breakpoint> = fresh
            .iter()
            .copied()
            .filter(|&at| {
                marks
                    .iter()
                    .filter(|mark| lasting.contains(&mark.at))
                    .any(|mark| yields(at, mark))
            })
            .collect();
        fresh.retain(|at| !skipped.contains(at));
        marks.retain(|mark| {
            !skipped.contains(&mark.at) || placed.contains(&mark.at)
        });

        let kept = kept(&marks, keep);
        let evicted: Vec<Breakpoint> = marks
            .iter()
            .map(|mark| mark.at)
            .filter(|at| !kept.contains(at))
            .collect();
        marks.retain(|mark| kept.contains(&mark.at));
        fresh.retain(|at| kept.contains(at));
        Plan {
            marks,
            target: self.auto_target(),
            fresh,
            evicted,
            cache_control,
        }
    }

    /// Carry out `plan`.
    pub(super) fn apply(&mut self, plan: Plan) {
        let Plan {
            fresh,
            evicted,
            cache_control,
            ..
        } = plan;
        for at in fresh {
            self.set_cache_control(at, Some(cache_control.clone()));
        }
        for at in evicted {
            self.set_cache_control(at, None);
        }
    }

    /// Mark (or, with `None`, unmark) the marker slot at `at`.
    fn set_cache_control(
        &mut self,
        at: Breakpoint,
        cache_control: Option<CacheControl>,
    ) {
        match at {
            Breakpoint::Tool(i) => {
                let tool =
                    self.tools.as_mut().and_then(|tools| tools.get_mut(i));
                // Tool markers are placed, never evicted.
                if let (Some(tool), Some(cache_control)) = (tool, cache_control)
                {
                    tool.cache_with(cache_control);
                }
            }
            Breakpoint::Block(index) => {
                if let Some(IndexMut::Block(block)) =
                    self.get_mut(Index::Block(index))
                {
                    match cache_control {
                        Some(cache_control) => block.cache_with(cache_control),
                        None => block.uncache(),
                    };
                }
            }
            Breakpoint::Auto => self.cache_control = cache_control,
        }
    }
}

/// The markers a request keeps within Anthropic's 4-marker budget: every one
/// outside `messages`, then the message markers in rank. 1-hour markers rank
/// first — theirs are the entries still there after a pause of more than
/// five minutes, and they must come first on the wire anyway — then, within
/// a TTL, those of the messages in `keep` (in its order, a message's last
/// block first), then the newest.
fn kept(marks: &[Mark], keep: &[usize]) -> Vec<Breakpoint> {
    let (messages, prefix): (Vec<&Mark>, Vec<&Mark>) =
        marks.iter().partition(|mark| mark.message().is_some());
    let budget = MAX_CACHE_CONTROLS_PER_REQUEST.saturating_sub(prefix.len());
    let window = |mark: &Mark| {
        let m = mark.message();
        let at = keep.iter().position(|&k| Some(k) == m);
        at.unwrap_or(keep.len())
    };
    let mut ranked: Vec<&Mark> = messages.into_iter().rev().collect();
    // Stable, so the newest go first within a rank.
    ranked.sort_by_key(|mark| (!mark.hour, window(mark)));
    let ranked = ranked.into_iter().take(budget);
    prefix
        .into_iter()
        .chain(ranked)
        .map(|mark| mark.at)
        .collect()
}

/// The first rule `marks` (in wire order) breaks, if any, with the
/// automatic slot landing on `target`.
fn check(marks: &[Mark], target: Option<Breakpoint>) -> Result<(), CacheError> {
    if marks.len() > MAX_CACHE_CONTROLS_PER_REQUEST {
        return Err(CacheError::TooMany { found: marks.len() });
    }

    let hour_at = |at| marks.iter().find(|mark| mark.at == at).map(|m| m.hour);
    if let (Some(auto), Some(at)) = (hour_at(Breakpoint::Auto), target)
        && hour_at(at).is_some_and(|own| own != auto)
    {
        return Err(CacheError::AutoMismatch { at });
    }

    let earlier = marks.iter().find(|mark| !mark.hour);
    let later = earlier
        .and_then(|earlier| marks.iter().find(|m| m.hour && m.at > earlier.at));
    match (earlier, later) {
        (Some(earlier), Some(later)) => Err(CacheError::TtlOrder {
            earlier: earlier.at,
            later: later.at,
        }),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        CachedPrompt,
        prompt::message::{Block, Content, Message, Role},
        tool::{CustomMethodDef, MethodDef},
    };

    fn five() -> Option<CacheControl> {
        Some(CacheControl::ephemeral())
    }

    fn hour() -> Option<CacheControl> {
        Some(CacheControl::one_hour())
    }

    /// An explicit `"ttl": "5m"`, which the wire treats as an omitted one.
    fn five_explicit() -> Option<CacheControl> {
        Some(CacheControl::Ephemeral {
            ttl: Some(CacheTtl::FiveMinutes),
        })
    }

    fn text(cache_control: Option<CacheControl>) -> Block {
        Block::Text {
            text: "x".into(),
            citations: None,
            cache_control,
        }
    }

    /// A prompt with `system` blocks (if any) and `turns`, alternating user
    /// and assistant, one block per `cache_control` given.
    fn prompt(
        system: &[Option<CacheControl>],
        turns: &[&[Option<CacheControl>]],
    ) -> Prompt {
        let blocks = |marks: &[Option<CacheControl>]| {
            Content(marks.iter().cloned().map(text).collect())
        };
        let messages = turns.iter().enumerate().map(|(i, marks)| Message {
            role: [Role::User, Role::Assistant][i % 2],
            content: blocks(marks),
        });
        Prompt {
            system: (!system.is_empty()).then(|| blocks(system)),
            messages: messages.collect(),
            ..Default::default()
        }
    }

    fn with_tool(
        mut prompt: Prompt,
        cache_control: Option<CacheControl>,
    ) -> Prompt {
        let mut tool = MethodDef::Custom(CustomMethodDef::simple("t", "d"));
        if let Some(cache_control) = cache_control {
            tool.cache_with(cache_control);
        }
        prompt.tools = Some(vec![tool]);
        prompt
    }

    fn with_auto(mut prompt: Prompt, auto: Option<CacheControl>) -> Prompt {
        prompt.cache_control = auto;
        prompt
    }

    fn system(i: usize) -> Breakpoint {
        Breakpoint::Block(BlockIndex::System(i))
    }

    fn message(m: usize, b: usize) -> Breakpoint {
        Breakpoint::Block(BlockIndex::Message((m, b)))
    }

    /// The `count_tokens` probes of 2026-09-30 (claude-haiku-4-5), each a
    /// request and what the API said: 200, or the 400 we report the same
    /// markers for.
    #[test]
    fn check_cache_agrees_with_the_probes() {
        use CacheError::*;

        let unmarked: &[Option<CacheControl>] = &[None];
        let cases: Vec<(&str, Prompt, Result<(), CacheError>)> = vec![
            (
                "5m system, 1h message",
                prompt(&[five()], &[&[hour()]]),
                Err(TtlOrder {
                    earlier: system(0),
                    later: message(0, 0),
                }),
            ),
            (
                "1h system, 5m message",
                prompt(&[hour()], &[&[five()]]),
                Ok(()),
            ),
            (
                "5m tool, 1h system",
                with_tool(prompt(&[hour()], &[unmarked]), five()),
                Err(TtlOrder {
                    earlier: Breakpoint::Tool(0),
                    later: system(0),
                }),
            ),
            (
                "1h tool, 5m system",
                with_tool(prompt(&[five()], &[unmarked]), hour()),
                Ok(()),
            ),
            (
                "explicit 5m, then 1h",
                prompt(&[], &[&[five_explicit()], &[hour()], unmarked]),
                Err(TtlOrder {
                    earlier: message(0, 0),
                    later: message(1, 0),
                }),
            ),
            (
                "1h, 5m, 1h",
                prompt(&[], &[&[hour()], &[five()], &[hour()]]),
                Err(TtlOrder {
                    earlier: message(1, 0),
                    later: message(2, 0),
                }),
            ),
            (
                "5m last block, 1h auto",
                with_auto(prompt(&[], &[&[five()]]), hour()),
                Err(AutoMismatch { at: message(0, 0) }),
            ),
            (
                "1h last block, 5m auto",
                with_auto(prompt(&[], &[&[hour()]]), five()),
                Err(AutoMismatch { at: message(0, 0) }),
            ),
            (
                "5m system, unmarked end, 1h auto",
                with_auto(prompt(&[five()], &[unmarked]), hour()),
                Err(TtlOrder {
                    earlier: system(0),
                    later: Breakpoint::Auto,
                }),
            ),
            (
                "5m last block, 5m auto",
                with_auto(prompt(&[], &[&[five()]]), five()),
                Ok(()),
            ),
            (
                "1h last block, 1h auto",
                with_auto(prompt(&[], &[&[hour()]]), hour()),
                Ok(()),
            ),
            (
                "5m last block, explicit 5m auto",
                with_auto(prompt(&[], &[&[five()]]), five_explicit()),
                Ok(()),
            ),
            (
                "earlier 1h, 5m auto",
                with_auto(
                    prompt(&[], &[&[hour()], unmarked, unmarked]),
                    five(),
                ),
                Ok(()),
            ),
            (
                "earlier 1h, 1h auto",
                with_auto(
                    prompt(&[], &[&[hour()], unmarked, unmarked]),
                    hour(),
                ),
                Ok(()),
            ),
            (
                "earlier 5m, last 1h, 1h auto",
                with_auto(
                    prompt(&[], &[&[five()], unmarked, &[hour()]]),
                    hour(),
                ),
                Err(TtlOrder {
                    earlier: message(0, 0),
                    later: message(2, 0),
                }),
            ),
            (
                "1h on the last message's first block, 5m auto",
                with_auto(prompt(&[], &[&[hour(), None]]), five()),
                Ok(()),
            ),
            (
                "1h user, unmarked assistant prefill, 5m auto",
                with_auto(prompt(&[], &[&[hour()], unmarked]), five()),
                Ok(()),
            ),
            (
                "1h system, unmarked message, 5m auto",
                with_auto(prompt(&[hour()], &[unmarked]), five()),
                Ok(()),
            ),
            (
                "four 5m blocks and a 5m auto",
                with_auto(
                    prompt(&[five()], &[&[five()], &[five()], &[five()]]),
                    five(),
                ),
                Err(TooMany { found: 5 }),
            ),
        ];

        for (name, prompt, expected) in cases {
            assert_eq!(prompt.check_cache(), expected, "{name}");
        }
    }

    #[test]
    fn breakpoints_print_anthropic_paths() {
        let printed = [
            Breakpoint::Tool(1),
            system(0),
            message(2, 3),
            Breakpoint::Auto,
        ]
        .map(|at| at.to_string());
        assert_eq!(
            printed,
            [
                "tools.1",
                "system.0",
                "messages.2.content.3",
                "cache_control"
            ]
        );
        let error = CacheError::TtlOrder {
            earlier: system(0),
            later: message(0, 0),
        };
        assert_eq!(
            error.to_string(),
            "messages.0.content.0: a 1h cache_control must not come after \
             the 5m one at system.0"
        );
    }

    /// A placement step, on a [`CachedPrompt`] (whose `&mut` placements
    /// keep the prompt on an error).
    #[derive(Clone, Copy, Debug)]
    enum Step {
        Cache,
        Cache1h,
        Window,
        Window1h,
        Auto,
        Auto1h,
        Turn,
    }

    impl Step {
        const ALL: [Step; 7] = [
            Step::Cache,
            Step::Cache1h,
            Step::Window,
            Step::Window1h,
            Step::Auto,
            Step::Auto1h,
            Step::Turn,
        ];

        fn take(self, cached: &mut CachedPrompt) -> Result<(), CacheError> {
            match self {
                Step::Cache => drop(cached.cache()),
                Step::Cache1h => drop(cached.cache_1h()?),
                Step::Window => cached.cache_windowed(2),
                Step::Window1h => cached.cache_windowed_1h(2)?,
                Step::Auto => cached.set_auto_cache()?,
                Step::Auto1h => cached.set_auto_cache_1h()?,
                Step::Turn => {
                    cached.push_message((Role::User, "u")).unwrap();
                    cached.push_message((Role::Assistant, "a")).unwrap();
                }
            }
            Ok(())
        }
    }

    /// Every sequence of 5 steps from a tool and a system prompt: whatever
    /// the order, a placement either leaves a request Anthropic accepts or
    /// refuses and changes nothing.
    #[test]
    fn placements_never_build_a_400() {
        let start = with_tool(prompt(&[None], &[]), None);
        let mut sequences = vec![vec![]];
        for _ in 0..5 {
            sequences = sequences
                .into_iter()
                .flat_map(|steps: Vec<Step>| {
                    Step::ALL.map(|step| [steps.clone(), vec![step]].concat())
                })
                .collect();
        }

        for steps in sequences {
            let mut cached = CachedPrompt::from(start.clone());
            for (i, &step) in steps.iter().enumerate() {
                let before = cached.clone();
                match step.take(&mut cached) {
                    Ok(()) => assert_eq!(
                        cached.check_cache(),
                        Ok(()),
                        "{:?}",
                        &steps[..=i]
                    ),
                    Err(_) => assert!(cached == before, "{:?}", &steps[..=i]),
                }
            }
        }
    }

    /// A 1-hour marker after a 5-minute one is refused, the prompt kept.
    #[test]
    fn cache_1h_after_a_five_minute_marker_is_refused() {
        let mut cached = CachedPrompt::cached(Prompt::default().system("s"));
        cached.push_message((Role::User, "hi")).unwrap();
        let before = cached.clone();

        let error = cached.cache_1h().unwrap_err();

        assert_eq!(
            error,
            CacheError::TtlOrder {
                earlier: system(0),
                later: message(0, 0),
            }
        );
        assert!(cached == before);
    }

    /// Under a 1-hour marker at or after the end, a 5-minute one adds
    /// nothing and would be a 400, so none is placed.
    #[test]
    fn five_minute_placements_yield_to_one_hour_markers() {
        let mut cached = CachedPrompt::from(Prompt::default());
        cached.push_message((Role::User, "u")).unwrap();
        cached.set_auto_cache_1h().unwrap();
        cached.cache();
        assert!(!cached.messages[0].content.has_cache(), "under the slot");

        let mut cached = CachedPrompt::from(Prompt::default());
        cached.push_message((Role::User, "u0")).unwrap();
        cached.push_message((Role::Assistant, "a0")).unwrap();
        cached.push_message((Role::User, "u1")).unwrap();
        cached.push_message((Role::Assistant, "a1")).unwrap();
        cached.cache_1h().unwrap();
        cached.cache_windowed(2);
        assert!(!cached.messages[1].content.has_cache(), "before the 1h");
        cached.cache();
        let end = cached.messages[3].content.last().unwrap();
        assert!(matches!(
            end.cache_control().unwrap().ttl(),
            CacheTtl::OneHour
        ));
        assert_eq!(cached.check_cache(), Ok(()));
    }

    /// The automatic slot must match a marker on the block it lands on.
    #[test]
    fn auto_cache_refuses_a_one_hour_end() {
        let prompt = Prompt::default()
            .add_message((Role::User, "hi"))
            .unwrap()
            .cache_1h()
            .unwrap();

        let error = prompt.auto_cache().unwrap_err();

        assert_eq!(error, CacheError::AutoMismatch { at: message(0, 0) });
    }

    /// A refused window changes nothing, not even its eviction.
    #[test]
    fn a_refused_window_changes_nothing() {
        let mut cached = CachedPrompt::from(Prompt::default());
        for i in 0..5 {
            cached.push_message((Role::User, format!("u{i}"))).unwrap();
            cached
                .push_message((Role::Assistant, format!("a{i}")))
                .unwrap();
            // The last turn is left for the window, whose marker would
            // evict the first.
            if i < 4 {
                cached.cache();
            }
        }
        let before = cached.clone();

        let error = cached.cache_windowed_1h(2).unwrap_err();

        assert!(matches!(error, CacheError::TtlOrder { .. }));
        assert!(cached == before);
    }
}
