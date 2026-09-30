//! What the chat does with a finished assistant turn: run its calls, seat it,
//! or drop it. Shared so the frontend and backend keep the same history.

use misanthropic::{
    prompt::{
        message::{Content, Role},
        Message, UserMessage,
    },
    response::{self, StopReason},
    tool,
};

/// What to do with a finished [`response::Message`].
#[derive(Debug)]
pub enum Disposition {
    /// Seat the turn; nothing to run.
    Keep,
    /// Seat the turn, run every call, and answer them all with one [`reply`].
    Dispatch(Vec<tool::Use>),
    /// Don't seat the turn, then [`rewind`]. It was refused, or it holds calls
    /// its stop reason says must not run — and an unanswered `tool_use` makes
    /// the next request a 400. Carries the stop reason for the notice.
    Drop(Option<StopReason>),
}

impl From<&response::Message> for Disposition {
    fn from(message: &response::Message) -> Self {
        let calls: Vec<_> = message.tool_uses().cloned().collect();
        let refused = matches!(message.stop_reason, Some(StopReason::Refusal));
        // Anthropic: discard a refused turn. Stripping just the calls could
        // strand a `server_tool_use`, so the whole turn goes.
        if refused || (calls.is_empty() && has_calls(&message.inner.content)) {
            Disposition::Drop(message.stop_reason)
        } else if calls.is_empty() {
            Disposition::Keep
        } else {
            Disposition::Dispatch(calls)
        }
    }
}

/// Every result of a [`Disposition::Dispatch`] as one user turn: parallel
/// calls are answered together, never one turn per result.
pub fn reply<Rs>(results: Rs) -> UserMessage
where
    Rs: IntoIterator<Item = tool::Result>,
{
    results.into_iter().collect()
}

/// After a [`Disposition::Drop`], pop trailing turns until a new user message
/// may follow — the history is empty or ends in an assistant turn with no
/// client calls — so the turn that prompted the dropped one goes too (a user
/// turn can't follow a user turn). Returns the removed turns, oldest first.
pub fn rewind(messages: &mut Vec<Message>) -> Vec<Message> {
    let keep = messages
        .iter()
        .rposition(|m| m.role == Role::Assistant && !has_calls(&m.content))
        .map_or(0, |i| i + 1);
    messages.split_off(keep)
}

/// Does the turn carry any client `tool_use` block?
fn has_calls(content: &Content) -> bool {
    content.iter().any(|block| block.tool_use().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two parallel calls, answerable: `stop_reason` is `tool_use`.
    const PARALLEL: &str = r#"{
        "id": "msg_parallel", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5", "stop_reason": "tool_use",
        "stop_sequence": null, "usage": {},
        "content": [
            {"type": "text", "text": "Checking both."},
            {"type": "tool_use", "id": "toolu_a", "name": "weather",
             "input": {"city": "Paris"}},
            {"type": "tool_use", "id": "toolu_b", "name": "weather",
             "input": {"city": "Rome"}}
        ]
    }"#;

    /// A closed call, then a refusal (synthetic: `PARALLEL`'s shape with the
    /// stop reason swapped).
    const REFUSED: &str = r#"{
        "id": "msg_refused", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5", "stop_reason": "refusal",
        "stop_sequence": null, "usage": {},
        "content": [
            {"type": "tool_use", "id": "toolu_a", "name": "weather",
             "input": {"city": "Paris"}}
        ]
    }"#;

    /// A call clipped by `max_tokens` mid-input (synthetic).
    const CLIPPED: &str = r#"{
        "id": "msg_clipped", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5", "stop_reason": "max_tokens",
        "stop_sequence": null, "usage": {},
        "content": [
            {"type": "text", "text": "Let me look"},
            {"type": "tool_use", "id": "toolu_a", "name": "weather",
             "input": {}}
        ]
    }"#;

    /// Plain text clipped by `max_tokens`: legal to seat and continue.
    const CLIPPED_TEXT: &str = r#"{
        "id": "msg_clipped_text", "type": "message", "role": "assistant",
        "model": "claude-haiku-4-5", "stop_reason": "max_tokens",
        "stop_sequence": null, "usage": {},
        "content": [{"type": "text", "text": "Once upon a"}]
    }"#;

    fn parse(json: &str) -> response::Message {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn parallel_dispatches_every_call() {
        let Disposition::Dispatch(calls) = (&parse(PARALLEL)).into() else {
            panic!("parallel calls should dispatch");
        };
        let ids: Vec<_> = calls.iter().map(|c| c.id.to_string()).collect();
        assert_eq!(ids, ["toolu_a", "toolu_b"]);

        let results = calls
            .into_iter()
            .map(|call| tool::Result::new(call.id, "sunny"));
        let reply = reply(results);
        let answered: Vec<_> = reply
            .content
            .iter()
            .filter_map(|block| match block {
                misanthropic::prompt::message::Block::ToolResult { result } => {
                    Some(result.tool_use_id.to_string())
                }
                _ => None,
            })
            .collect();
        assert_eq!(answered, ["toolu_a", "toolu_b"]);
    }

    #[test]
    fn refused_and_clipped_calls_drop() {
        assert!(matches!(
            (&parse(REFUSED)).into(),
            Disposition::Drop(Some(StopReason::Refusal))
        ));
        assert!(matches!(
            (&parse(CLIPPED)).into(),
            Disposition::Drop(Some(StopReason::MaxTokens))
        ));
    }

    #[test]
    fn text_refusal_drops_but_clipped_text_keeps() {
        let mut refused = parse(CLIPPED_TEXT);
        assert!(matches!((&refused).into(), Disposition::Keep));
        refused.stop_reason = Some(StopReason::Refusal);
        assert!(matches!((&refused).into(), Disposition::Drop(_)));
    }

    /// Dropping a turn rewinds past the user turn that prompted it, and past
    /// a tool-result reply to the call that led there.
    #[test]
    fn rewind_to_a_resting_point() {
        let user = |text: &str| Message::from(UserMessage::from(text));
        let assistant = |json| Message::from(parse(json).inner);
        let answered = parse(PARALLEL)
            .tool_uses()
            .map(|call| tool::Result::new(call.id.clone(), "sunny"))
            .collect::<UserMessage>();

        let mut messages = vec![
            user("hi"),
            assistant(CLIPPED_TEXT),
            user("weather?"),
            assistant(PARALLEL),
            answered.into(),
        ];
        let removed = rewind(&mut messages);
        assert_eq!(messages.len(), 2);
        assert_eq!(removed.len(), 3);
        assert!(removed[0].role == Role::User);

        let mut first = vec![user("hi")];
        assert_eq!(rewind(&mut first).len(), 1);
        assert!(first.is_empty());

        // Already resting: nothing to do.
        assert!(rewind(&mut messages).is_empty());
    }
}
