#![allow(dead_code)] // for doc tests

#[doc = include_str!("../../.claude/skills/misan-messages-api/SKILL.md")]
struct MessagesSkill;

#[doc = include_str!("../../.claude/skills/misan-streaming-api/SKILL.md")]
struct StreamingSkill;

// Per-tool resource files the messages-api SKILL.md points at (progressive
// disclosure). Doc-tested here so their code blocks can't drift either.
#[doc = include_str!("../../.claude/skills/misan-messages-api/MEMORY.md")]
struct MemorySkill;

#[doc = include_str!("../../.claude/skills/misan-messages-api/TEXT_EDITOR.md")]
struct TextEditorSkill;

#[doc = include_str!("../../.claude/skills/misan-messages-api/BASH.md")]
struct BashSkill;
