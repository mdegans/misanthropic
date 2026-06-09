#!/usr/bin/env bash
# Capture a streaming Messages API response as a wrapped SSE-JSONL fixture.
#
# Usage:
#   ./capture.sh request.json [beta[,beta...]] > fixture.sse.stream.jsonl
#
# The request JSON is sent as-is except `stream` is forced to true. Each SSE
# `data:` payload is emitted as one line, wrapped `{"Ok":<raw bytes>}` — or
# `{"Err":<raw bytes>}` for error events — by pure text concatenation. The
# payload bytes are NEVER parsed through this repo's types (that would hide
# dropped fields, the exact drift the fixtures exist to catch). See README.md.
#
# The API key is read from api.key in the crate root (same file the
# #[ignore]d live tests use).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
key_file="$here/../../api.key"
request="${1:?usage: capture.sh request.json [beta,beta...]}"
betas="${2:-}"

[[ -f "$key_file" ]] || { echo "no api.key at $key_file" >&2; exit 1; }

args=(
  -sN --fail-with-body
  https://api.anthropic.com/v1/messages
  -H "x-api-key: $(tr -d '[:space:]' <"$key_file")"
  -H "anthropic-version: 2023-06-01"
  -H "content-type: application/json"
)
[[ -n "$betas" ]] && args+=(-H "anthropic-beta: $betas")

# Force stream:true; jq here only touches the *request*, never the response.
jq -c '.stream = true' "$request" | curl "${args[@]}" -d @- |
  while IFS= read -r line; do
    payload="${line#data: }"
    [[ "$payload" == "$line" ]] && continue # not a data: line
    case "$payload" in
      '{"type":"error"'*) printf '{"Err":%s}\n' "$payload" ;;
      *) printf '{"Ok":%s}\n' "$payload" ;;
    esac
  done
