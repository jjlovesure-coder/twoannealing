#!/usr/bin/env bash
# SessionStart hook - injects superpowers using-superpowers bootstrap
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_FILE="$(cd "$SCRIPT_DIR/.." && pwd)/skills/using-superpowers/SKILL.md"

using_superpowers_content=$(cat "$SKILL_FILE" 2>&1 || echo "Error reading using-superpowers skill")

escape_for_json() {
    local s="$1"
    s="${s//\\/\\\\}"
    s="${s//\"/\\\"}"
    s="${s//$'\n'/\\n}"
    s="${s//$'\r'/\\r}"
    s="${s//$'\t'/\\t}"
    printf '%s' "$s"
}

session_context="<EXTREMELY_IMPORTANT>
You have superpowers.

**Below is the full content of your 'superpowers:using-superpowers' skill - your introduction to using skills. For all other skills, use the 'Skill' tool:**

${using_superpowers_content}
</EXTREMELY_IMPORTANT>"

escaped=$(escape_for_json "$session_context")
printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"%s"}}\n' "$escaped"
