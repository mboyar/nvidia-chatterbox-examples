#!/usr/bin/env bash
#
# Install Jetson Voice Clone Skill into Antigravity (AGY)
# Works for both Global configuration (~/.gemini/config/skills)
# and Workspace configuration (<project>/.agents/skills)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_NAME="jetson-voice-clone"

echo "=========================================================="
echo "🤖 Antigravity (AGY) Jetson Voice Clone Skill Installer"
echo "=========================================================="

DEST_MODE="${1:-global}"

if [ "$DEST_MODE" = "workspace" ] || [ "$DEST_MODE" = "local" ]; then
    TARGET_DIR="${2:-.}/.agents/skills/${SKILL_NAME}"
    echo "• Target: Workspace-local skill (${TARGET_DIR})"
else
    TARGET_DIR="${HOME}/.gemini/config/skills/${SKILL_NAME}"
    echo "• Target: Global AGY skill (${TARGET_DIR})"
fi

SCRIPT_REAL="$(readlink -f "$SCRIPT_DIR" || echo "$SCRIPT_DIR")"
TARGET_REAL="$(readlink -f "$TARGET_DIR" 2>/dev/null || echo "$TARGET_DIR")"

if [ "$SCRIPT_REAL" = "$TARGET_REAL" ]; then
    echo "✓ Skill is already in place at: $TARGET_DIR"
    echo "=========================================================="
    exit 0
fi

if ! mkdir -p "$(dirname "$TARGET_DIR")" 2>/dev/null; then
    echo "Notice: Global configuration directory is read-only in this environment."
    echo "The skill is active and available in your workspace at:"
    echo "  ${SCRIPT_DIR}"
    echo "Antigravity automatically discovers skills under .agents/skills/."
    echo "=========================================================="
    exit 0
fi

rm -rf "$TARGET_DIR"
cp -r "$SCRIPT_DIR" "$TARGET_DIR"

echo "✓ Skill successfully installed to: $TARGET_DIR"
echo ""
echo "Any Antigravity (AGY) agent will now automatically discover and activate"
echo "the '${SKILL_NAME}' skill when working on Jetson or voice cloning tasks!"
echo "=========================================================="

