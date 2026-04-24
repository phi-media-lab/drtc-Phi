#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${PORT:-18082}"
HOST="${HOST:-0.0.0.0}"
FPS="${FPS:-20}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CAMERA_COUNT="${CAMERA_COUNT:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_NAMES="${CAMERA_NAMES:-front}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_SERVER_MODE="${POLICY_SERVER_MODE:-real}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-checkpoints/jliu6718_lerobot-so101-act}"
VENV_PATH="${VENV_PATH:-$HOME/.venvs/drtc-rocm}"

cd "$PROJECT_ROOT"

if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo "ERROR: Python venv not found: $VENV_PATH" >&2
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-src}"

exec "$VENV_PATH/bin/python" tools/drtc_mock_smoke.py \
    --mode server \
    --host "$HOST" \
    --port "$PORT" \
    --fps "$FPS" \
    --policy-server-mode "$POLICY_SERVER_MODE" \
    --policy-type "$POLICY_TYPE" \
    --pretrained-name-or-path "$PRETRAINED_NAME_OR_PATH" \
    --policy-device "$POLICY_DEVICE" \
    --actions-per-chunk "$ACTIONS_PER_CHUNK" \
    --policy-no-act-pretrained-backbone-weights \
    --camera-count "$CAMERA_COUNT" \
    --camera-width "$CAMERA_WIDTH" \
    --camera-height "$CAMERA_HEIGHT" \
    --camera-names "$CAMERA_NAMES"
