#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

POLICY_SERVER_ADDRESS="${POLICY_SERVER_ADDRESS:-192.168.0.128:18082}"
DURATION_S="${DURATION_S:-60}"
FPS="${FPS:-20}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CAMERA_COUNT="${CAMERA_COUNT:-1}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_NAMES="${CAMERA_NAMES:-front}"
POLICY_TYPE="${POLICY_TYPE:-act}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-checkpoints/jliu6718_lerobot-so101-act}"
VENV_PATH="${VENV_PATH:-$PROJECT_ROOT/.venv-drtc}"
JSON_OUTPUT="${JSON_OUTPUT:-$PROJECT_ROOT/artifacts/drtc_amd_client_smoke.json}"

cd "$PROJECT_ROOT"
mkdir -p "$(dirname "$JSON_OUTPUT")"

if [ ! -x "$VENV_PATH/bin/python" ]; then
    echo "ERROR: Python venv not found: $VENV_PATH" >&2
    exit 1
fi

SERVER_HOST="${POLICY_SERVER_ADDRESS%%:*}"
export NO_PROXY="${SERVER_HOST},localhost,127.0.0.1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export PYTHONPATH="${PYTHONPATH:-src}"

exec "$VENV_PATH/bin/python" tools/drtc_mock_smoke.py \
    --mode client \
    --server-address "$POLICY_SERVER_ADDRESS" \
    --duration-s "$DURATION_S" \
    --fps "$FPS" \
    --policy-type "$POLICY_TYPE" \
    --pretrained-name-or-path "$PRETRAINED_NAME_OR_PATH" \
    --policy-device "$POLICY_DEVICE" \
    --actions-per-chunk "$ACTIONS_PER_CHUNK" \
    --policy-no-act-pretrained-backbone-weights \
    --camera-count "$CAMERA_COUNT" \
    --camera-width "$CAMERA_WIDTH" \
    --camera-height "$CAMERA_HEIGHT" \
    --camera-names "$CAMERA_NAMES" \
    --json-output "$JSON_OUTPUT"
