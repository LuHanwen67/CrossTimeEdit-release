#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-configs/rl/main.yaml}"
FF_TRAIN="${FF_TRAIN:-ff-train}"

cd "${ROOT}"
[[ -f "${CONFIG}" ]] || { echo "Configuration not found: ${CONFIG}" >&2; exit 2; }
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the environment before RL training}"
export GEMINI_BASE_URL="${GEMINI_BASE_URL:-https://generativelanguage.googleapis.com}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

"${FF_TRAIN}" "${CONFIG}"
