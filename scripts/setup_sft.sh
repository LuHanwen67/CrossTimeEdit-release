#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-${ROOT}/.venv}"
DIFFSYNTH_DIR="${DIFFSYNTH_DIR:-${ROOT}/external/DiffSynth-Studio}"
COMMIT="079e51c9f3f296bbe636aa74448a7e3637278232"

if [[ ! -d "${VENV}" ]]; then
  "${PYTHON}" -m venv "${VENV}"
fi
source "${VENV}/bin/activate"
python -m pip install --upgrade pip

if ! python -c 'import torch' >/dev/null 2>&1; then
  echo "Install the PyTorch build matching this machine's CUDA version, then rerun." >&2
  exit 2
fi

python -m pip install -r "${ROOT}/environment/requirements-sft.txt"

if [[ ! -d "${DIFFSYNTH_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${DIFFSYNTH_DIR}")"
  git clone https://github.com/modelscope/DiffSynth-Studio.git "${DIFFSYNTH_DIR}"
fi

git -C "${DIFFSYNTH_DIR}" fetch origin "${COMMIT}"
git -C "${DIFFSYNTH_DIR}" checkout --detach "${COMMIT}"

if git -C "${DIFFSYNTH_DIR}" apply --reverse --check "${ROOT}/patches/diffsynth-studio-079e51.patch" >/dev/null 2>&1; then
  echo "SFT patch is already applied."
else
  git -C "${DIFFSYNTH_DIR}" apply --check "${ROOT}/patches/diffsynth-studio-079e51.patch"
  git -C "${DIFFSYNTH_DIR}" apply "${ROOT}/patches/diffsynth-studio-079e51.patch"
fi

python -m pip install -e "${DIFFSYNTH_DIR}"
echo "SFT environment ready: ${VENV}"
