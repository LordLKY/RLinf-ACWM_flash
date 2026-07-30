#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)

DEFAULT_SAMPLE_PATH="${REPO_ROOT}/profile/wan_slice/data/group0/env_rank0000_pid1141683/samples/acwm_000005.pt"

SAMPLE_PATH=${1:-${SAMPLE_PATH:-${DEFAULT_SAMPLE_PATH}}}
if [[ $# -gt 0 ]]; then
  shift
fi

SLICE_NAME=group0
if [[ "${SAMPLE_PATH}" =~ (^|/)data/([^/]+)/ ]]; then
  SLICE_NAME="${BASH_REMATCH[2]}"
fi

OUTPUT_DIR=${1:-${OUTPUT_DIR:-"${REPO_ROOT}/profile/wan_slice/results/${SLICE_NAME}"}}
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}
CONFIG_NAME=${CONFIG_NAME:-wan_libero_spatial_grpo_openvlaoft_ngpu}
CONFIG_DIR=${CONFIG_DIR:-}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-0}
LOCAL_WAN_SRC=${LOCAL_WAN_SRC:-"${SCRIPT_DIR}/local_src/wan"}
PROFILE=${PROFILE:-0}
COMPARE=${COMPARE:-0}
SAVE_PT=${SAVE_PT:-0}
SAVE_OUTPUT_CURRENT_OBS_FRAMES=${SAVE_OUTPUT_CURRENT_OBS_FRAMES:-0}
GPU=${GPU:-}

is_true() {
  case "${1,,}" in
    1 | true | yes | y | on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

cmd=(
  "${PYTHON_BIN}"
  -m rlinf.models.embodiment.slice_model.run_wan_slice_inference
  --sample-path "${SAMPLE_PATH}"
  --config-name "${CONFIG_NAME}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --local-wan-src "${LOCAL_WAN_SRC}"
)

if [[ -n "${CONFIG_DIR}" ]]; then
  cmd+=(--config-dir "${CONFIG_DIR}")
fi

if is_true "${PROFILE}"; then
  cmd+=(--profile)
else
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if is_true "${COMPARE}"; then
  cmd+=(--compare)
fi

if is_true "${SAVE_PT}"; then
  cmd+=(--save-pt)
fi

if is_true "${SAVE_OUTPUT_CURRENT_OBS_FRAMES}"; then
  cmd+=(--save-output-current-obs-frames)
fi

cmd+=("$@")

echo "[wan-slice] repo: ${REPO_ROOT}"
echo "[wan-slice] sample: ${SAMPLE_PATH}"
echo "[wan-slice] slice: ${SLICE_NAME}"
echo "[wan-slice] local wan src: ${LOCAL_WAN_SRC}"
if is_true "${PROFILE}"; then
  echo "[wan-slice] mode: profile"
else
  echo "[wan-slice] output: ${OUTPUT_DIR}"
fi

exec "${cmd[@]}"
