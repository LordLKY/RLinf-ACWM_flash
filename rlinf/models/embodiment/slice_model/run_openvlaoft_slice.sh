#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)

DEFAULT_SAMPLE_PATH="${REPO_ROOT}/profile/openvlaoft_slice/data/group0/env_rank0000_pid1141683/samples/vla_000005.pt"

SAMPLE_PATH=${1:-${SAMPLE_PATH:-${DEFAULT_SAMPLE_PATH}}}
if [[ $# -gt 0 ]]; then
  shift
fi

SLICE_NAME=group0
if [[ "${SAMPLE_PATH}" =~ (^|/)data/([^/]+)/ ]]; then
  SLICE_NAME="${BASH_REMATCH[2]}"
fi

OUTPUT_DIR=${1:-${OUTPUT_DIR:-"${REPO_ROOT}/profile/openvlaoft_slice/results/${SLICE_NAME}"}}
if [[ $# -gt 0 ]]; then
  shift
fi

PYTHON_BIN=${PYTHON_BIN:-"${REPO_ROOT}/.venv/bin/python"}
CONFIG_NAME=${CONFIG_NAME:-wan_libero_spatial_grpo_openvlaoft_ngpu}
CONFIG_DIR=${CONFIG_DIR:-}
DEVICE=${DEVICE:-cuda}
SEED=${SEED:-0}
MODE=${MODE:-train}
MODEL_SOURCE=${MODEL_SOURCE:-actor}
LOCAL_PRISMATIC_SRC=${LOCAL_PRISMATIC_SRC:-"${SCRIPT_DIR}/local_src/openvla_oft"}
PROFILE=${PROFILE:-0}
COMPARE=${COMPARE:-0}
SAVE_PT=${SAVE_PT:-0}
CKPT_PATH=${CKPT_PATH:-}
GPU=${GPU:-}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

is_true() {
  case "${1,,}" in
    1 | true | yes | y | on) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ -n "${GPU}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

clean_output_dir() {
  local dir="$1"
  case "${dir}" in
    "${REPO_ROOT}/profile/openvlaoft_slice/results/"*)
      rm -rf -- "${dir}"
      mkdir -p -- "${dir}"
      ;;
    *)
      echo "[openvlaoft-slice] refusing to clean output outside profile/openvlaoft_slice/results: ${dir}" >&2
      return 1
      ;;
  esac
}

if ! is_true "${PROFILE}" && is_true "${CLEAN_OUTPUT}"; then
  clean_output_dir "${OUTPUT_DIR}"
fi

cmd=(
  "${PYTHON_BIN}"
  -m rlinf.models.embodiment.slice_model.run_openvlaoft_slice_inference
  --sample-path "${SAMPLE_PATH}"
  --config-name "${CONFIG_NAME}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --mode "${MODE}"
  --model-source "${MODEL_SOURCE}"
  --local-prismatic-src "${LOCAL_PRISMATIC_SRC}"
)

if [[ -n "${CONFIG_DIR}" ]]; then
  cmd+=(--config-dir "${CONFIG_DIR}")
fi

if [[ -n "${CKPT_PATH}" ]]; then
  cmd+=(--ckpt-path "${CKPT_PATH}")
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

cmd+=("$@")

echo "[openvlaoft-slice] repo: ${REPO_ROOT}"
echo "[openvlaoft-slice] sample: ${SAMPLE_PATH}"
echo "[openvlaoft-slice] slice: ${SLICE_NAME}"
echo "[openvlaoft-slice] local prismatic src: ${LOCAL_PRISMATIC_SRC}"
if is_true "${PROFILE}"; then
  echo "[openvlaoft-slice] mode: profile"
else
  echo "[openvlaoft-slice] output: ${OUTPUT_DIR}"
fi

exec "${cmd[@]}"
