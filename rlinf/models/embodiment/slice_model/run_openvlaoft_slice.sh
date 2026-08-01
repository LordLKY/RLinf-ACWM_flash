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

# profile for nsys
PROFILE=${PROFILE:-0}

# profile for scale
PROFILE_SCALE=${PROFILE_SCALE:-1}
PROFILE_SCALE_MODULES=${PROFILE_SCALE_MODULES:-1}
SCALE_BATCH_SIZES=${SCALE_BATCH_SIZES:-1,2,4,8,16,24,32}
PROFILE_SCALE_ITERS=${PROFILE_SCALE_ITERS:-10}
PROFILE_SCALE_WARMUP=${PROFILE_SCALE_WARMUP:-4}
PROFILE_SCALE_STOP_ON_OOM=${PROFILE_SCALE_STOP_ON_OOM:-1}
PROFILE_SCALE_EMPTY_CACHE=${PROFILE_SCALE_EMPTY_CACHE:-1}

# other profile
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

if ! is_true "${PROFILE}" && ! is_true "${PROFILE_SCALE}" && is_true "${CLEAN_OUTPUT}"; then
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
elif is_true "${PROFILE_SCALE}"; then
  cmd+=(
    --profile-scale
    --scale-batch-sizes "${SCALE_BATCH_SIZES}"
    --profile-scale-iters "${PROFILE_SCALE_ITERS}"
    --profile-scale-warmup "${PROFILE_SCALE_WARMUP}"
  )
  if is_true "${PROFILE_SCALE_STOP_ON_OOM}"; then
    cmd+=(--profile-scale-stop-on-oom)
  else
    cmd+=(--no-profile-scale-stop-on-oom)
  fi
  if is_true "${PROFILE_SCALE_EMPTY_CACHE}"; then
    cmd+=(--profile-scale-empty-cache)
  else
    cmd+=(--no-profile-scale-empty-cache)
  fi
  if is_true "${PROFILE_SCALE_MODULES}"; then
    cmd+=(--profile-scale-modules)
  fi
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
elif is_true "${PROFILE_SCALE}"; then
  if is_true "${PROFILE_SCALE_MODULES}"; then
    echo "[openvlaoft-slice] mode: profile_scale_modules"
  else
    echo "[openvlaoft-slice] mode: profile_scale"
  fi
  echo "[openvlaoft-slice] scale batch sizes: ${SCALE_BATCH_SIZES}"
else
  echo "[openvlaoft-slice] output: ${OUTPUT_DIR}"
fi

exec "${cmd[@]}"
