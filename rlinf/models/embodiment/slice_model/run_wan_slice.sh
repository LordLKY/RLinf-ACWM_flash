#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." >/dev/null 2>&1 && pwd)

DEFAULT_SAMPLE_PATH="${REPO_ROOT}/profile/wan_slice/data/group0/env_rank0000_pid1141683/samples/acwm_000007.pt"

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
COMPARE=${COMPARE:-0}
SAVE_PT=${SAVE_PT:-0}
SAVE_OUTPUT_CURRENT_OBS_FRAMES=${SAVE_OUTPUT_CURRENT_OBS_FRAMES:-0}
SEQUENCE=${SEQUENCE:-0}
SEQUENCE_MODE=${SEQUENCE_MODE:-teacher_forced}
MAX_CHUNKS=${MAX_CHUNKS:-}
GPU=${GPU:-}
CLEAN_OUTPUT=${CLEAN_OUTPUT:-1}

# profile for nsys
PROFILE=${PROFILE:-0}

# profile for scale
PROFILE_SCALE=${PROFILE_SCALE:-0}
PROFILE_SCALE_MODULES=${PROFILE_SCALE_MODULES:-1}
SCALE_BATCH_SIZES=${SCALE_BATCH_SIZES:-1,2,4,6,8,10,12}
PROFILE_SCALE_ITERS=${PROFILE_SCALE_ITERS:-10}
PROFILE_SCALE_WARMUP=${PROFILE_SCALE_WARMUP:-4}
PROFILE_SCALE_STOP_ON_OOM=${PROFILE_SCALE_STOP_ON_OOM:-1}
PROFILE_SCALE_EMPTY_CACHE=${PROFILE_SCALE_EMPTY_CACHE:-1}

# profile for cache
DUMP_DIT_RESIDUALS=${DUMP_DIT_RESIDUALS:-0}
DIT_RESIDUAL_DIR=${DIT_RESIDUAL_DIR:-"${REPO_ROOT}/profile/wan_slice/dit_residual"}
SHARE_INITIAL_NOISE=${SHARE_INITIAL_NOISE:-1}

# profile for prefix denoise-step quality
PROFILE_PREFIX_STEP=${PROFILE_PREFIX_STEP:-1}
PREFIX_STEPS=${PREFIX_STEPS:-2}
PREFIX_REFERENCE_BATCH_ID=${PREFIX_REFERENCE_BATCH_ID:-0}

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
    "${REPO_ROOT}/profile/wan_slice/results/"*)
      rm -rf -- "${dir}"
      mkdir -p -- "${dir}"
      ;;
    *)
      echo "[wan-slice] refusing to clean output outside profile/wan_slice/results: ${dir}" >&2
      return 1
      ;;
  esac
}

if ! is_true "${PROFILE}" && ! is_true "${PROFILE_SCALE}" && is_true "${CLEAN_OUTPUT}"; then
  clean_output_dir "${OUTPUT_DIR}"
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
elif is_true "${PROFILE_PREFIX_STEP}"; then
  cmd+=(
    --output-dir "${OUTPUT_DIR}"
    --profile-prefix-step
    --prefix-steps "${PREFIX_STEPS}"
    --prefix-reference-batch-id "${PREFIX_REFERENCE_BATCH_ID}"
  )
else
  cmd+=(--output-dir "${OUTPUT_DIR}")
fi

if ! is_true "${PROFILE}" && ! is_true "${PROFILE_SCALE}" && ! is_true "${PROFILE_PREFIX_STEP}" && is_true "${SEQUENCE}"; then
  cmd+=(--sequence --sequence-mode "${SEQUENCE_MODE}")
fi

if [[ -n "${MAX_CHUNKS}" ]]; then
  cmd+=(--max-chunks "${MAX_CHUNKS}")
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

if ! is_true "${PROFILE}" && ! is_true "${PROFILE_SCALE}" && ! is_true "${PROFILE_PREFIX_STEP}" && is_true "${DUMP_DIT_RESIDUALS}"; then
  cmd+=(--dump-dit-residuals --dit-residual-dir "${DIT_RESIDUAL_DIR}")
fi

if is_true "${SHARE_INITIAL_NOISE}"; then
  cmd+=(--share-initial-noise)
fi

cmd+=("$@")

echo "[wan-slice] repo: ${REPO_ROOT}"
echo "[wan-slice] sample: ${SAMPLE_PATH}"
echo "[wan-slice] slice: ${SLICE_NAME}"
echo "[wan-slice] local wan src: ${LOCAL_WAN_SRC}"
if is_true "${PROFILE}"; then
  echo "[wan-slice] mode: profile"
elif is_true "${PROFILE_SCALE}"; then
  if is_true "${PROFILE_SCALE_MODULES}"; then
    echo "[wan-slice] mode: profile_scale_modules"
  else
    echo "[wan-slice] mode: profile_scale"
  fi
  echo "[wan-slice] scale batch sizes: ${SCALE_BATCH_SIZES}"
elif is_true "${PROFILE_PREFIX_STEP}"; then
  echo "[wan-slice] mode: profile_prefix_step"
  echo "[wan-slice] prefix steps: ${PREFIX_STEPS}"
  echo "[wan-slice] prefix reference batch id: ${PREFIX_REFERENCE_BATCH_ID}"
  echo "[wan-slice] output: ${OUTPUT_DIR}"
elif is_true "${SEQUENCE}"; then
  echo "[wan-slice] mode: sequence/${SEQUENCE_MODE}"
  echo "[wan-slice] output: ${OUTPUT_DIR}"
else
  echo "[wan-slice] output: ${OUTPUT_DIR}"
fi
if ! is_true "${PROFILE}" && ! is_true "${PROFILE_SCALE}" && ! is_true "${PROFILE_PREFIX_STEP}" && is_true "${DUMP_DIT_RESIDUALS}"; then
  echo "[wan-slice] dit residuals: ${DIT_RESIDUAL_DIR}"
fi
if is_true "${SHARE_INITIAL_NOISE}"; then
  echo "[wan-slice] shared initial noise: enabled"
fi

exec "${cmd[@]}"
