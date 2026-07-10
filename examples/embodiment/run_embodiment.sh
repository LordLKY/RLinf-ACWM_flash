#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL=${MUJOCO_GL:-"egl"}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-"egl"}
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# POLARIS dataset
export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_ppo_openvlaoft"}
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
SECOND_ARG=${2:-""}
if [[ "${SECOND_ARG,,}" == "true" || "${SECOND_ARG}" == "1" || "${SECOND_ARG,,}" == "yes" || "${SECOND_ARG,,}" == "false" || "${SECOND_ARG}" == "0" || "${SECOND_ARG,,}" == "no" ]]; then
    ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
    USE_NSYS=${SECOND_ARG}
else
    ROBOT_PLATFORM=${SECOND_ARG:-${ROBOT_PLATFORM:-"LIBERO"}}
    USE_NSYS=${3:-${USE_NSYS:-"false"}}
fi

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using USE_NSYS=$USE_NSYS"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
NSYS_DIR="${REPO_PATH}/profile/nsys"
mkdir -p "${NSYS_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
if [[ "${USE_NSYS,,}" == "true" || "${USE_NSYS}" == "1" || "${USE_NSYS,,}" == "yes" ]]; then
    export RLINF_USE_NVTX=1
    NSYS_OUTPUT="${NSYS_DIR}/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
    CMD="nsys profile --force-overwrite=true --trace=cuda,cudnn,cublas,nvtx --sample=none --cpuctxsw=none --cuda-memory-usage=false --output=${NSYS_OUTPUT} ${CMD}"  # --trace=cuda,cudnn,cublas,nvtx,osrt --sample=process-tree --cpuctxsw=process-tree --cuda-memory-usage=true --osrt-threshold=1000 
else
    export RLINF_USE_NVTX=0
fi
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
