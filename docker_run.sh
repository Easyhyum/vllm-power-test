#!/usr/bin/env bash
set -euo pipefail

IMAGE="power-test"
WORKDIR="/workspace"
MEM_LIMIT="16g"
SHM_SIZE="4g"
USE_IPC_HOST=1           # 1 = use --ipc=host
USE_ULIMIT_MEMLOCK=1     # 1 = use --ulimit memlock=-1:-1

## NAME을 현재 user 이름으로 설정
########################################
# 3) SET NAME
########################################
NAME=${IMAGE}-$(whoami)
TARGET_GPUS=${1:-"0"}
GPU_OPTS=("-e" "CUDA_VISIBLE_DEVICES=${TARGET_GPUS}")
########################################
# 4) Execution flags
########################################
IPC_FLAG=""; [[ "${USE_IPC_HOST}" -eq 1 ]] && IPC_FLAG="--ipc=host"
ULIMIT_FLAG=""; [[ "${USE_ULIMIT_MEMLOCK}" -eq 1 ]] && ULIMIT_FLAG="--ulimit memlock=-1:-1"

########################################
# 5) docker run
########################################
CMD=(docker run -d -it
  --name "${NAME}"
  --gpus all
  --privileged
  "${GPU_OPTS[@]}"
  --memory="${MEM_LIMIT}"
  --shm-size="${SHM_SIZE}"
  ${IPC_FLAG}
  ${ULIMIT_FLAG}
  -w "${WORKDIR}"
  -v ${PWD}:/workspace
  "${IMAGE}"
  bash
)

echo "== Container Name : ${NAME}"
echo "== GPU     : ${TARGET_GPUS}"
echo "== Command to run =="
printf '%q ' "${CMD[@]}"; echo
exec "${CMD[@]}"