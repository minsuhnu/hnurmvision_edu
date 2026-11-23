#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
POSE_PYTHON=${RM_POSE_PYTHON:-$HOME/miniforge3/envs/rm-vision/bin/python}
WEIGHTS=${RM_POSE_WEIGHTS:-$REPO_ROOT/models/armor_pose.pt}
MODE=hybrid
CONFIG="$REPO_ROOT/configs/simulation.yaml"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --weights) WEIGHTS=$2; shift 2 ;;
        --python) POSE_PYTHON=$2; shift 2 ;;
        --mode) MODE=$2; shift 2 ;;
        --config) CONFIG=$2; shift 2 ;;
        *) echo "Usage: $0 [--weights best.pt] [--python python] [--mode hybrid|pose|opencv] [--config config.yaml]" >&2; exit 2 ;;
    esac
done
[[ "$MODE" == hybrid || "$MODE" == pose || "$MODE" == opencv ]] || { echo "Invalid detector mode" >&2; exit 2; }
RUNTIME_DIR=$(mktemp -d "${TMPDIR:-/tmp}/rm-pose.XXXXXX")
GPU_PID=
TRAINING_PID=
cleanup() {
    trap - INT TERM
    if [[ -n "$TRAINING_PID" ]] && kill -0 "$TRAINING_PID" 2>/dev/null; then
        kill -INT "$TRAINING_PID" 2>/dev/null || true
        wait "$TRAINING_PID" 2>/dev/null || true
    fi
    if [[ -n "$GPU_PID" ]] && kill -0 "$GPU_PID" 2>/dev/null; then
        kill -TERM "$GPU_PID" 2>/dev/null || true
        wait "$GPU_PID" 2>/dev/null || true
    fi
    echo "Runtime logs: $RUNTIME_DIR"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export RM_DETECTOR="$MODE"
if [[ "$MODE" != opencv ]]; then
    [[ -f "$WEIGHTS" && -x "$POSE_PYTHON" ]] || { echo "Provide existing weights and the CUDA-enabled Python environment" >&2; exit 2; }
    export PYTHONDONTWRITEBYTECODE=1
    export YOLO_CONFIG_DIR=${YOLO_CONFIG_DIR:-$RUNTIME_DIR/ultralytics}
    "$POSE_PYTHON" -B "$REPO_ROOT/scripts/pose_gpu_server.py" --weights "$WEIGHTS" \
        --socket "$RUNTIME_DIR/pose.sock" --ready "$RUNTIME_DIR/ready.json" > "$RUNTIME_DIR/gpu.log" 2>&1 &
    GPU_PID=$!
    for _ in {1..200}; do
        [[ -f "$RUNTIME_DIR/ready.json" && -S "$RUNTIME_DIR/pose.sock" ]] && break
        if ! kill -0 "$GPU_PID" 2>/dev/null; then cat "$RUNTIME_DIR/gpu.log" >&2; exit 1; fi
        sleep 0.2
    done
    [[ -f "$RUNTIME_DIR/ready.json" && -S "$RUNTIME_DIR/pose.sock" ]] || { echo "CUDA worker startup timed out" >&2; exit 1; }
    export RM_POSE_SOCKET="$RUNTIME_DIR/pose.sock"
    cat "$RUNTIME_DIR/ready.json"
else
    unset RM_POSE_SOCKET
fi
# ROS setup scripts contain optional unset variables.
set +u
source /opt/ros/humble/setup.bash
source "$REPO_ROOT/simulation/install_interfaces/setup.bash"
set -u
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-89}
export FASTRTPS_DEFAULT_PROFILES_FILE=${FASTRTPS_DEFAULT_PROFILES_FILE:-$REPO_ROOT/configs/fastdds_local.xml}
export MALLOC_ARENA_MAX=${MALLOC_ARENA_MAX:-2}
cd "$REPO_ROOT"
./build/training "$CONFIG" &
TRAINING_PID=$!
wait "$TRAINING_PID"
