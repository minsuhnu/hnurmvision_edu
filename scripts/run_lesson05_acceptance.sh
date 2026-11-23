#!/usr/bin/env bash
set -eo pipefail
LESSON05_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
source "$LESSON05_ROOT/simulation/install_interfaces/setup.bash"
export ROS_DOMAIN_ID=89
export FASTRTPS_DEFAULT_PROFILES_FILE="$LESSON05_ROOT/configs/fastdds_local.xml"
export BEVY_ASSET_ROOT="$LESSON05_ROOT/simulation/simulator"
export LD_LIBRARY_PATH="$LESSON05_ROOT/simulation/simulator/bin/lib:${LD_LIBRARY_PATH:-}"
export RUST_LOG=warn
export MALLOC_ARENA_MAX=2
unset WAYLAND_DISPLAY WAYLAND_SOCKET
cd "$LESSON05_ROOT"
exec /usr/bin/python3 submission/week05/run_acceptance.py "$@"
