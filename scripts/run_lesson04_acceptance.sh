#!/usr/bin/env bash
set -eo pipefail
LESSON04_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/humble/setup.bash
source "$LESSON04_ROOT/simulation/install_interfaces/setup.bash"
export ROS_DOMAIN_ID=89
export FASTRTPS_DEFAULT_PROFILES_FILE="$LESSON04_ROOT/configs/fastdds_local.xml"
export BEVY_ASSET_ROOT="$LESSON04_ROOT/simulation/simulator"
export LD_LIBRARY_PATH="$LESSON04_ROOT/simulation/simulator/target/release/deps:$(rustc +1.95.0 --print target-libdir):${LD_LIBRARY_PATH:-}"
export RUST_LOG=warn,daedalus::training=info
unset WAYLAND_DISPLAY WAYLAND_SOCKET
cd "$LESSON04_ROOT"
exec /usr/bin/python3 submission/tools/run_lesson04_acceptance.py "$@"
