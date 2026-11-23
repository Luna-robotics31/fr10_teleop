#!/usr/bin/env bash
# Terminal 1 (start this FIRST): the pinch gripper. It activates the gripper once (can take several
# seconds), then follows the hand data shared by the teleoperation program.
# NOTE: while the teleoperation streams, the controller takes only the first gripper move from this tool (it
# cannot pause the arm). Prefer ./scripts/start_teleop_real.sh, whose built-in gripper pauses the arm for each move.
# Extra arguments are passed on, e.g.  ./scripts/start_pinch_gripper.sh --dry-run
set -e
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xr
exec python tools/pinch_gripper.py --from-teleop --proportional "$@"
