#!/usr/bin/env bash
# The arm teleoperation with hand tracking. One terminal is enough: the gripper runs in a process of its
# own inside this program (it activates the gripper first, which takes a few seconds, then follows your
# right hand: pinch or close the hand). Use --no-gripper together with start_pinch_gripper.sh instead
# if you want the gripper in a separate program.
# Extra arguments are passed on, e.g.  ./scripts/start_teleop_real.sh --set control.speed_scale=0.3
set -e
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xr
IP=$(hostname -I | awk '{print $1}')
echo "Quest browser:  https://${IP}:8012/?ws=wss://${IP}:8012   (accept the warning, then Enter VR)"
echo "Wait for the gripper to activate, then press r. Left fist = arm follows your right hand; close the right hand = gripper"
echo "(the arm pauses ~1 s for each gripper move: the controller needs it). o / g = open / close. Space = stop, q = quit."
exec python teleop_fr10.py --robot real --set xr.input_mode=hand --set control.speed_scale=0.5 --set mapping.smoothing_time_s=0.05 "$@"
