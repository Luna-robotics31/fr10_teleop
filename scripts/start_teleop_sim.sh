#!/usr/bin/env bash
# Terminal 2: start the teleoperation program for the simulator (Meta Quest headset).
# Start ./scripts/start_sim.sh first and wait for "FR10 simulator running".
# Extra arguments are passed on, e.g.  ./scripts/start_teleop_sim.sh --record
set -e
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate xr
IP=$(hostname -I | awk '{print $1}')
echo "Quest browser:  https://${IP}:8012/?ws=wss://${IP}:8012"
echo "Then press r here, and hold the right grip button to move the arm."
exec python teleop_fr10.py --robot sim "$@"
