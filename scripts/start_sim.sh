#!/usr/bin/env bash
# Terminal 1: start the Isaac Lab FR10 simulator (with the Isaac Sim window).
# Extra arguments are passed on, e.g.  ./scripts/start_sim.sh --headless
set -e
cd "$(dirname "$0")/.."
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
exec python sim_fr10.py \
  --set "sim.camera.eye=[1.55,-0.8,0.8]" \
  --set "sim.camera.target=[0.5,0.0,0.15]" \
  "$@"
