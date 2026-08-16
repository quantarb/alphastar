#!/bin/zsh
# Double-click this file in Finder to watch the trained policy play in SC2.
set -e
cd "$(dirname "$0")/.."
export SC2PATH="/Applications/StarCraft II"
".conda-alphastar/bin/python" -u mac_sc2/play_hybrid_bc_python_sc2.py --realtime
echo
echo "The match has finished. Press Return to close this window."
read
