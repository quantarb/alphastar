#!/bin/zsh
# Launch the executable current-patch Transformer policy in a visible SC2 match.
cd "$(dirname "$0")/.."
export SC2PATH="/Applications/StarCraft II"
exec .conda-alphastar/bin/python -u mac_sc2/play_current_patch_transformer.py --realtime --difficulty easy
