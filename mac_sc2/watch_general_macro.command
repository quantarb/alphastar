#!/bin/zsh
cd "$(dirname "$0")/.."
exec .conda-alphastar-py314/bin/python -u mac_sc2/play_general_macro.py \
  --checkpoint mac_sc2/artifacts/general_macro_1000_v2.pt \
  --difficulty hard --realtime
