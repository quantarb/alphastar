#!/usr/bin/env python3
"""Export the installed client's action vocabulary for general-policy training."""
from pathlib import Path
import torch
from sc2.ids.ability_id import AbilityId

ROOT = Path(__file__).resolve().parents[1]
abilities = sorted((int(a.value), a.name) for a in AbilityId)
out = ROOT / 'mac_sc2/artifacts/current_sc2_ability_schema.pt'
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({'abilities': abilities, 'vocab_size': 2048, 'client_patch': '5.0.16'}, out)
print(f'Wrote {len(abilities)} abilities to {out}')
