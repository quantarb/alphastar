#!/usr/bin/env python3
"""Train the shared multi-race structured policy on every extracted replay action."""
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from mac_sc2.legacy.multi_race_policy import MultiRaceSC2Policy

ROOT = Path(__file__).resolve().parents[1]


class RaceRows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        entities = torch.zeros(32, 16)
        # Replay selections are the directly observed entity component.  Keep
        # type IDs and prior actions as separate bounded numeric features.
        for slot, unit_type in enumerate(row['selected_types'][:32]):
            entities[slot, 0] = min(unit_type, 2047) / 2047
            entities[slot, 1] = slot / 31
        history = row['history'][-8:]
        entities[0, 2:10] = torch.tensor([min(value, 8191) / 8191 for value in history])
        entities[0, 10] = min(row['frame'], 20000) / 20000
        mask = torch.ones(32, dtype=torch.bool)
        mask[:max(1, len(row['selected_types'][:32]))] = False
        return entities, mask, min(row['ability'], 2047)


data = torch.load(ROOT / 'mac_sc2/artifacts/structured_replay_actions.pt', map_location='cpu', weights_only=False)
by_race = defaultdict(list)
for row in data['rows']: by_race[row['race']].append(row)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = MultiRaceSC2Policy(width=96, heads=4, layers=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
total_seen = total_correct = 0
for race, rows in sorted(by_race.items()):
    loader = DataLoader(RaceRows(rows), batch_size=128, shuffle=True, num_workers=0)
    seen = correct = 0
    for entities, mask, ability in loader:
        entities, mask, ability = entities.to(device), mask.to(device), ability.to(device)
        race_ids = torch.full((len(ability),), race, device=device, dtype=torch.long)
        logits = model(entities, race_ids, mask)['ability']
        loss = nn.functional.cross_entropy(logits, ability)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        seen += len(ability); correct += logits.argmax(-1).eq(ability).sum().item()
        if seen % 100000 < 128: print(f'race={race} examples={seen}/{len(rows)} accuracy={correct/seen:.3f}', flush=True)
    total_seen += seen; total_correct += correct
    print(f'race={race} complete examples={seen} accuracy={correct/seen:.3f}', flush=True)
out = ROOT / 'mac_sc2/artifacts/multi_race_structured_policy_1021_games.pt'
torch.save({'state_dict': model.cpu().state_dict(), 'games': data.get('games', 1021), 'examples': total_seen,
            'train_accuracy': total_correct / total_seen, 'architecture': 'shared 96d/2-layer trunk; race ability heads'}, out)
print(f'saved={out} examples={total_seen} accuracy={total_correct/total_seen:.3f}', flush=True)
