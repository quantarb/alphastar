#!/usr/bin/env python3
"""Train the shared live-compatible MTL macro policy."""
import argparse
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from live_mtl_policy import LiveMTLPolicy

class Rows(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        race, state, label = self.rows[i]
        # A short repeated state history is used until trajectory windows are
        # materialized; state itself already contains the prior action history.
        return torch.tensor([state] * 8, dtype=torch.float32), label

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data', required=True); p.add_argument('--output', required=True); p.add_argument('--epochs', type=int, default=2); p.add_argument('--batch-size', type=int, default=256); a=p.parse_args()
    data=torch.load(a.data, map_location='cpu', weights_only=False); by_race=defaultdict(list)
    for row in data['rows']: by_race[row[0]].append(row)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=LiveMTLPolicy().to(device); opt=torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    for epoch in range(a.epochs):
        seen=correct=0
        for race, rows in sorted(by_race.items()):
            for states, labels in DataLoader(Rows(rows), batch_size=a.batch_size, shuffle=True):
                states, labels=states.to(device), labels.to(device); ids=torch.full((len(labels),), race, dtype=torch.long, device=device)
                logits=model(states, ids); loss=F.cross_entropy(logits, labels); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                seen += len(labels); correct += logits.argmax(-1).eq(labels).sum().item()
        print(f'epoch={epoch+1} examples={seen} accuracy={correct/seen:.3f}', flush=True)
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(), 'games':data['games'], 'examples':len(data['rows']), 'actions':7, 'architecture':'shared live-state Transformer + race macro heads'}, a.output)
if __name__ == '__main__': main()
