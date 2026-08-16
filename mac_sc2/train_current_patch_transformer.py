#!/usr/bin/env python3
"""Train only on winning live 5.0.16 teacher trajectories."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from current_patch_transformer import ACTIONS, CurrentPatchTransformer

ROOT = Path(__file__).resolve().parents[1]
data = torch.load(ROOT / 'mac_sc2/artifacts/current_patch_teacher_trajectories.pt', weights_only=False)
wins = [e for e in data['episodes'] if e['result'].endswith('Victory')]
episodes = wins or data['episodes']
if not wins:
    print('No winning episode available; training the Hard curriculum attempt as a bootstrap.', flush=True)
windows, labels = [], []
for episode in episodes:
    x, y = episode['features'], episode['labels']
    for i in range(len(y)):
        chunk = x[max(0, i - 7):i + 1]
        if len(chunk) < 8:
            chunk = torch.cat([torch.zeros(8 - len(chunk), 7, dtype=torch.long), chunk])
        windows.append(chunk)
        labels.append(y[i])
x, y = torch.stack(windows), torch.tensor(labels)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = CurrentPatchTransformer().to(device)
counts = torch.bincount(y, minlength=len(ACTIONS)).float(); weight = counts.sum() / counts.clamp_min(1)
loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)
opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
for epoch in range(30):
    total = correct = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device)
        loss = nn.functional.cross_entropy(model(bx), by, weight=weight.to(device))
        opt.zero_grad(); loss.backward(); opt.step()
        correct += model(bx).argmax(-1).eq(by).sum().item(); total += len(by)
    if epoch in (0, 9, 19, 29): print(f'epoch={epoch+1} accuracy={correct/total:.3f}')
out = ROOT / 'mac_sc2/artifacts/current_patch_transformer.pt'
torch.save({'state_dict': model.cpu().state_dict(), 'patch': data['patch'], 'winning_episodes': len(wins),
            'episodes': len(episodes), 'frames': len(y)}, out)
print(f'Saved {out} from {len(episodes)} episodes / {len(y)} frames')
