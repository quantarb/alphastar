#!/usr/bin/env python3
"""Learn a tactical teacher policy from randomized local combat states."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from micro_policy import MicroTransformer

torch.manual_seed(7)
n = 50000
# self health, cooldown, closest enemy, closest baneling, friendly count,
# enemy count, focus target health -- all compact categorical features.
x = torch.stack([torch.randint(0, 32, (n,)), torch.randint(0, 5, (n,)), torch.randint(1, 18, (n,)), torch.randint(1, 32, (n,)), torch.randint(1, 16, (n,)), torch.randint(1, 20, (n,)), torch.randint(0, 32, (n,))], 1)
y = torch.zeros(n, dtype=torch.long)  # attack focused low-health enemy
y[(x[:, 2] < 5) | (x[:, 3] < 8) | ((x[:, 1] > 0) & (x[:, 2] < 10))] = 1  # kite danger / reload
y[(x[:, 4] * 2 < x[:, 5]) | (x[:, 0] < 6)] = 2  # regroup when outnumbered
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
m = MicroTransformer().to(device); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
loader = DataLoader(TensorDataset(x, y), batch_size=512, shuffle=True)
for epoch in range(12):
    correct = total = 0
    for bx, by in loader:
        bx, by = bx.to(device), by.to(device); logits=m(bx); loss=nn.functional.cross_entropy(logits,by)
        opt.zero_grad(); loss.backward(); opt.step(); correct += logits.argmax(1).eq(by).sum().item(); total += len(by)
    if epoch in (0, 5, 11): print(f'epoch={epoch+1} tactical-teacher accuracy={correct/total:.3f}')
out=Path('mac_sc2/artifacts/micro_transformer.pt'); torch.save({'state_dict':m.cpu().state_dict()},out); print(out)
