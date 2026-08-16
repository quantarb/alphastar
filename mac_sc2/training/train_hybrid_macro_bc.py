#!/usr/bin/env python3
"""Train a replay-BC macro policy with targeted correction demonstrations.

Human replay labels cover production actions. The correction set adds only the
states where a real-game policy must make a legal strategic choice: an early
two-Barracks defense, supply, production, and a 10-Marine counterattack. This
is a compact DAgger-style curriculum, not a replacement for the human demos.
"""
import random
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ACTIONS = ('train_scv', 'supply', 'barracks', 'marine', 'attack')


class HybridMacroTransformer(nn.Module):
    def __init__(self, width=64, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(32, width)
        self.position = nn.Embedding(7, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(width, len(ACTIONS))

    def forward(self, state):
        pos = torch.arange(state.shape[1], device=state.device)
        return self.head(self.encoder(self.embedding(state) + self.position(pos))[..., -1, :])


def corrective_example():
    time_bin = random.randrange(0, 32)
    minerals = random.randrange(0, 32)
    free = random.randrange(0, 10)
    workers = random.randrange(8, 31)
    depots = random.randrange(0, 5)
    barracks = random.randrange(0, 4)
    marines = random.randrange(0, 32)
    state = [time_bin, minerals, free, workers, depots, barracks, marines]
    desired_rax = 1 if barracks == 0 else (2 if workers >= 12 else 1)
    if marines >= 8:
        desired_rax = 3
    if marines >= 10:
        label = 4
    elif free <= 5 and minerals >= 2:
        label = 1
    elif barracks < desired_rax and minerals >= 3:
        label = 2
    elif workers < 20 and minerals >= 1:
        label = 0
    else:
        label = 3
    return state, label


def main():
    random.seed(7); torch.manual_seed(7)
    replay = torch.load('mac_sc2/artifacts/sc2egset_macro_examples.pt', weights_only=False)
    replay_x, replay_y = replay['features'], replay['labels']
    # Upweight real examples, then add corrective states needed to execute them legally.
    corrective = [corrective_example() for _ in range(max(20000, len(replay_x) * 4))]
    correction_x = torch.tensor([x for x, _ in corrective]); correction_y = torch.tensor([y for _, y in corrective])
    x = torch.cat([replay_x, replay_x, correction_x])
    y = torch.cat([replay_y, replay_y, correction_y])
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = HybridMacroTransformer().to(device)
    counts = torch.bincount(y, minlength=len(ACTIONS)).float(); weights = (counts.sum() / counts).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=192, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    for epoch in range(1, 19):
        correct = total = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx); loss = nn.functional.cross_entropy(logits, by, weight=weights)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += logits.argmax(-1).eq(by).sum().item(); total += len(by)
        if epoch == 1 or epoch % 4 == 0:
            print(f'epoch {epoch}: hybrid BC accuracy={correct / total:.3f}')
    output = Path('mac_sc2/artifacts/hybrid_macro_bc.pt')
    torch.save({'state_dict': model.cpu().state_dict(), 'actions': ACTIONS,
                'replay_examples': int(len(replay_x)), 'correction_examples': int(len(correction_x))}, output)
    print(f'Saved hybrid replay-BC policy: {output.resolve()}')


if __name__ == '__main__':
    main()
