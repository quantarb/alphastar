#!/usr/bin/env python3
"""Train the Terran macro Transformer from build-order demonstration states."""
from pathlib import Path
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from mac_sc2.legacy.real_game_macro import MacroTransformer, encode_state, teacher_action


def main():
    random.seed(7); torch.manual_seed(7)
    examples, labels = [], []
    for _ in range(12000):
        minerals = random.randrange(0, 601, 25)
        free_supply = random.randrange(0, 13)
        scvs = random.randrange(8, 31)
        depots = random.randrange(0, 5)
        barracks = random.randrange(0, 3)
        marines = random.randrange(0, 31)
        examples.append(encode_state(minerals, free_supply, scvs, depots, barracks, marines))
        labels.append(teacher_action(minerals, free_supply, scvs, depots, barracks, marines))
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = MacroTransformer().to(device)
    loader = DataLoader(TensorDataset(torch.tensor(examples), torch.tensor(labels)), batch_size=128, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(1, 13):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = nn.functional.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += logits.argmax(-1).eq(y).sum().item(); total += len(y)
        print(f'epoch {epoch}: teacher accuracy={correct / total:.3f}')
    output = Path('mac_sc2/artifacts/real_game_macro_transformer.pt'); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'action_names': __import__('real_game_macro').ACTION_NAMES}, output)
    print(f'Saved trained real-game policy: {output.resolve()}')


if __name__ == '__main__':
    main()
