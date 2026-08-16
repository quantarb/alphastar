#!/usr/bin/env python3
"""Train the small live-SC2 control Transformer on all coarse screen positions."""
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from beacon_policy import GRID, BeaconTransformer, encode_position


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='mac_sc2/artifacts/beacon_transformer.pt')
    parser.add_argument('--epochs', type=int, default=25)
    args = parser.parse_args()
    tokens = torch.tensor([encode_position(x, y) for y in range(GRID) for x in range(GRID)])
    labels = torch.arange(GRID * GRID)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = BeaconTransformer().to(device)
    loader = DataLoader(TensorDataset(tokens, labels), batch_size=64, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for epoch in range(1, args.epochs + 1):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x); loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += logits.argmax(-1).eq(y).sum().item(); total += len(y)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f'epoch {epoch}: accuracy={correct/total:.3f}')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'grid': GRID}, output)
    print(f'Saved checkpoint: {output.resolve()}')


if __name__ == '__main__':
    torch.manual_seed(7); main()
