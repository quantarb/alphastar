#!/usr/bin/env python3
"""Train a Transformer strictly on SC2EGSet human macro examples."""
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReplayMacroTransformer(nn.Module):
    def __init__(self, width=64, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(32, width)
        self.position = nn.Embedding(7, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(width, 4)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        return self.head(self.encoder(self.embedding(tokens) + self.position(positions))[..., -1, :])


def main():
    torch.manual_seed(7)
    data = torch.load('mac_sc2/artifacts/sc2egset_macro_examples.pt', weights_only=False)
    x, y = data['features'], data['labels']
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = ReplayMacroTransformer().to(device)
    counts = torch.bincount(y, minlength=4).float(); class_weights = (counts.sum() / counts).to(device)
    loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-3)
    for epoch in range(1, 21):
        model.train(); correct = total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x); loss = nn.functional.cross_entropy(logits, batch_y, weight=class_weights)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += logits.argmax(-1).eq(batch_y).sum().item(); total += len(batch_y)
        if epoch == 1 or epoch % 5 == 0:
            print(f'epoch {epoch}: train accuracy={correct / total:.3f}')
    output = Path('mac_sc2/artifacts/sc2egset_macro_transformer.pt')
    torch.save({'state_dict': model.cpu().state_dict(), 'class_labels': data['class_labels']}, output)
    print(f'Saved replay-trained behavior clone: {output.resolve()}')


if __name__ == '__main__':
    main()
