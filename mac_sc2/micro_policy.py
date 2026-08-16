"""Per-unit tactical Transformer: attack, kite, or regroup (no APM throttle)."""
import torch
from torch import nn

MODES = ('attack', 'kite', 'regroup')

class MicroTransformer(nn.Module):
    def __init__(self, width=64, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(32, width)
        self.slot = nn.Embedding(7, width)
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.net = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(width, len(MODES))
    def forward(self, x):
        slots = torch.arange(7, device=x.device)
        return self.head(self.net(self.embed(x) + self.slot(slots))[:, 0])
