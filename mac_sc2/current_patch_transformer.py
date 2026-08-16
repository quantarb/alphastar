"""Small causal Transformer used for the current-patch curriculum."""
import torch
from torch import nn

ACTIONS = ('scv', 'supply', 'barracks', 'marine', 'attack', 'wait')


class CurrentPatchTransformer(nn.Module):
    def __init__(self, width=96, heads=4, layers=2):
        super().__init__()
        self.value = nn.Embedding(64, width)
        self.feature = nn.Embedding(7, width)
        self.time = nn.Embedding(8, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(width, len(ACTIONS))

    def forward(self, states):
        # [batch, sequence, 7 categorical state features]
        feat = torch.arange(7, device=states.device)
        time = torch.arange(states.shape[1], device=states.device)
        x = self.value(states).sum(-2) + self.feature(feat).sum(0) + self.time(time)[None]
        mask = torch.triu(torch.ones(states.shape[1], states.shape[1], device=states.device, dtype=torch.bool), 1)
        return self.head(self.encoder(x, mask=mask)[:, -1])
