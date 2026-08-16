"""A tiny Transformer policy used for an end-to-end SC2 client integration test.

It predicts a coarse screen coordinate for the beacon from coordinate tokens. The
purpose is to prove the complete Mac path: checkpoint -> PyTorch inference ->
PySC2 action. It is not presented as a competitive SC2 policy.
"""
import torch
from torch import nn

GRID = 16
VOCAB = 1 + GRID + GRID


class BeaconTransformer(nn.Module):
    def __init__(self, width=48, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(3, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 2,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(width, GRID * GRID)

    def forward(self, tokens):
        positions = torch.arange(3, device=tokens.device)
        return self.head(self.encoder(self.embedding(tokens) + self.position(positions))[..., -1, :])


def encode_position(x, y):
    return [0, 1 + int(x), 1 + GRID + int(y)]
