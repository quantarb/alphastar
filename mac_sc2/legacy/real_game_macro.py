"""A small learned macro policy for a genuine SC2 1v1 game.

The policy's training labels are a transparent Terran build-order teacher. It is
not a claim of replay-trained grandmaster play; it is an end-to-end trained
policy that can make legal SC2 raw actions in a real melee game.
"""
import torch
from torch import nn

ACTION_NAMES = ('train_scv', 'supply', 'barracks', 'marine', 'attack', 'noop')
VOCAB = 12


class MacroTransformer(nn.Module):
    def __init__(self, width=48, heads=4, layers=2):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, width)
        self.position = nn.Embedding(6, width)
        block = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers)
        self.head = nn.Linear(width, len(ACTION_NAMES))

    def forward(self, state):
        pos = torch.arange(state.shape[1], device=state.device)
        return self.head(self.encoder(self.embedding(state) + self.position(pos))[..., -1, :])


def encode_state(minerals, free_supply, scvs, depots, barracks, marines):
    """Six small categorical tokens, all in range [0, 11]."""
    return [min(11, minerals // 50), min(11, free_supply), min(11, scvs // 2),
            min(11, depots), min(11, barracks), min(11, marines // 2)]


def teacher_action(minerals, free_supply, scvs, depots, barracks, marines):
    if free_supply <= 2 and minerals >= 100:
        return 1
    if barracks < 1 and minerals >= 150:
        return 2
    if scvs < 22 and minerals >= 50:
        return 0
    if marines < 16 and barracks and minerals >= 50 and free_supply >= 1:
        return 3
    if marines >= 8:
        return 4
    return 5
