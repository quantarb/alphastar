"""Transfer-learning tactical controller for the runnable MTL macro agent.

The controller deliberately reuses the shared backbone learned by the macro
checkpoint.  A compact adapter turns local combat observations into the
backbone's 17-dimensional input and four factorized heads decode commands
which the live runner can execute: squad, intent, target policy, and movement
direction.  It never emits raw/patch-dependent ability IDs.
"""
import torch
from torch import nn

from multirace_general_policy import RACES

TACTICAL_FEATURES = 20
GROUPS = ("all_army", "wounded", "nearby")
INTENTS = ("attack", "kite", "regroup", "hold")
TARGETS = ("nearest", "lowest_health", "highest_health")
DIRECTIONS = ("toward_enemy", "away_from_enemy", "toward_home", "hold_position")


class FactorizedMicroPolicy(nn.Module):
    """Macro-backbone transfer model with independently trainable micro heads."""

    def __init__(self, backbone: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Sequential(
            nn.Linear(TACTICAL_FEATURES, 48), nn.GELU(), nn.LayerNorm(48),
            nn.Linear(48, 17),
        )
        self.heads = nn.ModuleDict({
            race: nn.ModuleDict({
                "group": nn.Linear(224, len(GROUPS)),
                "intent": nn.Linear(224, len(INTENTS)),
                "target": nn.Linear(224, len(TARGETS)),
                "direction": nn.Linear(224, len(DIRECTIONS)),
            })
            for race in RACES
        })
        self.freeze_backbone(freeze_backbone)

    def freeze_backbone(self, freeze: bool = True) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze

    def forward(self, features: torch.Tensor, race: torch.Tensor):
        # The stored macro shared trunk is an executable, learned initializer;
        # only its input adapter and tactical heads need replay-micro training.
        latent = self.backbone(self.adapter(features))
        outputs = {}
        for key in ("group", "intent", "target", "direction"):
            all_logits = torch.stack([self.heads[name][key](latent) for name in RACES], dim=1)
            outputs[key] = all_logits[torch.arange(features.shape[0], device=features.device), race]
        return outputs


def checkpoint_metadata():
    return {
        "architecture": "macro-backbone transfer + factorized tactical heads",
        "feature_count": TACTICAL_FEATURES,
        "groups": GROUPS,
        "intents": INTENTS,
        "targets": TARGETS,
        "directions": DIRECTIONS,
    }
