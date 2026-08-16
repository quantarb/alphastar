"""Composable playable MTL policy with independently executable task heads."""
from __future__ import annotations

import torch
from torch import nn

from mac_sc2.architectures.macro_placement import PlacementRanker
from mac_sc2.architectures.patch_race_rich_mtl import PatchRaceRichMTLPolicy
from mac_sc2.architectures.repair import RepairPolicy


class PlayableMultiTaskPolicy(nn.Module):
    """One checkpoint, three task heads, no forced shared latent contract.

    Macro uses its replay-trained patch/race trunk.  Placement and repair
    retain their snapshot-trained encoders because those checkpoints have
    different, valid input representations.  Joint training coordinates tasks
    without discarding either learned task-specific representation.
    """
    def __init__(self, task_specs: dict[str, list[dict]]):
        super().__init__()
        self.macro = PatchRaceRichMTLPolicy(task_specs)
        self.placement = PlacementRanker()
        self.repair = RepairPolicy()

    def load_initializers(self, macro_state: dict, placement_repair_state: dict) -> None:
        self.macro.load_streaming_backbone(macro_state)
        self.placement.load_state_dict(placement_repair_state["placement_state_dict"])
        self.repair.load_state_dict(placement_repair_state["repair_state_dict"])

    def task_logits(self, state: torch.Tensor, task: str) -> torch.Tensor:
        return self.macro.task_logits(state, task)
