"""Research-only multi-patch replay pretraining model."""
from __future__ import annotations

from torch import nn

from mac_sc2.architectures.patch_race_rich_mtl import PatchRaceRichMTLPolicy, module_key
from mac_sc2.contracts.semantic_schema import FAMILIES


class HistoricalReplayMTL(nn.Module):
    """Task-local historical heads with a shared transferable trunk.

    This model is deliberately not imported by the live runner.
    """
    def __init__(self, specs: dict[str, list[dict]]):
        super().__init__()
        self.specs = specs
        self.micro = PatchRaceRichMTLPolicy(specs)
        self.macro = nn.ModuleDict({module_key(task): nn.Linear(224, len(FAMILIES)) for task in specs})
        self.build_ids = {task: tuple(index for index, row in enumerate(vocab) if row["family"] == "build" or row["replay_ability"].lower().startswith("land"))
                          for task, vocab in specs.items()}
        self.build = nn.ModuleDict({module_key(task): nn.Linear(224, len(indices)) for task, indices in self.build_ids.items() if indices})

    def load_backbone(self, state_dict: dict) -> None:
        self.micro.load_streaming_backbone(state_dict)

    def micro_logits(self, state, task: str):
        return self.micro.task_logits(state, task)

    def macro_logits(self, state, task: str):
        return self.macro[module_key(task)](self.micro.hidden(state))

    def build_logits(self, state, task: str):
        return self.build[module_key(task)](self.micro.hidden(state))
