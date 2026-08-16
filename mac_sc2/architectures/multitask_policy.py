"""Composable playable MTL policy with independently executable task heads."""
from __future__ import annotations

import torch
from torch import nn

from mac_sc2.architectures.macro_placement import PlacementRanker
from mac_sc2.architectures.patch_race_rich_mtl import PatchRaceRichMTLPolicy
from mac_sc2.architectures.patch_race_rich_mtl import module_key
from mac_sc2.contracts.multitask import task_key
from mac_sc2.contracts.semantic_schema import FAMILIES


class PlayableMultiTaskPolicy(nn.Module):
    """One checkpoint with a primary action policy and auxiliary task heads.

    ``<patch>/<race>/micro`` is the only head that chooses the next command.
    Macro family, build order, placement, and repair are auxiliary losses or
    rankers scoped to the same live patch and race; they never substitute for
    action selection.
    """
    def __init__(self, task_specs: dict[str, list[dict]], routes: dict[str, tuple[str, ...]]):
        super().__init__()
        micro_specs = {task_key(*base.split("/"), "micro"): vocab for base, vocab in task_specs.items()}
        self.micro = PatchRaceRichMTLPolicy(micro_specs)
        self.routes = {base: tuple(tasks) for base, tasks in routes.items()}
        # Task-local tuple IDs cannot be shared across patches/races.  Their
        # embeddings are therefore task-local, while the recurrent sequence
        # encoder is shared across all task streams.
        self.micro_history = nn.ModuleDict({module_key(task): nn.Embedding(len(vocab) + 1, 224, padding_idx=0)
                                            for task, vocab in micro_specs.items()})
        self.sequence = nn.GRU(224, 224, batch_first=True)
        self.build_action_ids = {base: tuple(index for index, row in enumerate(vocab)
                                             if row["family"] == "build" or row["replay_ability"].lower().startswith("land"))
                                 for base, vocab in task_specs.items()}
        self.build_heads = nn.ModuleDict({module_key(task_key(*base.split("/"), "build")): nn.Linear(224, len(self.build_action_ids[base]))
                                          for base, tasks in self.routes.items() if "build" in tasks})
        self.macro_heads = nn.ModuleDict({module_key(task_key(*base.split("/"), "macro")): nn.Linear(224, len(FAMILIES))
                                          for base, tasks in self.routes.items() if "macro" in tasks})
        self.build_placement_heads = nn.ModuleDict({module_key(task_key(*base.split("/"), "build")): PlacementRanker()
                                                    for base, tasks in self.routes.items() if "build" in tasks})

    def load_initializers(self, macro_state: dict, placement_repair_state: dict | None = None) -> None:
        self.micro.load_streaming_backbone(macro_state)
        if placement_repair_state is not None:
            for head in self.build_placement_heads.values(): head.load_state_dict(placement_repair_state["placement_state_dict"])

    def micro_hidden(self, state: torch.Tensor, patch: str, race: str, history: torch.Tensor | None = None) -> torch.Tensor:
        """Score the next exact tuple with a short, task-local action history.

        History stores tuple ID + 1; zero is padding.  The live runner and raw
        replay extractor both keep only the preceding commands from this same
        patch/race ActionSpec.
        """
        task = task_key(patch, race, "micro")
        hidden = self.micro.hidden(state)
        if history is not None and history.numel():
            embedded = self.micro_history[module_key(task)](history)
            _, sequence_hidden = self.sequence(embedded)
            hidden = hidden + sequence_hidden[-1]
        return hidden

    def micro_logits(self, state: torch.Tensor, patch: str, race: str, history: torch.Tensor | None = None) -> torch.Tensor:
        task = task_key(patch, race, "micro")
        return self.micro.task_heads[module_key(task)](self.micro_hidden(state, patch, race, history))

    def build_logits(self, state: torch.Tensor, patch: str, race: str, history: torch.Tensor | None = None) -> torch.Tensor:
        key = module_key(task_key(patch, race, "build"))
        if key not in self.build_heads: raise RuntimeError(f"build is not a valid task for {patch}/{race}")
        return self.build_heads[key](self.micro_hidden(state, patch, race, history))

    def macro_logits(self, state: torch.Tensor, patch: str, race: str) -> torch.Tensor:
        key = module_key(task_key(patch, race, "macro"))
        return self.macro_heads[key](self.micro.hidden(state))

    def build_placement_scores(self, entities: torch.Tensor, mask: torch.Tensor, candidates: torch.Tensor, patch: str, race: str) -> torch.Tensor:
        key = module_key(task_key(patch, race, "build"))
        if key not in self.build_placement_heads: raise RuntimeError(f"build is not a valid task for {patch}/{race}")
        return self.build_placement_heads[key](entities, mask, candidates)
