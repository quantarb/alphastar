"""One playable MTL policy with historical auxiliary task heads."""
from __future__ import annotations

from torch import Tensor, nn

from mac_sc2.architectures.multitask_policy import PlayableMultiTaskPolicy
from mac_sc2.architectures.patch_race_rich_mtl import module_key
from mac_sc2.contracts.semantic_schema import FAMILIES


class UnifiedPlayableMTLPolicy(PlayableMultiTaskPolicy):
    """A live 4.9.2 policy whose trunk is jointly trained by all replay tasks.

    The inherited current-patch heads are the only executable outputs. These
    historical heads are auxiliary classifiers that train the same trunk, but
    are never consulted by the SC2 runner.
    """
    def __init__(self, live_specs: dict[str, list[dict]], live_routes: dict[str, tuple[str, ...]],
                 historical_specs: dict[str, list[dict]]):
        super().__init__(live_specs, live_routes)
        self.historical_specs = historical_specs
        self.historical_micro_heads = nn.ModuleDict({module_key(task): nn.Linear(224, len(vocab))
                                                     for task, vocab in historical_specs.items()})
        self.historical_macro_heads = nn.ModuleDict({module_key(task): nn.Linear(224, len(FAMILIES))
                                                     for task in historical_specs})
        self.historical_build_ids = {
            task: tuple(index for index, row in enumerate(vocab)
                        if row["family"] == "build" or row["replay_ability"].lower().startswith("land"))
            for task, vocab in historical_specs.items()
        }
        self.historical_build_heads = nn.ModuleDict({module_key(task): nn.Linear(224, len(indices))
                                                     for task, indices in self.historical_build_ids.items() if indices})

    def historical_micro_logits(self, state: Tensor, task: str) -> Tensor:
        return self.historical_micro_heads[module_key(task)](self.micro.hidden(state))

    def historical_macro_logits(self, state: Tensor, task: str) -> Tensor:
        return self.historical_macro_heads[module_key(task)](self.micro.hidden(state))

    def historical_build_logits(self, state: Tensor, task: str) -> Tensor:
        return self.historical_build_heads[module_key(task)](self.micro.hidden(state))
