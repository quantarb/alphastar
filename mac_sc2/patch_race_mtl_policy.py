"""Patch-and-race multitask policy with task-local action vocabularies.

Each head is keyed by ``<patch family>/<race>`` and has exactly the labels
observed for that task.  The shared trunk is the transfer mechanism; labels
are never shared by integer position across patch-specific heads.
"""
from __future__ import annotations

import re
import torch
from torch import nn

from general_macro_policy import STATE_SIZE


def module_key(task: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", task)


class PatchRaceMTLPolicy(nn.Module):
    def __init__(self, task_vocabs: dict[str, list[str]], width: int = 224):
        super().__init__()
        self.task_vocabs = {task: list(vocab) for task, vocab in task_vocabs.items()}
        self.shared = nn.Sequential(
            nn.Linear(STATE_SIZE, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            module_key(task): nn.Linear(width, len(vocab))
            for task, vocab in self.task_vocabs.items()
        })

    def forward_task(self, state: torch.Tensor, task: str) -> torch.Tensor:
        return self.heads[module_key(task)](self.shared(state))

    def ensure_labels(self, task: str, labels: list[str], max_actions: int = 160) -> bool:
        """Create/expand one head online; returns whether parameters changed."""
        old_vocab = self.task_vocabs.get(task, [])
        additions = [label for label in labels if label not in old_vocab]
        if not additions:
            return False
        new_vocab = (old_vocab + additions)[:max_actions]
        if new_vocab == old_vocab:
            return False
        key = module_key(task)
        old_head = self.heads[key] if key in self.heads else None
        device = self.shared[0].weight.device
        new_head = nn.Linear(self.shared[0].out_features, len(new_vocab)).to(device)
        if old_head is not None:
            with torch.no_grad():
                new_head.weight[:len(old_vocab)].copy_(old_head.weight)
                new_head.bias[:len(old_vocab)].copy_(old_head.bias)
        self.task_vocabs[task] = new_vocab
        self.heads[key] = new_head
        return True
