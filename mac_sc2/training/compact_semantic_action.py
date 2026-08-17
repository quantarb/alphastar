"""One-cache E2E fine-tuning of the patch-pinned, executable semantic policy."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.semantic_action_mtl import SemanticActionMTL
from mac_sc2.contracts.semantic_action import BASELINE_SPEC_HASH, spec_hash
from mac_sc2.contracts.semantic_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, TARGET_KINDS
from mac_sc2.data.compact_semantic_action import examples

FIELDS = {"actor": ACTOR_ROLES, "family": FAMILIES, "payload": PAYLOAD_ROLES, "target": TARGET_KINDS}


def train(cache: str, race: str, output: str, baseline: str) -> dict:
    source = torch.load(baseline, map_location="cpu", weights_only=False)
    if source.get("action_contract_hash") != BASELINE_SPEC_HASH:
        raise RuntimeError("baseline is not the approved semantic initializer")
    rows = list(examples(cache, race, discarded := Counter()))
    if not rows:
        raise RuntimeError(f"no complete live-contract labels; discarded={dict(discarded)}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = SemanticActionMTL(); model.load_state_dict(source["state_dict"]); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    for start in range(0, len(rows), 128):
        chunk = rows[start:start + 128]
        state = torch.tensor([x[0] for x in chunk], dtype=torch.float32, device=device)
        race_ids = torch.tensor([x[1] for x in chunk], device=device)
        out = model(state, race_ids)
        loss = F.cross_entropy(out["queued"], torch.tensor([x[3] for x in chunk], device=device))
        for index, (name, choices) in enumerate(FIELDS.items()):
            target = torch.tensor([choices.index(x[2][index]) for x in chunk], device=device)
            loss = loss + F.cross_entropy(out[name], target)
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "games": 1, "resumed_from": str(Path(baseline).resolve()),
                "action_contract_hash": spec_hash(), "labels": len(rows), "discarded": dict(discarded),
                "source_cache": str(Path(cache).resolve())}, destination)
    return {"checkpoint": str(destination.resolve()), "labels": len(rows), "discarded": dict(discarded)}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("cache"); parser.add_argument("--race", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--baseline", default="mac_sc2/artifacts/semantic_contract_all_replays.pt")
    args = parser.parse_args(); print(train(args.cache, args.race, args.output, args.baseline))


if __name__ == "__main__": main()
