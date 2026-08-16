#!/usr/bin/env python3
"""Train the factorized tactical heads while retaining a learned macro trunk.

The teacher is deliberately explicit: it provides safe tactical supervision
from complete local combat state.  This is used only because the historical
replays cannot be replay-observed by the installed SC2 client; metadata stored
in the checkpoint makes that provenance impossible to confuse with pro BC.
"""
import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.legacy.factorized_micro_policy import FactorizedMicroPolicy, TACTICAL_FEATURES, checkpoint_metadata
from mac_sc2.legacy.multirace_general_policy import MultiRaceGeneralMacroPolicy


def combat_teacher(n: int, device: torch.device):
    """Generate bounded, physically coherent local-combat observations."""
    g = torch.Generator().manual_seed(97563)
    race = torch.randint(0, 3, (n,), generator=g)
    x = torch.rand(n, TACTICAL_FEATURES, generator=g)
    army = torch.randint(2, 55, (n,), generator=g).float()
    enemy = torch.randint(1, 55, (n,), generator=g).float()
    low = torch.rand(n, generator=g)
    reload = torch.rand(n, generator=g)
    distance = torch.rand(n, generator=g)
    friendly_health = .25 + .75 * torch.rand(n, generator=g)
    enemy_health = .2 + .8 * torch.rand(n, generator=g)
    x[:, 0] = race.float() / 2
    x[:, 5] = army / 60; x[:, 6] = enemy / 60; x[:, 7] = low; x[:, 8] = reload
    x[:, 9] = distance; x[:, 10] = distance * .7; x[:, 11] = friendly_health; x[:, 12] = enemy_health
    # Priority order captures the crucial tactical distinction: retreat when
    # overwhelmed, kite during reload under close pressure, otherwise attack.
    overwhelmed = (enemy > army * 1.45) | (friendly_health < .38)
    kite = ~overwhelmed & (reload > .42) & (distance < .42)
    hold = ~overwhelmed & ~kite & (distance > .88)
    intent = torch.zeros(n, dtype=torch.long)
    intent[kite] = 1; intent[overwhelmed] = 2; intent[hold] = 3
    group = torch.zeros(n, dtype=torch.long)
    group[low > .38] = 1
    group[(low <= .38) & (distance < .48)] = 2
    target = torch.zeros(n, dtype=torch.long)
    target[(enemy_health < .52) & ~overwhelmed] = 1
    target[(enemy_health >= .52) & (enemy > army * 1.15)] = 2
    direction = torch.zeros(n, dtype=torch.long)
    direction[kite] = 1; direction[overwhelmed] = 2; direction[hold] = 3
    labels = {"group": group, "intent": intent, "target": target, "direction": direction}
    return x.to(device), race.to(device), {key: value.to(device) for key, value in labels.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--macro-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--examples", type=int, default=60000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=8e-4)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    macro_data = torch.load(args.macro_checkpoint, map_location="cpu", weights_only=False)
    macro = MultiRaceGeneralMacroPolicy()
    macro.load_state_dict(macro_data["state_dict"])
    model = FactorizedMicroPolicy(macro.shared, freeze_backbone=True).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=.01)
    x, race, labels = combat_teacher(args.examples, device)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(args.examples, device=device)
        correct = {key: 0 for key in labels}; total = 0
        for start in range(0, args.examples, args.batch_size):
            ids = order[start:start + args.batch_size]
            out = model(x[ids], race[ids])
            loss = sum(F.cross_entropy(out[key], labels[key][ids]) for key in labels)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += len(ids)
            for key in labels: correct[key] += out[key].argmax(-1).eq(labels[key][ids]).sum().item()
        print(f"epoch={epoch} " + " ".join(f"{key}_acc={correct[key]/total:.3f}" for key in labels), flush=True)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state,
        "macro_checkpoint": str(args.macro_checkpoint),
        "backbone_frozen": True,
        "trained_micro_examples": args.examples,
        "training_source": "explicit local-combat tactical-teacher distillation; not replay behavioral cloning",
        **checkpoint_metadata(),
    }, output)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
