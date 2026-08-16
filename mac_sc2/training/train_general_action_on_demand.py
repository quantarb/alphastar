#!/usr/bin/env python3
"""Train the complete 4.9.2 ActionSpec directly from raw replays on demand."""
import argparse, json, subprocess, sys
from collections import defaultdict
from pathlib import Path

import sc2reader
import torch
from torch.nn import functional as F

from mac_sc2.legacy.general_action_checkpoint import metadata
from mac_sc2.legacy.general_action_policy import GeneralActionPolicy
from mac_sc2.legacy.general_action_registry import ActionRegistry
from mac_sc2.contracts.semantic_schema import ACTOR_ROLES, TARGET_KINDS
from mac_sc2.training.train_general_macro_on_demand import cat, event_pid, vec


def norm_point(location):
    if not location:
        return (0.0, 0.0)
    return tuple(max(-1.0, min(1.0, float(v) / 100.0 - 1.0)) for v in location[:2])


def rows(path, registry):
    replay = sc2reader.load_replay(path, load_level=4)
    races = {p.pid: p.play_race for p in replay.players}
    latest, counts, selected, previous = {}, defaultdict(lambda: [0] * 8), {}, {}
    for event in replay.events:
        pid = event_pid(event)  # CommandEvent ownership comes from event.player.pid.
        if pid not in races or races[pid] not in ("Terran", "Protoss", "Zerg"):
            continue
        kind = type(event).__name__
        if kind == "PlayerStatsEvent":
            latest[pid] = event; continue
        if kind in ("UnitBornEvent", "UnitInitEvent"):
            q = cat(getattr(event, "unit_type_name", "")); counts[pid] = [a + b for a, b in zip(counts[pid], q)]; continue
        if kind == "SelectionEvent":
            selected[pid] = [str(unit) for unit in (getattr(event, "objects", []) or [])]; continue
        if "CommandEvent" not in kind or pid not in latest:
            continue
        from semantic_action_schema import from_event
        action = from_event(event, registry.patch, races[pid], selected.get(pid, []))
        row = registry.lookup(races[pid], action.actor_role, action.ability_name,
                              action.target_kind, action.target_name, action.queued)
        if row is None:
            continue
        delay = min(7, max(0, int(getattr(event, "frame", 0) - previous.get(pid, getattr(event, "frame", 0))) // 16))
        previous[pid] = int(getattr(event, "frame", 0))
        yield (vec(latest[pid], counts[pid], getattr(event, "second", 0)), registry.task_id(races[pid]), row, norm_point(action.location), delay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True); p.add_argument("--registry", required=True); p.add_argument("--output", required=True)
    p.add_argument("--max-games", type=int); p.add_argument("--batch-size", type=int, default=256); p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--checkpoint-every", type=int, default=200)
    p.add_argument("--eval-difficulty", default="easy")
    args = p.parse_args()
    if args.checkpoint_every <= 0 or args.checkpoint_every > 200:
        raise ValueError("checkpoint-every must be between 1 and 200")
    registry = ActionRegistry(args.registry)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GeneralActionPolicy(17, len(registry.abilities), len(registry.target_types), len(registry.tasks)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    manifest = json.loads(Path(args.manifest).read_text())["valid"]
    files = [x for x in manifest if x["version"].startswith(registry.patch)]
    if args.max_games: files = files[:args.max_games]
    if not files: raise ValueError(f"No {registry.patch} replays selected")
    aidx = {x: i for i, x in enumerate(registry.abilities)}; tidx = {x: i + 1 for i, x in enumerate(registry.target_types)}
    ridx = {x: i for i, x in enumerate(ACTOR_ROLES)}; kidx = {x: i for i, x in enumerate(TARGET_KINDS)}
    seen = discarded = 0; evaluator = None
    # Frequency balancing is derived from the observed ActionSpec labels.  It
    # has no ability-specific exception: repair receives weight only because
    # it is rare in the same raw replay stream as every other action.
    ability_counts = torch.ones(len(registry.abilities), device=device)
    def save(game):
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save({"state_dict": state, "games": game, "decisions": seen, "discarded_commands": discarded,
                    "action_spec_hash": registry.hash, "registry": str(Path(args.registry).resolve()),
                    "architecture": "factorized live 4.9.2 ActionSpec policy", **metadata(registry.hash, args.registry)}, args.output)
    for game, item in enumerate(files, 1):
        try: examples = list(rows(item["path"], registry))
        except Exception as exc: print(f"skip game={game} {type(exc).__name__}", flush=True); continue
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start:start + args.batch_size]
            if not batch: continue
            state = torch.tensor([x[0] for x in batch], dtype=torch.float32, device=device)
            task = torch.tensor([x[1] for x in batch], device=device)
            out = model(state, task)
            labels = {"actor_role": torch.tensor([ridx[x[2].actor_role] for x in batch], device=device),
                      "ability": torch.tensor([aidx[x[2].ability] for x in batch], device=device),
                      "target_kind": torch.tensor([kidx[x[2].target_kind] for x in batch], device=device),
                      "target_type": torch.tensor([tidx.get(x[2].target_type, 0) for x in batch], device=device),
                      "queued": torch.tensor([int(x[2].queued) for x in batch], device=device),
                      "delay_loops": torch.tensor([x[4] for x in batch], device=device)}
            ability_counts += torch.bincount(labels["ability"], minlength=len(registry.abilities)).float()
            ability_weight = (ability_counts.sum() / ability_counts).sqrt().clamp(max=20)
            loss = (F.cross_entropy(out["ability"], labels["ability"], weight=ability_weight) +
                    sum(F.cross_entropy(out[key], value) for key, value in labels.items() if key != "ability") +
                    F.mse_loss(out["target_point"], torch.tensor([x[3] for x in batch], device=device)))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); opt.step(); seen += len(batch)
        if game % 25 == 0: print(f"games={game} decisions={seen}", flush=True)
        if game % args.checkpoint_every == 0:
            save(game); print(f"checkpoint={args.output} games={game}", flush=True)
            if evaluator is None:
                replay = str(Path(args.output).with_suffix(".first_eval.SC2Replay"))
                evaluator = subprocess.Popen([sys.executable, "mac_sc2/play_general_action.py", "--checkpoint", args.output,
                                               "--registry", args.registry, "--race", "terran", "--difficulty", args.eval_difficulty, "--replay", replay])
                print(f"first_checkpoint_live_eval_pid={evaluator.pid} replay={replay}", flush=True)
    save(len(files)); print(f"saved={args.output} games={len(files)} decisions={seen}", flush=True)


if __name__ == "__main__": main()
