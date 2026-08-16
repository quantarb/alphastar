#!/usr/bin/env python3
"""On-demand BC with one task-local action head per patch/race pair.

The short catalog pass reads raw replays only to discover each task's labels;
it writes no preprocessed dataset.  Training then reparses one replay at a
time and immediately discards its features.
"""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import sc2reader
import torch
from torch.nn import functional as F

from mac_sc2.legacy.patch_race_mtl_policy import PatchRaceMTLPolicy
from mac_sc2.training.train_general_macro_on_demand import RID, cat, event_pid, vec

IGNORED = {"", "RightClick", "SetWorkerRally", "SetRallyPoint", "Stop", "HoldPosition"}


def patch_family(version: str) -> str:
    return ".".join(version.split(".")[:3])


def task_name(version: str, race: str) -> str:
    return f"{patch_family(version)}/{race}"


def ability_name(event) -> str | None:
    name = (getattr(event, "ability_name", "") or "").strip()
    return None if name in IGNORED else name


def replay_rows(item: dict, window: int, winner_only: bool):
    replay = sc2reader.load_replay(item["path"], load_level=4)
    race = {p.pid: p.play_race for p in replay.players if p.play_race in RID}
    winners = {p.pid for p in replay.players if getattr(p, "result", None) == "Win"}
    latest, counts, emitted, out = {}, defaultdict(lambda: [0] * 8), defaultdict(lambda: -1), []
    for event in replay.events:
        pid = event_pid(event)
        if pid not in race:
            continue
        typ = type(event).__name__
        if typ == "PlayerStatsEvent":
            latest[pid] = event
        elif typ in ("UnitBornEvent", "UnitInitEvent"):
            unit = cat(getattr(event, "unit_type_name", ""))
            counts[pid] = [a + b for a, b in zip(counts[pid], unit)]
        elif "CommandEvent" in typ:
            label = ability_name(event)
            if label is None or pid not in latest or (winner_only and pid not in winners):
                continue
            bucket = int(getattr(event, "second", 0) // window)
            if bucket == emitted[pid]:
                continue
            emitted[pid] = bucket
            out.append((task_name(item["version"], race[pid]), vec(latest[pid], counts[pid], getattr(event, "second", 0)), label))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", help="Compatible base patch/race checkpoint to fine-tune")
    parser.add_argument("--start-game", type=int, default=0)
    parser.add_argument("--max-games", type=int, default=1000)
    parser.add_argument("--patch", help="Train only one patch family, e.g. 4.9.2")
    parser.add_argument("--patches", help="Comma-separated patch families for a compact MTL pilot")
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-label-count", type=int, default=3)
    parser.add_argument("--winner-only", action="store_true")
    args = parser.parse_args()
    files = json.loads(Path(args.manifest).read_text())["valid"]
    if args.patch:
        files = [item for item in files if patch_family(item["version"]) == args.patch]
    if args.patches:
        selected = set(args.patches.split(","))
        files = [item for item in files if patch_family(item["version"]) in selected]
    files = files[args.start_game:args.start_game + args.max_games]

    # Catalog is in-memory only. It gives every head a patch/race-valid output set.
    counts = defaultdict(Counter)
    for game, item in enumerate(files, 1):
        try:
            for task, _, label in replay_rows(item, args.window, args.winner_only):
                counts[task][label] += 1
        except Exception as exc:
            print(f"catalog skip game={game} {type(exc).__name__}", flush=True)
        if game % 25 == 0:
            print(f"catalog games={game}/{len(files)} tasks={len(counts)}", flush=True)
    vocabs = {task: sorted(label for label, n in labels.items() if n >= args.min_label_count)
              for task, labels in counts.items()}
    vocabs = {task: vocab for task, vocab in vocabs.items() if vocab}
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        # Fine-tuning must retain the base pass's exact task-local label IDs.
        # Winner-only examples may omit rare labels, but must not renumber heads.
        vocabs = checkpoint["task_vocabs"]
    if not vocabs:
        raise RuntimeError("No task vocabularies found")
    label_ids = {task: {label: i for i, label in enumerate(vocab)} for task, vocab in vocabs.items()}

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PatchRaceMTLPolicy(vocabs).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    seen = Counter()
    for game, item in enumerate(files, 1):
        try:
            rows = replay_rows(item, args.window, args.winner_only)
        except Exception as exc:
            print(f"train skip game={game} {type(exc).__name__}", flush=True)
            continue
        by_task = defaultdict(list)
        for task, state, label in rows:
            if label in label_ids.get(task, {}):
                by_task[task].append((state, label_ids[task][label]))
        for task, data in by_task.items():
            for start in range(0, len(data), args.batch_size):
                batch = data[start:start + args.batch_size]
                state = torch.tensor([x for x, _ in batch], dtype=torch.float32, device=device)
                target = torch.tensor([y for _, y in batch], dtype=torch.long, device=device)
                logits = model.forward_task(state, task)
                loss = F.cross_entropy(logits, target)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
                seen[task] += len(batch)
        if game % 25 == 0:
            print(f"games={game} tasks={len(vocabs)} decisions={sum(seen.values())}", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "task_vocabs": vocabs,
                "decisions_by_task": dict(seen), "winner_only": args.winner_only, "resumed_from": args.resume,
                "architecture": "shared backbone + patch/race-specific valid-action heads"}, args.output)
    print(f"saved={args.output} tasks={len(vocabs)} decisions={sum(seen.values())}", flush=True)


if __name__ == "__main__":
    main()
