#!/usr/bin/env python3
"""Print patch/race action-head predictions on a real replay trajectory."""
import argparse, json
from pathlib import Path

import torch

from patch_race_mtl_policy import PatchRaceMTLPolicy
from train_patch_race_mtl_on_demand import replay_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--patch", default="4.9.2")
    parser.add_argument("--race", choices=("Terran", "Protoss", "Zerg"), default="Terran")
    parser.add_argument("--game-index", type=int, default=0)
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PatchRaceMTLPolicy(checkpoint["task_vocabs"])
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    items = [x for x in json.loads(Path(args.manifest).read_text())["valid"] if x["version"].startswith(args.patch)]
    item = items[args.game_index]
    task = f"{args.patch}/{args.race}"
    vocab = checkpoint["task_vocabs"][task]
    rows = [row for row in replay_rows(item, 8, winner_only=False) if row[0] == task]
    print(f"replay={Path(item['path']).name} task={task} rows={len(rows)}")
    for index in torch.linspace(0, max(len(rows) - 1, 0), args.examples).long().tolist():
        _, state, actual = rows[index]
        with torch.no_grad():
            probabilities = model.forward_task(torch.tensor([state], dtype=torch.float32), task)[0].softmax(-1)
        top = probabilities.topk(min(3, len(vocab)))
        choices = ", ".join(f"{vocab[i]} {p:.0%}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"step={index:>4} replay_action={actual:<28} predicted=[{choices}]")


if __name__ == "__main__":
    main()
