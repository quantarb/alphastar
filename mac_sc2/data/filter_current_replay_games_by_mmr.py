#!/usr/bin/env python3
"""Filter a current-patch replay manifest at game level by recorded MMR."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import sc2reader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--filtered-replay-manifest", type=Path)
    parser.add_argument("--move-unselected-to-trash", action="store_true")
    parser.add_argument("--trash-dir", type=Path, default=Path("/Users/johnnylee/.Trash"))
    parser.add_argument("--min-mmr", type=int, default=1)
    args = parser.parse_args()

    source = json.loads(args.replay_manifest.read_text())
    unique_paths = sorted({Path(row["path"]).resolve() for row in source["rows"]})
    selected = []
    unselected = []
    for replay_path in unique_paths:
        replay = sc2reader.load_replay(str(replay_path), load_level=2)
        mmrs = [int(getattr(player, "mmr", 0) or 0) for player in replay.players]
        if max(mmrs, default=0) >= args.min_mmr:
            selected.append((replay_path, mmrs))
        else:
            unselected.append(replay_path)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\n" for path, _ in selected))
    if args.filtered_replay_manifest:
        selected_paths = {path for path, _ in selected}
        filtered = dict(source)
        filtered["rows"] = [row for row in source["rows"] if Path(row["path"]).resolve() in selected_paths]
        args.filtered_replay_manifest.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n")
    trash_destination = None
    if args.move_unselected_to_trash:
        trash_destination = args.trash_dir / "alphastar_current_5_0_16_mmr_lte_4000"
        if trash_destination.exists():
            raise RuntimeError(f"refusing to merge into existing trash destination: {trash_destination}")
        trash_destination.mkdir(parents=True)
        for replay_path in unselected:
            shutil.move(str(replay_path), str(trash_destination / replay_path.name))
    report = {
        "source_manifest": str(args.replay_manifest.resolve()),
        "min_mmr": args.min_mmr,
        "games": len(selected),
        "player_trajectories_with_mmr": sum(mmr >= args.min_mmr for _, mmrs in selected for mmr in mmrs),
        "output_manifest": str(args.output.resolve()),
        "filtered_replay_manifest": (str(args.filtered_replay_manifest.resolve())
                                     if args.filtered_replay_manifest else None),
        "moved_to_trash": len(unselected) if args.move_unselected_to_trash else 0,
        "trash_destination": str(trash_destination) if trash_destination else None,
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
