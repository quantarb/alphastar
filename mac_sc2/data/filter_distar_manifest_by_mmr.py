#!/usr/bin/env python3
"""Filter explicit DI-star replay/player rows to trajectories with recorded MMR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sc2reader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--min-mmr", type=int, default=1)
    args = parser.parse_args()

    selected = []
    for line in args.manifest.read_text().splitlines():
        replay_path, player_index = line.rsplit("\t", 1)
        replay = sc2reader.load_replay(replay_path, load_level=2)
        player = replay.players[int(player_index)]
        mmr = int(getattr(player, "mmr", 0) or 0)
        if mmr >= args.min_mmr:
            selected.append((replay_path, int(player_index), mmr))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(f"{path}\t{index}\n" for path, index, _ in selected))
    report = {
        "source_manifest": str(args.manifest.resolve()),
        "min_mmr": args.min_mmr,
        "trajectories": len(selected),
        "raw_replay_count": len({path for path, _, _ in selected}),
        "mmr_min": min((mmr for _, _, mmr in selected), default=None),
        "mmr_max": max((mmr for _, _, mmr in selected), default=None),
        "output_manifest": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
