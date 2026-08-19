#!/usr/bin/env python3
"""Build a raw-replay manifest for current-patch DI-star Zerg fine-tuning.

The manifest is not a training shard: DI-star replays these source files on
demand.  The accompanying report records every command that its versioned
action table cannot train or execute.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import sc2reader
from s2clientprotocol import sc2api_pb2 as sc_pb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--replay-manifest", required=True, type=Path)
    parser.add_argument("--distar-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--winner-only", action="store_true",
                        help="write explicit replay/player rows for winning Zerg trajectories only")
    parser.add_argument("--explicit-players", action="store_true",
                        help="write explicit replay/player rows for every selected Zerg trajectory")
    args = parser.parse_args()

    sys.path.insert(0, str(args.distar_root))
    from distar.agent.default.lib.actions import FUNC_ID_TO_ACTION_TYPE_DICT
    from distar.agent.default.lib.current_patch_actions import ENABLED_LEGACY_ACTION_IDS
    from distar.pysc2.lib import actions

    rows = json.loads(args.replay_manifest.read_text())["rows"]
    replay_paths = {Path(item["path"]).name: Path(item["path"]).resolve() for item in rows}
    races: dict[str, dict[int, str]] = {}
    for name, replay_path in replay_paths.items():
        replay = sc2reader.load_replay(str(replay_path), load_level=2)
        races[name] = {pid: value.get("Race", "") for pid, value in replay.attributes.items() if pid in (1, 2)}
        winners = {player.pid for player in replay.players if player.result == "Win"}
        races[name]["winners"] = winners

    def supported(ability_id: int, target_kind: str) -> bool:
        candidates = actions.RAW_ABILITY_IDS.get(ability_id)
        if not candidates:
            return False
        general = next(iter(candidates)).general_id
        candidates = actions.RAW_ABILITY_IDS.get(general or ability_id, ())
        required = {
            "unit": actions.raw_cmd_unit,
            "point": actions.raw_cmd_pt,
            "quick": actions.raw_cmd,
        }[target_kind]
        return any(
            item.function_type is required
            and FUNC_ID_TO_ACTION_TYPE_DICT.get(item.id) in ENABLED_LEGACY_ACTION_IDS
            for item in candidates
        )

    trajectories = 0
    selected_trajectories: set[tuple[Path, int]] = set()
    commands = Counter()
    rejected = Counter()
    for cache in sorted(args.cache_dir.glob("*/player_*.compact.jsonl.gz")):
        with gzip.open(cache, "rt", encoding="utf-8") as stream:
            header = json.loads(next(stream))
            replay_name, player = header["replay"], int(header["player"])
            if races.get(replay_name, {}).get(player) != "Zerg":
                continue
            if args.winner_only and player not in races[replay_name]["winners"]:
                continue
            trajectories += 1
            selected_trajectories.add((replay_paths[replay_name], player))
            for line in stream:
                row = json.loads(line)
                for encoded in row.get("actions", []):
                    action = sc_pb.Action()
                    action.ParseFromString(base64.b64decode(encoded))
                    if not (action.HasField("action_raw") and action.action_raw.HasField("unit_command")):
                        continue
                    command = action.action_raw.unit_command
                    target_kind = "unit" if command.HasField("target_unit_tag") else (
                        "point" if command.HasField("target_world_space_pos") else "quick"
                    )
                    commands["total"] += 1
                    if supported(command.ability_id, target_kind):
                        commands["supported"] += 1
                    else:
                        rejected[f"ability_{command.ability_id}_{target_kind}"] += 1

    if args.winner_only or args.explicit_players:
        # DI-star's stock manifest format means "decode both players".  The
        # tab-delimited PID makes winner-only ownership unambiguous.
        output_rows = [f"{path}\t{player - 1}" for path, player in sorted(selected_trajectories)]
    else:
        output_rows = [str(path) for path in sorted({path for path, _ in selected_trajectories})]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_rows) + "\n")
    report = {
        "patch": "5.0.16.97563",
        "race": "Zerg",
        "raw_replay_count": len({path for path, _ in selected_trajectories}),
        "zerg_trajectories": trajectories,
        "winner_only": args.winner_only,
        "explicit_players": args.winner_only or args.explicit_players,
        "commands": dict(commands),
        "rejected": dict(sorted(rejected.items())),
        "manifest": str(args.output.resolve()),
    }
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
