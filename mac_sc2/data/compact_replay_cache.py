"""Bounded action-tick cache for exact-build replay training.

The cache deliberately stores only policy inputs and labels, never full game
loops, score/debug protobuf fields, or rendered images.  It is an explicitly
requested, versioned cache; raw replays remain the source of truth.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import io
import json
import sys
from pathlib import Path

import mpyq
from absl import flags
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.terran_entity_ar import PATCH, contract_hash


def _game_version(path: Path) -> str:
    archive = mpyq.MPQArchive(io.BytesIO(path.read_bytes())).extract()
    return json.loads(archive[b"replay.gamemetadata.json"])["GameVersion"]


def _unit(unit) -> list[float | int]:
    """Fields consumed by the entity policy; tags are intentionally omitted."""
    order = unit.orders[0].ability_id if unit.orders else 0
    return [unit.tag, unit.unit_type, unit.alliance, round(unit.pos.x, 2), round(unit.pos.y, 2),
            round(unit.health, 2), round(unit.health_max, 2), round(unit.shield, 2),
            round(unit.energy, 2), round(unit.build_progress, 4), int(unit.is_selected),
            int(unit.is_flying), order]


def _scalar(observation) -> list[int]:
    common = observation.player_common
    return [common.minerals, common.vespene, common.food_cap, common.food_used,
            common.food_army, common.food_workers, common.idle_worker_count,
            common.army_count, common.warp_gate_count, common.larva_count]


def convert(replay_path: Path, player: int, output: Path, research_patch_family: bool = False) -> dict[str, object]:
    version = _game_version(replay_path)
    if version != PATCH and not (research_patch_family and version.startswith("5.0.16.")):
        raise ValueError(f"expected exact {PATCH} replay (or explicit 5.0.16 research cache): {replay_path}")
    data = replay_path.read_bytes()
    request = sc_pb.RequestStartReplay(
        replay_data=data,
        observed_player_id=player,
        disable_fog=False,
        options=sc_pb.InterfaceOptions(raw=True, raw_affects_selection=True,
                                        raw_crop_to_playable_area=True),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # Never leave a syntactically plausible but truncated cache after a missing
    # map, protocol failure, or interrupted replay.  The completed artifact is
    # atomically published only after the player trajectory reaches its result.
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.unlink(missing_ok=True)
    records = 0
    completed = False
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as stream:
        stream.write(json.dumps({
            "format": "terr_entity_action_tick_v1",
            "game_version": version,
            "live_eligible": version == PATCH,
            "action_contract_hash": contract_hash(),
            "entity_snapshot_hash": snapshot_hash(),
            "replay": replay_path.name,
            "player": player,
            "skill_conditioning": "player_mmr_div_7000_v1",
            "player_mmr_record": "first_metadata_record",
            "unit_fields": ["tag", "unit_type", "alliance", "x", "y", "health", "health_max", "shield", "energy", "build_progress", "selected", "flying", "first_order_ability"],
            "scalar_fields": ["minerals", "vespene", "food_cap", "food_used", "food_army", "food_workers", "idle_worker_count", "army_count", "warp_gate_count", "larva_count"],
        }, separators=(",", ":")) + "\n")
        config = run_configs.get()
        with config.start() as controller:
            info = controller.replay_info(data)
            player_mmr = int(info.player_info[player - 1].player_mmr) if len(info.player_info) >= player else 0
            stream.write(json.dumps({"metadata": {"player_mmr": player_mmr}}, separators=(",", ":")) + "\n")
            if info.local_map_path:
                request.map_data = config.map_data(info.local_map_path)
            controller.start_replay(request)
            controller.step(1)
            while True:
                response = controller.observe()
                if response.actions:
                    observation = response.observation
                    stream.write(json.dumps({
                        "loop": observation.game_loop,
                        "scalar": _scalar(observation),
                        "upgrades": list(observation.raw_data.player.upgrade_ids),
                        "units": [_unit(unit) for unit in observation.raw_data.units],
                        "actions": [base64.b64encode(action.SerializeToString()).decode("ascii") for action in response.actions],
                    }, separators=(",", ":")) + "\n")
                    records += 1
                if response.player_result:
                    completed = True
                    break
                controller.step(1)
    if not completed:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"replay did not reach a player result: {replay_path}")
    temporary.replace(output)
    return {"output": str(output.resolve()), "game_version": version, "live_eligible": version == PATCH, "action_ticks": records,
            "compressed_bytes": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research-patch-family", action="store_true",
                        help="allow 5.0.16.* cache extraction; these examples cannot train the live policy")
    args = parser.parse_args()
    flags.FLAGS([sys.argv[0]])
    print(json.dumps(convert(args.replay, args.player, args.output, args.research_patch_family), indent=2))


if __name__ == "__main__":
    main()
