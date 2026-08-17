"""Compile compact raw actions into complete, executable rich-V2 labels."""
from __future__ import annotations

import base64
import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

from sc2.ids.ability_id import AbilityId
from s2clientprotocol import sc2api_pb2 as sc_pb

from mac_sc2.contracts.race_rich_actions import intents_for
from mac_sc2.contracts.terran_entity_ar import PATCH, REGIONS

ABILITY_TO_INTENT = {
    "Terran": {"COMMANDCENTERTRAIN_SCV": "train_scv", "TERRANBUILD_SUPPLYDEPOT": "build_supply",
        "TERRANBUILD_REFINERY": "build_refinery", "TERRANBUILD_BARRACKS": "build_barracks",
        "TERRANBUILD_FACTORY": "build_factory", "TERRANBUILD_COMMANDCENTER": "build_command_center",
        "BARRACKSTRAIN_MARINE": "train_marine", "TRAIN_HELLION": "train_hellion",
        "UPGRADETOORBITAL_ORBITALCOMMAND": "morph_orbital", "CALLDOWNMULE_CALLDOWNMULE": "call_mule",
        "ATTACK_ATTACK": "attack", "MOVE_MOVE": "retreat"},
    "Protoss": {"NEXUSTRAIN_PROBE": "train_probe", "PROTOSSBUILD_PYLON": "build_pylon",
        "PROTOSSBUILD_ASSIMILATOR": "build_assimilator", "PROTOSSBUILD_GATEWAY": "build_gateway",
        "PROTOSSBUILD_CYBERNETICSCORE": "build_cybernetics", "PROTOSSBUILD_NEXUS": "build_nexus",
        "GATEWAYTRAIN_ZEALOT": "train_zealot", "WARPGATETRAIN_ZEALOT": "train_zealot",
        "GATEWAYTRAIN_STALKER": "train_stalker", "WARPGATETRAIN_STALKER": "train_stalker",
        "RESEARCH_WARPGATE": "research_warpgate", "EFFECT_CHRONOBOOSTENERGYCOST": "chronoboost",
        "ATTACK_ATTACK": "attack", "MOVE_MOVE": "retreat"},
    "Zerg": {"LARVATRAIN_DRONE": "train_drone", "LARVATRAIN_OVERLORD": "train_overlord",
        "ZERGBUILD_EXTRACTOR": "build_extractor", "ZERGBUILD_SPAWNINGPOOL": "build_spawning_pool",
        "ZERGBUILD_ROACHWARREN": "build_roach_warren", "ZERGBUILD_HATCHERY": "build_hatchery",
        "LARVATRAIN_ZERGLING": "train_zergling", "LARVATRAIN_ROACH": "train_roach",
        "EFFECT_INJECTLARVA": "inject_larva", "BUILD_CREEPTUMOR_QUEEN": "spread_creep",
        "ATTACK_ATTACK": "attack", "MOVE_MOVE": "retreat", "MORPH_OVERSEER": "morph_overseer"},
}


def _region(command) -> int:
    if not command.HasField("target_world_space_pos"):
        return 0
    point = command.target_world_space_pos
    # A deterministic coarse target label; live policy refines it through map tokens.
    return 2 if point.x + point.y > 128 else 0


def examples(path: str | Path, race: str, discarded: Counter):
    """Yield complete labels; unsupported/non-alignable commands are counted."""
    wanted = {item.name: index for index, item in enumerate(intents_for(race))}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if not header.get("game_version", "").startswith("5.0.16.") or header.get("format") != "terr_entity_action_tick_v1":
            raise ValueError("not a supported 5.0.16 compact cache")
        mmr = 0
        for line in stream:
            row = json.loads(line)
            if "metadata" in row:
                mmr = int(row["metadata"].get("player_mmr", 0)); continue
            # Mirror the live encoder: friendly and visible enemy units first;
            # neutral minerals/geysers are not pointer candidates and must not
            # consume the fixed entity budget.
            entities = [unit for unit in row["units"] if int(unit[2]) in (1, 4)]
            entities = sorted(entities, key=lambda unit: (int(unit[2]) != 1, int(unit[0])))[:96]
            tags = {int(unit[0]): index for index, unit in enumerate(entities)}
            for encoded in row.get("actions", []):
                action = sc_pb.Action(); action.ParseFromString(base64.b64decode(encoded))
                if not action.HasField("action_raw") or not action.action_raw.HasField("unit_command"):
                    discarded["non_unit_command"] += 1; continue
                command = action.action_raw.unit_command
                try: ability = AbilityId(command.ability_id).name
                except ValueError: discarded["unknown_ability"] += 1; continue
                name = ABILITY_TO_INTENT[race].get(ability)
                if name not in wanted:
                    discarded["outside_live_contract"] += 1; continue
                if not command.unit_tags or int(command.unit_tags[0]) not in tags:
                    discarded["actor_not_in_96_slots"] += 1; continue
                if int(entities[tags[int(command.unit_tags[0])]][2]) != 1:
                    discarded["actor_not_owned_by_observed_player"] += 1; continue
                target = int(command.target_unit_tag) if command.HasField("target_unit_tag") else int(command.unit_tags[0])
                if target not in tags:
                    discarded["target_not_in_96_slots"] += 1; continue
                yield {"scalar": row["scalar"], "entities": entities, "mmr": mmr,
                       "source_patch": header["game_version"], "live_eligible": header["game_version"] == PATCH,
                       "intent": wanted[name], "actor": tags[int(command.unit_tags[0])], "target": tags[target],
                       "region": _region(command), "queued": int(command.queue_command)}


def _header(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.loads(next(stream))


def _race_for_cache(path: Path, replay_paths: dict[str, Path]) -> str:
    """Read a player's declared race from the corresponding source replay."""
    import sc2reader
    header = _header(path)
    replay = replay_paths[header["replay"]]
    parsed = sc2reader.load_replay(str(replay), load_level=1)
    return parsed.attributes[int(header["player"])]["Race"]


def compile_caches(cache_dir: str | Path, manifest: str | Path, output: str | Path,
                   additional_manifests: tuple[str | Path, ...] = ()) -> dict:
    """Compile completed atomic caches into per-race, pointer-validated labels."""
    cache_dir, output = Path(cache_dir), Path(output)
    replay_paths = {}
    for manifest_path in (Path(manifest), *(Path(item) for item in additional_manifests)):
        source = json.loads(manifest_path.read_text())
        manifest_rows = source.get("rows", source.get("valid"))
        if manifest_rows is None:
            raise ValueError(f"manifest needs a 'rows' or 'valid' list: {manifest_path}")
        replay_paths.update({Path(item["path"]).name: Path(item["path"]) for item in manifest_rows})
    rows_by_race = {race: [] for race in ("Terran", "Protoss", "Zerg")}
    discarded: Counter[str] = Counter(); files = 0
    for cache in sorted(cache_dir.glob("*/player_*.compact.jsonl.gz")):
        header = _header(cache)
        if header["replay"] not in replay_paths:
            discarded["replay_missing_from_manifest"] += 1
            continue
        race = _race_for_cache(cache, replay_paths)
        if race not in rows_by_race:
            discarded["unsupported_race"] += 1
            continue
        files += 1
        for row in examples(cache, race, discarded):
            if not (len(row["scalar"]) == 10 and len(row["entities"]) <= 96 and
                    0 <= row["actor"] < len(row["entities"]) and 0 <= row["target"] < len(row["entities"])):
                discarded["invalid_compiled_pointer"] += 1
                continue
            rows_by_race[race].append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format": "rich_v2_compiled_labels_v1", "live_eligible": False, "files": files, "rows": rows_by_race,
               "discarded": dict(discarded)}
    output.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"output": str(output.resolve()), "files": files,
            "labels": {race: len(rows) for race, rows in rows_by_race.items()},
            "discarded": dict(discarded)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile completed rich-V2 compact caches")
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--additional-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compile_caches(args.cache_dir, args.manifest, args.output,
                                   tuple(args.additional_manifest)), indent=2))


if __name__ == "__main__":
    main()
