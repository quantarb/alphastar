"""Read the bounded raw-observation cache into the executable semantic policy.

Only commands with an explicit, complete mapping to the live decoder are
labels.  Everything else is counted and discarded; this is deliberately not
a best-effort ability vocabulary.
"""
from __future__ import annotations

import base64
import gzip
import json
from collections import Counter
from pathlib import Path

from sc2.ids.ability_id import AbilityId
from s2clientprotocol import sc2api_pb2 as sc_pb

from mac_sc2.contracts.semantic_action import PATCH
from mac_sc2.runtime.macro_decoder_config import RACE_CONFIG, RACE_IDS


def _label(ability: str, race: str):
    """Return a decoder-supported (actor, family, payload, target) tuple."""
    if ability == "ATTACK_ATTACK":
        return ("combat", "attack", "spell", "point")
    if ability in {"MOVE_MOVE", "PATROL_PATROL"}:
        return ("combat", "move", "utility", "point")
    if race == "protoss":
        if ability in {"NEXUSTRAIN_PROBE", "TRAIN_PROBE"}:
            return ("production", "train_morph", "worker", "none")
        if ability == "PROTOSSBUILD_PYLON":
            return ("worker", "build", "supply", "point")
        if ability == "PROTOSSBUILD_GATEWAY":
            return ("worker", "build", "production", "point")
        if ability == "PROTOSSBUILD_ASSIMILATOR":
            return ("worker", "build", "gas", "unit")
        if ability in {"PROTOSSBUILD_CYBERNETICSCORE", "PROTOSSBUILD_ROBOTICSFACILITY"}:
            return ("worker", "build", "tech", "point")
        if ability in {"WARPGATETRAIN_ZEALOT", "GATEWAYTRAIN_ZEALOT"}:
            return ("production", "train_morph", "basic_army", "none")
        if ability in {"WARPGATETRAIN_STALKER", "GATEWAYTRAIN_STALKER"}:
            return ("production", "train_morph", "ranged_army", "none")
        if ability in {"ROBOTICSFACILITYTRAIN_IMMORTAL"}:
            return ("production", "train_morph", "advanced_army", "none")
    return None


def _state(row: dict, race: str) -> list[float]:
    scalar = row["scalar"]
    config = RACE_CONFIG[race]
    own = [unit[1] for unit in row["units"] if unit[2] == 1]
    count = lambda kind: own.count(config[kind].value) if config.get(kind) else 0
    return [min(row["loop"] / (22.4 * 900), 1), min(scalar[0] / 1500, 1),
            min(scalar[1] / 1000, 1), min(scalar[3] / 200, 1),
            min(scalar[2] / 200, 1), min(max(scalar[2] - scalar[3], 0) / 30, 1),
            min(scalar[5] / 80, 1), min(count("supply") / 20, 1),
            min(count("prod") / 20, 1), min(count("gas") / 20, 1),
            min(count("tech") / 20, 1), min(count("basic") / 20, 1),
            min(count("ranged") / 20, 1), min(count("advanced") / 20, 1), 0, 0, 0]


def examples(path: str | Path, race: str, discarded: Counter):
    """Yield complete, patch-matching semantic labels from one compact cache."""
    race = race.lower()
    if race not in RACE_IDS:
        raise ValueError(f"unsupported race: {race}")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        header = json.loads(next(stream))
        if header.get("format") != "terr_entity_action_tick_v1" or header.get("game_version") != PATCH:
            raise ValueError("compact cache contract/version mismatch")
        for line in stream:
            row = json.loads(line)
            state = _state(row, race)
            for encoded in row["actions"]:
                action = sc_pb.Action(); action.ParseFromString(base64.b64decode(encoded))
                if not action.HasField("action_raw") or not action.action_raw.HasField("unit_command"):
                    discarded["non_unit_command"] += 1; continue
                try:
                    ability = AbilityId(action.action_raw.unit_command.ability_id).name
                except ValueError:
                    discarded["unknown_ability"] += 1; continue
                label = _label(ability, race)
                if label is None:
                    discarded["outside_live_contract"] += 1; continue
                yield state, RACE_IDS[race], label, int(action.action_raw.unit_command.queue_command)
