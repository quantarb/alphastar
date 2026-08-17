"""Concrete, fail-closed Protoss and Zerg rich-V2 command execution.

The policy selects only semantic intents and entity pointers.  This module owns
the versioned SC2 ability, actor role, target validation, and command emission
for the non-Terran V2 tasks.  It deliberately asks the live client for an
actor's available abilities immediately before issuing a command.
"""
from __future__ import annotations

from collections import Counter

from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from mac_sc2.contracts.race_rich_actions import intents_for


ABILITY_NAMES = {
    "Protoss": {
        "train_probe": "NEXUSTRAIN_PROBE", "build_pylon": "PROTOSSBUILD_PYLON",
        "build_assimilator": "PROTOSSBUILD_ASSIMILATOR", "build_gateway": "PROTOSSBUILD_GATEWAY",
        "build_cybernetics": "PROTOSSBUILD_CYBERNETICSCORE", "build_nexus": "PROTOSSBUILD_NEXUS",
        "train_zealot": "GATEWAYTRAIN_ZEALOT", "train_stalker": "GATEWAYTRAIN_STALKER",
        "research_warpgate": "RESEARCH_WARPGATE", "chronoboost": "EFFECT_CHRONOBOOSTENERGYCOST",
        "attack": "ATTACK_ATTACK", "retreat": "MOVE_MOVE", "scout": "MOVE_MOVE",
    },
    "Zerg": {
        "train_drone": "LARVATRAIN_DRONE", "train_overlord": "LARVATRAIN_OVERLORD",
        "build_extractor": "ZERGBUILD_EXTRACTOR", "build_spawning_pool": "ZERGBUILD_SPAWNINGPOOL",
        "build_roach_warren": "ZERGBUILD_ROACHWARREN", "build_hatchery": "ZERGBUILD_HATCHERY",
        "train_zergling": "LARVATRAIN_ZERGLING", "train_roach": "LARVATRAIN_ROACH",
        "inject_larva": "EFFECT_INJECTLARVA", "spread_creep": "BUILD_CREEPTUMOR_QUEEN",
        "attack": "ATTACK_ATTACK", "retreat": "MOVE_MOVE", "scout": "MOVE_MOVE",
        "morph_overseer": "MORPH_OVERSEER",
    },
}


def ability_for(race: str, intent: str) -> AbilityId:
    return AbilityId[ABILITY_NAMES[race][intent]]


def validate_race_live_contract(race: str) -> None:
    """Check coverage and enum resolution before a trainer can use a task."""
    if race not in ABILITY_NAMES:
        raise ValueError(f"no rich-V2 executor for {race}")
    declared = {intent.name for intent in intents_for(race)}
    mapped = set(ABILITY_NAMES[race])
    if declared != mapped:
        raise RuntimeError(f"{race} decoder coverage mismatch: contract={declared}, executor={mapped}")
    for name in mapped:
        ability_for(race, name)


class RaceRichExecutor(BotAI):
    """Reusable live legality guard and emitter for a non-Terran V2 task."""
    def __init__(self, race: str):
        super().__init__()
        validate_race_live_contract(race)
        self.race_name = race
        self.telemetry: Counter[str] = Counter()

    def actors(self, role: str):
        if role == "worker": return self.workers
        if role == "townhall": return self.townhalls
        if role == "gateway": return self.structures(UnitTypeId.GATEWAY) | self.structures(UnitTypeId.WARPGATE)
        if role == "cybernetics": return self.structures(UnitTypeId.CYBERNETICSCORE)
        if role == "larva": return self.units(UnitTypeId.LARVA)
        if role == "queen": return self.units(UnitTypeId.QUEEN)
        if role == "overlord": return self.units(UnitTypeId.OVERLORD)
        if role == "combat": return self.units.exclude_type({UnitTypeId.PROBE, UnitTypeId.DRONE, UnitTypeId.LARVA, UnitTypeId.OVERLORD})
        raise ValueError(f"unsupported actor role: {role}")

    def _default_target(self, name: str, actor):
        if name in ("build_assimilator", "build_extractor"):
            return self.vespene_geyser.closest_to(actor) if self.vespene_geyser else None
        if name == "chronoboost":
            targets = self.structures.ready.filter(lambda unit: unit.tag != actor.tag)
            return targets.first if targets else None
        if name == "inject_larva": return self.townhalls.closest_to(actor) if self.townhalls else None
        return None

    def _point_target(self, name: str):
        if name in ("attack", "scout") and self.enemy_start_locations:
            return self.enemy_start_locations[0]
        if name == "retreat": return self.start_location
        return self.start_location

    async def legal(self, intent_name: str, actor=None, target=None) -> tuple[object | None, object | None]:
        """Return a live-legal actor and target, or ``(None, None)``."""
        intent = next(item for item in intents_for(self.race_name) if item.name == intent_name)
        ability = ability_for(self.race_name, intent_name)
        candidates = [actor] if actor is not None else sorted(self.actors(intent.actor_role), key=lambda unit: unit.tag)
        candidates = [unit for unit in candidates if unit is not None]
        if not candidates or not self.can_afford(ability):
            return None, None
        available = await self.get_available_abilities(candidates)
        actor = next((unit for unit, abilities in zip(candidates, available) if ability in abilities), None)
        if actor is None:
            return None, None
        if intent.target_kind == "entity":
            target = target or self._default_target(intent_name, actor)
            if target is None:
                return None, None
        elif intent.target_kind == "point":
            target = target or self._point_target(intent_name)
            if intent_name.startswith("build_") and not await self.can_place_single(ability, target):
                target = await self.find_placement(ability, target, placement_step=2)
            if target is None:
                return None, None
        return actor, target

    async def issue(self, intent_name: str, actor=None, target=None, queued: bool = False) -> bool:
        """Emit only a command accepted by the current live ability query."""
        actor, target = await self.legal(intent_name, actor, target)
        if actor is None:
            self.telemetry[f"reject_{intent_name}"] += 1
            return False
        intent = next(item for item in intents_for(self.race_name) if item.name == intent_name)
        command = actor(ability_for(self.race_name, intent_name), target=target, queue=queued)
        try:
            accepted = self.do(command)
        except Exception:
            self.telemetry["reject_execution"] += 1
            return False
        if not accepted:
            self.telemetry["reject_command"] += 1
            return False
        self.telemetry[f"issued_{intent_name}"] += 1
        return True
