"""Live 5.0.16.97563 executor for the Terran entity autoregressive contract."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

from mac_sc2.architectures.terran_entity_ar import TerranEntityARPolicy
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.terran_entity_ar import INTENTS, PATCH, REGIONS, contract_hash, intent_id
from mac_sc2.runtime.entity_snapshot import encode

DECODER_INTENTS = frozenset({"train_scv", "build_supply", "build_refinery", "build_barracks", "build_factory",
                             "build_command_center", "train_marine", "train_hellion", "morph_orbital", "call_mule",
                             "attack", "retreat", "scout", "repair"})


def validate_live_contract() -> None:
    """Fail before training if a permitted output lacks decoding or legality."""
    contract_names = {intent.name for intent in INTENTS}
    if contract_names != DECODER_INTENTS:
        raise RuntimeError(f"decoder coverage mismatch: contract={contract_names} decoder={DECODER_INTENTS}")


class TerranEntityARBot(BotAI):
    """Fail-closed live runner.  ``checkpoint=None`` uses only the executor.

    The deterministic mode is deliberately provided for validation: it proves
    every action family can become a real SC2 action before any new model is
    trained.  A learned checkpoint must carry the exact contracts below.
    """
    def __init__(self, checkpoint: str | None = None, smoke_steps: int | None = None):
        super().__init__()
        self.smoke_steps = smoke_steps
        self.history: list[int] = []
        self.telemetry: Counter[str] = Counter()
        self.model: TerranEntityARPolicy | None = None
        if checkpoint:
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if data.get("action_contract_hash") != contract_hash() or data.get("entity_snapshot_hash") != snapshot_hash():
                raise RuntimeError("Terran entity checkpoint contract mismatch")
            self.model = TerranEntityARPolicy(**data.get("architecture", {}))
            self.model.load_state_dict(data["state_dict"]); self.model.eval()

    def feature_tensor(self) -> torch.Tensor:
        return torch.tensor([[min(self.time / 900, 1), min(self.minerals / 1500, 1), min(self.vespene / 1000, 1),
            min(self.supply_used / 200, 1), min(self.supply_cap / 200, 1), min(max(self.supply_left, 0) / 30, 1),
            min(self.workers.amount / 80, 1), min(self.structures(UnitTypeId.SUPPLYDEPOT).amount / 20, 1),
            min(self.structures(UnitTypeId.BARRACKS).amount / 20, 1), min(self.structures(UnitTypeId.REFINERY).amount / 20, 1),
            min(self.structures(UnitTypeId.FACTORY).amount / 20, 1), min(self.units(UnitTypeId.MARINE).amount / 40, 1),
            min(self.units(UnitTypeId.HELLION).amount / 20, 1), min(self.army_count / 100, 1),
            min(self.enemy_units.amount / 50, 1), min(self.enemy_structures.amount / 20, 1), 0]], dtype=torch.float32)

    def _region(self, region: str):
        if region in ("enemy_start", "enemy_army") and self.enemy_units:
            return self.enemy_units.closest_to(self.start_location).position
        if region == "enemy_start" and self.enemy_start_locations:
            return self.enemy_start_locations[0]
        if region == "retreat": return self.start_location
        if region == "natural": return self.expansion_locations_list[1] if len(self.expansion_locations_list) > 1 else self.start_location
        return self.start_location

    def _rule_intent(self) -> int:
        if self.townhalls.idle and self.can_afford(UnitTypeId.SCV) and self.workers.amount < 65: return intent_id("train_scv")
        if self.supply_left < 5 and self.can_afford(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(UnitTypeId.SUPPLYDEPOT): return intent_id("build_supply")
        if self.structures(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 3 and self.can_afford(UnitTypeId.BARRACKS): return intent_id("build_barracks")
        if self.structures(UnitTypeId.REFINERY).amount < 2 and self.structures(UnitTypeId.BARRACKS) and self.can_afford(UnitTypeId.REFINERY): return intent_id("build_refinery")
        if self.structures(UnitTypeId.FACTORY).amount < 1 and self.structures(UnitTypeId.BARRACKS).ready and self.can_afford(UnitTypeId.FACTORY): return intent_id("build_factory")
        if self.townhalls.amount < 2 and self.can_afford(UnitTypeId.COMMANDCENTER): return intent_id("build_command_center")
        if self.structures(UnitTypeId.BARRACKS).idle and self.can_afford(UnitTypeId.MARINE) and self.supply_left: return intent_id("train_marine")
        if self.structures(UnitTypeId.FACTORY).idle and self.can_afford(UnitTypeId.HELLION) and self.supply_left: return intent_id("train_hellion")
        if self.army_count >= 12: return intent_id("attack")
        return intent_id("scout")

    def _legal(self, name: str) -> bool:
        """Cheap legality mask before pointer decoding and command emission."""
        if name == "train_scv": return bool(self.townhalls.idle and self.can_afford(UnitTypeId.SCV) and self.supply_left)
        if name == "build_supply": return bool(self.workers and self.supply_left < 5 and self.can_afford(UnitTypeId.SUPPLYDEPOT) and not self.already_pending(UnitTypeId.SUPPLYDEPOT))
        if name == "build_refinery": return bool(self.workers and self.townhalls and self.structures(UnitTypeId.REFINERY).amount + self.already_pending(UnitTypeId.REFINERY) < 2 and self.can_afford(UnitTypeId.REFINERY) and self._target(name))
        if name == "build_barracks": return bool(self.workers and self.structures(UnitTypeId.BARRACKS).amount + self.already_pending(UnitTypeId.BARRACKS) < 3 and self.can_afford(UnitTypeId.BARRACKS))
        if name == "build_factory": return bool(self.workers and self.structures(UnitTypeId.BARRACKS).ready and self.structures(UnitTypeId.FACTORY).amount + self.already_pending(UnitTypeId.FACTORY) < 1 and self.can_afford(UnitTypeId.FACTORY))
        if name == "build_command_center": return bool(self.workers and self.townhalls.amount + self.already_pending(UnitTypeId.COMMANDCENTER) < 2 and self.can_afford(UnitTypeId.COMMANDCENTER))
        if name == "train_marine": return bool(self.structures(UnitTypeId.BARRACKS).idle and self.can_afford(UnitTypeId.MARINE) and self.supply_left)
        if name == "train_hellion": return bool(self.structures(UnitTypeId.FACTORY).idle and self.can_afford(UnitTypeId.HELLION) and self.supply_left)
        if name == "morph_orbital": return bool(self.townhalls(UnitTypeId.COMMANDCENTER).idle and self.can_afford(UnitTypeId.ORBITALCOMMAND))
        if name == "call_mule": return bool(self.townhalls(UnitTypeId.ORBITALCOMMAND).filter(lambda unit: unit.energy >= 50) and self._target(name))
        if name in ("attack", "retreat", "scout"): return bool(self.units.exclude_type({UnitTypeId.SCV}))
        if name == "repair": return bool(self.workers and self._target(name))
        return False

    def _actor(self, name: str):
        role = INTENTS[intent_id(name)].actor_role
        choices = {"worker": self.workers.gathering or self.workers, "townhall": self.townhalls.idle,
                   "barracks": self.structures(UnitTypeId.BARRACKS).idle, "factory": self.structures(UnitTypeId.FACTORY).idle,
                   "combat": self.units.exclude_type({UnitTypeId.SCV})}
        group = choices[role]
        return group.first if group else None

    def _ranked_actor(self, name: str, owned, scores: torch.Tensor):
        """Use the pointer head only among entities valid for this intent."""
        role = INTENTS[intent_id(name)].actor_role
        groups = {"worker": self.workers.gathering or self.workers, "townhall": self.townhalls.idle,
                  "barracks": self.structures(UnitTypeId.BARRACKS).idle, "factory": self.structures(UnitTypeId.FACTORY).idle,
                  "combat": self.units.exclude_type({UnitTypeId.SCV})}
        allowed = {unit.tag: unit for unit in groups[role]}
        for index in scores.argsort(descending=True).tolist():
            if index < len(owned) and owned[index].tag in allowed:
                return allowed[owned[index].tag]
        return self._actor(name)

    def _target(self, name: str):
        if name in ("build_refinery", "call_mule"):
            return (self.vespene_geyser.closer_than(12, self.townhalls.first).first
                    if name == "build_refinery" and self.vespene_geyser else
                    (self.mineral_field.closer_than(12, self.townhalls.first).first if self.mineral_field else None))
        if name == "repair":
            damaged = (self.units | self.structures).filter(lambda unit: unit.health < unit.health_max)
            return damaged.first if damaged else None
        return None

    def _ranked_target(self, name: str, owned, scores: torch.Tensor):
        if name == "repair":
            allowed = {unit.tag: unit for unit in (self.units | self.structures)
                       if unit.health < unit.health_max}
        elif name == "attack":
            allowed = {unit.tag: unit for unit in (self.enemy_units | self.enemy_structures)}
        else:
            return self._target(name)
        for index in scores.argsort(descending=True).tolist():
            if index < len(owned) and owned[index].tag in allowed:
                return allowed[owned[index].tag]
        return self._target(name)

    async def _issue(self, name: str, actor, target=None) -> bool:
        if actor is None: return False
        try:
            if name == "train_scv": actor.train(UnitTypeId.SCV)
            elif name == "build_supply": await self.build(UnitTypeId.SUPPLYDEPOT, near=self.start_location, placement_step=3)
            elif name == "build_barracks": await self.build(UnitTypeId.BARRACKS, near=self.start_location, placement_step=3)
            elif name == "build_factory": await self.build(UnitTypeId.FACTORY, near=self.start_location, placement_step=3)
            elif name == "build_command_center": await self.expand_now()
            elif name == "build_refinery":
                if target is None: return False
                actor.build_gas(target)
            elif name == "train_marine": actor.train(UnitTypeId.MARINE)
            elif name == "train_hellion": actor.train(UnitTypeId.HELLION)
            elif name == "morph_orbital": actor(AbilityId.UPGRADETOORBITAL_ORBITALCOMMAND)
            elif name == "call_mule":
                if target is None: return False
                actor(AbilityId.CALLDOWNMULE_CALLDOWNMULE, target)
            elif name == "attack":
                destination = target or self._region("enemy_start")
                for unit in self.units.exclude_type({UnitTypeId.SCV}): unit.attack(destination)
            elif name in ("retreat", "scout"):
                actor.move(self._region("retreat" if name == "retreat" else "enemy_start"))
            elif name == "repair":
                if target is None: return False
                actor.repair(target)
            else: return False
            self.telemetry[f"issued_{name}"] += 1
            return True
        except Exception:
            self.telemetry["reject_execution"] += 1
            return False

    async def on_step(self, iteration: int) -> None:
        if iteration % 16 or not self.townhalls: return
        entities, padding, owned = encode(self)
        actions = [self._rule_intent()]
        output = None
        if self.model:
            history = torch.tensor([[0] * (16 - min(16, len(self.history))) + [value + 1 for value in self.history[-16:]]])
            with torch.no_grad(): output = self.model(self.feature_tensor(), entities[None], padding[None], history)
            actions = output.intent[0].argsort(descending=True).tolist()
        for action in actions:
            name = INTENTS[action].name
            if not self._legal(name):
                self.telemetry["masked_illegal"] += 1
                continue
            actor = self._ranked_actor(name, owned, output.actor[0]) if output else self._actor(name)
            target = self._ranked_target(name, owned, output.target[0]) if output else self._target(name)
            if await self._issue(name, actor, target):
                self.history.append(action); self.telemetry["decisions"] += 1
                break
        if self.smoke_steps is not None and iteration >= self.smoke_steps: await self.client.leave()

    async def on_end(self, result):
        if getattr(self, "result_path", None): Path(self.result_path).write_text(json.dumps({"result": str(result), "telemetry": dict(self.telemetry)}))
        print(result, flush=True)
