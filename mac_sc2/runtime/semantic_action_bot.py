"""Live 4.9.2 decoder for the all-race semantic-action MTL checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from sc2.bot_ai import BotAI
from sc2.ids.upgrade_id import UpgradeId
from sc2.position import Point2
from sc2.unit import UnitOrder

from mac_sc2.architectures.semantic_action_mtl import SemanticActionMTL
from mac_sc2.contracts.semantic_action import spec_hash, supports
from mac_sc2.contracts.semantic_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES
from mac_sc2.runtime.macro_decoder_config import RACE_CONFIG, RACE_IDS


class _UnknownObservedAbility:
    """Observation-only placeholder for an ability newer than python-sc2's enum."""
    def __init__(self, ability_id: int): self.ability_id = ability_id
    def __repr__(self) -> str: return f"UnknownObservedAbility({self.ability_id})"
    @property
    def id(self) -> int: return self.ability_id
    @property
    def exact_id(self) -> int: return self.ability_id


_unit_order_from_proto = UnitOrder.from_proto.__func__


def _safe_unit_order_from_proto(cls, proto, bot_object):
    """Do not crash on a new *observed* order that this wrapper cannot name.

    This path does not authorize or emit that ability: decoded policy commands
    still come solely from the explicit action contract below.
    """
    try:
        return _unit_order_from_proto(cls, proto, bot_object)
    except KeyError:
        target = Point2.from_proto(proto.target_world_space_pos) if proto.HasField("target_world_space_pos") else (
            proto.target_unit_tag if proto.HasField("target_unit_tag") else None)
        return cls(_UnknownObservedAbility(proto.ability_id), target, proto.progress)


UnitOrder.from_proto = classmethod(_safe_unit_order_from_proto)


class SemanticActionBot(BotAI):
    def __init__(self, checkpoint: str, race: str):
        super().__init__()
        self.race_name = race.lower(); self.race_id = RACE_IDS[self.race_name]; self.c = RACE_CONFIG[self.race_name]
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if data.get("action_contract_hash") != spec_hash():
            raise RuntimeError("semantic action contract mismatch")
        self.model = SemanticActionMTL(); self.model.load_state_dict(data["state_dict"]); self.model.eval()
        self.result_path: str | None = None

    def feat(self) -> torch.Tensor:
        c = self.c; amount = lambda kind: self.structures(kind).amount if kind else 0
        army = lambda kind: self.units.of_type({kind}).amount
        return torch.tensor([[min(self.time / 900, 1), min(self.minerals / 1500, 1), min(self.vespene / 1000, 1),
            min(self.supply_used / 200, 1), min(self.supply_cap / 200, 1), min(max(self.supply_left, 0) / 30, 1),
            min(self.workers.amount / 80, 1), min(amount(c["supply"]) / 20, 1), min(amount(c["prod"]) / 20, 1),
            min(amount(c["gas"]) / 20, 1), min(amount(c["tech"]) / 20, 1), min(army(c["basic"]) / 20, 1),
            min(army(c["ranged"]) / 20, 1), min(army(c["advanced"]) / 20, 1), 0, 0, 0]], dtype=torch.float32)

    async def research_ready(self) -> bool:
        upgrade = {"terran": UpgradeId.STIMPACK, "protoss": UpgradeId.WARPGATERESEARCH,
                   "zerg": UpgradeId.ZERGLINGMOVEMENTSPEED}[self.race_name]
        if self.already_pending_upgrade(upgrade): return False
        ability = self.game_data.upgrades[upgrade.value].research_ability.id
        structures = self.structures.ready.idle
        if not structures: return False
        return any(ability in abilities for abilities in await self.get_available_abilities(structures))

    async def on_step(self, iteration: int) -> None:
        if iteration % 16 or not self.townhalls:
            return
        with torch.no_grad(): out = self.model(self.feat(), torch.tensor([self.race_id]))
        ids = ({value: index for index, value in enumerate(ACTOR_ROLES)}, {value: index for index, value in enumerate(FAMILIES)},
               {value: index for index, value in enumerate(PAYLOAD_ROLES)})
        choices = []
        def add(actor, family, payload, code, legal):
            target = "unit" if code in ("gas", "rally") else ("point" if code in ("supply", "prod", "tech", "attack", "expand", "scout") else "none")
            if legal and supports(actor, family, payload, target):
                choices.append((float(out["actor"][0, ids[0][actor]] + out["family"][0, ids[1][family]] + out["payload"][0, ids[2][payload]]), code))
        c = self.c
        add("production", "train_morph", "worker", "worker", self.townhalls.idle and self.can_afford(c["worker"]) and self.workers.amount < 70)
        add("worker", "build", "supply", "supply", self.can_afford(c["supply"]) and self.supply_left <= 5 and not self.already_pending(c["supply"]))
        add("worker", "build", "production", "prod", self.can_afford(c["prod"]) and self.structures(c["prod"]).amount < 4)
        add("worker", "build", "gas", "gas", self.can_afford(c["gas"]) and self.structures(c["gas"]).amount < 2)
        add("worker", "build", "tech", "tech", self.can_afford(c["tech"]) and not self.structures(c["tech"]).amount)
        add("production", "train_morph", "basic_army", "basic", self.structures(c["prod"]).ready.idle and self.can_afford(c["basic"]) and self.supply_left > 0)
        add("production", "train_morph", "ranged_army", "ranged", self.structures(c["ranged_prod"]).ready.idle and self.can_afford(c["ranged"]) and self.supply_left > 0)
        army = self.units.of_type({c["basic"], c["ranged"], c["advanced"]})
        add("combat", "attack", "spell", "attack", army.amount >= 8)
        add("worker", "build", "townhall", "expand", self.can_afford(c["townhall"]) and self.townhalls.amount < 3)
        add("production", "research", "upgrade", "research", await self.research_ready())
        add("production", "train_morph", "advanced_army", "advanced", self.can_afford(c["advanced"]))
        add("combat", "move", "utility", "scout", bool(army) and self.enemy_start_locations)
        if not choices: return
        code = max(choices)[1]
        if code == "worker": self.townhalls.idle.first.train(c["worker"])
        elif code in ("supply", "prod", "tech"): await self.build({"supply": c["supply"], "prod": c["prod"], "tech": c["tech"]}[code], near=self.townhalls.first, placement_step=3)
        elif code == "gas":
            for geyser in self.vespene_geyser.closer_than(12, self.townhalls.first): self.workers.closest_to(geyser).build(c["gas"], geyser); break
        elif code in ("basic", "ranged"):
            unit = c["basic"] if code == "basic" else c["ranged"]; buildings = self.structures(c["prod"] if code == "basic" else c["ranged_prod"]).ready.idle
            if buildings: buildings.first.train(unit)
        elif code == "attack":
            for unit in army: unit.attack(self.enemy_start_locations[0])
        elif code == "expand": await self.expand_now()
        elif code == "advanced":
            building = c["advanced_build"]
            if building and not self.structures(building).ready:
                await self.build(building, near=self.townhalls.first, placement_step=3)
            else:
                producers = self.structures(building).ready.idle if building else self.townhalls.ready.idle
                if producers and self.supply_left > 0: producers.first.train(c["advanced"])
        elif code == "scout": army.closest_to(self.townhalls.first).move(self.enemy_start_locations[0])
        elif code == "research":
            upgrade = {"terran": UpgradeId.STIMPACK, "protoss": UpgradeId.WARPGATERESEARCH,
                       "zerg": UpgradeId.ZERGLINGMOVEMENTSPEED}[self.race_name]
            if not self.already_pending_upgrade(upgrade):
                for structure in self.structures.ready.idle:
                    if structure.research(upgrade): break
        print(f"t={self.time:.0f} semantic={code}", flush=True)

    async def on_end(self, result):
        if self.result_path: Path(self.result_path).write_text(json.dumps({"result": str(result)}))
        print(result, flush=True)
