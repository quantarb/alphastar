"""Live 4.9.2 decoder for the primary patch/race/micro policy."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

import torch

os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId

from mac_sc2.architectures.multitask_policy import PlayableMultiTaskPolicy
from mac_sc2.contracts.multitask import task_routes, validate_checkpoint as validate_multitask
from mac_sc2.contracts.patch_race_mtl import build_specs, task_key
from mac_sc2.contracts.semantic_schema import FAMILIES
from mac_sc2.runtime.entity_snapshot import encode
from mac_sc2.runtime.macro_decoder_config import RACE_CONFIG
from mac_sc2.runtime.placement_candidates import candidates

PATCH = "4.9.2"
HISTORY_SIZE = 16


class PatchRaceBot(BotAI):
    """Decoder that fail-closes before issuing any non-live-valid command."""

    def __init__(self, checkpoint: str, registry: str, race: str, smoke_steps: int | None = None,
                 decision_head: str = "micro"):
        super().__init__()
        self.race_name = race.title()
        self.smoke_steps = smoke_steps
        if decision_head not in ("micro", "macro"):
            raise ValueError(f"unknown decision head: {decision_head}")
        self.decision_head = decision_head
        self.task = task_key(PATCH, self.race_name)
        self.specs = build_specs(registry)
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate_multitask(data, registry)
        self.routes = task_routes(registry)
        self.model = PlayableMultiTaskPolicy(self.specs, self.routes)
        # Unified checkpoints may retain historical auxiliary heads. They are
        # intentionally absent from the live model and cannot affect decoding.
        live_state = {key: value for key, value in data["state_dict"].items()
                      if not key.startswith("historical_")}
        self.model.load_state_dict(live_state)
        self.model.eval()
        self.c = RACE_CONFIG[race.lower()]
        self.history: list[int] = []
        self.telemetry: Counter[str] = Counter()

    def feat(self) -> torch.Tensor:
        config = self.c
        structures = lambda unit_type: self.structures(unit_type).amount if unit_type else 0
        units = lambda unit_type: self.units.of_type({unit_type}).amount
        return torch.tensor([[min(self.time / 900, 1), min(self.minerals / 1500, 1), min(self.vespene / 1000, 1),
                              min(self.supply_used / 200, 1), min(self.supply_cap / 200, 1), min(max(self.supply_left, 0) / 30, 1),
                              min(self.workers.amount / 80, 1), min(structures(config["supply"]) / 20, 1),
                              min(structures(config["prod"]) / 20, 1), min(structures(config["gas"]) / 20, 1),
                              min(structures(config["tech"]) / 20, 1), min(units(config["basic"]) / 20, 1),
                              min(units(config["ranged"]) / 20, 1), min(units(config["advanced"]) / 20, 1), 0, 0, 0]], dtype=torch.float32)

    def actors(self, role: str):
        if role == "worker":
            return self.workers
        if role == "production":
            return self.structures.ready
        if role == "combat":
            return self.units.exclude_type({self.c["worker"]})
        return self.units | self.structures

    def history_tensor(self) -> torch.Tensor:
        values = [0] * (HISTORY_SIZE - min(HISTORY_SIZE, len(self.history)))
        values += [tuple_id + 1 for tuple_id in self.history[-HISTORY_SIZE:]]
        return torch.tensor([values], dtype=torch.long)

    async def legal_actor(self, actors, ability: AbilityId, actor_scores=None):
        ranked = sorted(actors, key=lambda unit: unit.tag)
        if actor_scores is not None:
            by_tag = {unit.tag: unit for unit in actors}
            ranked = [unit for index in actor_scores.argsort(descending=True).tolist()
                      if index < len(self._owned) and (unit := self._owned[index]).tag in by_tag]
        available = await self.get_available_abilities(ranked)
        return next((unit for unit, abilities in zip(ranked, available) if ability in abilities), None)

    async def legal_tuple_ids(self) -> list[int]:
        """Mask tuple logits with actual per-unit 4.9.2 availability."""
        available_by_role: dict[str, set[AbilityId]] = {}
        for role in ("worker", "combat", "production", "transport"):
            actors = self.actors(role)
            if actors:
                available = await self.get_available_abilities(actors)
                available_by_role[role] = {ability for abilities in available for ability in abilities}
        legal = []
        for index, row in enumerate(self.specs[self.task]):
            if AbilityId(row["ability"]) not in available_by_role.get(row["actor"], set()):
                continue
            if row["target_mode"] == "point" and not self.townhalls:
                continue
            if row["target_mode"] == "unit":
                if row["family"] == "gather" and not (self.mineral_field or self.vespene_geyser):
                    continue
                if row["family"] == "repair" and not (self.units or self.structures):
                    continue
                if row["family"] == "attack" and not (self.enemy_units or self.enemy_structures):
                    continue
            legal.append(index)
        self.telemetry["legal_tuples"] += len(legal)
        return legal

    async def issue(self, row: dict) -> bool:
        actors = self.actors(row["actor"])
        if not actors:
            self.telemetry["reject_no_actor"] += 1
            return False
        ability = AbilityId(row["ability"])
        entities, padding, self._owned = encode(self)
        actor = await self.legal_actor(actors, ability)
        if actor is None:
            self.telemetry["reject_ability"] += 1
            return False

        target = None
        if row["target_mode"] == "point":
            if row["family"] == "build" or row["replay_ability"].lower().startswith("land"):
                valid = await candidates(self, ability.value, self.townhalls.first.position) if self.townhalls else []
                if not valid:
                    self.telemetry["reject_no_placement"] += 1
                    return False
                home = self.townhalls.first.position
                coordinates = torch.tensor([((point.x - home.x) / 64, (point.y - home.y) / 64) for point in valid])
                with torch.no_grad():
                    scores = self.model.build_placement_scores(entities[None], padding[None], coordinates[None], PATCH, self.race_name)[0]
                target = valid[int(scores.argmax())]
            elif row["actor"] == "combat" and self.enemy_start_locations:
                target = next(iter(self.enemy_start_locations))
            elif self.townhalls:
                target = self.townhalls.first.position
            else:
                self.telemetry["reject_no_point"] += 1
                return False
        elif row["target_mode"] == "unit":
            if row["family"] == "gather":
                targets = self.mineral_field | self.vespene_geyser
            elif row["family"] == "repair":
                targets = self.units | self.structures
            elif row["family"] == "attack":
                targets = self.enemy_units | self.enemy_structures
            else:
                self.telemetry["reject_undecoded_unit_target"] += 1
                return False
            if not targets:
                self.telemetry["reject_no_target"] += 1
                return False
            target = targets.closest_to(actor)
            if target is None:
                self.telemetry["reject_no_target"] += 1
                return False
        try:
            if not self.do(actor(ability, target=target, queue=row["queue"])):
                self.telemetry["reject_command"] += 1
                return False
            self.telemetry[f"issued_{row['family']}"] += 1
            return True
        except Exception:
            self.telemetry["reject_execution"] += 1
            return False

    async def on_step(self, iteration: int):
        if iteration % 16 or not self.townhalls:
            return
        self.telemetry["decisions"] += 1
        legal = await self.legal_tuple_ids()
        if not legal:
            self.telemetry["reject_no_legal_tuple"] += 1
            return
        with torch.no_grad():
            logits = self.model.micro_logits(self.feat(), PATCH, self.race_name, self.history_tensor())[0]
            if self.decision_head == "macro":
                family_id = int(self.model.macro_logits(self.feat(), PATCH, self.race_name)[0].argmax())
                family = FAMILIES[family_id]
                routed = [index for index in legal if self.specs[self.task][index]["family"] == family]
                if routed:
                    legal = routed
                    self.telemetry[f"macro_family_{family}"] += 1
                else:
                    self.telemetry["macro_no_legal_family"] += 1
        for index in sorted(legal, key=lambda item: float(logits[item]), reverse=True):
            if await self.issue(self.specs[self.task][index]):
                self.history.append(index)
                print(f"t={self.time:.0f} tuple={index}", flush=True)
                break
        if self.smoke_steps is not None and iteration >= self.smoke_steps:
            await self.client.leave()

    async def on_end(self, result):
        if getattr(self, "result_path", None):
            Path(self.result_path).write_text(json.dumps({"result": str(result), "telemetry": dict(self.telemetry)}))
        print(result, flush=True)
