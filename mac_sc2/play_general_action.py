#!/usr/bin/env python3
"""Live 4.9.2 runner for a checkpoint trained against ``ActionRegistry``."""
import argparse, json, os
from pathlib import Path
os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")

import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.ability_id import AbilityId

from general_action_checkpoint import validate
from general_action_decoder import execute
from general_action_policy import GeneralActionPolicy
from general_action_registry import ActionRegistry
from general_action_runtime import actor_candidates, point_from_normalized, target_candidates
from semantic_action_schema import ACTOR_ROLES, TARGET_KINDS


class GeneralActionBot(BotAI):
    def __init__(self, checkpoint, registry_path, race):
        super().__init__()
        self.registry = ActionRegistry(registry_path); self.race_name = race.title()
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        validate(data, self.registry.hash)
        self.model = GeneralActionPolicy(17, len(self.registry.abilities), len(self.registry.target_types), len(self.registry.tasks))
        self.model.load_state_dict(data["state_dict"]); self.model.eval()
        self.history = []; self.result_path = None

    async def on_start(self):
        self.game_data.abilities.setdefault(4135, self.game_data.abilities[1])
        self.prevent_double_actions = lambda action: True

    def features(self):
        workers = self.workers.amount; army = max(0, self.units.amount - workers)
        structures = self.structures.amount
        h = (self.history + [0] * 8)[-8:]
        values = [min(self.time / 900, 1), min(self.minerals / 1500, 1), min(self.vespene / 1000, 1),
                  min(self.supply_used / 200, 1), min(self.supply_cap / 200, 1), min(max(self.supply_left, 0) / 30, 1),
                  min(workers / 80, 1), min(army / 80, 1), min(structures / 30, 1)]
        values.extend(x / max(len(self.registry.abilities), 1) for x in h)
        return torch.tensor([values], dtype=torch.float32)

    async def on_step(self, iteration):
        if iteration % 16 or not self.townhalls:
            return
        task = torch.tensor([self.registry.task_id(self.race_name)])
        with torch.no_grad(): out = self.model(self.features(), task)
        aid = {x: i for i, x in enumerate(self.registry.abilities)}
        rid = {x: i for i, x in enumerate(ACTOR_ROLES)}
        kid = {x: i for i, x in enumerate(TARGET_KINDS)}
        target_types = ("",) + self.registry.target_types
        ranked = []
        for action in self.registry.candidates(self.race_name):
            target_index = target_types.index(action.target_type) if action.target_type else 0
            score = (out["actor_role"][0, rid[action.actor_role]] + out["ability"][0, aid[action.ability]] +
                     out["target_kind"][0, kid[action.target_kind]] + out["target_type"][0, target_index] +
                     out["queued"][0, int(action.queued)])
            ranked.append((float(score), action))
        for _, action in sorted(ranked, reverse=True, key=lambda x: x[0]):
            actors = actor_candidates(self, action.actor_role)
            if action.requires_flying_actor:
                actors = actors.filter(lambda unit: unit.is_flying)
            elif action.requires_grounded_actor:
                actors = actors.filter(lambda unit: not unit.is_flying)
            if not actors:
                continue
            target = None
            if action.target_kind == "unit":
                candidates = target_candidates(self, action.target_type)
                if not candidates:
                    continue
                point = point_from_normalized(self, out["target_point"][0].tolist())
                target = candidates.closest_to(point)
            elif action.target_kind == "point":
                target = point_from_normalized(self, out["target_point"][0].tolist())
                # Placement is a generic legality predicate from the action
                # contract's command family, not a scripted build location.
                # SC2 chooses a nearby legal location or rejects the proposal.
                if action.requires_placement:
                    target = await self.find_placement(AbilityId(action.ability_id), target, placement_step=3)
                    if target is None:
                        continue
            # The actor is selected from the model's role and the model's
            # target/location, not from ability-specific bot logic.
            chosen = [actors.closest_to(target if target is not None else self.townhalls.first)]
            sent = await execute(self, chosen, action.ability_id, target, action.queued)
            if sent:
                self.history.append(aid[action.ability]); print(f"t={self.time:.0f} ability={action.ability} sent={sent}", flush=True)
                return

    async def on_end(self, result):
        if self.result_path:
            Path(self.result_path).write_text(json.dumps({"result": str(result)}))
        print(result, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True); p.add_argument("--registry", required=True)
    p.add_argument("--race", choices=("terran", "protoss", "zerg"), default="terran"); p.add_argument("--difficulty", default="easy")
    p.add_argument("--replay", required=True); p.add_argument("--result")
    a = p.parse_args(); bot = GeneralActionBot(a.checkpoint, a.registry, a.race); bot.result_path = a.result or str(Path(a.replay).with_suffix(".json"))
    result = run_game(maps.get("Simple64"), [Bot(getattr(Race, a.race.title()), bot), Computer(Race.Zerg, getattr(Difficulty, a.difficulty.title()))],
                      realtime=False, save_replay_as=a.replay, game_time_limit=1800)
    print(result)


if __name__ == "__main__": main()
