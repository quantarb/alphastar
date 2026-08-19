"""Live runner for the V2 rich transformer policy."""
from __future__ import annotations

from pathlib import Path

import torch
from sc2.ids.unit_typeid import UnitTypeId

from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.contracts.race_rich_actions import intents_for
from mac_sc2.contracts.rich_transformer_action import contract_hash
from mac_sc2.contracts.rich_transformer_snapshot import snapshot_hash
from mac_sc2.contracts.terran_entity_ar import INTENTS
from mac_sc2.runtime.rich_transformer_snapshot import encode
from mac_sc2.runtime.terran_entity_ar_bot import TerranEntityARBot, validate_live_contract
from mac_sc2.runtime.race_rich_executor import RaceRichExecutor, validate_race_live_contract


class RichTransformerBot(TerranEntityARBot):
    """Uses the established patch-valid Terran executor with transformer scores."""
    def __init__(self, checkpoint: str | None = None, smoke_steps: int | None = None, target_mmr: int = 4500):
        super().__init__(checkpoint=None, smoke_steps=smoke_steps)
        self.target_mmr = target_mmr
        # The tick compiler has no positive ``scout`` labels yet.  Do not let
        # an untrained intent become an always-legal fallback in a live game.
        self.enable_tick_scout = False
        self.scout_tag: int | None = None
        self.scout_complete = False
        # ``attack`` is a group command in the executor.  A one-unit pointer
        # label must never be allowed to turn into a stream of lone attacks.
        self.attack_min_units = 6
        self.attack_group_tags: set[int] = set()
        validate_live_contract()
        if checkpoint:
            data = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if data.get("architecture_name") != "RichEntityTransformerPolicy":
                raise RuntimeError("checkpoint is not a rich transformer policy")
            if data.get("action_contract_hash") != contract_hash() or data.get("entity_snapshot_hash") != snapshot_hash():
                raise RuntimeError("rich transformer checkpoint contract mismatch")
            self.model = RichEntityTransformerPolicy(**data.get("architecture", {}))
            self.model.load_state_dict(data["state_dict"]); self.model.eval()

    def feature_tensor(self) -> torch.Tensor:
        # This is exactly the scalar layout stored by compact_replay_cache.py.
        return torch.tensor([[self.minerals, self.vespene, self.supply_cap, self.supply_used,
                              self.supply_army, self.supply_workers, self.idle_worker_count,
                              self.army_count, 0, 0]], dtype=torch.float32)

    async def _issue_prerequisite(self, name: str) -> bool:
        """Emit a contract action while satisfying a learned Terran goal."""
        # Supply depots are a Barracks prerequisite even before the normal
        # supply-low policy would consider building one.
        if name == "build_supply":
            if not self.workers or not self.can_afford(UnitTypeId.SUPPLYDEPOT):
                return False
        elif not self._legal(name):
            return False
        return await self._issue(name, self._actor(name), self._target(name))

    async def _resolve_goal(self, goal: str) -> str | None:
        """Turn a blocked Terran goal into its first missing live prerequisite."""
        depot = self.structures(UnitTypeId.SUPPLYDEPOT)
        barracks = self.structures(UnitTypeId.BARRACKS)
        factory = self.structures(UnitTypeId.FACTORY)
        needs_depot = {"build_barracks", "build_factory", "train_marine", "train_hellion",
                       "morph_orbital", "call_mule", "attack"}
        needs_barracks = {"build_factory", "train_marine", "train_hellion", "morph_orbital",
                          "call_mule", "attack"}
        if goal in needs_depot:
            if not depot:
                return "build_supply" if await self._issue_prerequisite("build_supply") else None
            if not depot.ready:
                return None
        if goal in needs_barracks:
            if not barracks:
                return "build_barracks" if await self._issue_prerequisite("build_barracks") else None
            if not barracks.ready:
                return None
        if goal == "train_hellion":
            if not factory:
                return "build_factory" if await self._issue_prerequisite("build_factory") else None
            if not factory.ready:
                return None
        if goal == "call_mule" and not self.townhalls(UnitTypeId.ORBITALCOMMAND):
            return "morph_orbital" if await self._issue_prerequisite("morph_orbital") else None
        if goal == "attack" and not self.units.exclude_type({UnitTypeId.SCV}):
            return "train_marine" if await self._issue_prerequisite("train_marine") else None
        return None

    def _refresh_scout(self) -> None:
        """Close a scout assignment after arrival or loss; never stack scouts."""
        if self.scout_tag is None or self.scout_complete:
            return
        scout = self.units.find_by_tag(self.scout_tag)
        if scout is None:
            self.scout_complete = True
            self.telemetry["scout_lost"] += 1
        elif scout.distance_to(self._region("enemy_start")) <= 10:
            self.scout_complete = True
            self.telemetry["scout_complete"] += 1

    def _refresh_attack_group(self) -> None:
        """Keep reinforcements home until the committed squad is gone."""
        if not self.attack_group_tags:
            return
        alive = {unit.tag for unit in self.units.exclude_type({UnitTypeId.SCV})}
        self.attack_group_tags.intersection_update(alive)
        if not self.attack_group_tags:
            self.telemetry["attack_group_finished"] += 1

    def _legal(self, name: str) -> bool:
        if name == "scout":
            if not self.enable_tick_scout:
                self.telemetry["masked_untrained_scout"] += 1
                return False
            return bool(not self.scout_complete and self.scout_tag is None and
                        self.units.exclude_type({UnitTypeId.SCV}))
        if name == "attack":
            return bool(not self.attack_group_tags and
                        self.units.exclude_type({UnitTypeId.SCV}).amount >= self.attack_min_units)
        return super()._legal(name)

    async def _issue(self, name: str, actor, target=None) -> bool:
        issued = await super()._issue(name, actor, target)
        if issued and name == "scout":
            self.scout_tag = actor.tag
            self.telemetry["scout_assigned"] += 1
        if issued and name == "attack":
            self.attack_group_tags = {unit.tag for unit in self.units.exclude_type({UnitTypeId.SCV})}
            self.telemetry["attack_group_started"] += 1
        return issued

    async def on_step(self, iteration: int) -> None:
        if iteration % 16 or not self.townhalls:
            return
        self._refresh_scout()
        self._refresh_attack_group()
        entities, padding, owned = encode(self)
        actions, output = [self._rule_intent()], None
        if self.model:
            history_actions = torch.tensor([[0] * (16 - min(16, len(self.history))) +
                                            [value + 1 for value in self.history[-16:]]])
            history_scalars = self.feature_tensor().unsqueeze(1).expand(-1, 16, -1)
            with torch.no_grad():
                output = self.model(self.feature_tensor(), entities.unsqueeze(0), padding.unsqueeze(0),
                                    history_scalars=history_scalars, history_actions=history_actions,
                                    target_mmr=torch.tensor([[self.target_mmr]], dtype=torch.float32))
            actions = output.intent[0].argsort(descending=True).tolist()
        # Apply state- and SC2-legality masking *before* choosing an intent.
        # The old loop used the first legal fallback after scoring illegal
        # macro goals, which made the untrained scout class disproportionately
        # likely whenever production was unavailable.
        legal_actions = []
        blocked_actions = []
        for action in actions:
            name = INTENTS[action].name
            if not self._legal(name):
                self.telemetry["masked_illegal"] += 1
                blocked_actions.append(action)
            else:
                legal_actions.append(action)
        if legal_actions:
            action = legal_actions[0]; name = INTENTS[action].name
            actor = self._ranked_actor(name, owned, output.actor[0]) if output else self._actor(name)
            target = self._ranked_target(name, owned, output.target[0]) if output else self._target(name)
            if await self._issue(name, actor, target):
                self.history.append(action); self.telemetry["decisions"] += 1
        elif blocked_actions:
            # If there is no legal output at all, resolve only the highest
            # ranked blocked goal rather than attempting every fallback.
            action = blocked_actions[0]; name = INTENTS[action].name
            resolved = await self._resolve_goal(name)
            if resolved:
                self.history.append(action)
                self.telemetry["prerequisite_issued"] += 1
                self.telemetry[f"prerequisite_for_{name}"] += 1
        if self.smoke_steps is not None and iteration >= self.smoke_steps:
            await self.client.leave()


class RichTransformerRaceBot(RaceRichExecutor):
    """Load and execute the shared V2 transformer for Protoss or Zerg."""
    def __init__(self, race: str, checkpoint: str, smoke_steps: int | None = None, target_mmr: int = 4500,
                 decision_log: str | None = None):
        super().__init__(race); self.smoke_steps, self.target_mmr, self.history = smoke_steps, target_mmr, []
        self.decision_log = Path(decision_log) if decision_log else None
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if data.get("architecture_name") != "RichEntityTransformerPolicy" or data.get("action_contract_hash") != contract_hash() or data.get("entity_snapshot_hash") != snapshot_hash():
            raise RuntimeError("rich V2 checkpoint contract mismatch")
        self.model = RichEntityTransformerPolicy(**data["architecture"]); self.model.load_state_dict(data["state_dict"]); self.model.eval()

    def _log_decision(self, message: str) -> None:
        if self.decision_log:
            with self.decision_log.open("a", encoding="utf-8") as stream:
                stream.write(message + "\n")

    async def _fallback(self) -> str | None:
        """Keep a live match progressing when no learned action is currently legal.

        This is deliberately limited to actions in the versioned V2 contract.
        It is recorded separately from learned decisions so demonstrations do
        not present scripted macro as model behavior.
        """
        if self.race_name == "Protoss":
            candidates = []
            if self.supply_left <= 2:
                candidates.append("build_pylon")
            candidates.append("train_probe")
            if not self.actors("gateway"):
                candidates.append("build_gateway")
            candidates.extend(("train_zealot", "train_stalker"))
            if self.actors("combat").amount >= 4:
                candidates.append("attack")
        else:
            candidates = []
            if self.supply_left <= 2:
                candidates.append("train_overlord")
            candidates.append("train_drone")
            if not self.structures(UnitTypeId.SPAWNINGPOOL):
                candidates.append("build_spawning_pool")
            candidates.extend(("train_zergling", "train_roach"))
            if self.actors("combat").amount >= 6:
                candidates.append("attack")
        for name in candidates:
            if await self.issue(name):
                self.telemetry["fallback_issued"] += 1
                self.telemetry[f"fallback_{name}"] += 1
                return name
        self.telemetry["fallback_no_legal_action"] += 1
        return None

    async def _resolve_goal(self, goal: str) -> str | None:
        """Issue the first missing executable prerequisite for a model goal."""
        if self.race_name == "Zerg":
            pool = self.structures(UnitTypeId.SPAWNINGPOOL)
            warren = self.structures(UnitTypeId.ROACHWARREN)
            needs_pool = {"build_roach_warren", "train_zergling", "train_roach", "attack"}
            if goal in needs_pool and not pool:
                return "build_spawning_pool" if await self.issue("build_spawning_pool") else None
            if goal in needs_pool and not pool.ready:
                return None
            if goal == "train_roach" and not warren:
                return "build_roach_warren" if await self.issue("build_roach_warren") else None
            if goal == "train_roach" and not warren.ready:
                return None
            if goal == "attack" and not self.actors("combat"):
                return "train_zergling" if await self.issue("train_zergling") else None
            return None
        if self.race_name != "Protoss":
            return None

        needs_gateway = {"build_cybernetics", "train_zealot", "train_stalker",
                         "research_warpgate", "attack"}
        needs_cybernetics = {"train_stalker", "research_warpgate"}
        pylon = self.structures(UnitTypeId.PYLON)
        gateway = self.structures(UnitTypeId.GATEWAY) | self.structures(UnitTypeId.WARPGATE)
        cybernetics = self.structures(UnitTypeId.CYBERNETICSCORE)

        # A Gateway itself needs a completed Pylon.  Do not duplicate an
        # already-in-progress prerequisite; wait for it to complete instead.
        if goal == "build_gateway" or goal in needs_gateway:
            if not pylon:
                return "build_pylon" if await self.issue("build_pylon") else None
            if not pylon.ready:
                return None
        if goal in needs_gateway and not gateway:
            return "build_gateway" if await self.issue("build_gateway") else None
        if goal in needs_gateway and not gateway.ready:
            return None
        if goal in needs_cybernetics and not cybernetics:
            return "build_cybernetics" if await self.issue("build_cybernetics") else None
        if goal in needs_cybernetics and not cybernetics.ready:
            return None
        # An attack goal without an army is resolved into the least-assumptive
        # contract action that creates one; its own prerequisites are handled
        # by the earlier branches on subsequent ticks.
        if goal == "attack" and not self.actors("combat"):
            return "train_zealot" if await self.issue("train_zealot") else None
        return None

    def feature_tensor(self) -> torch.Tensor:
        return torch.tensor([[self.minerals, self.vespene, self.supply_cap, self.supply_used, self.supply_army,
                              self.supply_workers, self.idle_worker_count, self.army_count, 0, 0]], dtype=torch.float32)

    async def on_step(self, iteration: int) -> None:
        if iteration % 16: return
        entities, padding, owned = encode(self)
        history = torch.tensor([[0] * (16 - min(16, len(self.history))) + [item + 1 for item in self.history[-16:]]])
        with torch.no_grad():
            output = self.model(self.feature_tensor(), entities.unsqueeze(0), padding.unsqueeze(0),
                history_scalars=self.feature_tensor().unsqueeze(1).expand(-1, 16, -1), history_actions=history,
                race=torch.tensor([RACES.index(self.race_name)]), target_mmr=torch.tensor([[self.target_mmr]], dtype=torch.float32))
        names = intents_for(self.race_name)
        ranked = output.intent[0, :len(names)].argsort(descending=True).tolist()
        issued = None
        resolved = None
        for index in ranked:
            intent = names[index]
            allowed = {unit.tag: unit for unit in self.actors(intent.actor_role)}
            actor = next((allowed[owned[pointer].tag] for pointer in output.actor[0].argsort(descending=True).tolist()
                          if pointer < len(owned) and owned[pointer].tag in allowed), None)
            # Legal-mask each candidate using the same live availability and
            # placement checks as emission. This prevents an unavailable top
            # prediction from appearing like a valid policy choice.
            legal_actor, target = await self.legal(intent.name, actor=actor)
            if legal_actor is None:
                self.telemetry["masked_illegal"] += 1
                self.telemetry[f"masked_{intent.name}"] += 1
                resolved = await self._resolve_goal(intent.name)
                if resolved:
                    self.telemetry["prerequisite_issued"] += 1
                    self.telemetry[f"prerequisite_for_{intent.name}"] += 1
                    self.history.append(index)
                    break
                continue
            if await self.issue(intent.name, actor=legal_actor, target=target,
                                queued=bool(output.queued[0].argmax())):
                self.history.append(index)
                self.telemetry["learned_issued"] += 1
                issued = intent.name
                break
        fallback = None
        if issued is None and resolved is None:
            fallback = await self._fallback()
        self._log_decision(
            f"iteration={iteration} ranked={[names[index].name for index in ranked[:4]]} "
            f"learned_issued={issued} prerequisite_issued={resolved} fallback_issued={fallback} "
            f"masked_illegal={self.telemetry['masked_illegal']}"
        )
        if self.smoke_steps is not None and iteration >= self.smoke_steps: await self.client.leave()
