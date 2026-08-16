#!/usr/bin/env python3
"""Verify whether a historical replay yields live entity observations.

Spatial BC is allowed only if these observations are available from the exact
replay/client build.  This probe never writes a dataset or trains a model.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
from sc2.main import run_replay
from sc2.observer_ai import ObserverAI


class Probe(ObserverAI):
    async def on_step(self, iteration):
        if iteration < 2:
            own = list(self.units) + list(self.structures)
            enemy = list(self.enemy_units) + list(self.enemy_structures)
            sample = own[0] if own else None
            print({
                "iteration": iteration,
                "game_loop": self.state.game_loop,
                "own_entities": len(own),
                "visible_enemy_entities": len(enemy),
                "sample": None if sample is None else {
                    "type": sample.type_id.name, "x": sample.position.x,
                    "y": sample.position.y, "health": sample.health,
                    "cooldown": sample.weapon_cooldown,
                },
            }, flush=True)
        if iteration >= 16:
            await self.client.leave()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--replay", required=True)
    p.add_argument("--player", type=int, default=1, choices=(1, 2))
    args = p.parse_args()
    print(run_replay(Probe(), Path(args.replay).resolve(), realtime=False, observed_id=args.player))


if __name__ == "__main__":
    main()
