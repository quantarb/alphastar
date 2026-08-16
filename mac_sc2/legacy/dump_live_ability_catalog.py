#!/usr/bin/env python3
"""Print the installed SC2 client's live ability metadata as JSON."""
import asyncio
import argparse
import json
import os

os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer


class CatalogBot(BotAI):
    async def on_start(self):
        rows = []
        for ability_id, data in self.game_data.abilities.items():
            rows.append({"id": ability_id, "button_name": data.button_name,
                         "link_name": data.link_name, "friendly_name": data.friendly_name,
                         "target": data._proto.target})
        with open(self.output, "w") as handle:
            json.dump(rows, handle)
        print(f"saved={self.output} abilities={len(rows)}", flush=True)
        await self.client.leave()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--output", required=True); args = parser.parse_args()
    bot = CatalogBot(); bot.output = args.output
    try:
        run_game(maps.get("Simple64"), [Bot(Race.Terran, bot), Computer(Race.Zerg, Difficulty.VeryEasy)], realtime=False)
    except Exception as exc:
        # Leaving after cataloguing ends the game intentionally.
        if "Not in a game" not in str(exc): raise
