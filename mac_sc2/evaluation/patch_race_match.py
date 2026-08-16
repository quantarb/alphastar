"""Launch a live patch/race match and persist its literal result beside replay."""
from __future__ import annotations

import json
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from mac_sc2.runtime.patch_race_rich_bot import PatchRaceBot


def run(checkpoint: str, registry: str, race: str, difficulty: str, replay: str, smoke_steps: int | None = None):
    bot = PatchRaceBot(checkpoint, registry, race, smoke_steps)
    result_path = Path(replay).with_suffix(".json")
    bot.result_path = str(result_path)
    result = run_game(maps.get("Simple64"), [Bot(getattr(Race, race.title()), bot), Computer(Race.Zerg, getattr(Difficulty, difficulty.title()))], realtime=False, save_replay_as=replay)
    # on_end normally writes this; keep result recording reliable on startup failures too.
    if not result_path.exists():
        result_path.write_text(json.dumps({"result": str(result)}))
    return result
