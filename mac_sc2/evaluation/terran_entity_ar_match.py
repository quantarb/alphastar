"""Live evaluation for the Terran entity-action executor."""
from __future__ import annotations

import json
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from mac_sc2.runtime.terran_entity_ar_bot import TerranEntityARBot


def run_match(replay: str, difficulty: str = "veryeasy", checkpoint: str | None = None,
              smoke_steps: int | None = None) -> dict:
    bot = TerranEntityARBot(checkpoint=checkpoint, smoke_steps=smoke_steps)
    replay_path = Path(replay).resolve(); replay_path.parent.mkdir(parents=True, exist_ok=True)
    bot.result_path = str(replay_path.with_suffix(".json"))
    difficulty_id = {"veryeasy": Difficulty.VeryEasy, "easy": Difficulty.Easy, "medium": Difficulty.Medium,
                     "hard": Difficulty.Hard, "veryhard": Difficulty.VeryHard}[difficulty.lower()]
    result = run_game(maps.get("Simple64"), [Bot(Race.Terran, bot), Computer(Race.Zerg, difficulty_id)],
                      realtime=False, save_replay_as=str(replay_path))
    record = {"checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None, "difficulty": difficulty,
              "result": str(result), "replay": str(replay_path), "telemetry": dict(bot.telemetry)}
    replay_path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    return record
