"""Real SC2 smoke/evaluation entry point for the rich transformer runner."""
from __future__ import annotations

import json
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from mac_sc2.runtime.rich_transformer_bot import RichTransformerBot


def run_match(replay: str, difficulty: str = "veryeasy", checkpoint: str | None = None,
              smoke_steps: int | None = None) -> dict:
    bot = RichTransformerBot(checkpoint=checkpoint, smoke_steps=smoke_steps)
    path = Path(replay).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    levels = {"veryeasy": Difficulty.VeryEasy, "easy": Difficulty.Easy, "medium": Difficulty.Medium,
              "hard": Difficulty.Hard, "veryhard": Difficulty.VeryHard}
    result = run_game(maps.get("Simple64"), [Bot(Race.Terran, bot), Computer(Race.Zerg, levels[difficulty.lower()])],
                      realtime=False, save_replay_as=str(path))
    record = {"checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None, "difficulty": difficulty,
              "result": str(result), "replay": str(path), "telemetry": dict(bot.telemetry)}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2))
    return record
