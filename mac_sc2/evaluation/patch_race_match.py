"""Reusable live-match evaluation and result/replay recording."""
from __future__ import annotations

import json
from multiprocessing import Process
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from mac_sc2.runtime.patch_race_rich_bot import PatchRaceBot


def run_match(checkpoint: str, registry: str, race: str, difficulty: str, replay: str, smoke_steps: int | None = None) -> dict:
    bot = PatchRaceBot(checkpoint, registry, race, smoke_steps)
    replay_path = Path(replay).resolve(); replay_path.parent.mkdir(parents=True, exist_ok=True)
    result_path = replay_path.with_suffix(".json")
    bot.result_path = str(result_path)
    result = run_game(maps.get("Simple64"), [Bot(getattr(Race, race.title()), bot), Computer(Race.Zerg, getattr(Difficulty, difficulty.title()))], realtime=False, save_replay_as=str(replay_path))
    record = {"checkpoint": str(Path(checkpoint).resolve()), "race": race, "difficulty": difficulty,
              "result": str(result), "replay": str(replay_path), "telemetry": dict(bot.telemetry)}
    result_path.write_text(json.dumps(record, indent=2))
    return record


def run_easy_suite(checkpoint: str, registry: str, output_dir: str | Path) -> list[dict]:
    """Run races serially: SC2 clients cannot safely share auto-selected ports."""
    records = []
    for race in ("terran", "protoss", "zerg"):
        replay = Path(output_dir) / f"{Path(checkpoint).stem}_{race}_easy.SC2Replay"
        records.append(run_match(checkpoint, registry, race, "easy", str(replay)))
    return records


def launch_easy_suite(checkpoint: str, registry: str, output_dir: str | Path) -> Process:
    process = Process(target=run_easy_suite, args=(checkpoint, registry, output_dir))
    process.start()
    return process
