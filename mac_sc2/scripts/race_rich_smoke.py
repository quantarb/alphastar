"""Run actual current-client legality/emitter smoke tests for rich-V2 races."""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from mac_sc2.contracts.race_rich_actions import intents_for
from mac_sc2.runtime.race_rich_executor import RaceRichExecutor, validate_race_live_contract
from mac_sc2.runtime.rich_transformer_bot import RichTransformerRaceBot


class SmokeBot(RaceRichExecutor):
    """A deterministic driver that exercises live ability and target checks."""
    def __init__(self, race: str, steps: int):
        super().__init__(race)
        self.steps = steps

    async def on_step(self, iteration: int) -> None:
        if iteration % 16:
            return
        # Every candidate reaches ``get_available_abilities`` and can emit only
        # if its current-client prerequisites and target are valid.
        names = {intent.name for intent in intents_for(self.race_name)}
        for name in ("train_probe", "train_drone", "train_overlord", "build_pylon",
                     "build_spawning_pool", "attack", "scout", "retreat"):
            if name in names:
                if await self.issue(name):
                    break
        if iteration >= self.steps:
            await self.client.leave()

    async def on_end(self, result) -> None:
        self.result = str(result)


def _watch_log(path: Path, message: str) -> None:
    """Append an immediately readable lifecycle record for visible matches."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with path.with_suffix(".watch.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{timestamp} {message}\n")


def smoke(race: str, replay: str, steps: int | None = 128, checkpoint: str | None = None,
          realtime: bool = False) -> dict:
    race = race.title()
    path = Path(replay).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    _watch_log(path, f"launch requested race={race} realtime={realtime} step_limit={steps} checkpoint={checkpoint}")
    try:
        if race not in ("Protoss", "Zerg"):
            raise ValueError("smoke supports Protoss and Zerg")
        _watch_log(path, "validating live action contract")
        validate_race_live_contract(race)
        _watch_log(path, "constructing policy bot and loading checkpoint")
        bot = (RichTransformerRaceBot(race, checkpoint, steps,
                                      decision_log=str(path.with_suffix(".decisions.log")))
               if checkpoint else SmokeBot(race, steps))
        _watch_log(path, "entering SC2 run_game")
        result = run_game(maps.get("Simple64"), [Bot(Race[race], bot), Computer(Race.Terran, Difficulty.VeryEasy)],
                          realtime=realtime, save_replay_as=str(path))
        _watch_log(path, f"SC2 run_game returned result={result}")
    except Exception as error:
        # A failed launch must be distinguishable from an intentional smoke
        # leave or an SC2 match result before a watcher is retried.
        path.with_suffix(".json").write_text(json.dumps({
            "race": race,
            "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
            "replay": str(path),
            "launch_error": repr(error),
            "traceback": traceback.format_exc(),
        }, indent=2) + "\n")
        _watch_log(path, f"launch failed error={error!r}\n{traceback.format_exc()}")
        raise
    record = {"race": race, "checkpoint": str(Path(checkpoint).resolve()) if checkpoint else None,
              "result": str(result), "replay": str(path), "telemetry": dict(bot.telemetry)}
    path.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    _watch_log(path, f"result record written telemetry={record['telemetry']}")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race", required=True, choices=("Protoss", "Zerg"))
    parser.add_argument("--replay", required=True)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--full", action="store_true", help="run without a smoke step limit")
    parser.add_argument("--realtime", action="store_true", help="play at wall-clock speed for watching")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    print(json.dumps(smoke(args.race, args.replay, None if args.full else args.steps, args.checkpoint,
                           args.realtime), indent=2))


if __name__ == "__main__":
    main()
