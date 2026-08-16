#!/usr/bin/env python3
"""Minimal current-patch replay-observer check for state extraction."""
import json
import os
from pathlib import Path

os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
from sc2.main import run_replay
from sc2.observer_ai import ObserverAI

ROOT = Path(__file__).resolve().parents[1]


class StateProbe(ObserverAI):
    async def on_start(self):
        self.client.game_step = 112
        self.rows = []

    async def on_step(self, iteration):
        self.rows.append((round(self.time, 1), self.units.amount, self.structures.amount,
                          self.enemy_units.amount, self.enemy_structures.amount))
        if iteration >= 12:
            await self.client.leave()

    async def on_end(self, result):
        print(f'result={result} samples={self.rows}', flush=True)


manifest = json.loads((ROOT / 'local_data/current_replays/manifest_spawningtool_pro_2026_5_0_16.json').read_text())
probe = StateProbe()
run_replay(probe, str(Path(manifest['valid'][0]['path']).resolve()), realtime=False, observed_id=1)
