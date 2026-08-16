#!/usr/bin/env python3
"""Launch one runnable MTL bot in a three-player FFA against two computers."""
import argparse, os

os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

from play_multirace_general import MultiRaceBot


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--tactical-checkpoint', help='Transfer-trained factorized tactical checkpoint.')
    p.add_argument('--race', choices=('terran', 'protoss', 'zerg'), default='terran')
    p.add_argument('--difficulty', default='hard')
    p.add_argument('--map', default='Simple64',
                   help='Installed three-player-compatible map (default: Simple64).')
    p.add_argument('--game-time-limit', type=int, default=1800,
                   help='Maximum simulated game seconds; prevents realtime loop rollover.')
    p.add_argument('--realtime', action='store_true')
    p.add_argument('--replay', required=True)
    a = p.parse_args()
    difficulty = getattr(Difficulty, a.difficulty.title())
    race = getattr(Race, a.race.title())
    players = [
        Bot(race, MultiRaceBot(a.checkpoint, a.race, 16, a.tactical_checkpoint), name='Trained MTL Bot'),
        Computer(Race.Zerg, difficulty),
        Computer(Race.Protoss, difficulty),
    ]
    print(f'Launching FFA on {a.map}: trained {a.race} vs {a.difficulty} Zerg and Protoss')
    print(run_game(maps.get(a.map), players, realtime=a.realtime,
                   save_replay_as=a.replay, game_time_limit=a.game_time_limit))


if __name__ == '__main__':
    main()
