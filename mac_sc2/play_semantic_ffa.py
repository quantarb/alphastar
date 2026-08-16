#!/usr/bin/env python3
"""Watch a semantic-contract checkpoint in a three-player SC2 FFA."""
import argparse, os
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from play_semantic_transfer import SemanticBot

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--race',default='terran',choices=('terran','protoss','zerg'));p.add_argument('--difficulty',default='hard');p.add_argument('--realtime',action='store_true');p.add_argument('--replay',required=True);a=p.parse_args()
 d=getattr(Difficulty,a.difficulty.title());r=getattr(Race,a.race.title())
 print(run_game(maps.get('Simple64'),[Bot(r,SemanticBot(a.checkpoint,a.race),name='Semantic BC Bot'),Computer(Race.Zerg,d),Computer(Race.Protoss,d)],realtime=a.realtime,save_replay_as=a.replay,game_time_limit=1800))
if __name__=='__main__':main()
