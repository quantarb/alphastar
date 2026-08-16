#!/usr/bin/env python3
"""CLI entry point for a recorded patch/race live evaluation."""
import argparse
from mac_sc2.evaluation.patch_race_match import run

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--registry',required=True);parser.add_argument('--race',choices=('terran','protoss','zerg'),required=True);parser.add_argument('--difficulty',choices=('easy','medium','hard'),default='easy');parser.add_argument('--replay',required=True);parser.add_argument('--smoke-steps',type=int);args=parser.parse_args()
    print(run(args.checkpoint,args.registry,args.race,args.difficulty,args.replay,args.smoke_steps))
if __name__ == "__main__": main()
