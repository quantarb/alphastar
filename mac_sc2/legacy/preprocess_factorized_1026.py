#!/usr/bin/env python3
"""The sole canonical preprocessing entry point for detailed SC2 BC data.

It creates a stable 1,026-way ability vocabulary, then canonical shards that
retain ability, selection, target and history fields.  Do not use the retired
macro-only preprocessors for new training runs.
"""
import argparse, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--max-games',type=int,required=True);a=p.parse_args()
 # The vocabulary is created once per corpus and then reused by every model
 # head, preventing label-space drift across training experiments.
 vocab=Path(a.out)/'ability_vocab_1026.json';Path(a.out).mkdir(parents=True,exist_ok=True)
 if not vocab.exists():
  raise RuntimeError('Create the frozen vocabulary first; existing 2k run uses mac_sc2/artifacts/ability_vocab_2k.json.')
 subprocess.run([sys.executable,str(ROOT/'mac_sc2/build_ability_1026_shards.py'),'--manifest',a.manifest,'--vocab',str(vocab),'--out',str(Path(a.out)/'shards'),'--max-games',str(a.max_games)],check=True)
if __name__=='__main__':main()
