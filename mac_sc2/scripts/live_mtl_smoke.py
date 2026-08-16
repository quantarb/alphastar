"""Thin CLI for an executable checkpoint smoke match."""
from __future__ import annotations

import argparse

from mac_sc2.evaluation.patch_race_match import run_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--race", choices=("terran", "protoss", "zerg"), required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--steps", type=int, default=64)
    args = parser.parse_args()
    print(run_match(args.checkpoint, args.registry, args.race, "easy", args.replay, args.steps))


if __name__ == "__main__":
    main()
