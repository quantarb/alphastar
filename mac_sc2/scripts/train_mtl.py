"""Thin CLI for the reusable playable MTL fine-tuning lifecycle."""
from __future__ import annotations

import argparse

from mac_sc2.training.multitask import MultiTaskConfig, fine_tune


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int)
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--start-game", type=int, default=0)
    args = parser.parse_args()
    print(fine_tune(MultiTaskConfig(manifest=args.manifest, registry=args.registry, output=args.output, games=args.games,
                                    resume_checkpoint=args.resume_checkpoint, start_game=args.start_game)))


if __name__ == "__main__":
    main()
