"""Thin CLI for research-only all-patch shared-backbone pretraining."""
from __future__ import annotations

import argparse

from mac_sc2.training.historical_pretrain import HistoricalPretrainConfig, pretrain


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--games", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    print(pretrain(HistoricalPretrainConfig(args.manifest, args.registry, args.output,
                                             games=args.games, batch_size=args.batch_size)))


if __name__ == "__main__":
    main()
