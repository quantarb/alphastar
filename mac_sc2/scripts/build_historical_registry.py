"""Thin CLI for research-only multi-patch replay ActionSpecs."""
from __future__ import annotations

import argparse

from mac_sc2.data.action_registry import write_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--patch", action="append", dest="patches")
    args = parser.parse_args()
    registry = write_registry(args.manifest, args.output, set(args.patches or ()), args.workers, args.max_games)
    print({"replays": registry["replays"], "tasks": len(registry["tasks"])})


if __name__ == "__main__":
    main()
