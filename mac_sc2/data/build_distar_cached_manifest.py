#!/usr/bin/env python3
"""List only completed DI-star native trajectory cache files."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def cache_path(cache_dir: Path, replay_path: str, player_index: int) -> Path:
    identity = f"{Path(replay_path).resolve()}\t{player_index}".encode("utf-8")
    return cache_dir / f"{hashlib.sha256(identity).hexdigest()}.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory-manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cached = []
    for line in args.trajectory_manifest.read_text().splitlines():
        replay_path, player_index = line.rsplit("\t", 1)
        path = cache_path(args.cache_dir, replay_path, int(player_index))
        if path.exists():
            cached.append(str(path.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(cached) + ("\n" if cached else ""))
    print({"cached_trajectories": len(cached), "manifest": str(args.output.resolve())})


if __name__ == "__main__":
    main()
