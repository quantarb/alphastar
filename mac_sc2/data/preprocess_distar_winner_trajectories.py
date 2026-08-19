#!/usr/bin/env python3
"""Persist exact DI-star tensors for explicit replay/player manifest rows."""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
from pathlib import Path

import torch


def cache_path(cache_dir: Path, replay_path: str, player_index: int) -> Path:
    identity = f"{Path(replay_path).resolve()}\t{player_index}".encode("utf-8")
    return cache_dir / f"{hashlib.sha256(identity).hexdigest()}.pt"


def preprocess_chunk(distar_root: str, entries: list[tuple[str, int]], cache_dir: str, offset: int, total: int, race: str) -> None:
    import sys
    sys.path.insert(0, distar_root)
    from distar.agent.default.replay_decoder import ReplayDecoder
    from distar.ctools.utils import read_config

    root = Path(distar_root)
    destination = Path(cache_dir)
    config = read_config(str(root / "distar/bin/sl_user_config.yaml"))
    config.learner.data.parse_race = [{"zerg": "Z", "terran": "T", "protoss": "P"}[race]]
    config.current_patch_race = race
    decoder = ReplayDecoder(config)
    for local_index, (replay_path, player_index) in enumerate(entries, start=1):
        target = cache_path(destination, replay_path, player_index)
        global_index = offset + local_index
        if target.exists():
            print(f"[{global_index}/{total}] cached {target.name}", flush=True)
            continue
        trajectory = decoder.run(replay_path, player_index)
        if trajectory is None:
            raise RuntimeError(f"failed to preprocess {replay_path} player {player_index}")
        temporary = target.with_suffix(".tmp")
        torch.save(trajectory, temporary)
        os.replace(temporary, target)
        print(f"[{global_index}/{total}] saved {target.name} ({len(trajectory)} steps)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distar-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--race", choices=("zerg", "terran", "protoss"), default="zerg")
    args = parser.parse_args()

    entries = []
    for line in args.manifest.read_text().splitlines():
        replay_path, player_index = line.rsplit("\t", 1)
        entries.append((replay_path, int(player_index)))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.workers, len(entries))
    processes = []
    for worker_index in range(workers):
        chunk = entries[worker_index::workers]
        process = mp.Process(
            target=preprocess_chunk,
            args=(str(args.distar_root.resolve()), chunk, str(args.cache_dir.resolve()), worker_index, len(entries), args.race),
        )
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
        if process.exitcode:
            raise RuntimeError(f"preprocess worker exited with {process.exitcode}")
    print({"completed": len(list(args.cache_dir.glob("*.pt"))), "cache_dir": str(args.cache_dir.resolve())}, flush=True)


if __name__ == "__main__":
    main()
