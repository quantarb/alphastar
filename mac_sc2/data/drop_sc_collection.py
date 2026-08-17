#!/usr/bin/env python3
"""Collect the public recent-replay feed from Drop.sc.

Files are admitted only after checking their embedded GameVersion.  This keeps
an external public feed from polluting the runnable, exact-client corpus.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import re
import urllib.request
from pathlib import Path

from mac_sc2.data.replay_collection import game_version

FEED = "https://drop.sc/"
USER_AGENT = "alphastar-local-research/1.0 (public-replay-collector)"


def fetch(url: str, timeout: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def downloads(timeout: int) -> dict[int, str]:
    html = fetch(FEED, timeout).decode("utf-8", "replace")
    # The hosted download URLs carry a per-replay hash, so retain each complete
    # URL rather than trying to reconstruct it from the replay id.
    pairs = re.findall(
        r'href="/replay/(\d+)".*?href="(//sc2replaystats\.com/download/[^"]+)"',
        html,
        flags=re.DOTALL,
    )
    return {int(replay_id): "https:" + url for replay_id, url in pairs}


def download(item: tuple[int, str], destination: Path, required: str, timeout: int) -> tuple[int, str]:
    replay_id, url = item
    target = destination / f"drop_{replay_id}.SC2Replay"
    try:
        if target.is_file() and target.stat().st_size > 1024:
            return replay_id, "exists" if game_version(target.read_bytes()) == required else "rejected_version"
        payload = fetch(url, timeout)
        if len(payload) < 1024 or game_version(payload) != required:
            return replay_id, "rejected_version"
        temporary = target.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(target)
        return replay_id, "downloaded"
    except Exception as error:
        return replay_id, f"failed:{type(error).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--require-version", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    items = downloads(args.timeout)
    counts: dict[str, int] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for replay_id, status in pool.map(
            lambda item: download(item, args.destination, args.require_version, args.timeout), items.items()
        ):
            key = status.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
            print(f"replay={replay_id} {status}", flush=True)
    print({"feed_items": len(items), **counts, "destination": str(args.destination.resolve())})


if __name__ == "__main__":
    main()
