#!/usr/bin/env python3
"""Download a reproducible, public, pro-only replay corpus from SpawningTool.

The site labels these listings as "pro replays".  That is a high-skill proxy,
not an MMR claim.  `validate_current_replays.py` must still be run afterwards:
the game client can only load replays from its exact 5.0.16 patch family.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path

import mpyq

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://lotv.spawningtool.com'
USER_AGENT = 'alphastar-local-research/1.0 (public-replay-collector)'


def fetch(url: str, timeout: int = 90) -> bytes:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def listing_ids(page: int, timeout: int, patch: int | None, pro_only: bool) -> list[int]:
    query = {"p": page}
    if patch is not None:
        query["patch"] = patch
    if pro_only:
        query["pro_only"] = "on"
    html = fetch(f'{BASE}/replays/?{urllib.parse.urlencode(query)}', timeout).decode('utf-8', 'replace')
    return sorted({int(value) for value in re.findall(r'href="/(\d+)/download/"', html)}, reverse=True)


def game_version(payload: bytes) -> str:
    archive = mpyq.MPQArchive(io.BytesIO(payload)).extract()
    return json.loads(archive[b'replay.gamemetadata.json'])['GameVersion']


def download(replay_id: int, destination: Path, require_version: str | None = None,
             timeout: int = 90) -> tuple[int, str]:
    target = destination / f'{replay_id}.SC2Replay'
    if target.is_file() and target.stat().st_size > 1_024:
        if not require_version or game_version(target.read_bytes()) == require_version:
            return replay_id, 'exists'
        return replay_id, 'rejected_version'
    temporary = target.with_suffix('.partial')
    try:
        payload = fetch(f'{BASE}/{replay_id}/download/', timeout)
        # Valid replays are binary and substantially larger than an HTML error page.
        if len(payload) < 1_024 or payload.lstrip().lower().startswith(b'<!doctype html'):
            raise ValueError(f'non-replay response ({len(payload)} bytes)')
        if require_version and game_version(payload) != require_version:
            return replay_id, 'rejected_version'
        temporary.write_bytes(payload)
        temporary.replace(target)
        return replay_id, 'downloaded'
    except Exception as error:
        temporary.unlink(missing_ok=True)
        return replay_id, f'failed:{type(error).__name__}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-page', type=int, default=1)
    parser.add_argument('--pages', type=int, default=50, help='pro-only listing pages to inspect')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--delay', type=float, default=0.2, help='pause between listing-page requests')
    parser.add_argument('--destination', type=Path,
                        help='isolated output directory; defaults to the shared pro corpus')
    parser.add_argument('--require-version',
                        help='retain only this exact embedded replay GameVersion')
    parser.add_argument('--patch', type=int, help='SpawningTool patch category (170 is 5.0.16)')
    parser.add_argument('--include-all', action='store_true',
                        help='include public games, not only listings marked pro')
    parser.add_argument('--timeout', type=int, default=30, help='per-request network timeout in seconds')
    arguments = parser.parse_args()
    destination = arguments.destination or ROOT / 'local_data/current_replays/spawningtool_pro_2026'
    destination.mkdir(parents=True, exist_ok=True)

    records, ids = [], set()
    for page in range(arguments.start_page, arguments.start_page + arguments.pages):
        try:
            page_ids = listing_ids(page, arguments.timeout, arguments.patch, not arguments.include_all)
            ids.update(page_ids)
            records.append({'page': page, 'replay_ids': page_ids})
            print(f'listing page={page} ids={len(page_ids)} total_unique={len(ids)}', flush=True)
        except Exception as error:
            records.append({'page': page, 'error': type(error).__name__})
            print(f'listing page={page} failed={type(error).__name__}', flush=True)
        time.sleep(arguments.delay)

    index = destination / 'source_index.json'
    index.write_text(json.dumps({
        'source': f'{BASE}/replays/', 'patch': arguments.patch, 'pro_only': not arguments.include_all,
        'pages_requested': arguments.pages,
        'start_page': arguments.start_page,
        'replay_ids': sorted(ids, reverse=True),
        'listing_records': records,
    }, indent=2))
    counts = {'exists': 0, 'downloaded': 0, 'rejected_version': 0, 'failed': 0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        for replay_id, status in pool.map(lambda value: download(value, destination, arguments.require_version, arguments.timeout), sorted(ids, reverse=True)):
            key = status.split(':', 1)[0]
            counts[key] = counts.get(key, 0) + 1
            print(f'replay={replay_id} {status}', flush=True)
    print(json.dumps({'requested': len(ids), **counts, 'directory': str(destination)}, indent=2))


if __name__ == '__main__':
    main()
