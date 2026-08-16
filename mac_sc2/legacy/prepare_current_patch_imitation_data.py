#!/usr/bin/env python3
"""Create a deduplicated manifest for playable current-patch imitation data.

Files are never copied or modified.  The manifest records provenance and a
SHA-256 digest so downstream state/action extraction is reproducible.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'local_data/current_replays'
SOURCES = (
    ('spawningtool_pro_2026', True),
    ('spawningtool_2026', False),
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


rows, seen = [], set()
for source, high_skill_proxy in SOURCES:
    manifest = DATA / f'manifest_{source}_5_0_16.json'
    if not manifest.exists():
        continue
    for record in json.loads(manifest.read_text())['valid']:
        replay = Path(record['path'])
        checksum = digest(replay)
        if checksum in seen:
            continue
        seen.add(checksum)
        rows.append({
            'path': str(replay), 'sha256': checksum, 'bytes': replay.stat().st_size,
            'version': record['version'], 'source': source,
            'high_skill_proxy': high_skill_proxy,
        })
rows.sort(key=lambda row: (not row['high_skill_proxy'], row['path']))
out = DATA / 'playable_imitation_5_0_16_index.json'
out.write_text(json.dumps({
    'patch_family': '5.0.16',
    'description': 'Raw replays for observation/action/selection/target extraction.',
    'rows': rows,
}, indent=2))
high = sum(row['high_skill_proxy'] for row in rows)
print(f'unique_current_patch_replays={len(rows)} high_skill_source={high} index={out}')
