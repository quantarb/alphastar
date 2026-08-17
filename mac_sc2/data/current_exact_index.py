"""Build a deduplicated manifest of all locally available exact-client games."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mac_sc2.contracts.terran_entity_ar import PATCH
from mac_sc2.data.replay_collection import game_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("roots", type=Path, nargs="+")
    args = parser.parse_args()
    seen: set[str] = set(); rows = []; rejected = []
    for root in args.roots:
        for replay in sorted(root.glob("*.SC2Replay")):
            payload = replay.read_bytes(); digest = hashlib.sha256(payload).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            version = game_version(payload)
            record = {"path": str(replay.resolve()), "version": version, "sha256": digest,
                      "source": root.name}
            if version == PATCH:
                rows.append(record)
            else:
                rejected.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"patch": PATCH, "rows": rows, "rejected": rejected}, indent=2) + "\n")
    print(json.dumps({"exact_unique_games": len(rows), "rejected": len(rejected), "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
