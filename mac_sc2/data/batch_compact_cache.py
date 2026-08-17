"""Bounded two-player compact-cache batch with current-client eligibility tags."""
from __future__ import annotations

import argparse
import json
import plistlib
import sys
from pathlib import Path

from absl import flags
from mac_sc2.data.compact_replay_cache import convert
from mac_sc2.contracts.terran_entity_ar import PATCH


def _installed_versions() -> set[str]:
    """Discover replay builds the local SC2 installation can actually launch."""
    root = Path("/Applications/StarCraft II/Versions")
    versions = set()
    for info in root.glob("Base*/SC2.app/Contents/Info.plist"):
        with info.open("rb") as stream:
            bundle = plistlib.load(stream)
        short, build = bundle.get("CFBundleShortVersionString"), bundle.get("CFBundleVersion")
        if short and build:
            versions.add(f"{short.split()[0]}.{build}")
    if not versions:
        raise RuntimeError("no installed SC2 replay clients found")
    return versions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("index", type=Path)
    parser.add_argument("--additional-index", type=Path, action="append", default=[],
                        help="additional manifest(s); paths are deduplicated before conversion")
    parser.add_argument("--games", type=int, default=0,
                        help="maximum games; 0 converts every eligible manifest row")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exact-current", action="store_true",
                        help="convert only replays exactly matching the installed live client")
    parser.add_argument("--workers", type=int, default=1, help="number of disjoint converter workers")
    parser.add_argument("--worker-index", type=int, default=0, help="zero-based worker partition")
    args = parser.parse_args()
    # ``run_configs.get`` in the converter reads PySC2's global flags even
    # when called as a library function rather than through its CLI entrypoint.
    flags.FLAGS([sys.argv[0]])
    # The bundled protocol decoder has no schema for 5.0.16.97337.  The
    # remaining 211 5.0.16 replays are supported by the installed client.
    source_rows = []
    for index in [args.index, *args.additional_index]:
        manifest = json.loads(index.read_text())
        rows = manifest.get("rows", manifest.get("valid"))
        if rows is None:
            raise ValueError(f"manifest needs a 'rows' or 'valid' list: {index}")
        source_rows.extend(rows)
    seen = set(); source_rows = [row for row in source_rows if not (row["path"] in seen or seen.add(row["path"]))]
    installed = _installed_versions()
    if args.exact_current:
        rows = [row for row in source_rows if row["version"] == PATCH]
    else:
        rows = [row for row in source_rows if row["version"] in installed]
    if args.games:
        rows = rows[:args.games]
    if args.workers < 1 or not 0 <= args.worker_index < args.workers:
        raise ValueError("worker-index must be in [0, workers)")
    rows = rows[args.worker_index::args.workers]
    summary = {"requested_games": len(rows), "worker_index": args.worker_index, "workers": args.workers,
               "installed_versions": sorted(installed),
               "completed_trajectories": 0, "failed": []}
    for game, row in enumerate(rows, 1):
        replay = Path(row["path"])
        for player in (1, 2):
            output = args.output_dir / replay.stem / f"player_{player}.compact.jsonl.gz"
            if output.exists():
                summary["completed_trajectories"] += 1
                continue
            try:
                result = convert(replay, player, output, research_patch_family=True)
                summary["completed_trajectories"] += 1
                print(json.dumps({"game": game, "player": player, **result}), flush=True)
            except Exception as error:
                summary["failed"].append({"replay": replay.name, "player": player, "error": str(error)})
                print(json.dumps({"game": game, "player": player, "failed": type(error).__name__}), flush=True)
        summary_path = args.output_dir / f"batch_summary_worker_{args.worker_index}.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__": main()
