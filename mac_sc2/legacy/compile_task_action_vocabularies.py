#!/usr/bin/env python3
"""Compile the raw replay registry into declarative patch/race action masks.

No examples are copied or transformed here.  The result is a small manifest
used by the on-demand reader, model, checkpoint, and live legality masker.
"""
import argparse
import hashlib
import json
from pathlib import Path

from general_action_spec import schema_hash


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--min-count", type=int, default=1)
    args = p.parse_args()
    raw = json.loads(Path(args.registry).read_text())
    tasks, global_abilities, global_targets = {}, set(), set()
    for task, rows in raw["tasks"].items():
        legal = [r for r in rows if r["count"] >= args.min_count and r["ability_name"]]
        abilities = sorted({r["ability_name"] for r in legal})
        ability_ids = {name: sorted({r["ability_id"] for r in legal if r["ability_name"] == name}) for name in abilities}
        targets = sorted({r["target_name"] for r in legal if r["target_name"]})
        tasks[task] = {
            "abilities": abilities,
            "ability_ids": ability_ids,
            "target_types": targets,
            "signatures": sorted({
                (r["actor"], r["ability_name"], r["target_kind"], bool(r["queued"])) for r in legal
            }),
            "observed_actions": sum(r["count"] for r in legal),
        }
        global_abilities.update(abilities); global_targets.update(targets)
    body = {
        "schema_hash": schema_hash(), "registry_schema_version": raw["schema_version"],
        "abilities": sorted(global_abilities), "target_types": sorted(global_targets), "tasks": tasks,
    }
    body["vocab_hash"] = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(body))
    print(f"saved={output} tasks={len(tasks)} abilities={len(body['abilities'])} targets={len(body['target_types'])}")


if __name__ == "__main__":
    main()
