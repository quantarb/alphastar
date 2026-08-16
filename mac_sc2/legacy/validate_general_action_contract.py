#!/usr/bin/env python3
"""Fail closed unless every ActionRegistry tuple has a 4.9.2 decoder."""
import argparse
import json
from pathlib import Path

from sc2.ids.ability_id import AbilityId
from general_action_registry import ActionRegistry


def main():
    p = argparse.ArgumentParser(); p.add_argument("--registry", required=True); p.add_argument("--catalog", required=True)
    a = p.parse_args(); registry = ActionRegistry(a.registry); catalog = {int(x["id"]): x for x in json.loads(Path(a.catalog).read_text())}
    modes = {1: "none", 2: "point", 3: "unit", 4: "either", 5: "point"}
    failures = []
    for row in registry.rows:
        try: AbilityId(row.ability_id)
        except ValueError: failures.append(f"unknown AbilityId {row.ability_id} for {row}"); continue
        live = catalog.get(row.ability_id)
        if live is None: failures.append(f"catalog missing {row.ability_id} for {row}"); continue
        mode = modes.get(live["target"], "unknown")
        if mode != "either" and mode != row.target_kind:
            failures.append(f"target mismatch {mode} != {row.target_kind} for {row}")
    if failures:
        raise SystemExit("ActionSpec invalid:\n" + "\n".join(failures[:20]))
    print(f"ActionSpec valid hash={registry.hash} tuples={len(registry.rows)} patch={registry.patch}")


if __name__ == "__main__": main()
