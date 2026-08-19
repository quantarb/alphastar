#!/usr/bin/env python3
"""Stamp unchanged released DI-star weights for a verified current-patch race."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distar-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--race", choices=("terran", "protoss"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.current_patch_contract import legacy_race_contract_hash

    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    checkpoint["current_patch_contract_hash"] = legacy_race_contract_hash(args.race)
    checkpoint["policy_action_encoding"] = f"released_327_id_with_current_{args.race}_mask"
    checkpoint["resumed_from"] = str(args.source.resolve())
    checkpoint["weights_changed"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print({"output": str(args.output), "race": args.race,
           "contract": checkpoint["current_patch_contract_hash"], "weights_changed": False})


if __name__ == "__main__":
    main()
