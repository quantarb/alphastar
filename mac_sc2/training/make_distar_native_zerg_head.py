#!/usr/bin/env python3
"""Create a compact executable Zerg action head from released DI-star weights.

The released model's observation/history inputs remain 327-way, preserving
their pretrained representation.  Only the policy output and its immediate
action-conditioning one-hot are compacted to the current patch's 112
executable Zerg actions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distar-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.agent.default.lib.current_patch_actions import LEGACY_ACTION_INDICES
    from distar.agent.default.model.model import Model
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import (
        NATIVE_POLICY_ACTION_ENCODING,
        contract_hash,
    )

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    cfg = read_config(str(args.distar_root / "distar/bin/sl_user_config.yaml"))
    cfg.common.type = "sl"
    cfg.native_zerg_action_head = True
    target = Model(cfg).state_dict()
    source_state = source["model"]
    index = torch.tensor(LEGACY_ACTION_INDICES, dtype=torch.long)

    for name, tensor in target.items():
        if name in {
            "policy.action_type_head.action_fc.layer2.0.weight",
            "policy.action_type_head.action_fc.layer2.0.bias",
            "policy.action_type_head.action_map_fc1.0.weight",
        }:
            continue
        old = source_state.get(name)
        if old is None or old.shape != tensor.shape:
            raise RuntimeError(f"unexpected incompatible tensor: {name}")
        target[name] = old

    target["policy.action_type_head.action_fc.layer2.0.weight"] = source_state[
        "policy.action_type_head.action_fc.layer2.0.weight"
    ].index_select(0, index)
    target["policy.action_type_head.action_fc.layer2.0.bias"] = source_state[
        "policy.action_type_head.action_fc.layer2.0.bias"
    ].index_select(0, index)
    target["policy.action_type_head.action_map_fc1.0.weight"] = source_state[
        "policy.action_type_head.action_map_fc1.0.weight"
    ].index_select(1, index)

    output = {
        "model": target,
        "last_iter": 0,
        "current_patch_contract_hash": contract_hash(NATIVE_POLICY_ACTION_ENCODING),
        "policy_action_encoding": NATIVE_POLICY_ACTION_ENCODING,
        "resumed_from": str(args.source.resolve()),
        "native_zerg_action_head": {
            "action_count": len(LEGACY_ACTION_INDICES),
            "local_to_released_action_id": list(LEGACY_ACTION_INDICES),
            "copied_from_unfine_tuned_source": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print({"output": str(args.output), "actions": len(LEGACY_ACTION_INDICES),
           "contract": output["current_patch_contract_hash"],
           "resumed_from": output["resumed_from"]})


if __name__ == "__main__":
    main()
