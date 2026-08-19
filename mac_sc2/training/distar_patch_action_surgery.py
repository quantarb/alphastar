#!/usr/bin/env python3
"""Convert released DI-star Zerg weights to the 5.0.16 action vocabulary."""
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
    from distar.agent.default.model.model import Model
    from distar.agent.default.lib.actions import (
        BEGINNING_ORDER_ACTIONS as LEGACY_BEGINNING_ORDER_ACTIONS,
        CUMULATIVE_STAT_ACTIONS as LEGACY_CUMULATIVE_STAT_ACTIONS,
    )
    from distar.agent.default.lib.current_patch_actions import (
        BEGINNING_ORDER_ACTIONS,
        CUMULATIVE_STAT_ACTIONS,
        LEGACY_ACTION_INDICES,
    )
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import contract_hash

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    source_state = source["model"]
    model_cfg = read_config(str(args.distar_root / "distar/bin/sl_user_config.yaml"))
    model_cfg.common.type = "sl"
    target = Model(model_cfg).state_dict()
    index = torch.tensor(LEGACY_ACTION_INDICES, dtype=torch.long)
    transferred, missing = [], []
    for name, tensor in target.items():
        old = source_state.get(name)
        if old is not None and old.shape == tensor.shape:
            target[name] = old
            transferred.append(name)
        else:
            missing.append(name)

    # The only changed learned dimensions are the action label embedding, the
    # classifier rows, and the action-conditioning matrix columns.
    row_keys = [
        "encoder.scalar_encoder.encode_modules.last_action_type.weight",
        "policy.action_type_head.action_fc.layer2.0.weight",
        "policy.action_type_head.action_fc.layer2.0.bias",
    ]
    for name in row_keys:
        target[name] = source_state[name].index_select(0, index)
        missing.remove(name)
    name = "policy.action_type_head.action_map_fc1.0.weight"
    target[name] = source_state[name].index_select(1, index)
    missing.remove(name)
    cumulative_index = torch.tensor(
        [LEGACY_CUMULATIVE_STAT_ACTIONS.index(LEGACY_ACTION_INDICES[i])
         for i in CUMULATIVE_STAT_ACTIONS], dtype=torch.long
    )
    name = "encoder.scalar_encoder.encode_modules.cumulative_stat.0.weight"
    target[name] = source_state[name].index_select(1, cumulative_index)
    missing.remove(name)
    beginning_index = torch.tensor(
        [LEGACY_BEGINNING_ORDER_ACTIONS.index(LEGACY_ACTION_INDICES[i])
         for i in BEGINNING_ORDER_ACTIONS], dtype=torch.long
    )
    name = "encoder.scalar_encoder.encode_modules.beginning_order.action_one_hot.weight"
    target[name] = source_state[name].index_select(0, beginning_index).index_select(1, beginning_index)
    missing.remove(name)
    name = "encoder.scalar_encoder.encode_modules.beginning_order.transformer.embedding.0.weight"
    old = source_state[name]
    action_columns = old[:, :len(LEGACY_BEGINNING_ORDER_ACTIONS)].index_select(1, beginning_index)
    target[name] = torch.cat((action_columns, old[:, len(LEGACY_BEGINNING_ORDER_ACTIONS):]), dim=1)
    missing.remove(name)
    # Queue-order embeddings have a legacy, inconsistent 49-way layout.  They
    # are input adapters, not executable outputs; retain their fresh 5.0.16
    # initialization rather than inventing a false row correspondence.
    for name in [
        "encoder.entity_encoder.encode_modules.order_id_0.weight",
        "encoder.entity_encoder.encode_modules.order_id_1.weight",
        "encoder.entity_encoder.encode_modules.order_id_2.weight",
        "encoder.entity_encoder.encode_modules.order_id_3.weight",
    ]:
        if name != "encoder.entity_encoder.encode_modules.order_id_0.weight":
            missing.remove(name)
    name = "encoder.entity_encoder.encode_modules.order_id_0.weight"
    target[name] = source_state[name].index_select(0, index).index_select(1, index)
    missing.remove(name)
    # This is the entity input projection.  Its columns concatenate the four
    # resized action/order one-hots, so no row-wise legacy correspondence is
    # sound; preserve the downstream transformer and initialize this adapter
    # for the new 5.0.16 feature width.
    missing.remove("encoder.entity_encoder.transformer.embedding.0.weight")
    if missing:
        raise RuntimeError(f"unexpected incompatible tensors: {missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": target,
        "last_iter": 0,
        "current_patch_contract_hash": contract_hash(),
        "resumed_from": str(args.source.resolve()),
        "action_head_surgery": {
            "legacy_action_count": 327,
            "current_patch_action_count": len(LEGACY_ACTION_INDICES),
            "legacy_indices": list(LEGACY_ACTION_INDICES),
        },
    }, args.output)
    print({"output": str(args.output), "contract": contract_hash(),
           "actions": len(LEGACY_ACTION_INDICES), "transferred": len(transferred) + 4})


if __name__ == "__main__":
    main()
