#!/usr/bin/env python3
"""Fine-tune DI-star's released Zerg policy on current-patch raw replays."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distar-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment", default="distar_5_0_16_zerg_finetune")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--updates", type=int, default=200,
                        help="bounded supervised updates; checkpoint remains overwrite-only")
    parser.add_argument("--preprocessed-cache-dir", type=Path,
                        default=Path("local_data/preprocessed_distar_5_0_16_zerg_winners"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-zerg-action-head", action="store_true")
    parser.add_argument("--native-race", choices=("zerg", "terran", "protoss"))
    parser.add_argument("--multi-race-action-heads", action="store_true",
                        help="train one task-local projection inside a shared three-race checkpoint")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.agent.default.sl_learner import SLLearner
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import NATIVE_POLICY_ACTION_ENCODING, contract_hash

    cfg = read_config(str(args.distar_root / "distar/bin/sl_user_config.yaml"))
    cfg.common.type = "sl"
    cfg.common.experiment_name = args.experiment
    native_race = args.native_race or ("zerg" if args.native_zerg_action_head else None)
    cfg.native_zerg_action_head = native_race == "zerg"
    cfg.native_action_race = native_race
    cfg.multi_race_action_heads = args.multi_race_action_heads
    if args.multi_race_action_heads:
        from distar.current_patch_contract import multi_race_contract_hash
        cfg.policy_action_encoding = "multi_race_native_action_heads_v1"
        cfg.current_patch_contract_hash = multi_race_contract_hash()
    elif native_race in ("terran", "protoss"):
        from distar.current_patch_contract import legacy_race_contract_hash
        cfg.policy_action_encoding = f"native_current_patch_{native_race}_action_head"
        cfg.current_patch_contract_hash = legacy_race_contract_hash(native_race)
    else:
        cfg.policy_action_encoding = (NATIVE_POLICY_ACTION_ENCODING if native_race == "zerg"
                                      else "released_327_id_with_current_zerg_mask")
        cfg.current_patch_contract_hash = contract_hash(cfg.policy_action_encoding)
    cfg.checkpoint_overwrite_path = str(args.output.resolve())
    cfg.learner.use_cuda = False
    cfg.learner.load_path = str(args.checkpoint.resolve())
    cfg.learner.load_optimizer = False
    cfg.learner.load_last_iter = False
    cfg.learner.freeze_except_action_logits = True
    cfg.learner.max_iterations = args.updates
    cfg.learner.data.train_data_file = str(args.manifest.resolve())
    # The cached feature extractor filters event ownership with this value;
    # keeping it aligned with the compact action head prevents cross-race
    # labels from ever reaching the learner.
    race_code = {"zerg": "Z", "terran": "T", "protoss": "P"}
    cfg.learner.data.parse_race = [race_code.get(native_race, "Z")]
    cfg.current_patch_race = native_race or "zerg"
    cfg.learner.data.epochs = args.epochs
    cfg.learner.data.num_workers = args.workers
    cfg.learner.data.batch_size = args.batch_size
    cfg.learner.data.preprocessed_cache_dir = str(args.preprocessed_cache_dir.resolve())
    # This 92-trajectory run needs an early playable snapshot.  The DI-star
    # hook overwrites one stable path rather than retaining a checkpoint per
    # interval.
    cfg.learner.hook.save_ckpt_after_iter.ext_args.freq = 50
    print({"resumed_from": cfg.learner.load_path, "manifest": cfg.learner.data.train_data_file,
           "epochs": args.epochs, "updates": args.updates, "contract": cfg.current_patch_contract_hash,
           "trainable": "policy.action_type_head.action_fc.layer2",
           "native_action_race": native_race, "multi_race_action_heads": args.multi_race_action_heads,
           "preprocessed_cache_dir": cfg.learner.data.preprocessed_cache_dir}, flush=True)
    SLLearner(cfg, "single_node").run()


if __name__ == "__main__":
    main()
