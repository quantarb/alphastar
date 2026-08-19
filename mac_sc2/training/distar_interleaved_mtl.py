#!/usr/bin/env python3
"""Interleaved three-race supervised MTL for DI-star compact action heads."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import torch


def reset_hidden(hidden, new_episodes):
    for layer in range(len(hidden)):
        h, c = hidden[layer]
        h = h.clone().detach(); c = c.clone().detach()
        for index, is_new in enumerate(new_episodes):
            if is_new:
                h[index] *= 0; c[index] *= 0
        hidden[layer] = (h, c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--distar-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--zerg-manifest', type=Path, required=True)
    parser.add_argument('--terran-manifest', type=Path, required=True)
    parser.add_argument('--protoss-manifest', type=Path, required=True)
    parser.add_argument('--zerg-cache', type=Path, required=True)
    parser.add_argument('--terran-cache', type=Path, required=True)
    parser.add_argument('--protoss-cache', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=600,
                        help='bounded interleaved updates; ignored by --full-epochs')
    parser.add_argument('--full-epochs', type=int, default=0,
                        help='consume every trajectory window this many times per race')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--races', nargs='+', choices=('zerg', 'terran', 'protoss'),
                        default=('zerg', 'terran', 'protoss'),
                        help='Task races to train; keeps the same shared MTL model format.')
    args = parser.parse_args()
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.agent.default.model.model import Model
    from distar.agent.default.sl_training.sl_dataloader import SLDataloader
    from distar.agent.default.sl_training.sl_loss import SupervisedLoss
    from distar.agent.default.lib.current_patch_actions import race_legacy_to_current
    from distar.ctools.utils import read_config
    from distar.ctools.torch_utils.grad_clip import build_grad_clip
    from distar.ctools.torch_utils.lr_scheduler_util import GradualWarmupScheduler
    from distar.current_patch_contract import multi_race_contract_hash

    base = read_config(str(args.distar_root / 'distar/bin/sl_user_config.yaml'))
    base.common.type = 'sl'; base.learner.use_cuda = False
    base.multi_race_action_heads = True
    base.learner.data.batch_size = args.batch_size
    base.learner.data.num_workers = args.workers
    base.learner.data.epochs = args.full_epochs or 1
    spec = {
        'zerg': (args.zerg_manifest, args.zerg_cache, 'Z'),
        'terran': (args.terran_manifest, args.terran_cache, 'T'),
        'protoss': (args.protoss_manifest, args.protoss_cache, 'P'),
    }
    loaders = {}
    selected_races = tuple(args.races)
    for race in selected_races:
        manifest, cache, code = spec[race]
        cfg = deepcopy(base)
        cfg.current_patch_race = race; cfg.native_action_race = race
        cfg.learner.data.train_data_file = str(manifest.resolve())
        cfg.learner.data.preprocessed_cache_dir = str(cache.resolve())
        cfg.learner.data.parse_race = [code]
        loaders[race] = iter(SLDataloader(cfg))
    cfg = deepcopy(base)
    cfg.current_patch_race = 'zerg'; cfg.native_action_race = 'zerg'
    model = Model(cfg)
    loaded = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if loaded.get('current_patch_contract_hash') != multi_race_contract_hash():
        raise RuntimeError('initializer ActionSpec mismatch')
    model.load_state_dict(loaded['model'], strict=True)
    trainable = [p for name, p in model.named_parameters()
                 if name.endswith('.layer2.0.weight') or name.endswith('.layer2.0.bias')]
    # Only the three named compact action-logit heads are trainable.
    trainable = [p for name, p in model.named_parameters()
                 if name.startswith('policy.action_type_head.action_heads.') and
                 (name.endswith('.layer2.0.weight') or name.endswith('.layer2.0.bias'))]
    for p in model.parameters(): p.requires_grad_(False)
    for p in trainable: p.requires_grad_(True)
    # Match DI-star SLLearner exactly.  The released STL Zerg fine-tune used
    # this long warmup, six-batch initial hold, and momentum-norm clipping;
    # applying 1e-3 from step one was the source of the MTL regression.
    optimizer = torch.optim.Adam(trainable, lr=base.learner.learning_rate,
                                 weight_decay=base.learner.weight_decay)
    after_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(range(base.learner.lr_decay_interval,
                              base.learner.lr_decay_interval * 40,
                              base.learner.lr_decay_interval)),
        gamma=base.learner.lr_decay,
    )
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=base.learner.get('multiplier', 1),
        total_epoch=base.learner.warm_up_steps,
        after_scheduler=after_scheduler,
    )
    grad_clip = build_grad_clip(base.learner.grad_clip)
    ignored_updates = 0
    loss_fn = SupervisedLoss(cfg)
    hidden_size, layers = model.cfg.encoder.core_lstm.hidden_size, model.cfg.encoder.core_lstm.num_layers
    hidden = {race: [(torch.zeros(args.batch_size, hidden_size), torch.zeros(args.batch_size, hidden_size))
                     for _ in range(layers)] for race in selected_races}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    order = selected_races
    active = set(order)
    step = 0
    cursor = 0
    target_updates = None if args.full_epochs else args.updates
    while active and (target_updates is None or step < target_updates):
        race = order[cursor % len(order)]
        cursor += 1
        if race not in active:
            continue
        try:
            data = next(loaders[race])
        except StopIteration:
            active.remove(race)
            print({'race_epoch_complete': race, 'remaining_races': sorted(active)}, flush=True)
            continue
        step += 1
        cfg.current_patch_race = race; cfg.native_action_race = race
        model.whole_cfg.current_patch_race = race; model.whole_cfg.native_action_race = race
        reset_hidden(hidden[race], data.pop('new_episodes'))
        logits, infer, next_hidden = model.sl_train(**data, hidden_state=hidden[race])
        labels = dict(data['action_info'])
        labels['action_type'] = race_legacy_to_current(labels['action_type'], race)
        stats = loss_fn.compute_loss(logits, labels, data['action_mask'], data['selected_units_num'],
                                     data['entity_num'], infer)
        if ignored_updates > 5:
            optimizer.zero_grad()
            stats['total_loss'].backward()
            grad_clip.apply(model.parameters())
            optimizer.step()
            scheduler.step()
        ignored_updates += 1
        hidden[race] = [(h.detach(), c.detach()) for h, c in next_hidden]
        if step % 25 == 0 or step == 1:
            print({'update': step, 'race': race, 'loss': round(float(stats['total_loss']), 4)}, flush=True)
        if step % 50 == 0 or (target_updates is not None and step == target_updates):
            torch.save({'model': model.state_dict(), 'last_iter': step,
                        'current_patch_contract_hash': multi_race_contract_hash(),
                        'policy_action_encoding': 'multi_race_native_action_heads_v1',
                        'multi_race_action_heads': True, 'interleaved_order': list(order),
                        'trajectories': {'zerg': 21, 'terran': 12, 'protoss': 13},
                        'full_epochs': args.full_epochs,
                        'optimizer': optimizer.state_dict(),
                        'optimizer_schedule': 'distar_sl_warmup_20000_momentum_norm',
                        'resumed_from': str(args.checkpoint.resolve())}, args.output)
            print({'checkpoint': str(args.output.resolve()), 'update': step}, flush=True)


if __name__ == '__main__':
    main()
