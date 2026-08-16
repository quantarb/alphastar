#!/usr/bin/env python3
"""Train and evaluate a command-sequence Transformer on 1,000 current pro replays.

This is an honest first behavioural-cloning stage: it learns the next *raw
ability command* from past commands.  It does not yet reconstruct game state,
unit selections, targets, or turn it into a playable policy; those are the
next data-extraction stages.  Replay-level splitting avoids leakage.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import sc2reader

from mac_sc2.training.train_replay_transformer import (
    BOS, PAD, UNK, ReplayTransformer, make_examples,
)

ROOT = Path(__file__).resolve().parents[1]


def replay_action_sequences(paths):
    """Extract actual player command abilities using the 5.0.16-capable reader.

    The older `s2protocol` package bundled in this environment stops at older
    game builds.  sc2reader decodes the installed 5.0.16 replay builds and
    exposes ability_id on command events, which is sufficient for this first
    command-history behavioural-cloning stage.
    """
    sequences = []
    for path in paths:
        try:
            replay = sc2reader.load_replay(str(path), load_level=4)
            player_ids = {player.pid for player in replay.players}
            by_player = {pid: [] for pid in player_ids}
            for event in replay.events:
                if 'CommandEvent' not in type(event).__name__:
                    continue
                ability = getattr(event, 'ability_id', None)
                if event.pid in by_player and ability is not None:
                    by_player[event.pid].append(ability)
            sequences.extend(sequence for sequence in by_player.values() if len(sequence) >= 8)
        except Exception as error:
            print(f'skipping={path.name} error={type(error).__name__}', flush=True)
    return sequences


def run_epoch(model, loader, optimizer, device):
    train = optimizer is not None
    model.train(train)
    loss_total = correct = total = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if train:
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_total += loss.item() * len(y)
            correct += logits.argmax(-1).eq(y).sum().item()
            total += len(y)
    return loss_total / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='local_data/current_replays/manifest_spawningtool_pro_2026_5_0_16.json')
    parser.add_argument('--games', type=int, default=1000)
    parser.add_argument('--context', type=int, default=32)
    parser.add_argument('--max-vocab', type=int, default=512)
    parser.add_argument('--max-train-examples', type=int, default=200000)
    parser.add_argument('--max-validation-examples', type=int, default=40000)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--output', default='mac_sc2/artifacts/current_patch_1000_pro_action_transformer.pt')
    args = parser.parse_args()

    manifest = json.loads((ROOT / args.manifest).read_text())
    paths = [Path(row['path']) for row in manifest['valid']][:args.games]
    if len(paths) < args.games:
        raise SystemExit(f'Need {args.games} validated games, found {len(paths)} in {args.manifest}')
    split = int(len(paths) * 0.9)
    train_sequences = replay_action_sequences(paths[:split])
    validation_sequences = replay_action_sequences(paths[split:])
    counts = Counter(ability for sequence in train_sequences for ability in sequence)
    vocab = {ability: index + 3 for index, (ability, _) in enumerate(counts.most_common(args.max_vocab))}
    train_x, train_y = make_examples(train_sequences, vocab, args.context, args.max_train_examples)
    validation_x, validation_y = make_examples(validation_sequences, vocab, args.context, args.max_validation_examples)
    if not len(train_x) or not len(validation_x):
        raise SystemExit('No usable command examples were extracted.')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'games={len(paths)} train_games={split} heldout_games={len(paths)-split} '
          f'train_examples={len(train_x)} validation_examples={len(validation_x)} '
          f'vocab={len(vocab)+3} device={device}', flush=True)
    model = ReplayTransformer(len(vocab) + 3, args.context, width=192, layers=4, heads=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True)
    validation_loader = DataLoader(TensorDataset(validation_x, validation_y), batch_size=256)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, optimizer, device)
        validation_loss, validation_accuracy = run_epoch(model, validation_loader, None, device)
        metrics = {'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                   'validation_loss': validation_loss, 'validation_accuracy': validation_accuracy}
        history.append(metrics); print(metrics, flush=True)
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'vocab': vocab, 'context': args.context,
                'games': len(paths), 'manifest': str(ROOT / args.manifest), 'history': history,
                'architecture': '192d/4-layer/6-head command Transformer',
                'limitations': 'command history only; not executable without state/selection/target heads'}, output)
    print(f'saved={output}', flush=True)


if __name__ == '__main__':
    torch.manual_seed(7); main()
