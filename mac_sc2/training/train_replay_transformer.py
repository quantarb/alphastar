#!/usr/bin/env python3
"""Train a compact next-action Transformer directly from SC2 replay events.

This is deliberately Mac-native: it uses s2protocol to read retail replay game
events and PyTorch for training.  It does not require AlphaStar's Linux-only
PySC2 C++ converter.
"""
import argparse
from collections import Counter
from pathlib import Path
import random

from mpyq import MPQArchive
from s2protocol import versions
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PAD, BOS, UNK = 0, 1, 2


def replay_action_sequences(replay_paths, max_replays):
    """Return per-player sequences of SC2 ability-link integers."""
    result = []
    for replay_path in list(replay_paths)[:max_replays]:
        try:
            archive = MPQArchive(str(replay_path))
            header = versions.latest().decode_replay_header(
                archive.header['user_data_header']['content'])
            protocol = versions.build(header['m_version']['m_baseBuild'])
            by_user = {}
            for event in protocol.decode_replay_game_events(
                    archive.read_file('replay.game.events')):
                if event.get('_event') != 'NNet.Game.SCmdEvent':
                    continue
                ability = event.get('m_abil')
                user = event.get('_userid', {}).get('m_userId')
                if ability is not None and user in (0, 1):
                    by_user.setdefault(user, []).append(ability['m_abilLink'])
            result.extend(seq for seq in by_user.values() if len(seq) >= 8)
        except Exception as exc:  # A corrupt replay should not stop a dataset job.
            print(f'Skipping {replay_path.name}: {exc}')
    return result


def make_examples(sequences, vocab, context, limit):
    encoded = [[vocab.get(a, UNK) for a in seq] for seq in sequences]
    examples, labels = [], []
    for seq in encoded:
        for index in range(1, len(seq)):
            prefix = ([BOS] + seq[:index])[-context:]
            examples.append([PAD] * (context - len(prefix)) + prefix)
            labels.append(seq[index])
            if len(examples) >= limit:
                return torch.tensor(examples), torch.tensor(labels)
    return torch.tensor(examples), torch.tensor(labels)


class ReplayTransformer(nn.Module):
    def __init__(self, vocab_size, context, width=96, layers=2, heads=4):
        super().__init__()
        self.token = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.position = nn.Embedding(context, width)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=width * 3,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None, :, :]
        hidden = self.encoder(hidden, src_key_padding_mask=tokens.eq(PAD))
        return self.head(hidden[:, -1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replays', default='local_data/replays/4.9.2/sample')
    parser.add_argument('--output', default='mac_sc2/artifacts/replay_transformer.pt')
    parser.add_argument('--max-replays', type=int, default=3)
    parser.add_argument('--max-vocab', type=int, default=64)
    parser.add_argument('--context', type=int, default=16)
    parser.add_argument('--max-examples', type=int, default=6000)
    parser.add_argument('--epochs', type=int, default=4)
    args = parser.parse_args()

    paths = sorted(Path(args.replays).rglob('*.SC2Replay'))
    if not paths:
        raise SystemExit(f'No .SC2Replay files under {args.replays}')
    sequences = replay_action_sequences(paths, args.max_replays)
    counts = Counter(a for seq in sequences for a in seq)
    frequent = [ability for ability, _ in counts.most_common(args.max_vocab)]
    vocab = {ability: index + 3 for index, ability in enumerate(frequent)}
    features, labels = make_examples(sequences, vocab, args.context, args.max_examples)
    if len(features) < 8:
        raise SystemExit('Not enough command events to train.')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Replays={len(paths[:args.max_replays])}, player sequences={len(sequences)}, '
          f'examples={len(features)}, vocab={len(vocab) + 3}, device={device}')
    loader = DataLoader(TensorDataset(features, labels), batch_size=64, shuffle=True)
    model = ReplayTransformer(len(vocab) + 3, args.context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    for epoch in range(1, args.epochs + 1):
        model.train(); total_loss = total_correct = total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * len(batch_y)
            total_correct += logits.argmax(-1).eq(batch_y).sum().item()
            total += len(batch_y)
        print(f'epoch {epoch}: loss={total_loss / total:.4f}, accuracy={total_correct / total:.3f}')
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'vocab': vocab,
                'context': args.context, 'architecture': '96d/2-layer/4-head Transformer'}, output)
    print(f'Saved checkpoint: {output.resolve()}')


if __name__ == '__main__':
    random.seed(7); torch.manual_seed(7); main()
