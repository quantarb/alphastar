#!/usr/bin/env python3
"""True replay-by-replay patch/race MTL: no catalog pass and no shards."""
import argparse, json, time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.legacy.patch_race_mtl_policy import PatchRaceMTLPolicy
from mac_sc2.training.train_patch_race_mtl_on_demand import patch_family, replay_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True); ap.add_argument('--output', required=True)
    ap.add_argument('--patches', required=True, help='Comma-separated patch families')
    ap.add_argument('--max-games', type=int, default=1000); ap.add_argument('--window', type=int, default=8)
    ap.add_argument('--batch-size', type=int, default=512); ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--max-actions', type=int, default=160); ap.add_argument('--winner-only', action='store_true')
    ap.add_argument('--status-file', help='JSON status updated every 25 replays')
    a = ap.parse_args()
    selected = set(a.patches.split(','))
    files = [x for x in json.loads(Path(a.manifest).read_text())['valid'] if patch_family(x['version']) in selected][:a.max_games]
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = PatchRaceMTLPolicy({}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    seen, correct = Counter(), Counter()
    started = time.time()
    def publish(game, state):
        if not a.status_file: return
        payload = {'state': state, 'games_completed': game, 'games_total': len(files),
                   'tasks': len(model.task_vocabs), 'decisions': sum(seen.values()),
                   'accuracy': sum(correct.values()) / max(sum(seen.values()), 1),
                   'elapsed_seconds': round(time.time() - started),
                   'task_names': sorted(model.task_vocabs)}
        Path(a.status_file).write_text(json.dumps(payload))
    publish(0, 'training')
    for game, item in enumerate(files, 1):
        try: rows = replay_rows(item, a.window, a.winner_only)
        except Exception as exc: print(f'skip game={game} {type(exc).__name__}', flush=True); continue
        grouped = defaultdict(list)
        for task, state, label in rows: grouped[task].append((state, label))
        changed = False
        for task, data in grouped.items(): changed |= model.ensure_labels(task, [label for _, label in data], a.max_actions)
        if changed:
            # Recreate optimizer only when an output layer grows; this permits
            # gradient updates to begin with the very first replay.
            optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
        for task, data in grouped.items():
            ids = {label: i for i, label in enumerate(model.task_vocabs[task])}
            data = [(state, ids[label]) for state, label in data if label in ids]
            for start in range(0, len(data), a.batch_size):
                batch = data[start:start + a.batch_size]
                x = torch.tensor([state for state, _ in batch], dtype=torch.float32, device=device)
                y = torch.tensor([label for _, label in batch], dtype=torch.long, device=device)
                logits = model.forward_task(x, task); loss = F.cross_entropy(logits, y)
                optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
                seen[task] += len(batch); correct[task] += logits.argmax(-1).eq(y).sum().item()
        if game % 25 == 0:
            print(f'games={game}/{len(files)} tasks={len(model.task_vocabs)} decisions={sum(seen.values())} acc={sum(correct.values()) / max(sum(seen.values()), 1):.3f}', flush=True)
            publish(game, 'training')
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': model.cpu().state_dict(), 'task_vocabs': model.task_vocabs,
                'decisions_by_task': dict(seen), 'winner_only': a.winner_only,
                'architecture': 'streaming shared backbone + online patch/race heads'}, a.output)
    publish(len(files), 'complete')
    print(f'saved={a.output} tasks={len(model.task_vocabs)} decisions={sum(seen.values())}', flush=True)


if __name__ == '__main__': main()
