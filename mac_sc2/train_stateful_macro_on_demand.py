#!/usr/bin/env python3
"""Train macro BC directly from raw replays, retaining no derived dataset.

Each label is one meaningful macro command in an 8-second window.  State comes
from replay PlayerStats plus per-player unit lifecycle counts, rather than from
the previous selection-history proxy.
"""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
import sc2reader, torch
from torch.nn import functional as F
from stateful_macro_policy import StatefulMacroPolicy, RACES, STATE_SIZE

RACE_ID = {r: i for i, r in enumerate(RACES)}
WORKERS = ("scv", "probe", "drone")
SUPPLY = ("supplydepot", "pylon", "overlord")
PRODUCTION = ("barracks", "gateway", "spawningpool", "factory", "cybernetics", "stargate", "robo", "hydraliskden")
ARMY = ("marine", "zealot", "zergling", "roach", "stalker", "marauder", "hellion", "immortal", "mutalisk")
TOWNHALL = ("commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive")

def macro(name):
    n = (name or "").lower()
    if "attack" in n or "scanmove" in n: return 5
    if any(x in n for x in TOWNHALL) or "expand" in n: return 4
    if any(x in n for x in SUPPLY): return 1
    if any(x in n for x in PRODUCTION): return 2
    if any(x in n for x in WORKERS): return 0
    if any(x in n for x in ARMY): return 3
    return None

def category(unit):
    n = (unit or "").lower()
    return [any(x in n for x in group) for group in (WORKERS, SUPPLY, PRODUCTION, ARMY, TOWNHALL)]

def vector(stats, counts, second):
    minerals = getattr(stats, "minerals_current", 0)
    gas = getattr(stats, "vespene_current", 0)
    used = getattr(stats, "food_used", 0)
    made = getattr(stats, "food_made", 0)
    workers = max(getattr(stats, "workers_active_count", 0), counts[0])
    return [min(second / 900, 1), min(minerals / 1500, 1), min(gas / 1000, 1),
            min(used / 200, 1), min(made / 200, 1), min(max(made-used, 0) / 30, 1),
            min(workers / 80, 1), min(counts[1] / 12, 1), min(counts[2] / 12, 1),
            min(counts[3] / 80, 1), min(counts[4] / 8, 1),
            min(getattr(stats, "minerals_collection_rate", 0) / 2500, 1),
            min(getattr(stats, "vespene_collection_rate", 0) / 1500, 1),
            min(getattr(stats, "resources_lost", 0) / 10000, 1)]

def examples(path, window=8):
    replay = sc2reader.load_replay(path, load_level=4)
    races = {p.pid: RACE_ID.get(p.play_race) for p in replay.players}
    latest, counts, emitted = {}, defaultdict(lambda: [0]*5), defaultdict(lambda: -1)
    out = []
    for event in replay.events:
        pid = getattr(event, "pid", getattr(event, "control_pid", None))
        if pid not in races or races[pid] is None: continue
        name = type(event).__name__
        if name == "PlayerStatsEvent": latest[pid] = event; continue
        if name in ("UnitBornEvent", "UnitInitEvent"):
            for i, hit in enumerate(category(getattr(event, "unit_type_name", ""))): counts[pid][i] += int(hit)
            continue
        if name == "UnitDiedEvent":
            owner = getattr(event, "unit", None)
            owner_pid = getattr(owner, "owner", None)
            if owner_pid in races:
                for i, hit in enumerate(category(getattr(event, "unit_type_name", ""))): counts[owner_pid][i] = max(0, counts[owner_pid][i] - int(hit))
            continue
        if "CommandEvent" not in name: continue
        label = macro(getattr(event, "ability_name", ""))
        if label is None or pid not in latest: continue
        bucket = int(getattr(event, "second", 0) // window)
        if bucket == emitted[pid]: continue
        emitted[pid] = bucket
        out.append((races[pid], vector(latest[pid], counts[pid], getattr(event, "second", 0)), label))
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest', required=True); ap.add_argument('--output', required=True); ap.add_argument('--max-games', type=int, default=1000); ap.add_argument('--batch-size', type=int, default=512); ap.add_argument('--window', type=int, default=8); args=ap.parse_args()
    files=json.loads(Path(args.manifest).read_text())['valid'][:args.max_games]
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=StatefulMacroPolicy().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01)
    labels=Counter(); seen=correct=0
    for gi,item in enumerate(files,1):
        try: rows=examples(item['path'],args.window)
        except Exception as exc: print(f'skip game={gi} {type(exc).__name__}',flush=True); continue
        if not rows: continue
        labels.update(y for _,_,y in rows)
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size]; state=torch.tensor([x for _,x,_ in batch],dtype=torch.float32,device=dev); race=torch.tensor([r for r,_,_ in batch],device=dev); target=torch.tensor([y for _,_,y in batch],device=dev)
            # Recomputed from seen labels: rare strategic actions remain learnable.
            weight=torch.tensor([max(seen,1)/(7*max(labels[i],1)) for i in range(7)],device=dev).clamp(max=12)
            logits=model(state,race); loss=F.cross_entropy(logits,target,weight=weight); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step()
            seen += len(batch); correct += logits.argmax(-1).eq(target).sum().item()
        if gi%25==0: print(f'games={gi} decisions={seen} accuracy={correct/max(seen,1):.3f} labels={dict(labels)}',flush=True)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.cpu().state_dict(),'games':args.max_games,'decisions':seen,'labels':dict(labels),'architecture':'stateful replay macro BC, class-balanced'},args.output)
    print(f'saved={args.output} decisions={seen} labels={dict(labels)}',flush=True)
if __name__=='__main__': main()
