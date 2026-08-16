#!/usr/bin/env python3
"""Stream cross-patch semantic BC into a macro-backbone transfer policy."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path

import sc2reader
import torch
from torch.nn import functional as F

from mac_sc2.legacy.multirace_general_policy import MultiRaceGeneralMacroPolicy, RACES
from mac_sc2.contracts.semantic_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, TARGET_KINDS, from_event
from mac_sc2.legacy.semantic_transfer_policy import SemanticTransferPolicy, metadata
from mac_sc2.training.train_general_macro_on_demand import RID, cat, event_pid, vec
from mac_sc2.legacy.semantic_action_contract import spec_hash, supports

INDEX = {"actor": {x: i for i, x in enumerate(ACTOR_ROLES)}, "family": {x: i for i, x in enumerate(FAMILIES)},
         "payload": {x: i for i, x in enumerate(PAYLOAD_ROLES)}, "target": {x: i for i, x in enumerate(TARGET_KINDS)}}


def rows(path, patch):
    replay = sc2reader.load_replay(path, load_level=4)
    races = {p.pid: RID.get(p.play_race) for p in replay.players}
    latest, counts, selected, groups, out = {}, defaultdict(lambda: [0] * 8), {}, defaultdict(dict), []
    for event in replay.events:
        pid = event_pid(event)
        if pid not in races or races[pid] is None:
            continue
        typ = type(event).__name__
        if typ == "PlayerStatsEvent": latest[pid] = event; continue
        if typ in ("UnitBornEvent", "UnitInitEvent"):
            q = cat(getattr(event, "unit_type_name", "")); counts[pid] = [a + b for a, b in zip(counts[pid], q)]; continue
        if typ == "SelectionEvent":
            selected[pid] = [str(unit) for unit in (getattr(event, "objects", []) or [])]; continue
        if "ControlGroupEvent" in typ:
            group = getattr(event, "control_group", 0); current = selected.get(pid, [])
            if typ == "SetControlGroupEvent": groups[pid][group] = list(current)
            elif typ == "AddToControlGroupEvent": groups[pid][group] = list(dict.fromkeys(groups[pid].get(group, []) + current))
            elif typ == "GetControlGroupEvent" and groups[pid].get(group): selected[pid] = list(groups[pid][group])
            continue
        if "CommandEvent" not in typ or pid not in latest:
            continue
        action = from_event(event, patch, replay.players[pid - 1].play_race or "Unknown", selected.get(pid, []))
        if action.ability_name.lower() == "cabil": continue
        if not supports(action.actor_role, action.family, action.payload_role, action.target_kind):
            continue
        out.append((races[pid], vec(latest[pid], counts[pid], getattr(event, "second", 0)),
                    INDEX["actor"][action.actor_role], INDEX["family"][action.family],
                    INDEX["payload"][action.payload_role], INDEX["target"][action.target_kind], int(action.queued)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True); parser.add_argument("--macro-checkpoint", required=True)
    parser.add_argument("--resume", default="mac_sc2/artifacts/semantic_contract_all_replays.pt",
                        help="Compatible semantic checkpoint to fine-tune; random initialization is not permitted")
    parser.add_argument("--output", required=True); parser.add_argument("--max-games", type=int, default=36910)
    parser.add_argument("--batch-size", type=int, default=512); parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint-every", type=int, default=200)
    args = parser.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    base = MultiRaceGeneralMacroPolicy(); base.load_state_dict(torch.load(args.macro_checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    model = SemanticTransferPolicy(base.shared)
    resume = torch.load(args.resume, map_location="cpu", weights_only=False)
    if resume.get("action_contract_hash") != spec_hash():
        raise ValueError("resume checkpoint ActionSpec is incompatible with the live semantic decoder")
    model.load_state_dict(resume["state_dict"])
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    files = json.loads(Path(args.manifest).read_text())["valid"][:args.max_games]
    seen, correct = 0, Counter()
    def save(games):
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": state, "games": games, "macro_checkpoint": args.macro_checkpoint, "resumed_from": args.resume,
                    "decisions": seen, "action_contract_hash": spec_hash(),
                    "training_source": "raw replay semantic factors streamed on demand", **metadata()}, args.output)
    for game, item in enumerate(files, 1):
        patch = ".".join(item["version"].split(".")[:3])
        try: data = rows(item["path"], patch)
        except Exception as exc: print(f"skip game={game} {type(exc).__name__}", flush=True); continue
        for start in range(0, len(data), args.batch_size):
            batch = data[start:start + args.batch_size]
            state = torch.tensor([x[1] for x in batch], dtype=torch.float32, device=device)
            race = torch.tensor([x[0] for x in batch], dtype=torch.long, device=device)
            labels = {"actor": torch.tensor([x[2] for x in batch], device=device), "family": torch.tensor([x[3] for x in batch], device=device),
                      "payload": torch.tensor([x[4] for x in batch], device=device), "target": torch.tensor([x[5] for x in batch], device=device),
                      "queued": torch.tensor([x[6] for x in batch], device=device)}
            out = model(state, race); loss = sum(F.cross_entropy(out[key], labels[key]) for key in labels)
            optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            seen += len(batch)
            for key in labels: correct[key] += out[key].argmax(-1).eq(labels[key]).sum().item()
        if game % 25 == 0: print(f"games={game} decisions={seen} " + " ".join(f"{key}_acc={correct[key]/max(seen,1):.3f}" for key in ("actor", "family", "payload", "target", "queued")), flush=True)
        if args.checkpoint_every and game % args.checkpoint_every == 0: save(game); print(f"checkpoint={args.output} games={game}", flush=True)
    save(len(files)); print(f"saved={args.output} decisions={seen}", flush=True)


if __name__ == "__main__": main()
