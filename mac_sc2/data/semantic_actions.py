#!/usr/bin/env python3
"""Validate patch-stable semantic action extraction directly from raw replays.

No shards are written.  Training imports ``actions_for_replay`` and streams
examples on demand once validation establishes that selection and target
coverage are adequate.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import sc2reader

from mac_sc2.contracts.semantic_schema import from_event


def event_pid(event):
    return getattr(getattr(event, "player", None), "pid", None)


def actions_for_replay(path, patch):
    replay = sc2reader.load_replay(path, load_level=4)
    races = {player.pid: player.play_race or "Unknown" for player in replay.players}
    selected, control_groups = {}, {}
    for event in replay.events:
        pid = event_pid(event)
        if pid not in races:
            continue
        if type(event).__name__ == "SelectionEvent":
            # ``Unit.type`` is the numeric type id in these historic replay
            # builds.  The unit representation retains the human-readable
            # race-independent name (for example ``Marine [..]``), which is
            # what the semantic role classifier intentionally consumes.
            selected[pid] = [str(unit) for unit in (getattr(event, "objects", []) or [])]
            continue
        # Many pro actions are issued through numbered control groups rather
        # than a fresh SelectionEvent. Reconstruct that lightweight UI state
        # so actor-role supervision remains tied to the real selected squad.
        if "ControlGroupEvent" in type(event).__name__:
            groups = control_groups.setdefault(pid, {})
            group = getattr(event, "control_group", 0)
            current = selected.get(pid, [])
            if type(event).__name__ == "SetControlGroupEvent":
                groups[group] = list(current)
            elif type(event).__name__ == "AddToControlGroupEvent":
                groups[group] = list(dict.fromkeys(groups.get(group, []) + current))
            elif type(event).__name__ == "GetControlGroupEvent":
                # An unseen control group carries no actor evidence; retain
                # the last direct selection instead of replacing it with an
                # artificial empty selection.
                if groups.get(group):
                    selected[pid] = list(groups[group])
            continue
        if "CommandEvent" not in type(event).__name__:
            continue
        action = from_event(event, patch, races[pid], selected.get(pid, []))
        # Sc2reader uses CAbil for client UI/camera records. They are not game
        # commands and must not become imitation targets.
        if action.ability_name.lower() != "cabil":
            yield action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-games", type=int, default=200)
    args = parser.parse_args()
    games = json.loads(Path(args.manifest).read_text())["valid"][:args.max_games]
    families, payloads, roles, targets, patches, total = Counter(), Counter(), Counter(), Counter(), Counter(), 0
    examples = []
    for index, item in enumerate(games, 1):
        patch = ".".join(item["version"].split(".")[:3])
        try:
            actions = actions_for_replay(item["path"], patch)
            for action in actions:
                total += 1; families[action.family] += 1; payloads[action.payload_role] += 1; roles[action.actor_role] += 1
                targets[action.target_kind] += 1; patches[action.patch] += 1
                if len(examples) < 12: examples.append(action.record())
        except Exception as exc:
            print(f"skip game={index} {type(exc).__name__}")
    print(json.dumps({"games_scanned": len(games), "actions": total, "families": dict(families), "payload_roles": dict(payloads),
                      "actor_roles": dict(roles), "target_kinds": dict(targets), "patches": dict(patches),
                      "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
