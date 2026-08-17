"""On-demand 5.0.16.97563 Terran labels for the entity autoregressive contract."""
from __future__ import annotations

from collections import Counter, defaultdict

import sc2reader

from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS
from mac_sc2.contracts.terran_entity_ar import PATCH, intent_id
from mac_sc2.contracts.semantic_schema import from_event
from mac_sc2.data.patch_race_exact import TOWN_HALL, _id, _name, _point, _snapshot, cat, event_owner, vec


def _intent(action) -> int | None:
    name = action.ability_name.lower()
    if "train_scv" in name: return intent_id("train_scv")
    if "supplydepot" in name and action.family == "build": return intent_id("build_supply")
    if "refinery" in name and action.family == "build": return intent_id("build_refinery")
    if "barracks" in name and action.family == "build": return intent_id("build_barracks")
    if "factory" in name and action.family == "build": return intent_id("build_factory")
    if "commandcenter" in name and action.family == "build": return intent_id("build_command_center")
    if "train_marine" in name: return intent_id("train_marine")
    if "train_hellion" in name: return intent_id("train_hellion")
    if "upgradetoorbital" in name: return intent_id("morph_orbital")
    if "calldownmule" in name: return intent_id("call_mule")
    if action.family == "attack": return intent_id("attack")
    if action.family == "repair": return intent_id("repair")
    if action.family == "move" and action.actor_role == "combat": return intent_id("scout")
    return None


def examples(path: str, version: str, stats: Counter | None = None):
    """Stream both players' exact-build Terran commands without writing shards."""
    if version != PATCH:
        return
    replay = sc2reader.load_replay(path, load_level=4)
    races = {player.pid: player.play_race for player in replay.players}
    latest, counts, selected, groups, units, homes = {}, defaultdict(lambda: [0] * 8), defaultdict(list), defaultdict(dict), {}, {}
    discarded = Counter()
    for event in replay.events:
        pid, typ = event_owner(event), type(event).__name__
        if typ in ("UnitBornEvent", "UnitInitEvent", "UnitDoneEvent", "UnitTypeChangeEvent"):
            unit = getattr(event, "unit", None); owner = getattr(getattr(unit, "owner", None), "pid", None); position = _point(unit)
            if unit and owner and position:
                units[_id(unit)] = (unit, owner, position)
                if owner not in homes and any(word in _name(unit).lower() for word in TOWN_HALL): homes[owner] = position
            if owner in races: counts[owner] = [a + b for a, b in zip(counts[owner], cat(getattr(event, "unit_type_name", _name(unit) if unit else "")))]
            continue
        if typ == "UnitDiedEvent": units.pop(int(getattr(event, "unit_id", 0) or 0), None); continue
        if typ == "UnitPositionsEvent":
            for unit, position in getattr(event, "units", {}).items():
                if _id(unit) in units: units[_id(unit)] = (units[_id(unit)][0], units[_id(unit)][1], (float(position[0]), float(position[1])))
            continue
        if pid not in races: continue
        if typ == "PlayerStatsEvent": latest[pid] = event; continue
        if typ == "SelectionEvent": selected[pid] = [_id(unit) for unit in (getattr(event, "objects", []) or []) if _id(unit)]; continue
        if "ControlGroupEvent" in typ:
            group = getattr(event, "control_group", 0)
            if typ == "SetControlGroupEvent": groups[pid][group] = list(selected[pid])
            elif typ == "AddToControlGroupEvent": groups[pid][group] = list(dict.fromkeys(groups[pid].get(group, []) + selected[pid]))
            elif typ == "GetControlGroupEvent" and groups[pid].get(group): selected[pid] = groups[pid][group]
            continue
        if "CommandEvent" not in typ or races.get(pid) != "Terran" or pid not in latest or pid not in homes: continue
        action = from_event(event, PATCH, "Terran", [str(units[tag][0]) for tag in selected[pid] if tag in units])
        label = _intent(action)
        if label is None: discarded["outside_contract"] += 1; continue
        snapshot, ids = _snapshot(units, pid, homes[pid])
        if not snapshot: discarded["empty_snapshot"] += 1; continue
        target = _id(getattr(event, "target", None))
        yield {"state": vec(latest[pid], counts[pid], getattr(event, "second", 0)), "snapshot": snapshot,
               "intent": label, "actor": next((ids.index(tag) for tag in selected[pid] if tag in ids), -1),
               "target": ids.index(target) if target in ids else -1, "queued": int(action.queued)}
    if stats is not None: stats.update(discarded)
