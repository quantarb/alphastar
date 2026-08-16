"""On-demand replay placement examples with bounded last-known entity state."""
import zlib
import sc2reader

from mac_sc2.contracts.placement import ENTITY_FEATURES, ENTITY_SLOTS, PlacementLabel
from mac_sc2.data.events import event_pid

TOWN_HALL = ("commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive")

def _name(unit): return str(getattr(unit, "type_name", unit)).split(" [", 1)[0]
def _key(unit): return int(getattr(unit, "id", getattr(unit, "unit_id", 0)) or 0)

def examples(path):
    """Yield only point-placement commands; no files or replay shards are made."""
    replay = sc2reader.load_replay(path, load_level=4)
    races = {p.pid: p.play_race for p in replay.players}
    units, homes = {}, {}
    for event in replay.events:
        pid, typ = event_pid(event), type(event).__name__
        if typ in ("UnitBornEvent", "UnitInitEvent", "UnitDoneEvent", "UnitTypeChangeEvent"):
            unit = getattr(event, "unit", None); owner = getattr(getattr(unit, "owner", None), "pid", None)
            pos = getattr(unit, "location", None)
            if unit and pos:
                units[_key(unit)] = (owner, _name(unit), float(pos[0]), float(pos[1]))
                if owner and owner not in homes and any(x in _name(unit).lower() for x in TOWN_HALL): homes[owner] = (float(pos[0]), float(pos[1]))
            continue
        if typ == "UnitDiedEvent": units.pop(int(getattr(event, "unit_id", 0) or 0), None); continue
        if typ == "UnitPositionsEvent":
            for unit, pos in getattr(event, "units", {}).items():
                key = _key(unit)
                if key in units: units[key] = (*units[key][:2], float(pos[0]), float(pos[1]))
            continue
        if "CommandEvent" not in typ or pid not in races:
            continue
        name, point = getattr(event, "ability_name", "") or "", getattr(event, "location", None)
        if not point or not (name.startswith("Build") or name.startswith("Land")):
            continue
        home = homes.get(pid)
        if not home: continue
        snapshot = []
        for owner, unit_name, x, y in units.values():
            if owner != pid: continue
            snapshot.append((zlib.crc32(unit_name.encode()) % 65535 / 65535, (x-home[0])/64, (y-home[1])/64, 1.0, 0, 0, 0, 0))
        yield snapshot[:ENTITY_SLOTS], PlacementLabel(races[pid], name, (float(point[0]), float(point[1])), int(event.frame)), home
