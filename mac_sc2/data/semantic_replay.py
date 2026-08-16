"""Raw-replay semantic macro examples using the executable eight-action contract."""
from collections import defaultdict
import sc2reader
from mac_sc2.contracts.semantic import RACE_IDS, supports
from mac_sc2.data.events import event_pid
from mac_sc2.semantic_action_schema import from_event

WORDS=[('scv','probe','drone'),('supplydepot','pylon','overlord'),('barracks','gateway','spawningpool'),('refinery','assimilator','extractor'),('cybernetics','robotics','stargate','forge','engineeringbay','armory','factory','spire','hydraliskden','roachwarren'),('marine','zealot','zergling'),('stalker','adept','sentry','roach','hydralisk','marauder','hellion'),('immortal','colossus','disruptor','templar','carrier','voidray','phoenix','siegetank','medivac','mutalisk','lurker')]
def cat(name):
    name=(name or '').lower(); return [int(any(word in name for word in group)) for group in WORDS]
def vec(stats, counts, second):
    minerals=getattr(stats,'minerals_current',0); gas=getattr(stats,'vespene_current',0); used=getattr(stats,'food_used',0); made=getattr(stats,'food_made',0); workers=max(getattr(stats,'workers_active_count',0),counts[0])
    return [min(second/900,1),min(minerals/1500,1),min(gas/1000,1),min(used/200,1),min(made/200,1),min(max(made-used,0)/30,1),min(workers/80,1),*[min(x/20,1) for x in counts[1:]],min(getattr(stats,'minerals_collection_rate',0)/2500,1),min(getattr(stats,'vespene_collection_rate',0)/1500,1),min(getattr(stats,'resources_lost',0)/10000,1)]

def examples(path, patch="4.9.2"):
    replay=sc2reader.load_replay(path,load_level=4); races={p.pid:p.play_race for p in replay.players}; latest={};counts=defaultdict(lambda:[0]*8);selected={};groups=defaultdict(dict)
    for event in replay.events:
        pid=event_pid(event); typ=type(event).__name__
        if pid not in races or races[pid].lower() not in RACE_IDS: continue
        if typ=='PlayerStatsEvent': latest[pid]=event; continue
        if typ in ('UnitBornEvent','UnitInitEvent'):
            inc=cat(getattr(event,'unit_type_name',''));counts[pid]=[a+b for a,b in zip(counts[pid],inc)];continue
        if typ=='SelectionEvent': selected[pid]=[str(x) for x in (getattr(event,'objects',[]) or [])];continue
        if 'ControlGroupEvent' in typ:
            group=getattr(event,'control_group',0)
            if typ=='SetControlGroupEvent': groups[pid][group]=list(selected.get(pid,[]))
            elif typ=='GetControlGroupEvent' and groups[pid].get(group): selected[pid]=groups[pid][group]
            continue
        if 'CommandEvent' not in typ or pid not in latest: continue
        action=from_event(event,patch,races[pid],selected.get(pid,[]))
        if supports(action.actor_role,action.family,action.payload_role,action.target_kind):
            yield RACE_IDS[races[pid].lower()], vec(latest[pid],counts[pid],getattr(event,'second',0)), action
