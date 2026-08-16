"""Declarative mapping from the fixed 10 MTL actions to each SC2 race.

The trainer and live runner share the same action ordering. New checkpoints
with that schema require no decoder edits; only a new checkpoint path.
"""
from sc2.ids.unit_typeid import UnitTypeId

RACE_IDS = {"terran": 0, "protoss": 1, "zerg": 2}
RACE_CONFIG = {
    "terran": dict(worker=UnitTypeId.SCV, supply=UnitTypeId.SUPPLYDEPOT,
        prod=UnitTypeId.BARRACKS, ranged_prod=UnitTypeId.FACTORY,
        gas=UnitTypeId.REFINERY, tech=UnitTypeId.FACTORY,
        basic=UnitTypeId.MARINE, ranged=UnitTypeId.HELLION,
        advanced_build=UnitTypeId.STARPORT, advanced=UnitTypeId.MEDIVAC,
        townhall=UnitTypeId.COMMANDCENTER),
    "protoss": dict(worker=UnitTypeId.PROBE, supply=UnitTypeId.PYLON,
        prod=UnitTypeId.GATEWAY, ranged_prod=UnitTypeId.GATEWAY,
        gas=UnitTypeId.ASSIMILATOR, tech=UnitTypeId.CYBERNETICSCORE,
        basic=UnitTypeId.ZEALOT, ranged=UnitTypeId.STALKER,
        advanced_build=UnitTypeId.ROBOTICSFACILITY, advanced=UnitTypeId.IMMORTAL,
        townhall=UnitTypeId.NEXUS),
    "zerg": dict(worker=UnitTypeId.DRONE, supply=UnitTypeId.OVERLORD,
        prod=UnitTypeId.SPAWNINGPOOL, ranged_prod=UnitTypeId.HATCHERY,
        gas=UnitTypeId.EXTRACTOR, tech=UnitTypeId.ROACHWARREN,
        basic=UnitTypeId.ZERGLING, ranged=UnitTypeId.ROACH,
        advanced_build=None, advanced=UnitTypeId.QUEEN,
        townhall=UnitTypeId.HATCHERY),
}
