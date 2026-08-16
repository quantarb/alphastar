"""Executable tactical actions shared by spatial BC training and live play.

This is intentionally smaller than the global replay vocabulary.  It contains
only commands whose selected units and targets can be represented by the
entity-pointer policy and issued safely by the 4.9.2 python-sc2 runner.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TacticalAction:
    command: str
    actor: str
    target: str
    offset: bool


ACTIONS = (
    TacticalAction("attack", "combat", "entity_or_point", False),
    TacticalAction("move", "combat", "point", True),
    TacticalAction("kite", "combat", "point", True),
    TacticalAction("regroup", "combat", "point", True),
    TacticalAction("defend_home", "combat", "point", False),
    TacticalAction("repair", "worker", "friendly_entity", False),
    TacticalAction("hold", "combat", "none", False),
)

COMMANDS = tuple(action.command for action in ACTIONS)


def validate(command: str, actor_count: int, target_count: int) -> bool:
    """Reject output combinations with no patch-valid live realization."""
    if command not in COMMANDS or actor_count <= 0:
        return False
    action = ACTIONS[COMMANDS.index(command)]
    return action.target == "none" or target_count > 0
