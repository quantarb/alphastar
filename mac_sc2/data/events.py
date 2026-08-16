"""Replay-event utilities shared by raw replay extractors."""

def event_pid(event):
    """Return a player's stable one-based replay id.

    Command events sometimes expose a zero-based ``pid``; their resolved
    player object is authoritative and keeps both players' trajectories
    correctly attributed.
    """
    player = getattr(event, "player", None)
    return getattr(player, "pid", None) if player is not None else getattr(event, "pid", getattr(event, "control_pid", None))
