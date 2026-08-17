"""Time a single exact-build SC2 raw-observation replay pass.

This is deliberately an inspection probe, not a persisted training shard.  It
shows that the replay can be rendered by the installed client and reports the
number of action-tick observations available to a future on-demand trainer.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import struct
import sys
import time
import zlib
from pathlib import Path

import mpyq
from absl import flags
from pysc2 import run_configs
from s2clientprotocol import sc2api_pb2 as sc_pb

from mac_sc2.contracts.terran_entity_ar import PATCH


def replay_version(path: Path) -> str:
    archive = mpyq.MPQArchive(io.BytesIO(path.read_bytes())).extract()
    return json.loads(archive[b"replay.gamemetadata.json"])["GameVersion"]


def probe(path: Path, observed_player: int, archive_path: Path | None = None) -> dict[str, object]:
    if replay_version(path) != PATCH:
        raise ValueError(f"{path} is not an exact {PATCH} replay")

    # PySC2's macOS run-config calls this single installed binary "latest".
    # The exact build is verified above from replay metadata.
    config = run_configs.get()
    replay_data = path.read_bytes()
    interface = sc_pb.InterfaceOptions(
        raw=True,
        score=True,
        raw_affects_selection=True,
        raw_crop_to_playable_area=True,
    )
    request = sc_pb.RequestStartReplay(
        replay_data=replay_data,
        options=interface,
        disable_fog=False,
        observed_player_id=observed_player,
    )

    started = time.perf_counter()
    loops = action_ticks = raw_units = actions = 0
    action_tick_proto_bytes = 0
    compressor = zlib.compressobj(level=6)
    action_tick_zlib_bytes = 0
    archive = None
    if archive_path:
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        archive = gzip.open(archive_path, "wb", compresslevel=6)
        archive.write(b"SC2RAWACT1\\n")
    with config.start() as controller:
        info = controller.replay_info(replay_data)
        if info.local_map_path:
            request.map_data = config.map_data(info.local_map_path)
        controller.start_replay(request)
        controller.step(1)
        while True:
            observation = controller.observe()
            loops += 1
            raw_units += len(observation.observation.raw_data.units)
            actions += len(observation.actions)
            if observation.actions:
                action_ticks += 1
                payload = observation.observation.SerializeToString()
                payload += b"".join(action.SerializeToString() for action in observation.actions)
                action_tick_proto_bytes += len(payload)
                action_tick_zlib_bytes += len(compressor.compress(payload))
                if archive:
                    action_payload = b"".join(
                        struct.pack(">I", action.ByteSize()) + action.SerializeToString()
                        for action in observation.actions
                    )
                    archive.write(struct.pack(">III", observation.observation.game_loop, observation.observation.ByteSize(), len(action_payload)))
                    archive.write(observation.observation.SerializeToString())
                    archive.write(action_payload)
            if observation.player_result:
                break
            controller.step(1)

    action_tick_zlib_bytes += len(compressor.flush())
    if archive:
        archive.close()
    elapsed = time.perf_counter() - started
    result = {
        "replay": str(path.resolve()),
        "game_version": PATCH,
        "observed_player": observed_player,
        "elapsed_seconds": round(elapsed, 3),
        "game_loops_observed": loops,
        "action_tick_observations": action_ticks,
        "recorded_actions": actions,
        "action_tick_proto_bytes": action_tick_proto_bytes,
        "action_tick_zlib_bytes": action_tick_zlib_bytes,
        "mean_visible_raw_units": round(raw_units / loops, 2),
    }
    if archive_path:
        result["archive"] = str(archive_path.resolve())
        result["archive_bytes"] = archive_path.stat().st_size
        result["archive_format"] = "SC2RAWACT1: gzip, loop+ObservationProto+ActionProto records"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--player", type=int, default=1, choices=(1, 2))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    # PySC2 accesses its global flags even when called from a normal script.
    flags.FLAGS([sys.argv[0]])
    result = probe(args.replay, args.player, args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
