#!/usr/bin/env python3
"""Load the trained checkpoint and control PySC2's MoveToBeacon mini-game."""
import argparse
import numpy
import time
import torch
from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from beacon_policy import GRID, BeaconTransformer, encode_position

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', 'mac_sc2/artifacts/beacon_transformer.pt', 'Torch checkpoint.')
flags.DEFINE_integer('steps', 64, 'Maximum agent steps.')
flags.DEFINE_bool('visualize', False, 'Show PySC2\'s live human renderer window.')
flags.DEFINE_float('seconds_per_step', 0.0, 'Wall-clock pause after each model action.')
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL


class TransformerBeaconAgent(base_agent.BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location='cpu')
        self.model = BeaconTransformer(); self.model.load_state_dict(payload['state_dict'])
        self.model.eval()

    def step(self, obs):
        super().step(obs)
        if actions.FUNCTIONS.Move_screen.id not in obs.observation.available_actions:
            return actions.FUNCTIONS.select_army('select')
        screen = obs.observation.feature_screen.player_relative
        y, x = (screen == _PLAYER_NEUTRAL).nonzero()
        if len(x) == 0:
            return actions.FUNCTIONS.no_op()
        # The state encoder exposes the visible beacon center as two coarse tokens.
        size_y, size_x = screen.shape
        x_bin = min(GRID - 1, int(numpy.mean(x) * GRID / size_x))
        y_bin = min(GRID - 1, int(numpy.mean(y) * GRID / size_y))
        with torch.no_grad():
            target = self.model(torch.tensor([encode_position(x_bin, y_bin)])).argmax(-1).item()
        target_x = (target % GRID + 0.5) * size_x / GRID
        target_y = (target // GRID + 0.5) * size_y / GRID
        return actions.FUNCTIONS.Move_screen('now', (target_x, target_y))


def main(argv):
    del argv
    agent = TransformerBeaconAgent(FLAGS.checkpoint)
    interface = features.AgentInterfaceFormat(feature_dimensions=features.Dimensions(screen=84, minimap=64))
    with sc2_env.SC2Env(map_name='MoveToBeacon', players=[sc2_env.Agent(sc2_env.Race.terran)],
                        agent_interface_format=interface, step_mul=8, game_steps_per_episode=0,
                        visualize=FLAGS.visualize) as env:
        timestep = env.reset()[0]
        for _ in range(FLAGS.steps):
            timestep = env.step([agent.step(timestep)])[0]
            if FLAGS.seconds_per_step:
                time.sleep(FLAGS.seconds_per_step)
            if timestep.last():
                break
    print(f'Finished SC2 inference run after {agent.steps} agent steps.')


if __name__ == '__main__':
    app.run(main)
