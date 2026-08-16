#!/usr/bin/env python3
"""Run the trained Terran macro Transformer against SC2's built-in Easy AI."""
import time
import numpy
import torch
from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from real_game_macro import ACTION_NAMES, MacroTransformer, encode_state

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', 'mac_sc2/artifacts/real_game_macro_transformer.pt', 'Trained checkpoint.')
flags.DEFINE_integer('steps', 900, 'Maximum agent decisions.')
flags.DEFINE_float('seconds_per_step', 0.25, 'Slow down visual play.')
flags.DEFINE_bool('visualize', True, 'Show the live PySC2 game viewer.')


class LearnedTerranAgent(base_agent.BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location='cpu')
        self.model = MacroTransformer(); self.model.load_state_dict(payload['state_dict']); self.model.eval()
        self.last_action = 'startup'

    @staticmethod
    def units_of(obs, unit_type):
        return [u for u in obs.observation.raw_units
                if u.alliance == features.PlayerRelative.SELF and u.unit_type == unit_type and u.build_progress >= 100]

    def step(self, obs):
        super().step(obs)
        player = obs.observation.player
        minerals = int(player[features.Player.minerals])
        free_supply = int(player[features.Player.food_cap] - player[features.Player.food_used])
        scvs = self.units_of(obs, units.Terran.SCV)
        ccs = self.units_of(obs, units.Terran.CommandCenter)
        depots = self.units_of(obs, units.Terran.SupplyDepot)
        barracks = self.units_of(obs, units.Terran.Barracks)
        marines = self.units_of(obs, units.Terran.Marine)
        state = torch.tensor([encode_state(minerals, free_supply, len(scvs), len(depots), len(barracks), len(marines))])
        with torch.no_grad():
            action_id = self.model(state).argmax(-1).item()
        intent = ACTION_NAMES[action_id]
        self.last_action = intent
        if intent == 'train_scv' and ccs and minerals >= 50:
            return actions.RAW_FUNCTIONS.Train_SCV_quick('now', [ccs[0].tag])
        if intent == 'supply' and scvs and minerals >= 100:
            cc = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt('now', [scvs[0].tag], (cc.x + 6, cc.y + 4))
        if intent == 'barracks' and scvs and minerals >= 150:
            cc = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt('now', [scvs[0].tag], (cc.x + 10, cc.y + 3))
        if intent == 'marine' and barracks and minerals >= 50:
            return actions.RAW_FUNCTIONS.Train_Marine_quick('now', [barracks[0].tag])
        if intent == 'attack' and marines:
            # Simple64 spawns at opposing corners; center first is a robust rally/attack route.
            return actions.RAW_FUNCTIONS.Attack_pt('now', [u.tag for u in marines], (32, 32))
        return actions.RAW_FUNCTIONS.no_op()


def main(argv):
    del argv
    agent = LearnedTerranAgent(FLAGS.checkpoint)
    interface = features.AgentInterfaceFormat(use_raw_units=True, use_raw_actions=True,
                                              feature_dimensions=features.Dimensions(screen=84, minimap=64))
    players = [sc2_env.Agent(sc2_env.Race.terran), sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)]
    with sc2_env.SC2Env(map_name='Simple64', players=players, agent_interface_format=interface,
                        step_mul=32, game_steps_per_episode=16 * 60 * 12, visualize=FLAGS.visualize,
                        disable_fog=False) as env:
        timestep = env.reset()[0]
        for index in range(FLAGS.steps):
            timestep = env.step([agent.step(timestep)])[0]
            if FLAGS.seconds_per_step:
                time.sleep(FLAGS.seconds_per_step)
            if index % 40 == 0:
                print(f'step={index:03d} policy={agent.last_action}')
            if timestep.last():
                break
    outcome = 'completed'
    print(f'Real-game run {outcome}: {agent.steps} decisions against built-in Easy AI.')


if __name__ == '__main__':
    app.run(main)
