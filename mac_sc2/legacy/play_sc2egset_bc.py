#!/usr/bin/env python3
"""Play a real SC2 game using the SC2EGSet-trained macro behavior clone."""
import time
import torch
from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from extract_sc2egset_macro import encoded_state
from train_sc2egset_macro import ReplayMacroTransformer

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', 'mac_sc2/artifacts/sc2egset_macro_transformer.pt', 'BC checkpoint.')
flags.DEFINE_integer('steps', 900, 'Maximum agent decisions.')
flags.DEFINE_float('seconds_per_step', 0.25, 'Slow down visual play.')
flags.DEFINE_bool('visualize', True, 'Show live game viewer.')


class ReplayClonedTerran(base_agent.BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = ReplayMacroTransformer(); self.model.load_state_dict(payload['state_dict']); self.model.eval()
        self.last_action = 'starting'

    @staticmethod
    def owned(obs, unit_type):
        return [u for u in obs.observation.raw_units if u.alliance == features.PlayerRelative.SELF
                and u.unit_type == unit_type and u.build_progress >= 100]

    def step(self, obs):
        super().step(obs)
        p = obs.observation.player
        minerals = int(p[features.Player.minerals]); used = int(p[features.Player.food_used]); cap = int(p[features.Player.food_cap])
        scvs = self.owned(obs, units.Terran.SCV); ccs = self.owned(obs, units.Terran.CommandCenter)
        depots = self.owned(obs, units.Terran.SupplyDepot); barracks = self.owned(obs, units.Terran.Barracks)
        marines = self.owned(obs, units.Terran.Marine)
        state = torch.tensor([encoded_state(self.steps * 32, {'scoreValueMineralsCurrent': minerals,
            'scoreValueFoodUsed': used, 'scoreValueFoodMade': cap, 'scoreValueWorkersActiveCount': len(scvs)},
            {'SCV': len(scvs), 'SupplyDepot': len(depots), 'Barracks': len(barracks), 'Marine': len(marines)})])
        with torch.no_grad(): label = self.model(state).argmax(-1).item()
        names = ('train_scv', 'supply', 'barracks', 'marine'); self.last_action = names[label]
        if label == 0 and ccs and minerals >= 50:
            return actions.RAW_FUNCTIONS.Train_SCV_quick('now', [ccs[0].tag])
        if label == 1 and scvs and minerals >= 100:
            base = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt('now', [scvs[0].tag], (base.x + 6, base.y + 4))
        if label == 2 and scvs and minerals >= 150:
            base = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt('now', [scvs[0].tag], (base.x + 10, base.y + 3))
        if label == 3 and barracks and minerals >= 50:
            return actions.RAW_FUNCTIONS.Train_Marine_quick('now', [barracks[0].tag])
        if len(marines) >= 12:
            self.last_action = 'attack (macro threshold)'
            return actions.RAW_FUNCTIONS.Attack_pt('now', [u.tag for u in marines], (32, 32))
        return actions.RAW_FUNCTIONS.no_op()


def main(argv):
    del argv
    agent = ReplayClonedTerran(FLAGS.checkpoint)
    interface = features.AgentInterfaceFormat(use_raw_units=True, use_raw_actions=True,
        feature_dimensions=features.Dimensions(screen=84, minimap=64))
    with sc2_env.SC2Env(map_name='Simple64', players=[sc2_env.Agent(sc2_env.Race.terran),
        sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)], agent_interface_format=interface,
        step_mul=32, game_steps_per_episode=16 * 60 * 12, visualize=FLAGS.visualize) as env:
        timestep = env.reset()[0]
        for index in range(FLAGS.steps):
            timestep = env.step([agent.step(timestep)])[0]
            if FLAGS.seconds_per_step: time.sleep(FLAGS.seconds_per_step)
            if index % 40 == 0: print(f'step={index:03d} BC action={agent.last_action}')
            if timestep.last(): break
    print(f'Replay-trained BC match finished after {agent.steps} decisions.')


if __name__ == '__main__':
    app.run(main)
