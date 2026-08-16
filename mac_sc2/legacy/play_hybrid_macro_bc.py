#!/usr/bin/env python3
"""Run the trained hybrid behavior-cloning policy against SC2's Easy AI."""
import time
import torch
from absl import app, flags
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from extract_sc2egset_macro import encoded_state
from train_hybrid_macro_bc import ACTIONS, HybridMacroTransformer

FLAGS = flags.FLAGS
flags.DEFINE_string('checkpoint', 'mac_sc2/artifacts/hybrid_macro_bc.pt', 'Hybrid BC checkpoint.')
flags.DEFINE_integer('steps', 1400, 'Maximum agent decisions.')
flags.DEFINE_float('seconds_per_step', 0.0, 'Slow down visual play.')
flags.DEFINE_bool('visualize', False, 'Show live PySC2 viewer.')
flags.DEFINE_string('replay_dir', 'mac_sc2/artifacts/replays', 'Where to save the SC2 replay.')


class HybridTerran(base_agent.BaseAgent):
    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        self.model = HybridMacroTransformer(); self.model.load_state_dict(payload['state_dict']); self.model.eval()
        self.last_action = 'starting'; self.last_counts = {}

    @staticmethod
    def all_owned(obs, unit_type):
        return [u for u in obs.observation.raw_units if u.alliance == features.PlayerRelative.SELF and u.unit_type == unit_type]

    @staticmethod
    def complete(units_):
        return [u for u in units_ if u.build_progress >= 100]

    @staticmethod
    def idle(units_):
        return [u for u in units_ if u.order_length == 0]

    def step(self, obs):
        super().step(obs)
        p = obs.observation.player
        minerals, used, cap = int(p[features.Player.minerals]), int(p[features.Player.food_used]), int(p[features.Player.food_cap])
        scvs = self.complete(self.all_owned(obs, units.Terran.SCV)); ccs = self.complete(self.all_owned(obs, units.Terran.CommandCenter))
        depots_all = self.all_owned(obs, units.Terran.SupplyDepot); barracks_all = self.all_owned(obs, units.Terran.Barracks)
        depots, barracks = self.complete(depots_all), self.complete(barracks_all)
        marines = self.complete(self.all_owned(obs, units.Terran.Marine))
        state = torch.tensor([encoded_state(self.steps * 32, {'scoreValueMineralsCurrent': minerals,
            'scoreValueFoodUsed': used, 'scoreValueFoodMade': cap, 'scoreValueWorkersActiveCount': len(scvs)},
            {'SCV': len(scvs), 'SupplyDepot': len(depots_all), 'Barracks': len(barracks_all), 'Marine': len(marines)})])
        with torch.no_grad(): logits = self.model(state)[0]
        # Legal-action mask: never let an impossible network decision stall the game.
        valid = torch.full_like(logits, -1e9)
        idle_cc = self.idle(ccs); idle_rax = self.idle(barracks)
        if idle_cc and minerals >= 50 and len(scvs) < 26: valid[0] = logits[0]
        if scvs and minerals >= 100 and cap - used <= 3 and len(depots_all) < 5: valid[1] = logits[1]
        desired_rax = 2 if len(marines) >= 8 else 1
        if scvs and minerals >= 150 and len(barracks_all) < desired_rax: valid[2] = logits[2]
        if idle_rax and minerals >= 50 and cap > used: valid[3] = logits[3]
        if len(marines) >= 12: valid[4] = logits[4] + 1.0
        if torch.all(valid < -1e8):
            self.last_action = 'no_op'; return actions.RAW_FUNCTIONS.no_op()
        intent = int(valid.argmax()); self.last_action = ACTIONS[intent]
        self.last_counts = {'scvs': len(scvs), 'depots': len(depots_all), 'barracks': len(barracks_all), 'marines': len(marines)}
        if intent == 0:
            return actions.RAW_FUNCTIONS.Train_SCV_quick('now', [idle_cc[0].tag])
        if intent == 1:
            base = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt('now', [scvs[0].tag], (base.x + 5 + len(depots_all) * 2, base.y + 4))
        if intent == 2:
            base = ccs[0] if ccs else scvs[0]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt('now', [scvs[0].tag], (base.x + 10 + len(barracks_all) * 4, base.y + 3))
        if intent == 3:
            return actions.RAW_FUNCTIONS.Train_Marine_quick('now', [u.tag for u in idle_rax])
        base = ccs[0] if ccs else marines[0]
        target = (64 - base.x, 64 - base.y)
        return actions.RAW_FUNCTIONS.Attack_pt('now', [u.tag for u in marines], target)


def main(argv):
    del argv
    agent = HybridTerran(FLAGS.checkpoint)
    interface = features.AgentInterfaceFormat(use_raw_units=True, use_raw_actions=True,
        feature_dimensions=features.Dimensions(screen=84, minimap=64))
    with sc2_env.SC2Env(map_name='Simple64', players=[sc2_env.Agent(sc2_env.Race.terran),
        sc2_env.Bot(sc2_env.Race.zerg, sc2_env.Difficulty.easy)], agent_interface_format=interface,
        step_mul=32, game_steps_per_episode=16 * 60 * 20, visualize=FLAGS.visualize,
        save_replay_episodes=1, replay_dir=FLAGS.replay_dir, replay_prefix='hybrid_bc') as env:
        timestep = env.reset()[0]
        for step in range(FLAGS.steps):
            timestep = env.step([agent.step(timestep)])[0]
            if FLAGS.seconds_per_step: time.sleep(FLAGS.seconds_per_step)
            if step % 100 == 0: print(f'step={step:04d} action={agent.last_action} units={agent.last_counts}')
            if timestep.last(): break
        reward = float(timestep.reward)
    print(f'Hybrid BC match finished: reward={reward:+.1f}, decisions={agent.steps}, final={agent.last_counts}')


if __name__ == '__main__':
    app.run(main)
