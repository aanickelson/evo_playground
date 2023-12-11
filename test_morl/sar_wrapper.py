import mo_gymnasium as mo_gym
import numpy as np
from mo_gymnasium.utils import MORecordEpisodeStatistics

from evo_playground.support.neuralnet import NeuralNetwork as NN
from evo_playground.ccea_base import CCEA
from evo_playground.parameters.learningparams01 import LearnParams as lp
# from beepy import beep

import warnings


class SARWrap:
    def __init__(self, env, hid, bh, ts=1000):
        self.env = env
        self.ts = ts
        self.bh_name = bh

        self.st_size = self.env.observation_space.shape[0]
        self.states = np.zeros((ts, self.st_size))
        self.st_low = self.env.observation_space.low
        self.st_high = self.env.observation_space.high

        self.act_size = self.env.action_space.shape[0]
        self.acts = np.zeros((ts, self.act_size))
        self.act_low = self.env.action_space.low
        self.act_high = self.env.action_space.high

        self.n_obj = self.env.unwrapped.reward_dim
        self.model = NN(self.st_size, hid, self.act_size)

    def reset(self):
        self.states = np.zeros((self.ts, self.st_size))
        self.acts = np.zeros((self.ts, self.act_size))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.env.reset_custom()
        return self.env.reset()

    def interpolate(self, vec, old_v, new_v):
        old_vals = np.transpose(old_v)
        new_vals = np.transpose(new_v)
        interpolated = np.zeros_like(vec)
        for i, act in enumerate(vec):
            interpolated[i] = np.interp(act, old_vals[i], new_vals[i])
        return interpolated

    def run(self, policy):
        st, _ = self.reset()

        for ts in range(self.ts):
            pol_out = policy(st).detach().numpy()

            # Bookkeeping for behaviors
            self.states[ts] = self.interpolate(st, [self.st_low, self.st_high], [[0]*self.st_size, [1]*self.st_size])  # Change to be between [0, 1] for behaviors
            self.acts[ts] = pol_out     # Policy output is between [0,1], which is what we need for behaviors

            # This changes the action to be between the lower and upper bounds for each item in the array
            action = self.interpolate(pol_out, [[0]*self.act_size, [1]*self.act_size], [self.act_low, self.act_high])
            st, vec_reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                break

        # Only keep the parts that were filled in, i.e. the number of time steps the sim ran
        self.acts = self.acts[:ts]
        self.states = self.states[:ts]

        # Replace nan with 0s
        return np.nan_to_num(self.env.get_wrapper_attr('fin_rw'))

    def run_bh(self, x):
        self.model.set_trained_network(x)
        rw = self.run(self.model)
        bh_vals = self.get_bh(self.bh_name)
        return rw, bh_vals

    def state_size(self):
        return self.st_size

    def action_size(self):
        return self.act_size

    def bh_size(self, bh_name):
        sizes = {'avg st': self.env.get_wrapper_attr("st_bh_size"),                # Average state
                 'fin st': self.env.get_wrapper_attr("st_bh_size"),                # Final state
                 'avg act': self.act_size,                  # Average action
                 'fin act': self.act_size,                  # Final action
                 'min avg max act': self.act_size * 3}      # Min, average, max actions
        return sizes[bh_name]

    def get_bh(self, bh_name):
        st_idx = self.env.get_wrapper_attr('st_bh_idxs')
        bhs = {'avg st': np.mean(self.states, axis=0)[st_idx[0]:st_idx[1]],      # Average state
               'fin st': self.states[-1][st_idx[0]:st_idx[1]],                   # Final state
               'avg act': np.mean(self.acts, axis=0),       # Average action
               'fin act': self.acts[-1],                    # Final action
               'min avg max act':                           # Min, average, max actions
                   np.concatenate((np.min(self.acts, axis=0), np.mean(self.acts, axis=0), np.max(self.acts, axis=0)))}

        return bhs[bh_name]


class Params:
    def __init__(self):
        self.n_agents = 1


if __name__ == '__main__':
    params = Params()
    env = MORecordEpisodeStatistics(mo_gym.make("mo-lunar-lander-continuous-v2"), gamma=0.99)
    eval_env = mo_gym.make("mo-lunar-lander-continuous-v2")
    st_size = 8
    act_size = 2
    reward_size = 4
    # x = eval_env.action_space
    wrap = SARWrap(env)
    ccea = CCEA(wrap, params, lp, 'G', st_size, act_size, '/home/anna/PycharmProjects/evo_playground/test_morl/test_lander', 0)
    ccea.run_evolution()
    # beep(8)

# Cont environments:
# water-reservoir-vo
# mo-mountaincarcontinuous-v0
# mo-lunar-lander-v2
# mo-hopper-v4
# mo-halfcheetah-v4

# Helpful links
# https://mo-gymnasium.farama.org/environments/water-reservoir/
# https://gymnasium.farama.org/api/wrappers/reward_wrappers/
# https://pymoo.org/interface/minimize.html

# Paper with a lot of the implementaiton details
# https://openreview.net/forum?id=AwWaBXLIJE
