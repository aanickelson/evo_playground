import os
import sys

import numpy as np
from torch import from_numpy

from evo_playground.learning.neuralnet import NeuralNetwork as NN
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
from pymap_elites_multiobjective.scripts_data.run_env import run_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class RoverWrapper:
    def __init__(self, env, param):
        self.env = env
        self.p = param
        self.st_size = env.state_size()
        self.hid = lp.hid
        self.act_size = env.action_size()
        self.w0_size = self.st_size * self.hid
        self.w2_size = self.hid * self.act_size
        self.b0_size = self.hid
        self.b2_size = self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)
        self.n_evals = 0
        self.vis = False
        self.use_bh = True

    def evaluate(self, x):
        self.env.reset()

        # Use these ONLY if you're using the old data (before 7/7/2023)
        # w0_wts = from_numpy(np.reshape(x[:self.w0_size], (self.hid, self.st_size)))
        # w2_wts = from_numpy(np.reshape(x[self.w0_size:], (self.act_size, self.hid)))

        # Use this block to set the weights AND the biases. Like a real puppet.
        cut0 = self.b0_size
        cut1 = cut0 + self.b2_size
        cut2 = cut1 + self.w0_size

        b0_wts = from_numpy(np.array(x[:cut0]))
        b1_wts = from_numpy(np.array(x[cut0:cut1]))
        w0_wts = from_numpy(np.reshape(x[cut1:cut2], (self.hid, self.st_size)))
        w2_wts = from_numpy(np.reshape(x[cut2:], (self.act_size, self.hid)))

        self.model.set_biases([b0_wts, b1_wts])
        self.model.set_weights([w0_wts, w2_wts])

        out_vals = run_env(self.env, [self.model], self.p, use_bh=self.use_bh, vis=self.vis)
        self.n_evals += 1

        if self.use_bh:
            fitness, bh = out_vals
            return fitness, bh[0]
        else:
            return out_vals