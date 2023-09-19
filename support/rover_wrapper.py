import os
import sys

import numpy as np
from torch import from_numpy

from evo_playground.support.neuralnet import NeuralNetwork as NN
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
from pymap_elites_multiobjective.scripts_data.run_env import run_env

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class AgPol:
    def __init__(self, st_size, hid, act_size):
        self.st_size = st_size
        self.hid = hid
        self.act_size = act_size
        self.w0_size = self.st_size * self.hid
        self.w2_size = self.hid * self.act_size
        self.b0_size = self.hid
        self.b2_size = self.act_size
        self.model = NN(self.st_size, self.hid, self.act_size)

    def set_trained_network(self, x):
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


class RoverWrapper:
    def __init__(self, env, behs):
        self.env = env
        self.p = env.params
        self.agents = self.setup_ag()
        self.vis = False
        self.use_bh = True
        self.behs = behs

    def setup_ag(self):
        st_size = self.env.state_size()
        hid = lp.hid
        act_size = self.env.action_size()
        ag = []
        for _ in range(self.p.n_agents):
            ag.append(AgPol(st_size, hid, act_size))
        return ag

    def setup(self, x):
        self.env.reset()
        wts = x
        if self.p.n_agents == 1 and len(x) > 1:
            wts = [x]
        try:
            if len(wts) != self.p.n_agents:
                print(f"given x is of length {len(x)}, should be {self.p.n_agents}")
                return
        except TypeError:
            blerg = 10

        models = []
        for i, w in enumerate(wts):
            self.agents[i].set_trained_network(w)
            models.append(self.agents[i].model)
        return models

    def _evaluate(self, x):
        models = self.setup(x)
        out_vals = run_env(self.env, models, self.p, self.behs, use_bh=self.use_bh, vis=self.vis)
        if self.use_bh:
            fitness, bh = out_vals
            return fitness, bh[0]
        else:
            return out_vals

    def _evaluate_multiple(self, x):
        n_eval = self.p.n_cf_evals
        models = self.setup(x)
        fit_vals = np.zeros((n_eval, self.p.n_poi_types))
        bh_vals = np.zeros((n_eval, self.p.n_bh))
        for i in range(n_eval):
            self.env.reset()
            out_vals = run_env(self.env, models, self.p, self.behs, use_bh=self.use_bh, vis=self.vis)
            # out_vals = run_env(self.env, models, self.p, use_bh=self.use_bh, vis=self.vis)
            if self.use_bh:
                fitness, bh = out_vals
                fit_vals[i] = fitness
                bh_vals[i] = bh
            else:
                fit_vals[i] = out_vals

        fits = np.average(fit_vals, axis=0)
        bhs = np.average(bh_vals, axis=0)
        if self.use_bh:
            return fits, bhs
        else:
            return fits
