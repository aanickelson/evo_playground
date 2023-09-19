"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""
import os

# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
import multiprocessing
from datetime import datetime
from time import time

# Custom packages
from evo_playground.run_env import AICWrapper
from evo_playground.support.binary_species import Species
from pymap_elites_multiobjective.parameters.learningparams01 import LearnParams as lp
import pymap_elites_multiobjective.parameters as Params
from evo_playground.radians_G import G_exp


class CCEA:
    def __init__(self, env, p, lp, rew_type, in_size, out_size, base_fpath, statnm):
        np.random.seed(statnm + int(datetime.now().timestamp()))
        self.p = p
        self.env = env
        self.lp = lp
        self.stat_num = statnm
        self.n_gen = self.lp.n_gen
        self.base_fpath = base_fpath
        self.wts_pth = self.base_fpath + "/wts.npy"
        self.fits_path = self.base_fpath + f"/fits_{self.stat_num}.npy"
        self.raw_g = np.zeros(self.lp.n_gen) - 1
        self.d = np.zeros((self.lp.n_gen, self.p.n_agents))
        self.rew_type = rew_type
        self.nn_in = in_size
        self.nn_hid = lp.hid
        self.nn_out = out_size
        self.species = self.species_setup()

    def save_data(self, g):
        np.save(self.fits_path, self.raw_g)

    def species_setup(self):
        species = [Species(self.lp, self.nn_in, self.nn_hid, self.nn_out) for _ in range(self.p.n_agents)]
        return species

    def run_once(self, pol_num):
        # Reset the environment
        self.env.reset()

        # For each species
        for idx, spec in enumerate(self.species):
            # Set the current policy
            spec.model.set_weights(spec.weights[pol_num])

        # Array of one NN per species to use as policies
        models = [sp.model for sp in self.species]

        # Run the simulation
        G = self.env.run(models)
        return G

    def tournament(self, rawG, dscores):
        # Index of the policies that performed best over G
        max_g = np.max(rawG)
        argmax_g = np.argmax(rawG)

        # Update the starting weights (the policy we keep between generations) for each species
        for idx, spec in enumerate(self.species):
            # If the best of this group is better than the previous best, replace the hall of famers
            if max_g > spec.hof_score:
                spec.hof_score = max_g
                spec.hof_wts = spec.weights[argmax_g]
            if 'G' in self.rew_type:
                # Use raw G because the scores may be more noisy
                spec.binary_tournament(np.array(rawG))
            elif 'D' in self.rew_type:
                spec.binary_tournament(np.array(dscores[idx]))

            # Reduce the learning rate
            spec.learning_rate /= 1.001

    def run_evolution(self):

        for gen in tqdm(range(self.lp.n_gen)):
        # for gen in range(self.lp.n_gen):
            batteries = np.zeros(self.lp.n_policies)
            # Bookkeeping
            d_scores = np.zeros((self.p.n_agents, self.lp.n_policies))
            raw_G = np.zeros(self.lp.n_policies)

            # Mutate weights for all species
            for spec in self.species:
                spec.mutate_weights()

            for p_num in range(self.lp.n_policies):
                G, D = self.run_once(p_num)
                # D = [G] * self.p.n_agents
                # Bookkeeping
                # d_scores[:, p_num] = np.sum(D, axis=1)
                # batteries[p_num] = self.env.agents[0].battery

                raw_G[p_num] = G
                d_scores[:, p_num] = D

            self.raw_g[gen] = np.max(raw_G)
            self.d[gen] = np.max(d_scores, axis=1)

            # print(gen, " BATTERY: ", min(batteries), max(batteries), np.mean(batteries))
            # Policies that performed best
            max_wts = [self.species[sp].weights[np.argmax(raw_G)] for sp in range(self.p.n_agents)]
            self.tournament(raw_G, d_scores)

            # Bookkeeping - save data every 100 generations
            # Save models every 1000 generations
            if not gen % 100:
                self.save_data(gen)
            #     for i, species in enumerate(self.species):
            #         species.save_model(max_wts[i], self.l2_wts_pth + f"spec{i}_{gen}.pth")

        self.save_data(self.n_gen)
        # save the models
        # for i, species in enumerate(self.species):
        #     species.save_model(max_wts[i], self.l2_wts_pth + f"spec{i}_{self.lp.n_gen}.pth")
        self.env.reset()


def main(batch_p):
    # start = time()
    [en, param, learnpar, rew, wt_sz, out_sz, base_pth, stat_nm] = batch_p
    ccea = CCEA(en, param, learnpar, rew, wt_sz, out_sz, base_pth, stat_nm)
    ccea.run_evolution()
    # print(time() - start, base_pth)


def multiprocess_main(batch_for_multi):
    cpus = multiprocessing.cpu_count() - 1
    # cpus = 1
    with multiprocessing.Pool(processes=cpus) as pool:
        pool.map(main, batch_for_multi)


def setup(param_set, b_path=None):
    if not b_path:
        b_path = os.getcwd()

    now = datetime.now()
    base_path = path.join(b_path, 'data')
    if not os.path.exists(base_path):
        mkdir(base_path)

    now_str = now.strftime("%Y%m%d_%H%M%S")
    dirpath = path.join(base_path, now_str)
    mkdir(dirpath)
    batch = []

    for p in param_set:
        env = AICWrapper(p)
        filepath = path.join(dirpath, f'{p.param_idx}')
        mkdir(filepath)
        for i in range(lp.n_stat_runs):
            batch.append([env, p, lp, 'G', env.state_size(), env.action_size(), filepath, i])

    return batch


if __name__ == '__main__':
    param_set = [Params.p100000]
    b = setup(param_set, '/home/toothless/workspaces/evo_playground/')
    multiprocess_main(b)
    # main(b[0])

    import beepy
    beepy.beep(sound=4)
