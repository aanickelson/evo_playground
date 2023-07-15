"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir

# Custom packages
from evo_playground.support.binary_species import Species
from evo_playground.parameters.learningparams01 import LearnParams
import pymap_elites_multiobjective.parameters as Params
from evo_playground.ccea_top_pol_redo import TopPolEnv


class CCEA:
    def __init__(self, env, p, lp, rew_type, in_size, out_size):
        self.p = p
        self.env = env
        self.lp = lp
        self.stat_num = 0
        # self.n_gen = lp.n_gen
        self.n_gen = self.lp.n_gen
        self.raw_g = np.zeros((self.lp.n_stat_runs, self.lp.n_gen)) - 1
        self.d = np.zeros((self.lp.n_stat_runs, self.lp.n_gen, self.p.n_agents))
        self.rew_type = rew_type
        self.nn_in = in_size
        self.nn_hid = lp.hid
        self.nn_out = out_size
        self.species = self.species_setup()

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

            # Bookkeeping
            d_scores = np.zeros((self.p.n_agents, self.lp.n_policies))
            raw_G = np.zeros(self.lp.n_policies)

            # Mutate weights for all species
            for spec in self.species:
                spec.mutate_weights()

            for p_num in range(self.lp.n_policies):
                G = self.run_once(p_num)
                # D = [G] * self.p.n_agents
                # Bookkeeping
                # d_scores[:, p_num] = np.sum(D, axis=1)
                raw_G[p_num] = G

            self.raw_g[0][gen] = np.max(raw_G)
            # Policies that performed best
            max_wts = [self.species[sp].weights[np.argmax(raw_G)] for sp in range(self.p.n_agents)]
            self.tournament(raw_G, d_scores)

            # Bookkeeping - save data every 100 generations
            # Save models every 1000 generations
            if not gen % 100:
                for i, species in enumerate(self.species):
                    species.save_model(self.p.param_idx, self.stat_num, gen, self.rew_type, max_wts[i], species=i)

        # save the models
        # for i, species in enumerate(self.species):
        #     species.save_model(self.p.param_idx, self.stat_num, self.lp.n_gen, self.rew_type, max_wts[i], species=i)
        self.env.reset()


if __name__ == '__main__':
    wts_path = "/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/112_run0/weights_100000.dat"
    cent_path = "/home/toothless/workspaces/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/112_run0/centroids_2000_6.dat"
    params = Params.p500
    learnp = LearnParams
    learnp.n_gen = 100
    bh_size = 6
    en = TopPolEnv(params, learnp, wts_path, cent_path, bh_size)
    ccea = CCEA(en, params, learnp, 'G', 2, bh_size)
    ccea.run_evolution()
    print(ccea.raw_g)