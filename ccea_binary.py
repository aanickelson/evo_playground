"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from scipy.stats import sem, entropy
from os import getcwd, path
from optimal_comparison import optimal_policy
from multiprocessing import Process, Pool
from random import seed
import heapq as hq
import torch


# Custom packages
import parameters as param
from evo_playground.learning.evolve_population import EvolveNN as evoNN
from teaming.domain import DiscreteRoverDomain as Domain
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from evo_playground.learning.binary_species import Species



class CCEA:
    def __init__(self, env, p):
        seed()
        self.n_gen = p.n_gen
        self.trial_num = p.trial_num
        self.n_agents = p.n_agents
        self.p = p
        self.env = env
        self.species = self.species_setup()
        self.generations = range(self.n_gen)
        self.raw_g = np.zeros(self.n_gen)
        self.multi_g = np.zeros((self.n_gen, self.env.n_poi_types))
        self.max_score = np.zeros(self.n_gen)
        self.avg_score = np.zeros(self.n_gen)
        self.sterr_score = np.zeros(self.n_gen)
        self.avg_false = np.zeros(self.n_gen)
        self.d = np.zeros((self.n_gen, self.n_agents))

    def species_setup(self):
        species = [Species(self.env, self.p) for _ in range(self.n_agents)]
        return species

    def save_data(self, gen):
        cwd = getcwd()
        attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
        attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            fp = path.join(cwd, "data")
            filename = self.p.fname_prepend + "trial{:03d}_{}".format(self.trial_num, nm)
            ext = "csv"
            path_nm = path.join(fp, "{}.{}".format(filename, ext))

            # do_not_overwrite(fp, filename, ext, att, isnp=True)
            np.savetxt(path_nm, att, delimiter=",")

    def update_logs(self, scores, falses, raw_G, multi_g, i):
        self.max_score[i] = max(scores)
        self.avg_score[i] = np.mean(scores)
        self.sterr_score[i] = sem(scores)
        self.avg_false[i] = np.mean(falses)
        self.raw_g[i] = np.max(raw_G)
        best_idx = np.argmax(scores)
        self.multi_g[i] = multi_g[best_idx]

    def run_evolution(self):
        # Comparison of theoretical max for simple G
        # IF CHANGING THE ENVIRONMENT, put this the loop
        theoretical_max_g = optimal_policy(self.env)
        # theoretical_max_g = self.env.theoretical_max_g
        for gen in tqdm(self.generations):

            # Bookkeeping
            scores = np.zeros(self.p.n_policies)
            d_scores = np.zeros((self.n_agents, self.p.n_policies))
            raw_G = np.zeros(self.p.n_policies)
            falses = np.zeros(self.p.n_policies)
            all_multi_g = np.zeros((self.p.n_policies, self.env.n_poi_types))

            # Mutate weights for all species

            for spec in self.species:
                spec.mutate_weights()

            for pol_num in range(self.p.n_policies):
                # Pick one policy from each species
                wts = [self.species[i].weights[pol_num] for i in range(self.n_agents)]

                # For each species
                for idx, spec in enumerate(self.species):
                    # Set the current policy
                    spec.model.set_weights(wts[idx])

                # Array of one NN per species to use as policies
                models = [sp.model for sp in self.species]

                # Reset the environment
                self.env.reset()
                self.env.visualize = False

                # Run the simulation
                G, multi_g, avg_false = self.env.run_sim(models, multi_g=True)

                # Bookkeeping
                d_vec = self.env.D()
                d_scores[:, pol_num] = d_vec
                rew = G / theoretical_max_g
                raw_G[pol_num] = G
                scores[pol_num] = rew
                falses[pol_num] = avg_false
                all_multi_g[pol_num] = multi_g
            # Index of the policies that performed best over G
            max_g = np.argmax(raw_G)
            # Policies that performed best
            max_wts = [self.species[sp].weights[max_g] for sp in range(self.n_agents)]

            # Bookkeeping
            self.update_logs(scores, falses, raw_G, all_multi_g, gen)

            # Update the starting weights (the policy we keep between generations) for each species
            for idx, spec in enumerate(self.species):
                if 'G_' in self.p.fname_prepend:
                    # Use raw G because the scores may be more noisy (since it's divided by the greedy policy)
                    spec.start_weights = spec.binary_tournament(np.array(raw_G))
                elif 'D_' in self.p.fname_prepend:
                    spec.start_weights = spec.binary_tournament(np.array(d_scores[idx]))

                # Reduce the learning rate
                spec.learning_rate /= 1.001

            # Bookkeeping - save data every 100 generations
            # Save models every 1000 generations
            if gen > 0 and not gen % 1000:
                self.save_data(gen)

                for i, species in enumerate(self.species):
                    species.save_model(self.trial_num, gen, self.p.fname_prepend, max_wts[i], species=i)

        self.env.visualize = False
        self.env.reset()
        # if random() < 0.05:
        #     # 5% of the time, change the location of the POIs slightly
        #     self.env.move_pois()

        self.save_data(gen=self.n_gen)
        # save the models
        for i, species in enumerate(self.species):
            species.save_model(self.trial_num, self.n_gen, self.p.fname_prepend, max_wts[i], species=i)
        # Run a rollout simulation
        self.env.reset()
        self.env.visualize = True
        for idx, spec in enumerate(self.species):
            spec.model.set_weights(max_wts[idx])
        models = [sp.model for sp in self.species]
        _ = self.env.run_sim(models, multi_g=True)

def main(p):
    for prepend in ['D_binary', 'G_binary']:
        p.fname_prepend = prepend
        print("TRIAL {}".format(p.trial_num))
        env = Domain(p)
        evo = CCEA(env, p)
        evo.run_evolution()


if __name__ == '__main__':
    pool = Pool()
    # trials = [param.p318, param.p319, param.p328, param.p329, param.p402, param.p403]
    trials = param.BIG_BATCH_01
    pool.map(main, trials)
    # p = param.p445
    # main(p)
    # multip = Process(target=main, args=(p,))
    # multip.start()
