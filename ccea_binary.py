"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from scipy.stats import sem, entropy
from os import getcwd, path, mkdir
from optimal_comparison import optimal_policy
from multiprocessing import Process, Pool
from random import seed
import heapq as hq
import torch
from datetime import datetime


# Custom packages
import parameters as param
from evo_playground.learning.evolve_population import EvolveNN as evoNN
from teaming.domain import DiscreteRoverDomain as Domain
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from evo_playground.learning.binary_species import Species



class CCEA:
    def __init__(self, env, p, rew_type, st_time, time_str, fpath):
        seed()
        self.n_gen = p.n_gen
        self.trial_num = p.trial_num
        self.n_agents = p.n_agents
        self.p = p
        self.env = env
        self.n_stat_runs = 2
        self.species = None
        self.generations = range(self.n_gen)
        self.raw_g = np.zeros((self.n_stat_runs, self.n_gen))
        # self.multi_g = np.zeros((self.n_gen, self.env.n_poi_types))
        self.max_score = np.zeros((self.n_stat_runs, self.n_gen))
        self.avg_score = np.zeros((self.n_stat_runs, self.n_gen))
        self.sterr_score = np.zeros((self.n_stat_runs, self.n_gen))
        # self.avg_false = np.zeros(self.n_gen)
        self.d = np.zeros((self.n_stat_runs, self.n_gen, self.n_agents))
        self.rew_type = rew_type
        self.st_time = st_time
        self.time_str = time_str
        self.fpath = fpath

    def species_setup(self, time_or_no):
        species = [Species(self.env, self.p, time_or_no) for _ in range(self.n_agents)]
        return species

    def save_data(self):
        # attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
        # attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
        attrs = [self.max_score, self.raw_g]
        attr_names = ["max_norm", 'max_G']
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            filename = self.rew_type + "_" + self.time_str + "_trial{:03d}_{}".format(self.trial_num, nm)
            ext = "npy"
            path_nm = path.join(self.fpath, "{}.{}".format(filename, ext))

            # do_not_overwrite(fp, filename, ext, att, isnp=True)
            np.save(path_nm, att)

    def update_logs(self, scores, raw_G, i, stat):
        self.max_score[stat][i] = max(scores)
        # self.avg_score[stat][i] = np.mean(scores)
        # self.sterr_score[stat][i] = sem(scores)
        self.raw_g[stat][i] = np.max(raw_G)

    def run_evolution(self):
        # Comparison of theoretical max for simple G
        # IF CHANGING THE ENVIRONMENT, put this the loop
        theoretical_max_g = optimal_policy(self.env)

        for stat_num in range(self.n_stat_runs):
            self.species = self.species_setup(self.st_time)
            for gen in tqdm(self.generations):

                # Bookkeeping
                scores = np.zeros(self.p.n_policies)
                d_scores = np.zeros((self.n_agents, self.p.n_policies))
                raw_G = np.zeros(self.p.n_policies)

                # Mutate weights for all species
                for spec in self.species:
                    spec.mutate_weights()

                for pol_num in range(self.p.n_policies):
                    # Reset the environment
                    self.env.reset()
                    self.env.visualize = False

                    # Pick one policy from each species
                    wts = [self.species[i].weights[pol_num] for i in range(self.n_agents)]

                    # For each species
                    for idx, spec in enumerate(self.species):
                        # Set the current policy
                        spec.model.set_weights(wts[idx])

                    # Array of one NN per species to use as policies
                    models = [sp.model for sp in self.species]

                    # Run the simulation
                    G, D = self.env.run_sim(models, use_time=self.st_time)

                    # Bookkeeping
                    d_scores[:, pol_num] = D
                    raw_G[pol_num] = G
                    scores[pol_num] = G / theoretical_max_g
                # Index of the policies that performed best over G
                max_g = np.argmax(raw_G)
                # Policies that performed best
                max_wts = [self.species[sp].weights[max_g] for sp in range(self.n_agents)]

                # Bookkeeping
                self.update_logs(scores, raw_G, gen, stat_num)

                # Update the starting weights (the policy we keep between generations) for each species
                for idx, spec in enumerate(self.species):
                    if 'G' in self.rew_type:
                        # Use raw G because the scores may be more noisy (since it's divided by the greedy policy)
                        spec.start_weights = spec.binary_tournament(np.array(raw_G))
                    elif 'D' in self.rew_type:
                        spec.start_weights = spec.binary_tournament(np.array(d_scores[idx]))

                    # Reduce the learning rate
                    spec.learning_rate /= 1.0001

                # Bookkeeping - save data every 100 generations
                # Save models every 1000 generations
                if gen > 0 and not gen % 200:
                    self.save_data()

                    for i, species in enumerate(self.species):
                        species.save_model(self.trial_num, stat_num, self.n_gen, self.rew_type + '_' + self.time_str, max_wts[i], species=i)

            self.env.visualize = False
            self.env.reset()
            # if random() < 0.05:
            #     # 5% of the time, change the location of the POIs slightly
            #     self.env.move_pois()

            self.save_data()
            # save the models
            for i, species in enumerate(self.species):
                species.save_model(self.trial_num, stat_num, self.n_gen, self.rew_type + '_' + self.time_str, max_wts[i], species=i)
            # # Run a rollout simulation
            # self.env.reset()
            # self.env.visualize = True
            # for idx, spec in enumerate(self.species):
            #     spec.model.set_weights(max_wts[idx])
            # models = [sp.model for sp in self.species]
            # _ = self.env.run_sim(models, multi_g=True)


class RunPool:
    def __init__(self, batch):
        self.batch = batch
        self.fpath = None
        self.make_dirs()

    def make_dirs(self):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M")
        filepath = path.join(getcwd(), 'data', now_str)
        poi_fpath = path.join(filepath, 'poi_xy')
        self.fpath = filepath
        mkdir(filepath)
        mkdir(poi_fpath)
        for rew in ['D_', 'G_']:
            for t in ['time', 'no_time']:
                fpath = path.join(filepath, rew + t)
                mkdir(fpath)

    def main(self, p):
        env = Domain(p)
        poi_fpath = path.join(self.fpath, 'poi_xy', f'poi_xy_trial{p.trial_num}')
        env.save_poi_locs(poi_fpath)
        print("TRIAL {}".format(p.trial_num))
        for rew in ['D', 'G']:
            for st_time in [False, True]:
                if st_time:
                    time_str = 'time'
                else:
                    time_str = 'no_time'
                print(rew, time_str)
                # p.fname_prepend = rew
                fpath = path.join(self.fpath, rew + '_' + time_str)
                evo = CCEA(env, p, rew, st_time, time_str, fpath)
                evo.run_evolution()

    def run_pool(self):
        pool = Pool()
        pool.map(self.main, self.batch)


if __name__ == '__main__':
    # trials = param.BIG_BATCH_01
    trials = [param.p98]
    pooling = RunPool(trials)
    pooling.main(trials[0])

