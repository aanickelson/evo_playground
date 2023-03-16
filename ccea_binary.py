"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
from multiprocessing import Process, Pool
from random import seed
from datetime import datetime


# Custom packages
# from teaming.domain import DiscreteRoverDomain as Domain
from AIC.aic import aic as Domain
from evo_playground.run_wrapper import run_env
from parameters.learningparams00 import LearnParams as lp
from evo_playground.learning.binary_species import Species



class CCEA:
    def __init__(self, env, p, rew_type, fpath, coll_data):
        seed()
        self.p = p
        self.param_idx = p.param_idx
        self.n_agents = p.n_agents
        self.env = env
        self.coll_data = coll_data

        self.n_gen = lp.n_gen
        self.generations = range(self.n_gen)
        self.n_stat_runs = lp.n_stat_runs

        self.stat_num = 0
        self.species = None
        self.raw_g = np.zeros((self.n_stat_runs, self.n_gen)) - 1
        self.norm_G = np.zeros((self.n_stat_runs, self.n_gen))
        self.avg_score = np.zeros((self.n_stat_runs, self.n_gen))
        self.sterr_score = np.zeros((self.n_stat_runs, self.n_gen))
        self.d = np.zeros((self.n_stat_runs, self.n_gen, self.n_agents))
        self.rew_type = rew_type
        self.base_fpath = fpath
        self.nn_in = self.env.state_size()
        self.nn_hid = lp.hid
        self.nn_out = self.env.action_size()
        self.fpath = path.join(fpath, rew_type)

    def species_setup(self):
        species = [Species(self.env, lp, self.nn_in, self.nn_hid, self.nn_out) for _ in range(self.n_agents)]
        return species

    # def save_data(self):
    #     # attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
    #     # attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
    #     attrs = [self.norm_G, self.raw_g]
    #     attr_names = ["norm_G", 'raw_G']
    #     for j in range(len(attrs)):
    #         nm = attr_names[j]
    #         att = attrs[j]
    #         filename = self.rew_type + "_trial{:03d}_{}".format(self.param_idx, nm)
    #         ext = "npy"
    #         path_nm = path.join(self.fpath, "{}.{}".format(filename, ext))
    #
    #         # do_not_overwrite(fp, filename, ext, att, isnp=True)
    #         np.save(path_nm, att)
    #
    # def update_logs(self, normalized_G, raw_G, i):
    #     self.norm_G[self.stat_num][i] = max(normalized_G)
    #     # self.avg_score[stat][i] = np.mean(scores)
    #     # self.sterr_score[stat][i] = sem(scores)
    #     self.raw_g[self.stat_num][i] = max(raw_G)

    def run_evolution(self):
        theoretical_max_g = self.p.n_pois
        self.species = self.species_setup()

        for gen in tqdm(self.generations):

            # Bookkeeping
            normalized_G = np.zeros(lp.n_policies)
            d_scores = np.zeros((self.n_agents, lp.n_policies))
            raw_G = np.zeros(lp.n_policies)

            # Mutate weights for all species
            for spec in self.species:
                spec.mutate_weights()
            one_gen_G = []
            for pol_num in range(len(self.species[0].weights)):
                # Reset the environment
                self.env.reset()
                self.env.vis = False
                # Pick one policy from each species
                # wts = [self.species[i].weights[pol_num] for i in range(self.n_agents)]

                # For each species
                for idx, spec in enumerate(self.species):
                    # Set the current policy
                    spec.model.set_weights(spec.weights[pol_num])

                # Array of one NN per species to use as policies
                models = [sp.model for sp in self.species]

                # Run the simulation
                G = sum(run_env(self.env, models, self.p))
                # multi = self.env.multiG()
                # G = self.env.high_level_G()

                D = self.env.D()
                # Bookkeeping
                d_scores[:, pol_num] = np.sum(D, axis=1)
                raw_G[pol_num] = G
                normalized_G[pol_num] = float(G) / theoretical_max_g

            # Index of the policies that performed best over G
            max_g = np.max(raw_G)
            argmax_g = np.argmax(raw_G)
            # Policies that performed best
            max_wts = [self.species[sp].weights[argmax_g] for sp in range(self.n_agents)]

            # Bookkeeping
            self.coll_data.update_logs(normalized_G, raw_G, gen, self.stat_num)

            # Update the starting weights (the policy we keep between generations) for each species
            for idx, spec in enumerate(self.species):
                # If the best of this group is better than the previous best, replace the hall of famers
                if max_g > spec.hof_score:
                    spec.hof_score = max_g
                    spec.hof_wts = spec.weights[argmax_g]
                if 'G' in self.rew_type:
                    # Use raw G because the scores may be more noisy
                    spec.binary_tournament(np.array(raw_G))
                elif 'D' in self.rew_type:
                    spec.binary_tournament(np.array(d_scores[idx]))

                # Reduce the learning rate
                spec.learning_rate /= 1.001

            # Bookkeeping - save data every 100 generations
            # Save models every 1000 generations
            if not gen % 100:
                self.coll_data.save_data()

                for i, species in enumerate(self.species):
                    species.save_model(self.param_idx, self.stat_num, gen, self.rew_type, max_wts[i], species=i)

        self.env.visualize = False
        self.env.reset()
        # if random() < 0.05:
        #     # 5% of the time, change the location of the POIs slightly
        #     self.env.move_pois()

        self.coll_data.save_data()
        # save the models
        for i, species in enumerate(self.species):
            species.save_model(self.param_idx, self.stat_num, gen, self.rew_type, max_wts[i], species=i)
        # # Run a rollout simulation
        self.env.reset()
        # self.env.vis = True
        # for idx, spec in enumerate(self.species):
        #     spec.model.set_weights(max_wts[idx])
        # models = [sp.model for sp in self.species]
        # _ = self.env.run_sim(models)


class CollectiveData:
    def __init__(self, n_stat_runs, n_gen, n_agents, rew_type, fpath, param_idx):
        self.raw_g = np.zeros((n_stat_runs, n_gen)) - 1
        self.norm_G = np.zeros((n_stat_runs, n_gen)) - 1
        self.avg_score = np.zeros((n_stat_runs, n_gen)) - 1
        self.sterr_score = np.zeros((n_stat_runs, n_gen)) - 1
        self.d = np.zeros((n_stat_runs, n_gen, n_agents)) - 1
        self.rew_type = rew_type
        self.fpath = fpath
        self.param_idx = param_idx

    def save_data(self):
        # attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
        # attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
        attrs = [self.norm_G, self.raw_g, self.d]
        attr_names = ["norm_G", 'raw_G', 'D']
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            filename = self.rew_type + "_trial{:03d}_{}".format(self.param_idx, nm)
            ext = "npy"
            path_nm = path.join(self.fpath, "{}.{}".format(filename, ext))

            # do_not_overwrite(fp, filename, ext, att, isnp=True)
            np.save(path_nm, att)

    def update_logs(self, normalized_G, raw_G, i, stat_num):
        # TODO: these names suck
        self.norm_G[stat_num][i] = max(normalized_G)
        # self.avg_score[stat][i] = np.mean(scores)
        # self.sterr_score[stat][i] = sem(scores)
        self.raw_g[stat_num][i] = max(raw_G)


class RunPool:
    def __init__(self, p):
        self.p = p
        self.batch = [[p, i] for i in range(lp.n_stat_runs)]
        self.fpath = None
        self.rew_types = ['G']
        self.make_dirs()
        self.rew = 'G'
        self.coll_data = CollectiveData(lp.n_stat_runs, lp.n_gen, p.n_agents, self.rew, path.join(self.fpath, self.rew), p.param_idx)

    def make_dirs(self):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        filepath = path.join(getcwd(), 'data', f"base_{self.p.param_idx:03d}_{now_str}")
        poi_fpath = path.join(filepath, 'poi_xy')
        self.fpath = filepath
        try:
            mkdir(filepath)
        except FileExistsError:
            mkdir(filepath+'_01')
        mkdir(poi_fpath)
        for rew in self.rew_types:
            fpath = path.join(filepath, rew)
            mkdir(fpath)

    def main(self, vals):
        p, stat_num = vals
        env = Domain(p)
        poi_fpath = path.join(self.fpath, 'poi_xy', f'poi_xy_trial{p.param_idx}')
        # env.save_poi_locs(poi_fpath)
        print("TRIAL {}".format(p.param_idx))

        for rew in self.rew_types:
            p.rew_str = rew
            # fpath = path.join(self.fpath, rew)
            evo = CCEA(env, p, rew, self.fpath, self.coll_data)
            evo.stat_num = stat_num
            evo.run_evolution()

    def run_pool(self):
        pool = Pool()
        pool.map(self.main, self.batch)


if __name__ == '__main__':
    # trials = param.BIG_BATCH_01
    from AIC.parameter import parameter as p
    for _ in range(1):
        pooling = RunPool(p)
        pooling.main(pooling.batch[0])
        # pooling.main(trials[1])
        # pooling.run_pool()
