# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
from multiprocessing import Process, Pool, shared_memory, cpu_count
from random import seed
from datetime import datetime
from time import sleep
from torch import manual_seed


# Custom packages
# from teaming.domain import DiscreteRoverDomain as Domain
from AIC.aic import aic as Domain
from evo_playground.run_env import run_env
from parameters.learningparams01 import LearnParams as lp
from evo_playground.learning.binary_species import Species


class CCEA:
    def __init__(self, stat, p, rew, fpath, data_nms):
        self.stat_nm = stat
        self.p = p
        self.rew = rew  # Reward type for policy selection
        self.fpath = fpath
        self.env = Domain(self.p)
        self.data_nms = data_nms
        self.G = np.zeros(lp.n_gen, dtype=np.float16) - 1   # subtract one to indicate missing data
        self.mG = np.zeros((lp.n_gen, p.n_poi_types), dtype=np.float16) - 1
        self.D = np.zeros((lp.n_gen, p.n_agents), dtype=np.float16) - 1
        self.data = [self.G, self.D, self.mG]
        self.species = self.create_species()

    def create_species(self):
        manual_seed(self.stat_nm)
        return [Species(lp, self.env.state_size(), lp.hid, self.env.action_size())
                for _ in range(self.p.n_agents)]

    def save_data(self):
        for i, nm in enumerate(self.data_nms):
            fpath = path.join(self.fpath, nm, f'{self.p.param_idx:03d}_{self.stat_nm:02d}')
            np.save(fpath, self.data[i])

    def run_evo(self):
        for gen in tqdm(range(lp.n_gen)):
            g_vec = np.zeros(lp.n_policies) - 1
            d_vec = np.zeros((lp.n_policies, self.p.n_agents)) - 1
            for pol_num in range(lp.n_policies):
                self.env.reset()

                pols = []
                for species in self.species:
                    species.model.set_weights(species.weights[pol_num])
                    pols.append(species.model)

                g_vec[pol_num] = np.sum(run_env(self.env, pols, self.p))
                d_vec[pol_num] = np.sum(self.env.D(), axis=1)

            self.G[gen] = max(g_vec)

            for i, species in enumerate(self.species):
                if self.rew == 'G':
                    species.binary_tournament(g_vec)
                elif self.rew == 'D':
                    species.binary_tournament((d_vec[:, i]))

                species.mutate_weights()

                # if not gen % 10:
                #     species.weights = species.weights[:-10]
                #     species.add_new_pols()

            if not gen % 200:
                self.save_data()

        self.save_data()


class RunPool:
    def __init__(self, param, rew, data_nms):
        self.p = param
        self.rew = rew
        self.data_nms = data_nms
        self.fpath = None
        self.mk_dir()
        self.batch = [i for i in range(lp.n_stat_runs)]

    def mk_dir(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.fpath = path.join(getcwd(), 'data', f"base_{self.rew}_{self.p.param_idx:03d}_{now}")
        mkdir(self.fpath)
        for data_nm in self.data_nms:
            data_fpath = path.join(self.fpath, data_nm)
            mkdir(data_fpath)

    def main(self, stat_nm):
        print(f'Starting {stat_nm}')
        evo = CCEA(stat_nm, self.p, self.rew, self.fpath, self.data_nms)
        evo.run_evo()

    def run_pool(self):
        with Pool(processes=cpu_count() - 1) as pool:
            pool.map(self.main, self.batch)


if __name__ == '__main__':
    from evo_playground.parameters.parameters01 import Parameters as p01
    from evo_playground.parameters.parameters02 import Parameters as p02

    rewards = ['G']  # , 'D']  # , 'multi_G']
    data_names = ['G', 'D', 'mG']
    for params in [p01, p02]:
        if params.n_agents > 1:
            rewards = ['G', 'D']
        for reward in rewards:
            print(reward, ' - ',  params.param_idx)
            pooling = RunPool(params, reward, data_names)
            # pooling.main(pooling.batch[0])
            pooling.run_pool()
