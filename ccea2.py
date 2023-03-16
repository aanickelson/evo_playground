# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
from multiprocessing import Process, Pool, shared_memory
from random import seed
from datetime import datetime
from time import sleep


# Custom packages
# from teaming.domain import DiscreteRoverDomain as Domain
from AIC.aic import aic as Domain
from evo_playground.run_env import run_env
from parameters.learningparams00 import LearnParams as lp
from evo_playground.learning.binary_species import Species


class CCEA:
    def __init__(self, stat, p, rew, fpath):
        self.stat_nm = stat
        self.p = p
        self.rew = rew  # Reward type for policy selection
        self.fpath = fpath + f'/{self.p.param_idx:03d}_{self.stat_nm:02d}'
        self.env = Domain(self.p)
        self.G = np.zeros(lp.n_gen, dtype=np.float16) - 1   # subtract one to indicate missing data
        self.mG = np.zeros((lp.n_gen, p.n_poi_types), dtype=np.float16) - 1
        self.D = np.zeros((lp.n_gen, p.n_agents), dtype=np.float16) - 1
        self.species = self.create_species()

    def create_species(self):
        return [Species(lp, self.env.state_size(), lp.hid, self.env.action_size())
                for _ in range(self.p.n_agents)]

    def run_evo(self):
        for gen in range(lp.n_gen):
            g_vec = np.zeros(lp.n_policies)
            for species in self.species:
                species.mutate_weights()

            for pol_num in range(lp.n_policies):
                self.env.reset()

                pols = []
                for species in self.species:
                    species.model.set_weights(species.weights[pol_num])
                    pols.append(species.model)

                g_vec[pol_num] = run_env(self.env, pols, self.p)

            self.G[gen] = max(g_vec)
            for species in self.species:
                species.binary_tournament(g_vec)

            if not gen % 200:
                np.save(self.fpath, self.G)

        np.save(self.fpath, self.G)


class RunPool:
    def __init__(self, param, rew):
        self.p = param
        self.rew = rew
        self.fpath = None
        self.mk_dir()
        self.batch = [i for i in range(lp.n_stat_runs)]

    def mk_dir(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = path.join(getcwd(), 'data', f"base_{self.p.param_idx:03d}_{now}")
        self.fpath = path.join(fpath, self.rew)
        mkdir(fpath)
        mkdir(self.fpath)

    def main(self, stat_nm):
        print(f'Starting {stat_nm}')
        evo = CCEA(stat_nm, self.p, self.rew, self.fpath)
        evo.run_evo()

    def run_pool(self):
        pool = Pool()
        pool.map(self.main, self.batch)


if __name__ == '__main__':
    from AIC.parameter import parameter as params
    rewards = ['G', 'D', 'multi_G']
    pooling = RunPool(params, 'G')
    # pooling.main(pooling.batch[0])
    pooling.run_pool()
