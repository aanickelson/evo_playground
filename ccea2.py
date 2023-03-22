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

    def random_pairs(self):
        # Each policy is paired up randomly and tested three times
        test_each = 3
        idxs = np.zeros((test_each * lp.n_policies, self.p.n_agents), dtype=int)
        for j in range(self.p.n_agents):
            for i in range(test_each):
                idx = np.arange(lp.n_policies)
                np.random.shuffle(idx)
                idxs[i * lp.n_policies: (i + 1) * lp.n_policies, j] = idx
        return idxs

    def run_evo(self):
        for gen in tqdm(range(lp.n_gen)):
            g_vec = np.zeros((lp.n_policies, self.p.n_agents))
            d_vec = np.zeros((lp.n_policies, self.p.n_agents))

            # Creates random pairings of policies - tests each policy three times
            pairs = self.random_pairs()

            for pol_nums in pairs:
                # pol_nums = [idxs[pol0], idxs[pol0 + 1]]
                self.env.reset()

                pols = []
                for i, species in enumerate(self.species):
                    species.model.set_weights(species.weights[pol_nums[i]])
                    pols.append(species.model)

                G = np.sum(run_env(self.env, pols, self.p))
                D = np.sum(self.env.D(), axis=1)

                # Have to do this for each agent individually because their policy numbers are different
                for ag in range(self.p.n_agents):
                    g_vec[pol_nums[ag], ag] += G
                    d_vec[pol_nums[ag], ag] += D[ag]

            self.G[gen] = np.max(g_vec)

            for i, species in enumerate(self.species):
                if self.rew == 'G':
                    species.binary_tournament(g_vec[:, i])
                elif self.rew == 'D':
                    species.binary_tournament((d_vec[:, i]))

                species.mutate_weights()

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
    from evo_playground.parameters.parameters03 import Parameters as p03

    rewards = ['G']  # , 'D']  # , 'multi_G']
    data_names = ['G', 'D', 'mG']
    for params in [p02, p03]:
        if params.n_agents > 1:
            rewards = ['G', 'D']
        for reward in rewards:
            print(reward, ' - ',  params.param_idx)
            pooling = RunPool(params, reward, data_names)
            pooling.main(pooling.batch[0])
            # pooling.run_pool()
