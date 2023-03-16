# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
from multiprocessing import Process, Pool, shared_memory
from random import seed
from datetime import datetime


# Custom packages
# from teaming.domain import DiscreteRoverDomain as Domain
from AIC.aic import aic as Domain
# from evo_playground.run_wrapper import run_env
from parameters.learningparams00 import LearnParams as lp
from evo_playground.learning.binary_species import Species


class CCEA:
    def __init__(self, stat, p, rew, fpath, sh_nms):
        self.stat_nm = stat
        self.p = p
        self.rew = rew
        self.fpath = fpath
        self.G = None
        self.mG = None
        self.D = None
        # self.shm_setup(sh_nms)

    def shm_setup(self, shm_names):
        # Shared memory names for G, multi-G, and D
        sh_G, sh_mG, sh_D = shm_names
        ex_shm_G = shared_memory.SharedMemory(name=sh_G)
        ex_shm_mG = shared_memory.SharedMemory(name=sh_mG)
        ex_shm_D = shared_memory.SharedMemory(name=sh_D)
        shG = np.ndarray((lp.n_stat_runs, lp.n_gen), dtype=np.float16, buffer=ex_shm_G.buf)
        shmG = np.ndarray((lp.n_stat_runs, lp.n_gen, p.n_poi_types), dtype=np.float16, buffer=ex_shm_mG.buf)
        shD = np.ndarray((lp.n_stat_runs, lp.n_gen, p.n_agents), dtype=np.float16, buffer=ex_shm_D.buf)
        return shG, shmG, shD

    def run_evo(self, shm_nms):
        G, mG, D = self.shm_setup(shm_nms)
        for gen in range(lp.n_gen):
            G[self.stat_nm, gen] = self.stat_nm + np.random.random()

            # if not gen % 200:
            #     np.save(self.fpath + 'G.npy', self.G)
        print(G)


class RunPool:
    def __init__(self, param, shm_names, rew):
        self.p = param
        # Shared memory names for G, multi-G, and D
        self.shm_names = shm_names
        self.rew = rew
        self.fpath = self.fpath_setup()
        self.mk_dir()
        self.batch = [[p, i] for i in range(lp.n_stat_runs)]

    def fpath_setup(self):
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.join(getcwd(), 'data', f"base_{self.p.param_idx:03d}_{now}")

    def mk_dir(self):
        try:
            mkdir(self.fpath)
        except FileExistsError:
            mkdir(self.fpath + '_01')

    def main(self, vals):
        params, stat_nm = vals
        print(f'Starting {stat_nm}')
        evo = CCEA(stat_nm, params, self.rew, self.fpath, self.shm_names)
        evo.run_evo(self.shm_names)

    def run_pool(self):
        pool = Pool()
        pool.map(self.main, self.batch)



if __name__ == '__main__':
    from AIC.parameter import parameter as p
    shm_G = shared_memory.SharedMemory(create=True, size=np.zeros((lp.n_stat_runs, lp.n_gen), dtype=np.float16).nbytes)
    shm_mG = shared_memory.SharedMemory(create=True, size=np.zeros((lp.n_stat_runs, lp.n_gen, p.n_poi_types), dtype=np.float16).nbytes)
    shm_D = shared_memory.SharedMemory(create=True, size=np.zeros((lp.n_stat_runs, lp.n_gen, p.n_agents), dtype=np.float16).nbytes)

    G = np.ndarray((lp.n_stat_runs, lp.n_gen), dtype=np.float16, buffer=shm_G.buf)
    mG = np.ndarray((lp.n_stat_runs, lp.n_gen, p.n_poi_types), dtype=np.float16, buffer=shm_mG.buf)
    D = np.ndarray((lp.n_stat_runs, lp.n_gen, p.n_agents), dtype=np.float16, buffer=shm_D.buf)

    G[:] = -1.0
    mG[:] = -1.0
    D[:] = -1.0

    shm_nms = [shm_G.name, shm_mG.name, shm_D.name]
    rewards = ['G', 'D', 'multi_G']

    pooling = RunPool(p, shm_nms, 'G')
    pooling.main(pooling.batch[0])
    # pooling.run_pool()