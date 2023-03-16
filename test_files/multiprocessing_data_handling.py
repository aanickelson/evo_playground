import numpy
import numpy as np
from multiprocessing import Process, Pool, shared_memory
from time import sleep


def alter_data(param):
    shm_name, stat_runs, gens, curr_stat = param
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    c = np.ndarray((stat_runs, gens), dtype=np.float16, buffer=existing_shm.buf)
    for i in range(gens):
        c[curr_stat][i] = np.random.uniform(curr_stat, curr_stat+1)
        sleep(0.001)


if __name__ == '__main__':

    n_stat_runs = 5
    n_gens = 100

    shm = shared_memory.SharedMemory(create=True, size=np.zeros((n_stat_runs, n_gens), dtype=np.float16).nbytes)

    # Now create a NumPy array backed by shared memory
    G = np.ndarray((n_stat_runs, n_gens), dtype=np.float16, buffer=shm.buf)
    G[:] = -1.0
    share_name = shm.name  # We did not specify a name so one was chosen for us
    batch = []
    for stat in range(n_stat_runs):
        batch.append([share_name, n_stat_runs, n_gens, stat])
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    print(G)
    pool = Pool()
    pool.map(alter_data, batch)

    print(G)
