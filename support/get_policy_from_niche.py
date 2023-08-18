from AIC.aic import aic
import pymap_elites_multiobjective.parameters as Params
from evo_playground.support.rover_wrapper import RoverWrapper
import numpy as np
from numpy import inf
from sklearn.neighbors import KDTree
import copy


class PolicyMap:
    def __init__(self, pname, cname, bh_size):
        self.bh_size = bh_size

        self.centroids = self.load_data(cname)[:, :self.bh_size]
        self.kdt = KDTree(self.centroids, leaf_size=30, metric='euclidean')

        self.pol_data = self.load_data(pname)
        self.p_fits, self.p_cents, self.p_desc, self.p_wts = self.unpack_pol_data()

        self.bh_dict = self.create_bh_dict()

    def load_data(self, fname):
        """
        Load trained policy data
        """
        data = np.loadtxt(fname)
        return data

    def norm_fits(self, fits):
        fits[:, 0] = fits[:, 0] / max(fits[:, 0])
        fits[:, 1] = fits[:, 1] / max(fits[:, 1])
        return fits

    def unpack_pol_data(self):
        """
        Unpack policy data to separate fitnesses, centroids, behaviors, and weights
        """
        # p_fits = self.norm_fits(self.pol_data[:, :2])
        p_fits = self.pol_data[:, :2]
        p_cents = self.pol_data[:, 2:2 + self.bh_size]
        p_desc = self.pol_data[:, 2 + self.bh_size: 2 + (2 * self.bh_size)]
        p_wts = self.pol_data[:, 2 + (2 * self.bh_size):]
        return p_fits, p_cents, p_desc, p_wts

    def create_bh_dict(self):
        """
        Create dictionary mapping policies to centroids
        """
        bh_dict = {tuple(cent): [] for cent in self.centroids}
        for i, c in enumerate(self.p_cents):
            hval = self.get_hash_from_cent(c)
            bh_dict[hval].append(i)
        return bh_dict

    def get_hash_from_cent(self, centroid):
        niche_index = self.kdt.query([centroid], k=1)[1][0][0]
        niche_kdt = self.kdt.data[niche_index]
        cent = tuple(map(float, niche_kdt))
        return cent

    def get_pols_from_bh(self, bh):
        """
        Get policies from centroid for given behavior
        """
        c = self.get_hash_from_cent(bh)
        return c, self.bh_dict[c]

    def select_pol(self, pols, wts):
        """
        Select from the set of policies
        """
        w = np.array(wts)
        f: object = np.array(self.p_fits[pols])
        if len(wts) > 1:
            # Calculate the Euclidean distance between the weights input and each fitness, then select the closest
            diff = np.linalg.norm(abs(f - w), axis=1)
        else:
            # avoid divide by 0 errors
            f[f == 0] = 1.0e-4
            # Find the ratio and normalize between [0, 1]
            f_ratio = np.array(f[:, 0] / f[:, 1])
            f_ratio[f_ratio == inf] = max(f_ratio[f_ratio != inf]) + 5
            norm = np.linalg.norm(f_ratio)
            f = f_ratio / norm
            diff = abs(f - w)
        pol_num = pols[np.argmin(diff)]
        return self.p_wts[pol_num]

    def get_pol(self, output):
        bh = output[:self.bh_size]
        wts = output[self.bh_size:]
        _, p_idxs = self.get_pols_from_bh(bh)
        if not p_idxs:
            return []
        selected_pol = self.select_pol(p_idxs, wts)
        return selected_pol



if __name__ == '__main__':
    pol_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/516_20230726_160858/219_run2/weights_200000.dat"
    cent_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/516_20230726_160858/219_run2/new_centroids_2000_6.dat"
    bh_sz = 6
    pol_map = PolicyMap(pol_file, cent_file, bh_sz)
    p = copy.deepcopy(Params.p219)
    p.counter = 0
    p.n_agents = 2
    en = aic(p)
    wra = RoverWrapper(en)
    wra.vis = False
    wra.use_bh = False

    # for _ in range(100):
    #     v = np.random.random(bh_sz)
    #     print(v)
    v1 = [0, 0, 0, 1, 1, 1]
    v2 = [1, 1, 1, 0, 0, 0]
    p1 = pol_map.get_pol(v1, [0.5, 0.5])
    p2 = pol_map.get_pol(v2, [0.5, 0.5])
    # if len(p1) > 0 and len(p2) > 0:
    #     print(wra._evaluate([p1, p2]))
        # break