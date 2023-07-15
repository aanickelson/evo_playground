from AIC.aic import aic
import pymap_elites_multiobjective.parameters as Params
from evo_playground.support.rover_wrapper import RoverWrapper
import numpy as np
from sklearn.neighbors import KDTree


class PolicyMap:
    def __init__(self, pname, cname, bh_size):
        self.bh_size = bh_size

        self.centroids = self.load_data(cname)
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

    def unpack_pol_data(self):
        """
        Unpack policy data to separate fitnesses, centroids, behaviors, and weights
        """
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

    def select_pol(self, pols):
        """
        Select from the set of policies
        """
        ch = np.random.choice(pols)
        return self.p_wts[ch]

    def get_pol(self, bh):
        _, p_idxs = self.get_pols_from_bh(bh)
        tries = 0
        # How to do closest distance if empty? Continually +/- v small values & retest until I find one?
        while not p_idxs and tries < 50:
            bh += np.random.uniform(-0.05, 0.05, self.bh_size)
            _, p_idxs = self.get_pols_from_bh(abs(bh))
            tries += 1
        if not p_idxs:
            return []
        selected_pol = self.select_pol(p_idxs)
        return selected_pol



if __name__ == '__main__':
    pol_file = "/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/111_run0/weights_100000.dat"
    cent_file = "/pymap_elites_multiobjective/scripts_data/data/518_20230713_151107/111_run0/centroids_2000_6.dat"
    bh_sz = 6
    pol_map = PolicyMap(pol_file, cent_file, bh_sz)
    en = aic(Params.p500)
    wra = RoverWrapper(en)
    wra.vis = False
    wra.use_bh = False

    # for _ in range(100):
    #     v = np.random.random(bh_sz)
    #     print(v)
    v1 = [0, 0, 0, 1, 1, 1]
    v2 = [1, 1, 1, 0, 0, 0]
    p1 = pol_map.get_pol(v1)
    p2 = pol_map.get_pol(v2)
    if len(p1) > 0 and len(p2) > 0:
        print(wra._evaluate([p1, p2]))
        # break