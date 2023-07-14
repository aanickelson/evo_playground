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
        p_wts = self.pol_data[2 + (2 * self.bh_size)]
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

    def get_pols(self, bh):
        """
        Get policies from centroid for given behavior
        """
        c = self.get_hash_from_cent(bh)
        return c, self.bh_dict[c]

    def select_pol(self, pols):
        """
        Select from the set of policies
        """
        pass

    # How to do closest distance if empty? Continually +/- v small values & retest until I find one?


if __name__ == '__main__':
    pol_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/015_20230711_151938/502_119_run0_max/fin_arch_reduced.dat"
    cent_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/015_20230711_151938/502_119_run0_max/centroids_5000_5.dat"
    p = PolicyMap(pol_file, cent_file, 5)
    val = np.random.random(5)
    print(val, p.get_pols(val))
