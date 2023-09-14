from AIC.aic import aic
import pymap_elites_multiobjective.parameters as Params
from evo_playground.support.rover_wrapper import RoverWrapper
from evo_playground.radians_G import G_exp
import numpy as np
from numpy import inf
from sklearn.neighbors import KDTree
import pymap_elites_multiobjective.scripts_data.often_used as util
import copy


class PolicyMap:
    def __init__(self, pname, cname, bh_size):
        self.bh_size = bh_size

        self.centroids = self.load_data(cname)[:, :self.bh_size]
        self.kdt = KDTree(self.centroids, leaf_size=30, metric='euclidean')

        self.pol_data = self.load_data(pname)
        self.p_fits, self.p_cents, self.p_desc, self.p_wts = self.unpack_pol_data()
        self.pol_idxs = np.arange(0, self.pol_data.shape[0])
        self.pareto_idxs = self.calc_pareto_data(self.p_fits)
        # Yes this is dumb you have to do it twice. But the first time is to see what centroids each value is in
        # The second time re-sets it after removing the empty centroids.
        self.bh_dict, self.centroids = self.create_bh_dict()
        self.kdt = KDTree(self.centroids, leaf_size=30, metric='euclidean')
        self.theta_fits = None
        self.exp_vals = None
        self.g_exp_setup()

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

    def calc_pareto_data(self, fitnesses, layers=10):
        fit = np.zeros(fitnesses.shape[0])
        fits = fitnesses.copy()
        for lay in range(layers, 0, -1):
            pareto = util.is_pareto_efficient_simple(fits)
            fit[pareto] = 1
            fits[pareto] = [0, 0]
        return np.nonzero(fit)[0]

    def g_exp_setup(self):
        self.p_fits = rem_div_zero(self.p_fits)
        self.theta_fits = np.arctan(self.p_fits[:, 1] / self.p_fits[:, 0])
        self.exp_vals = np.sqrt(self.p_fits[:, 0] ** 2 + self.p_fits[:, 1] ** 2)

    def G_exp(self, pol_nums, desired_wts):
        b = rem_div_zero(desired_wts)
        # Get the angle of the preferred balance and of the g balance
        theta_wts = np.arctan(b[1] / b[0])
        # theta_gs = np.arctan(g[1] / g[0])
        # Take the difference between the angles, then scale so closer to the angle has a higher value
        # -10 is a scalar you can play with. Higher scalar values give a steeper gradient near the ideal tradeoff
        exp_val = np.exp(-5. * abs(theta_wts - self.theta_fits[pol_nums]))
        # Then scale so further from the origin has a higher value
        return exp_val * self.exp_vals[pol_nums]

    def create_bh_dict(self):
        """
        Create dictionary mapping policies to centroids
        """
        bh_dict = {tuple(cent): [] for cent in self.centroids}
        for i, c in enumerate(self.p_cents):
            hval = self.get_hash_from_cent(c)
            bh_dict[hval].append(i)
        bhd = {k:v for k, v in bh_dict.items() if v}
        bhd_keys = [k for k, _ in bhd.items()]
        return bhd, bhd_keys

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

    def select_pol(self, pols, w):
        """
        Select from the set of policies
        """
        w = np.array(w)
        # f = np.array(self.p_fits[pols])
        if len(w) > 1:
            # this is the case where the output is [g0, g1] instead of g0/g1
            # diff = []
            diff = self.G_exp(pols, w)
        else:
            wts = np.array([w, 1 - w])
            diff = self.G_exp(pols, wts)
        # pick the one that is closest
        pol_num = pols[np.argmax(diff)]
        return self.p_wts[pol_num]

    def get_pol(self, nn_out, only_bh, only_obj):

        if only_obj:
            wts = nn_out
            selected_pol = self.select_pol(self.pareto_idxs, wts)
        else:
            bh = nn_out[:self.bh_size]
            _, p_idxs = self.get_pols_from_bh(bh)
            if not p_idxs:
                return []
            if only_bh:
                pol_idx = np.random.choice(p_idxs)
                selected_pol = self.p_wts[pol_idx]
            else:
                wts = nn_out[self.bh_size:]
                selected_pol = self.select_pol(p_idxs, wts)

        return selected_pol


def rem_div_zero(v, min_val=1e-20):
    vals = np.array(v)
    vals[vals < min_val] = min_val
    return vals


if __name__ == '__main__':
    pol_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/516_20230726_160858/219_run2/weights_200000.dat"
    cent_file = "/home/anna/PycharmProjects/pymap_elites_multiobjective/scripts_data/data/516_20230726_160858/219_run2/centroids_2000_6.dat"
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