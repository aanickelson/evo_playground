"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from os import getcwd, path, mkdir
from multiprocessing import Process, Pool
from datetime import datetime
from matplotlib import pyplot as plt

# Custom packages
from teaming.domain import DiscreteRoverDomain as Domain
from ccea_binary import CCEA


class CCEA_MOO(CCEA):
    def __init__(self, env, p, rew_type, fpath):
        super().__init__(env, p, rew_type, fpath)
        self.max_possible_G = sum(sum(np.array(self.p.rooms)))

    # def save_data(self):
    #     pass
    #
    # def update_logs(self, normalized_G, raw_G, i, stat):
    #     pass

    def run_evolution(self):

        self.species = self.species_setup()

        for gen in tqdm(self.generations):
            # Mutate weights for all species
            for spec in self.species:
                spec.mutate_weights()

            if not gen % 500:
                pass

            normalized_G, d_scores, raw_G, multi_G = self.test_policies(gen)
            if not gen % 100:
                pareto = self.is_pareto_efficient_simple(multi_G)
                g1 = np.array([i[0] for i in multi_G])
                g2 = np.array([j[1] for j in multi_G])
                self.plot_it(g1, g2, pareto, gen)

            # Index of the policies that performed best over G
            max_g = np.argmax(raw_G)

            # Policies that performed best
            max_wts = [self.species[sp].weights[max_g] for sp in range(self.n_agents)]

            # Update the starting weights (the policy we keep between generations) for each species
            for idx, spec in enumerate(self.species):
                if 'G' in self.rew_type:
                    # Use raw G because the scores may be more noisy (since it's divided by the greedy policy)
                    spec.start_weights = spec.binary_tournament(np.array(raw_G))
                elif 'D' in self.rew_type:
                    spec.start_weights = spec.binary_tournament(np.array(d_scores[idx]))
                elif 'multi' in self.rew_type:
                    spec.start_weights = spec.binary_multi(np.array(multi_G))
                # Reduce the learning rate
                spec.learning_rate /= 1.0001

            # Bookkeeping - save data every 100 generations
            # Save models every 1000 generations
            if gen > 0 and not gen % 200:
                self.save_data()

                for i, species in enumerate(self.species):
                    species.save_model(self.trial_num, self.stat_num, self.n_gen, self.rew_type, max_wts[i], species=i)

        self.env.visualize = False
        self.env.reset()

        self.save_data()
        # save the models
        for i, species in enumerate(self.species):
            species.save_model(self.trial_num, self.stat_num, self.n_gen, self.rew_type, max_wts[i], species=i)
        # # Run a rollout simulation
        # self.env.reset()
        # self.env.vis = True
        # for idx, spec in enumerate(self.species):
        #     spec.model.set_weights(max_wts[idx])
        # models = [sp.model for sp in self.species]
        # _ = self.env.run_sim(models)
        self.stat_num += 1

    def test_policies(self, gen):
        # Bookkeeping
        normalized_G = np.zeros(self.p.n_policies)
        d_scores = np.zeros((self.n_agents, self.p.n_policies))
        raw_G = np.zeros(self.p.n_policies)
        multi_G = np.zeros((self.p.n_policies, self.p.n_poi_types))
        
        for pol_num in range(self.p.n_policies):
            # Reset the environment
            self.env.reset()
            self.env.vis = False

            # Pick one policy from each species
            wts = [self.species[i].weights[pol_num] for i in range(self.n_agents)]

            # For each species
            for idx, spec in enumerate(self.species):
                # Set the current policy
                spec.model.set_weights(wts[idx])

            # Array of one NN per species to use as policies
            models = [sp.model for sp in self.species]

            if gen > 0 and not gen % 100:
                x = 1
                pass

            # Run the simulation
            self.env.run_sim(models)
            G = self.env.G()
            D = self.env.D()
            multiG = self.env.multiG()

            # Bookkeeping
            d_scores[:, pol_num] = D
            raw_G[pol_num] = G
            normalized_G[pol_num] = float(G) / self.max_possible_G
            multi_G[pol_num] = multiG

        # Bookkeeping
        self.update_logs(normalized_G, raw_G, gen, self.stat_num)

        return normalized_G, d_scores, raw_G, multi_G

    def plot_it(self, x, y, iseff, gen):
        plt.clf()
        plt.scatter(x, y, c='red')
        plt.scatter(x[iseff], y[iseff], c="blue")
        plt.savefig(self.fpath + f'pareto_gen{gen}')

    def is_pareto_efficient_simple(self, xyvals):
        """
        Find the pareto-efficient points
        This function copied from here: https://stackoverflow.com/a/40239615
        :param costs: An (n_points, n_costs) array
        :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
        """
        costs = np.array(xyvals)
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self
        return is_efficient


class RunPool:
    def __init__(self, batch):
        self.batch = batch
        self.fpath = None
        self.rewards_to_try = ['G']  # 'G', 'D', 'multi',
        self.make_dirs()

    def make_dirs(self):
        now = datetime.now()
        now_str = now.strftime("%Y%m%d_%H%M%S")
        filepath = path.join(getcwd(), 'data', now_str)
        poi_fpath = path.join(filepath, 'poi_xy')
        self.fpath = filepath
        try:
            mkdir(filepath)
        except FileExistsError:
            mkdir(filepath+'_01')
        mkdir(poi_fpath)
        for rew in self.rewards_to_try:
            fpath = path.join(filepath, rew)
            mkdir(fpath)

    def main(self, p):
        env = Domain(p)
        poi_fpath = path.join(self.fpath, 'poi_xy', f'poi_xy_trial{p.trial_num}')
        env.save_poi_locs(poi_fpath)
        print("TRIAL {}".format(p.trial_num))

        for rew in self.rewards_to_try:
            p.rew_str = rew
            fpath = path.join(self.fpath, rew)
            evo = CCEA_MOO(env, p, rew, fpath)
            evo.run_evolution()

    def run_pool(self):
        pool = Pool()
        pool.map(self.main, self.batch)


if __name__ == '__main__':
    # trials = param.BIG_BATCH_01
    from parameters import p04 as p
    trials = [p] * p.n_stat_runs
    pooling = RunPool(trials)
    pooling.main(trials[0])
    # pooling.main(trials[1])
    # pooling.run_pool()