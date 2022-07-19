"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""
from tqdm import tqdm
import numpy as np
from teaming.domain import DiscreteRoverDomain as Domain
from scipy.stats import sem, entropy
from evo_playground.learning.evolve_population import EvolveNN as evoNN
from os import getcwd, path
from parameters import BATCH2, TEST_BATCH, BATCH3_SM
from optimal_comparison import optimal_policy
from multiprocessing import Process


class CCEA:
    def __init__(self, env, p):
        self.n_gen = p.n_gen
        self.trial_num = p.trial_num
        self.n_agents = p.n_agents
        self.p = p
        self.env = env
        self.species = self.species_setup()
        self.generations = range(self.n_gen)
        self.raw_g = np.zeros(self.n_gen)
        self.multi_g = np.zeros((self.n_gen, self.env.n_poi_types))
        self.max_score = np.zeros(self.n_gen)
        self.avg_score = np.zeros(self.n_gen)
        self.sterr_score = np.zeros(self.n_gen)
        self.avg_false = np.zeros(self.n_gen)
        self.d = np.zeros((self.n_gen, self.n_agents))

    def species_setup(self):
        species = []
        for _ in range(self.n_agents):
            species.append(evoNN(self.env, self.p))
        return species

    def save_data(self, gen):
        if not gen % 1000:
            for i, species in enumerate(self.species):
                species.save_model(self.trial_num, gen, species=i)
        cwd = getcwd()
        attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
        attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            fp = path.join(cwd, "data")
            filename = "D_trial{:02d}_{}".format(self.trial_num, nm)
            ext = "csv"
            path_nm = path.join(fp, "{}.{}".format(filename, ext))

            # do_not_overwrite(fp, filename, ext, att, isnp=True)
            np.savetxt(path_nm, att, delimiter=",")

    def update_logs(self, scores, falses, raw_G, multi_g, i):
        self.max_score[i] = max(scores)
        self.avg_score[i] = np.mean(scores)
        self.sterr_score[i] = sem(scores)
        self.avg_false[i] = np.mean(falses)
        self.raw_g[i] = np.max(raw_G)
        best_idx = np.argmax(scores)
        self.multi_g[i] = multi_g[best_idx]

    def run_evolution(self):
        # self.env.draw()
        for gen in tqdm(self.generations):
            # print(self.env.theoretical_max_g)
            scores = np.zeros(self.p.n_policies)
            d_scores = np.zeros((self.n_agents, self.p.n_policies))
            raw_G = np.zeros(self.p.n_policies)
            falses = np.zeros(self.p.n_policies)
            all_multi_g = np.zeros((self.p.n_policies, self.env.n_poi_types))
            # Mutate weights for all species
            mutated = [sp.mutate_weights(sp.start_weights) for sp in self.species]
            theoretical_max_g = optimal_policy(self.env)
            for pol_num in range(self.p.n_policies):
                # Pick one policy from each species
                wts = [mutated[i][pol_num] for i in range(self.n_agents)]

                for idx, spec in enumerate(self.species):
                    spec.model.set_weights(wts[idx])
                models = [sp.model for sp in self.species]

                self.env.reset()
                G, multi_g, avg_false = self.env.run_sim(models, multi_g=True)
                d_scores[:, pol_num] = self.env.D()
                rew = G / theoretical_max_g
                raw_G[pol_num] = G
                scores[pol_num] = rew
                falses[pol_num] = avg_false
                all_multi_g[pol_num] = multi_g

            self.update_logs(scores, falses, raw_G, all_multi_g, gen)
            # entr_scores = [entropy(i) for i in all_multi_g]
            for idx, spec in enumerate(self.species):
                spec.start_weights = spec.update_weights(spec.start_weights, mutated[idx], np.array(d_scores[idx]))  # np.array(entr_scores))
            if gen > 0 and not gen % 100:
                self.save_data(gen)
            self.env.new_env()

        self.save_data()


def main(p):
    print("TRIAL {}".format(p.trial_num))
    env = Domain(p)
    evo = CCEA(env, p)
    evo.run_evolution()

# def get_best_arr(all_g, multi_g):
#     h = [(-all_g[i], (-entropy(multi_g[i]), i, multi_g[i])) for i in range(len(all_g))]
#     # for i, g in enumerate(all_g):
#     #     h.append((-g, (-entropy(all_g[i]), multi_g[i])))
#     heapq.heapify(h)
#     best = heapq.heappop(h)
#     print("Heap result", best)
#     print(best[1][2])
#     return best[1][2]

if __name__ == '__main__':

    for p in BATCH3_SM:
        multip = Process(target=main, args=(p,))
        multip.start()
