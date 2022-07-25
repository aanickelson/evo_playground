"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""

# Python packages
from tqdm import tqdm
import numpy as np
from scipy.stats import sem, entropy
from os import getcwd, path
import parameters
from optimal_comparison import optimal_policy
from multiprocessing import Process
from random import random

# Custom packages
from evo_playground.learning.evolve_population import EvolveNN as evoNN
from teaming.domain import DiscreteRoverDomain as Domain


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

        cwd = getcwd()
        attrs = [self.max_score, self.avg_score, self.sterr_score, self.avg_false, self.raw_g, self.multi_g]
        attr_names = ["max", "avg", "sterr", "false", 'avg_G', 'multi_g']
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            fp = path.join(cwd, "data")
            filename = self.p.fname_prepend + "trial{:02d}b_{}".format(self.trial_num, nm)
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
        for gen in tqdm(self.generations):
            scores = np.zeros(self.p.n_policies)
            d_scores = np.zeros((self.n_agents, self.p.n_policies))
            raw_G = np.zeros(self.p.n_policies)
            falses = np.zeros(self.p.n_policies)
            all_multi_g = np.zeros((self.p.n_policies, self.env.n_poi_types))
            # Mutate weights for all species
            mutated = [sp.mutate_weights(sp.start_weights) for sp in self.species]
            theoretical_max_g = optimal_policy(self.env)
            save_wts = []

            for pol_num in range(self.p.n_policies):
                # Pick one policy from each species
                wts = [mutated[i][pol_num] for i in range(self.n_agents)]

                for idx, spec in enumerate(self.species):
                    spec.model.set_weights(wts[idx])
                models = [sp.model for sp in self.species]

                self.env.reset()
                G, multi_g, avg_false = self.env.run_sim(models, multi_g=True)
                d_vec = self.env.D()
                d_scores[:, pol_num] = d_vec
                rew = G / theoretical_max_g
                raw_G[pol_num] = G
                scores[pol_num] = rew
                falses[pol_num] = avg_false
                all_multi_g[pol_num] = multi_g

            max_g = np.argmax(raw_G)
            max_wts = [mutated[sp][max_g] for sp in range(self.n_agents)]
            self.update_logs(scores, falses, raw_G, all_multi_g, gen)
            for idx, spec in enumerate(self.species):
                if 'G_' in p.fname_prepend:
                    spec.start_weights = spec.update_weights(spec.start_weights, mutated[idx], np.array(scores))
                elif 'D_' in p.fname_prepend:
                    _ = np.array(d_scores[idx])
                    spec.start_weights = spec.update_weights(spec.start_weights, mutated[idx], np.array(d_scores[idx]))
                spec.learning_rate *= 1.001
            if gen > 0 and not gen % 100:
                self.save_data(gen)
            if gen > 0 and not gen % 1000:
                for i, species in enumerate(self.species):
                    species.save_model(self.trial_num, gen, p.fname_prepend, max_wts[i], species=i)
            if gen == self.n_gen - 1:
                for i, species in enumerate(self.species):
                    species.save_model(self.trial_num, gen, p.fname_prepend, max_wts[i], species=i)
                self.env.visualize = True
                self.env.reset()
                for idx, spec in enumerate(self.species):
                    spec.model.set_weights(max_wts[idx])
                models = [sp.model for sp in self.species]
                _ = self.env.run_sim(models, multi_g=True)

            self.env.visualize = False
            # self.env.new_env()
            self.env.reset()
            # if random() < 0.05:
            #     # 5% of the time, change the location of the POIs slightly
            #     self.env.move_pois()
        self.save_data(gen=self.n_gen)


def main(p):
    print("TRIAL {}".format(p.trial_num))
    env = Domain(p)
    evo = CCEA(env, p)
    evo.run_evolution()


if __name__ == '__main__':
    for prepend in ['D_b', 'G_b']:
        for p in parameters.BATCH6:
            p.fname_prepend = prepend
            # main(p)
            multip = Process(target=main, args=(p,))
            multip.start()
