"""
Adapted from evolutionary code written by github user Sir-Batman
https://github.com/AADILab/PyTorch-Evo-Strategies
"""
from tqdm import tqdm
import numpy as np
from teaming.domain import DiscreteRoverDomain as Domain
from scipy.stats import sem
from evo_playground.learning.evolve_population import EvolveNN as evoNN
from evo_playground.parameters.parameters01 import Parameters
from os import getcwd, path
from evo_playground.parameters.parameters00 import Parameters as p0
from evo_playground.parameters.parameters01 import Parameters as p1
from evo_playground.parameters.parameters01 import Parameters as p2


class BasicEvo:
    def __init__(self, env, p):
        self.n_gen = p.n_gen
        self.trial_num = p.trial_num
        self.n_agents = p.n_agents
        self.p = p
        self.env = env
        self.species = self.species_setup()
        self.generations = range(self.n_gen)
        self.generations = range(self.n_gen)
        self.min_score = np.zeros(self.n_gen)
        self.max_score = np.zeros(self.n_gen)
        self.avg_score = np.zeros(self.n_gen)
        self.sterr_score = np.zeros(self.n_gen)
        self.avg_false = np.zeros(self.n_gen)

    def species_setup(self):
        species = []
        for _ in range(self.n_agents):
            species.append(evoNN(self.env, self.p))
        return species

    def save_data(self):
        for i, species in enumerate(self.species):
            species.save_model(self.trial_num, species=i)

    def update_logs(self, scores, falses, i):
        self.min_score[i] = min(scores)
        self.max_score[i] = max(scores)
        self.avg_score[i] = np.mean(scores)
        self.sterr_score[i] = sem(scores)
        self.avg_false[i] = np.mean(falses)

    def run_evolution(self):
        for gen in tqdm(self.generations):
            scores = []
            falses = []
            mutated = [sp.mutate_weights(sp.start_weights) for sp in self.species]

            for pol_num in range(self.p.n_policies):
                wts = [mutated[i][pol_num] for i in range(self.n_agents)]
                for idx, spec in enumerate(self.species):
                    spec.model.set_weights(wts[idx])
                models = [sp.model for sp in self.species]

                self.env.new_env()
                G, avg_false = self.env.run_sim(models)
                scores.append(G)
                falses.append(avg_false)

            self.update_logs(scores, falses, gen)
            for idx, spec in enumerate(self.species):
                spec.start_weights = spec.update_weights(spec.start_weights, mutated[idx], np.array(scores))


if __name__ == '__main__':
    for p in [p0, p1, p2]:
        env = Domain(p)
        evo = BasicEvo(env, p)
        evo.run_evolution()
