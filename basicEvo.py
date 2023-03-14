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
from evo_playground.parameters.parameters04 import Parameters as p4
from evo_playground.parameters.parameters01 import Parameters as p1


class BasicEvo:
    def __init__(self, env, p):
        self.n_gen = p.n_gen
        self.trial_num = p.param_idx
        self.env = env
        self.evoNN = evoNN(self.env, p)

        self.generations = range(self.n_gen)
        self.min_score = np.zeros(self.n_gen)
        self.max_score = np.zeros(self.n_gen)
        self.avg_score = np.zeros(self.n_gen)
        self.sterr_score = np.zeros(self.n_gen)
        self.avg_false = np.zeros(self.n_gen)

    def update_logs(self, scores, falses, i):
        # scores = [score_genome(c) for c in candidates]
        self.min_score[i] = min(scores)
        self.max_score[i] = max(scores)
        self.avg_score[i] = np.mean(scores)
        self.sterr_score[i] = sem(scores)
        self.avg_false[i] = np.mean(falses)

    def save_data(self):
        self.evoNN.save_model(self.trial_num)
        cwd = getcwd()

        attrs = [self.min_score, self.max_score, self.avg_score, self.sterr_score, self.avg_false]
        attr_names = ["min", "max", "avg", "sterr", "false"]
        for j in range(len(attrs)):
            nm = attr_names[j]
            att = attrs[j]
            fp = path.join(cwd, "data")
            filename = "trial{:03d}_{}".format(self.trial_num, nm)
            ext = "csv"
            path_nm = path.join(fp, "{}.{}".format(filename, ext))

            # do_not_overwrite(fp, filename, ext, att, isnp=True)
            np.savetxt(path_nm, att, delimiter=",")

    def run_evolution(self):
        for gen in tqdm(self.generations):
            scores = []
            falses = []
            candidates = self.evoNN.mutate_weights(self.evoNN.start_weights)

            for c in candidates:
                sc, fs = self.evoNN.score_genome(c)
                scores.append(sc)
                falses.append(fs)

            self.update_logs(scores, falses, gen)
            self.evoNN.start_weights = self.evoNN.update_weights(self.evoNN.start_weights, candidates, np.array(scores))
            # if gen > 0 and not gen % 100:
            #     self.save_data()

        print(f"ending score: {self.evoNN.score_genome(self.evoNN.start_weights)[0]}")
        self.evoNN.score_genome(self.evoNN.start_weights)
        # self.save_data()


if __name__ == '__main__':
    for p in [p4]:
        env = Domain(p)
        evo = BasicEvo(env, p)
        evo.run_evolution()
