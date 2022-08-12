import numpy as np


class Species:
    def __init__(self):
        self.start_weights = [1.01, 1.02, 1.03]
        self.curr_weights = []

    def mutate_weights(self, weights, n_pol):
        weights_to_try = []
        for i in range(n_pol):
            new_wt = [wt + (i+1)*0.1 for wt in weights]
            new_list = [round(item, 2) for item in new_wt]
            weights_to_try.append(new_list)
        return weights_to_try

    def set_weights(self, wts):
        self.curr_weights = wts


if __name__ == '__main__':

    species = []
    n_species = 4
    n_policies = 2

    for i in range(4):
        s = Species()
        s.start_weights = [j + i for j in s.start_weights]
        species.append(s)

    for gen in range(3):
        mutated = [sp.mutate_weights(sp.start_weights, n_policies) for sp in species]
        for pol_num in range(n_policies):
            wts = [mutated[i][pol_num] for i in range(n_species)]
            for idx, spec in enumerate(species):
                spec.set_weights(wts[idx])
            models = [sp.curr_weights for sp in species]
            print(models)
        #
        #     self.env.new_env()
        #     G, avg_false = self.env.run_sim(models)
        #     scores.append(G)
        #     falses.append(avg_false)
        #
        # self.update_logs(scores, falses, gen)
        for idx, spec in enumerate(species):
            spec.start_weights = spec.curr_weights
        # if gen > 0 and not gen % 100:
        #     self.save_data()
