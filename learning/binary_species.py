
# Python packages
import numpy as np
import heapq as hq
import torch
from os import getcwd, path

# Custom packages
from evo_playground.learning.neuralnet import NeuralNetwork as NN
from teaming.domain import DiscreteRoverDomain as Domain
import evo_playground.parameters as param


class Species:
    def __init__(self, env, p, nn_in, nn_hid, nn_out, thirds=False):
        self.n_pol = p.n_policies
        self.p = p
        self.env = env
        self.sigma = p.sigma
        self.learning_rate = p.learning_rate
        self.nn_in = nn_in
        self.nn_hid = nn_hid
        self.nn_out = nn_out
        # self.model = NN(self.env.state_size(), self.p.hid, self.env.get_action_size())
        self.model = NN(nn_in, nn_hid, nn_out)
        self.weights = self._species_setup()
        if thirds:
            self.weights = self._species_setup_thirds()


    def _species_setup(self):
        # a set of randomly initilaized policies
        species = [NN(self.nn_in, self.nn_hid, self.nn_out).get_weights()
                   for _ in range(int(self.n_pol / 2))]
        return species

    def _species_setup_thirds(self):
        # a set of randomly initilaized policies
        species = [NN(self.nn_in, self.nn_hid, self.nn_out).get_weights()
                   for _ in range(int(self.n_pol / 3))]
        return species

    def mutate_weights(self):
        # Mutate each set of weights that was kept
        new_weights = []
        for wts in self.weights:
            noise = self.sigma * torch.normal(-1, 1, size=wts[0].shape) * self.learning_rate
            noise2 = self.sigma * torch.normal(-1, 1, size=wts[1].shape) * self.learning_rate
            new_weights.append([wts[0] + noise, wts[1] + noise2])
        # Have to do this separately otherwise it creates an infinite loop (whoopsies)
        for wts_01 in new_weights:
            self.weights.append(wts_01)

    def add_new_pols(self):
        new_wts = [NN(self.nn_in, self.nn_hid, self.nn_out).get_weights()
                   for _ in range(int(self.n_pol - len(self.weights)))]
        for wts in new_wts:
            self.weights.append(wts)

    def binary_tournament(self, scores):
        """
        Run a binary tournament to keep half of the policies
        :param scores:
        :return:
        """
        dummy_ranking = np.random.randint(0, 10000, len(scores))

        # create priority queue to randomly match two policies
        pq = []
        for i in range(len(scores)):
            pq.append([dummy_ranking[i], i])
        hq.heapify(pq)

        # Compare two randomly matched policies and keep one
        keep_idx = []
        for j in range(int(len(scores)/2)):
            sc0, idx0 = hq.heappop(pq)
            sc1, idx1 = hq.heappop(pq)
            if scores[idx0] >= scores[idx1]:
                keep_idx.append(idx0)
            else:
                keep_idx.append(idx1)

        self.weights = [self.weights[k] for k in keep_idx]

    def binary_multi(self, scores, pareto):

        dummy_ranking = np.random.randint(0, 10000, len(scores))

        # create priority queue to randomly match two policies
        pq = [[dummy_ranking[i], i] for i in range(len(scores))]
        hq.heapify(pq)

        # Compare two randomly matched policies and keep one
        keep_idx = [i for i, val in enumerate(pareto) if val]
        for j in range(int(len(scores)/2)):
            sc0, idx0 = hq.heappop(pq)
            sc1, idx1 = hq.heappop(pq)
            [g0_a, g0_b] = scores[idx0]
            [g1_a, g1_b] = scores[idx1]

            # If one dominates the other, choose that one
            if g0_a == g1_a and g0_b == g1_b:
                pick_one = np.random.choice([idx0, idx1])
                keep_idx.append(pick_one)
            elif g0_a >= g1_a and g0_b >= g1_b:
                keep_idx.append(idx0)
            elif g1_a >= g0_a and g1_b >= g0_b:
                keep_idx.append(idx1)
            # Otherwise choose one at random
            else:
                pick_one = np.random.choice([idx0, idx1])
                keep_idx.append(pick_one)
        keep_vals = keep_idx[:int(self.n_pol / 3)]
        self.weights = [self.weights[k] for k in keep_vals]

    def save_model(self, trial, stat, gen, prepend, wts, species=''):
        pth = path.join(getcwd(), 'weights', 't{:03d}_{}_{}weights_s{}_g{}.pth'.format(trial, stat, prepend, species, gen))
        torch.save(wts, pth)


if __name__ == '__main__':
    p = param.p318
    env = Domain(p)
    spec = Species(env, p)
    spec.mutate_weights()
    dummy_scores = np.random.randint(0, 100, 50)
    spec.binary_tournament(dummy_scores)
    print("Nothing")