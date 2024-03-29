from os import path, getcwd
import torch
import parameters
from teaming.domain import DiscreteRoverDomain as Domain
from support.neuralnet import NeuralNetwork as NN


def load_model(trial, gen, prepend, species=0):
    pth = path.join(getcwd(), 'weights', 't{:03d}_{}weights_s{}_g{}.pth'.format(trial, prepend, species, gen))
    return torch.load(pth)


def main(p):
    env = Domain(p)
    env.visualize = True
    policies = []
    for i in range(env.n_agents):
        nn = NN(env.state_size(use_time=False), p.hid, env.get_action_size())
        nn.set_weights(load_model(p.param_idx, 3000, p.fname_prepend, i))
        policies.append(nn)

    G, _ = env.run_sim(policies)
    print(G)


if __name__ == '__main__':
    for p in [parameters.p011]:
        for prepend in ['G_']:
            p.fname_prepend = prepend
            main(p)


#98G 100.0
#98D 87.0
#99G 0
#99D 1.0
