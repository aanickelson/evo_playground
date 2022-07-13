import errno

from teaming.domain import DiscreteRoverDomain as Domain
from evo_playground.learning.evo2 import BasicEvo
from evo_playground.parameters.parameters00 import Parameters as p
import os
from support_functions.safe_file_save import write

def main():

    env = Domain(p)
    evo = BasicEvo(env, p)
    evo.run_evolution()
    # cwd = os.getcwd()
    #
    # num = 0
    # path = os.path.join(cwd, "data", "fname{}.csv".format(num))
    # saved = False
    # data = [1,2,3]
    #
    # while not saved:
    #     path = os.path.join(cwd, "data", "fname{}.csv".format(num))
    #     try:
    #         write(path, data, True)
    #         saved = True
    #         break
    #     except FileExistsError:
    #         num += 1



if __name__ == '__main__':
    main()
