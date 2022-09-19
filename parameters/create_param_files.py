from create_param_strings import string_to_save as str_gen
from os import getcwd, path
from itertools import combinations


def gen_all_files():
    n_ag = [3, 5]
    n_p = [[2, 3, 4], [3, 5, 7]]
    # n_agents = 10
    # n_poi = 60
    poi_options = ['square', 'sin', 'exp']
    combos = list(combinations(poi_options, 1)) + list(combinations(poi_options, 3))
    count = 300
    for combo in combos:
        for i, ag in enumerate(n_ag):
            for p in n_p[i]:
                poi = combo
                generated_string = str_gen(count, ag, p, poi)
                filesave(generated_string, count)
                count += 1


def filesave(str_to_save, filenum):
    filename = "parameters{:03d}.py".format(filenum)
    filepath = path.join(getcwd(), filename)
    # Writing to file
    with open(filepath, "w") as fl:
        # Writing data to a file
        fl.writelines(str_to_save)


if __name__ == '__main__':
    gen_all_files()
