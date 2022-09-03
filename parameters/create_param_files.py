from create_param_strings import string_to_save as str_gen
from os import getcwd, path
from itertools import combinations


def gen_all_files():
    n_agents = 3
    n_poi = 12
    poi_options = ['square', 'sin', 'exp']
    combos = list(combinations(poi_options, 1)) + list(combinations(poi_options, 2)) + list(combinations(poi_options, 3))
    count = 100
    for combo in combos:
        poi = combo
        generated_string = str_gen(count, n_agents, n_poi, poi)
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
