from create_param_strings import string_to_save as str_gen
from os import getcwd, path


def gen_all_files():
    n_agents = 1
    n_poi_each = 1
    # poi_options = [[60, 1, 1, 1],  # 0
    #                [60, 1, 2, 1],  # 1
    #                [60, 1, 1, 2],  # 2
    #                [60, 1, 2, 2],  # 3
    #                [30, 1, 1, 2],  # 4
    #                [30, 2, 1, 1],  # 5
    #                [30, 1, 2, 2],  # 6
    #                [30, 2, 2, 1],  # 7
    #                [20, 1, 1, 3],  # 8
    #                [20, 2, 1, 1.5],  # 9
    #                [20, 3, 1, 1],  # 10
    #                [20, 1, 2, 3],  # 11
    #                [20, 2, 2, 1.5],  # 12
    #                [20, 3, 2, 1],  # 13
    #                [10, 4, 1, 1],  # 14
    #                [10, 4, 2, 1]]  # 15

    poi_options = [[60, 1, 1, 1],  # 0
                   [30, 1, 1, 1],  # 5
                   [30, 2, 1, 1],  # 7
                   [20, 3, 1, 1],  # 10
                   [20, 2, 1, 1],  # 13
                   [10, 4, 1, 1],  # 14
                   [10, 2, 1, 1]]  # 15

    poi_base = [None]  #, [[60, 1, 1, 1]]]
    count = 710
    for n_ag in range(n_agents):
        n_ag += 3
        for p in range(n_poi_each):
            p += 4
            n_poi = n_ag * p
            for poi_opt in poi_options:
                # for base in poi_base:
                    # if base:
                    #     poi = base.copy()
                    #     poi.append(poi_opt)
                    # else:
                poi = [poi_opt]
                generated_string = str_gen(count, n_ag, n_poi, poi)
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