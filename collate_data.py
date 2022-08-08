import parameters as params
from os import path, getcwd
import numpy as np

poi_options = [[60, 1, 1, 1],   # 0
               [60, 1, 2, 1],   # 1
               [60, 1, 1, 2],   # 2
               [60, 1, 2, 2],   # 3
               [30, 1, 1, 2],   # 4
               [30, 2, 1, 1],   # 5
               [30, 1, 2, 2],   # 6
               [30, 2, 2, 1],   # 7
               [20, 1, 1, 3],   # 8
               [20, 2, 1, 1.5], # 9
               [20, 3, 1, 1],   # 10
               [20, 1, 2, 3],   # 11
               [20, 2, 2, 1.5], # 12
               [20, 3, 2, 1],   # 13
               [1, 60, 1, 1],   # 14
               [1, 60, 2, 1]]   # 15

print(poi_options)
exit()
preps = ['G', 'D']
path_out_nm = path.join(getcwd(), 'data', 'collating_data')
fname = "big_batch_collated"
data_out_path = path.join(path_out_nm, "{}.csv".format(fname))  # Done this way for csv so we can pass the filename to make the graphs

with open(data_out_path, 'w') as f:
    for p in params.BIG_BATCH:
        for pre in preps:
            filename = "{}_btrial{:02d}_max".format(pre, p.trial_num)
            path_nm = path.join(getcwd(), 'data')
            data_path = path.join(path_nm, "{}.csv".format(filename))  # Done this way for csv so we can pass the filename to make the graphs
            try:
                final_g_score = np.loadtxt(data_path)[-1]
            except FileNotFoundError:
                continue
            poi_types_to_save = []
            for poi_type in p.poi_options:
                for i, poi_opt in enumerate(poi_options):
                    if poi_type == poi_opt:
                        poi_types_to_save.append(i)
                        break
            data_to_save = [p.trial_num, pre, final_g_score, p.n_pois, p.n_agents, poi_types_to_save]
            data_to_save = str(data_to_save) + '\n'
            f.write(data_to_save)