import numpy as np
from os import path, getcwd

for trial_num in [12, 13, 14, 15]:
    fpath = path.join(getcwd(), '20221004_1138', 'G_time', f'G_time_trial0{trial_num}_max_G.npy')
    x = np.load(fpath)
    count = 0
    for row in x:
        # print(row)
        if row[-1] == 1:
            count += 1

    print(trial_num, count / 50.)