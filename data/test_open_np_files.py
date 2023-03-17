import numpy as np
from os import path, getcwd

all_data = []
for trial_num in range(5):
    fpath = path.join(getcwd(), 'base_G_000_20230316_145354', 'G', f'000_{trial_num:02d}.npy')
    x = np.load(fpath)
    all_data.append(x)


all_data = np.array(all_data)


print(all_data)