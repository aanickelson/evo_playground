import numpy as np
from os import path, getcwd

fpath = path.join(getcwd(), '20220824_1553', 'poi_xy', 'poi_xy_trial98.npy')
[x, y] = np.load(fpath)
print(x)
print(y)