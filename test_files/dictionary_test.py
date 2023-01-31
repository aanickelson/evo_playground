import numpy as np

archive = {}
keys = np.sort(np.random.randint(0, 1000, 40))

for key in keys:
    archive[key] = [[np.random.randint(0, 100), np.random.randint(0, 100)]]


for key2 in keys:
    archive[key2].append([0, 100])

print(archive)