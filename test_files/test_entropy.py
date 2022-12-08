
import numpy as np
from scipy import stats

groups = []

for n1 in range(4):
    for n2 in range(4):
        for n3 in range(4):
            groups.append([n1, n2, n3])

for g in groups:
    print(g, stats.entropy(g))

# for _ in range(100):
#     a = np.random.randint(0, 10, 3)
#     print(a, stats.entropy(a))
