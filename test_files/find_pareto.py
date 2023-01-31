import numpy as np


def is_pareto_efficient_simple(vals):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    costs = np.array(vals)
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] > c, axis=1)  # Keep any point with a lower cost
            eff_add = np.all(costs == c, axis=1)
            is_efficient += eff_add
            is_efficient[i] = True  # And keep self
    final_vals = [costs[j] for j in range(len(costs)) if is_efficient[j]]
    return final_vals


def add_to_pareto(archive, new):
    n_non_dom = 0
    par_bool = np.ones(len(archive))
    new_bool = False
    for i, val in enumerate(archive):
        if new[0] == val[0] and new[1] == val[1]:
            new_bool = True
            break
        elif new[0] >= val[0] and new[1] >= val[1]:
            par_bool[i] = False
            new_bool = True
        elif val[0] >= new[0] and val[1] >= new[1]:
            break
        else:
            n_non_dom += 1

    if n_non_dom == len(archive):
        new_bool = True

    final_pareto = [archive[j] for j in range(len(archive)) if par_bool[j]]
    if new_bool:
        final_pareto.append(new)

    return final_pareto

from time import time
a = [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [4, 0], [2, 1], [3, 0], [3, 1], [0, 0]]
new = [3, 2]
start = time()
# for _ in range(10000):
x = is_pareto_efficient_simple(a)
print(x)
# print(f"their method: {time() - start}")

a_p = [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [4, 0]]

start2 = time()
# for _ in range(10000):
y = add_to_pareto(a, [3, 2])
print(y)
# print(f'our method: {time() - start}')
