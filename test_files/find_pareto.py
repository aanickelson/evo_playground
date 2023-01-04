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
    return is_efficient


a = [[4, 0], [3, 1], [2, 2], [1, 3], [0, 4], [4, 0], [2, 1], [3, 0], [3, 1], [0, 0]]
print(is_pareto_efficient_simple(a))
