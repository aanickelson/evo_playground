import numpy as np


def G_exp(gs, desired_wts):
    def rem_div_zero(v, min_val=1e-20):
        vals = np.array(v)
        vals[vals < min_val] = min_val
        return vals

    b = rem_div_zero(desired_wts)
    g = rem_div_zero(gs)
    # Get the angle of the preferred balance and of the g balance
    theta_wts = np.arctan(b[1] / b[0])
    theta_gs = np.arctan(g[1] / g[0])
    # Take the difference between the angles, then scale so closer to the angle has a higher value
    # -10 is a scalar you can play with. Higher scalar values give a steeper gradient near the ideal tradeoff
    exp_val = np.exp(-10 * abs(theta_wts - theta_gs))
    # Then scale so further from the origin has a higher value
    return exp_val * np.sqrt(g[0] ** 2 + g[1] ** 2)