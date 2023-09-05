import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors


def f_exp(x):
    return np.exp(-10*x)


def f_tanh(x):
    x2 = rem_div_zero(np.tanh(x/2))
    return 1/x2


def rem_div_zero(v, min_val=1e-5):
    vals = np.array(v)
    vals[vals < min_val] = min_val
    return vals


def G(x, y, b):
    b = rem_div_zero(np.array(b))
    x = rem_div_zero(x)
    y = rem_div_zero(y)
    exp_val = f_exp(abs(np.arctan(b[1] / b[0]) - np.arctan(y / x)))
    return exp_val * np.sqrt(x ** 2 + y ** 2)


if __name__ == '__main__':
    bounds = np.array([-0.25, -0.125, 0, 0.5, 1])

    norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)

    N = 1000
    X, Y = np.meshgrid(np.linspace(0, 4, N), np.linspace(0, 4, N))
    cols = ['Purples', 'Oranges', 'GnBu', 'Reds', 'Greens']

    fig, ax = plt.subplots(5, 1, figsize=(6, 30), constrained_layout=True)
    for i, bal in enumerate([[1.0, 0.], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0., 1.0]]):  # [0.5, 0.5]
        Z = G(X, Y, bal)
        print(Z[250, 999], Z[200, 800], Z[100, 100], Z[500, 500], Z[-1, -1])
        pcm = ax[i].pcolormesh(X, Y, Z, cmap=cols[i])
        fig.colorbar(pcm, ax=ax[i], orientation='vertical')
        # plt.figure(figsize=(10, 10))
        # plt.title('exp(-10 * theta), x, y scaling')
        # plt.pcolormesh(X, Y, Z, cmap=cols[0])
        # plt.show()
    fig.show()


    # bounds = np.array([-0.25, -0.125, 0, 0.5, 1])
    #
    # norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)
    #
    # N = 1000
    # X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    # cols = ['YlGnBu', 'Purples', 'Greens', 'Oranges', 'Reds']
    # # bal = [0.8, 0.2]
    # bal = rem_div_zero(np.array([0.5, 0.5]))
    # X = rem_div_zero(X)
    # Y = rem_div_zero(Y)
    # theta_diff = abs(np.arctan(bal[1]/bal[0]) - np.arctan(Y / X))
    # theta_scale = rem_div_zero(x_fun(theta_diff))
    # Z = 1/theta_scale * np.sqrt(X**2 + Y**2)
    #
    # Z = Z / np.max(Z)
    # # Z = np.exp(-3*(X - bal[0])**2 - 3*(Y - bal[1])**2)
    # plt.figure(figsize=(10, 10))
    # plt.title('tanh(0.5 * theta), x, y scaling')
    # plt.pcolormesh(X, Y, Z, cmap=cols[0])
    # plt.show()
