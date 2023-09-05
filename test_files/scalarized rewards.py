import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors




bounds = np.array([-0.25, -0.125, 0, 0.5, 1])

norm = colors.BoundaryNorm(boundaries=bounds, ncolors=4)

N = 1000
X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
cols = ['Purples', 'GnBu', 'Greens', 'Oranges', 'Reds']
# Z1 = np.exp(-X**2 - Y**2)
fig, ax = plt.subplots(5, 1, figsize=(6, 30), constrained_layout=True)
parababals = [[1.0, 0.], [0.9, 0.3], [0.8, 0.6], [0.6, 0.8], [0.3, 0.9], [0., 1.0]]
linbals = [[1.0, 0.], [0.7, 0.3], [0.5, 0.5], [0.3, 0.7], [0., 1.0]]
for i, bal in enumerate(linbals):
    # Z2 = np.exp(-3*(X - bal[0])**2 - 3*(Y - bal[1])**2)
    Z = (X * bal[0]) + (Y * bal[1])
    # Z = ((Z1 - Z2) * 2)[:-1, :-1]
    # plt.clf()
    # ax = ax.flatten()
    print("##############")
    print(bal, Z[int(bal[1] * (N-1)), int(bal[0] * (N-1))])
    for bal_check in linbals:
        print(bal_check, Z[int(bal_check[1] * (N-1)), int(bal_check[0] * (N-1))])
    # print(Z[250, 999], Z[200, 800], Z[200, 200], Z[500, 500], Z[-1, -1])

    # Default norm:
    pcm = ax[i].pcolormesh(X, Y, Z, cmap=cols[i])
    fig.colorbar(pcm, ax=ax[i], orientation='vertical')
# ax.set_title('Default norm')
fig.show()