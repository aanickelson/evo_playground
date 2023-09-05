import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as colors

def z_funct(x, y, b):
    z = (x * b[0]) + (y * b[1])
    return z

cols = ['YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
N = 1000
X = np.linspace(0, 10, N)
Y = 10 - X
rws = 5
colms = 2
fig, ax = plt.subplots(rws, colms, figsize=(colms*5, rws*5), constrained_layout=True)
parababals = [[1.0, 0.], [0., 1.0]]
linbals = [[1.0, 0.], [0., 1.0],
           [0.9, 0.1], [0.1, 0.9],
           [0.8, 0.2], [0.2, 0.8],
           [0.7, 0.3], [0.3, 0.7],
           [0.5, 0.5]]
g0 = np.arange(0, rws)
g1 = np.arange(0, colms)
graph_num = []
for g00 in g0:
    for g11 in g1:
        graph_num.append([g00, g11])

for i, bal in enumerate(linbals):
    Z = z_funct(X, Y, bal)
    print("##############")
    g = graph_num[i]
    # print(Z[250, 999], Z[200, 800], Z[200, 200], Z[500, 500], Z[-1, -1])

    # Default norm:
    pcm = ax[g[0], g[1]].scatter(X, Y, c=Z, cmap=cols[6], vmin=0, vmax=10)
    fig.colorbar(pcm, ax=ax[g[0], g[1]], orientation='vertical')
    # for j, txt in enumerate(Z):
    #     ax[g[0], g[1]].text(X[j]+0.1, Y[j]+0.1, f"{txt:0.2f}")
# ax.set_title('Default norm')
fig.show()
