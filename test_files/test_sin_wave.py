from math import sin, cos, floor, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


t = 20

for _ in range(1):
    plt.clf()
    in_array = np.linspace(0, t-1, t)
    rand_num = np.random.uniform(0, 1)
    rand_shift = np.random.choice([-1, pi])
    # the rand shift either keeps the original shape or reverses it to be exponential growth (between [0,1])
    out_arrayexp = np.exp(-0.1 * in_array)
    out_arraysin = .5 * (1 - np.cos(.335*in_array - pi))  # - rand_num))
    # print((out_array))
    # out_array = np.exp(-0.1 * rand_num * in_array)
    # out_array = 0.5 * (1 + signal.square(.2 * in_array ))
    # mult_array = np.ones(t)
    # mid = int(t/2) - 2
    # mult_array[-mid:] *= 0.5
    #
    # out_array *= mult_array



    # print("in_array : ", in_array)
    # print("\nout_array : ", out_array)
    # print('\nmult_array : ', mult_array)
    # red for numpy.sin()

    plt.plot(in_array, out_arrayexp, color='red', marker="o", alpha=1.2)
    plt.plot(in_array, out_arraysin, color='red', marker="o", alpha=0.2)
    plt.title("e^x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
