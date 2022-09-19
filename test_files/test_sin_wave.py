from math import sin, cos, floor, pi
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


t = 60

for _ in range(30):
    plt.clf()
    in_array = np.linspace(0, t-1, t)
    rand_num = np.random.uniform(0, 1)
    rand_shift = np.random.choice([-1, 1])
    # the rand shift either keeps the original shape or reverses it to be exponential growth (between [0,1])
    out_array = np.exp(-0.1 * in_array)[::rand_shift]
    print(max(out_array))
    # out_array = .5 * (1 - np.cos(.21*in_array - rand_num))
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

    plt.plot(in_array, out_array, color='red', marker="o")
    plt.title("e^x")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
