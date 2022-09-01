from math import sin, cos
import matplotlib.pyplot as plt
import numpy as np

x = [i for i in range(50)]


in_array = np.linspace(0, 60, 61)
# out_array = .5 * (1 - np.cos(.21*in_array))
out_array = np.exp(-0.1 * in_array)
print("in_array : ", in_array)
print("\nout_array : ", out_array)

# red for numpy.sin()
plt.plot(in_array, out_array, color='red', marker="o")
plt.title("e^x")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
