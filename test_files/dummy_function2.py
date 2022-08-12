import numpy as np


if __name__ == '__main__':
    a = [[1, 2, 3], [3, 2, 1]]
    b = np.array(a)
    np.savetxt('dummy_save.csv', b, delimiter=',')
