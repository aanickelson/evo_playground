import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    make_n_models = 10000
    model_size = 50
    models = np.zeros((make_n_models, model_size))

    for i in range(make_n_models):
        rand_num = np.random.randint(0, model_size - 5)
        for j in range(5):
            models[i][rand_num + j] = 1
    sum_models = np.sum(models, axis=0)
    print(sum_models)
    model_nums = list(range(model_size))
    plt.bar(model_nums, sum_models)
    plt.show()