import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import pandas as pd
import itertools
import random
import config
from numpy.random import default_rng
from synthetic_functions import *
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
import GPy


def get_al_data(dataset, n_train, n_query, n_test, seed):
    if dataset == "styblinski_tang":
        x, y, pairs, x_bo, y_bo = get_styblinski_tang_data(1)
    elif dataset == "six_hump_camel":
        x, y, pairs, x_bo, y_bo = get_six_hump_camel_data(1)
    elif dataset == "forrester":
        x, y, pairs, x_bo, y_bo = get_forrester_data(1)
    elif dataset == "branin":
        x, y, pairs, x_bo, y_bo = get_branin_data(1)
    elif dataset == "levy":
        x, y, pairs, x_bo, y_bo = get_levy_data(1)
    elif dataset == "hartmann":
        x, y, pairs, x_bo, y_bo = get_hartmann_data(1)
    else:
        raise NotImplementedError

    # y = y.reshape(-1) + gp_noise(x, config.var_noise, seed)
    # y = gp_noise(x, config.var_noise, seed)
    train_pairs = pairs[:n_train]
    # query_pairs = pairs[n_train:n_train + n_query]
    query_pairs = []
    test_pairs = pairs[n_train: n_train + n_test]
    query_pairs_nb = 0
    for i in range(n_train+n_test, len(pairs)):
        x1 = x[pairs[i][0]][0]
        y1 = x[pairs[i][0]][1]
        x2 = x[pairs[i][1]][0]
        y2 = x[pairs[i][1]][1]
        dist = np.sqrt((x2-x1)**2+(y2-y1)**2)
        if dist > 0.89 and np.abs((y1-y2)) > 0.8:
            query_pairs.append(pairs[i])
            query_pairs_nb += 1
        if len(query_pairs) == n_query:
            break

    x_duels_train = np.array(
        [[x[train_pairs[index][0]], x[train_pairs[index][1]]] for index in range(len(train_pairs))])
    pref_train = []
    for index in range(len(train_pairs)):

        pref_train.append(1) if y[train_pairs[index][0]] < y[train_pairs[index][1]] else pref_train.append(0)

    x_duels_query = np.array(
        [[x[query_pairs[index][0]], x[query_pairs[index][1]]] for index in range(len(query_pairs))])
    pref_query = []
    for index in range(len(query_pairs)):
        pref_query.append(1) if y[query_pairs[index][0]] < y[query_pairs[index][1]] else pref_query.append(0)

    x_duels_test = np.array([[x[test_pairs[index][0]], x[test_pairs[index][1]]] for index in range(len(test_pairs))])
    pref_test = []
    for index in range(len(test_pairs)):
        pref_test.append(1) if y[test_pairs[index][0]] < y[test_pairs[index][1]] else pref_test.append(0)

    train_al = {'x_duels': x_duels_train, 'pref': pref_train}
    query_al = {'x_duels': x_duels_query, 'pref': pref_query}
    test_al = {'x_duels': x_duels_test, 'pref': pref_test}
    query_bo = {'x': x_bo, 'y': y_bo.reshape(-1)}
    return train_al, query_al, test_al, query_bo


def gp_noise(x, var, seed):
    # kernel = float(var) * RBF(length_scale=1)
    # gpr = GaussianProcessRegressor(kernel=kernel)
    #
    # noise = gpr.sample_y(x, 1, random_state=seed)
    kernel = GPy.kern.RBF(input_dim=x.shape[1], variance=var, lengthscale=0.1)
    mu = np.zeros((x.shape[0]))
    C = kernel.K(x, x)
    noise = np.random.multivariate_normal(mu, C, 1).reshape(-1)
    return noise


def logistic_function(x):
    return 1 / (1+np.e**(-x))


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data-np.min(data))/_range


def get_forrester_data(seed):
    rng = default_rng(seed)
    x = np.linspace(0, 1, 100).reshape(-1, 1)
    y = forrester_function(x)

    pairs = list(itertools.permutations(range(len(x)), 2))

    random.shuffle(pairs)
    x_bo = rng.uniform(0, 1, 10000).reshape(-1, 1)
    y_bo = forrester_function(x_bo).reshape(-1)
    return x, y, pairs, x_bo, y_bo


def get_branin_data(seed):
    rng = default_rng(seed)
    x1 = rng.uniform(low=-5, high=10, size=1000)
    x2 = rng.uniform(low=0, high=15, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = branin_function(x1, x2)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    x1_bo = np.random.uniform(-5, 10, 10000)
    x2_bo = np.random.uniform(0, 15, 10000)
    x_bo = np.hstack([x1_bo.reshape(-1, 1), x2_bo.reshape(-1, 1)])
    y_bo = branin_function(x1_bo, x2_bo)
    return x, y, pairs, x_bo, y_bo


def get_levy_data(seed):
    rng = default_rng(seed)
    x = rng.uniform(low=-2, high=2, size=(1000, 10))
    y = levy_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    x_bo = rng.uniform(low=-2, high=2, size=(10000, 10))

    y_bo = levy_function(x_bo)

    return x, y, pairs, x_bo, y_bo


def get_hartmann_data(seed):
    rng = default_rng(seed)
    x = np.random.uniform(low=0, high=1, size=(1000, 6))

    y = hartmann_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)
    x_bo = np.random.uniform(low=0, high=1, size=(10000, 6))
    plt.hist(y)
    plt.show()
    y_bo = hartmann_function(x_bo)
    return x, y, pairs, x_bo, y_bo


def get_six_hump_camel_data(seed):
    rng = default_rng(seed)
    x1 = rng.uniform(low=-1, high=1, size=1000)
    x2 = rng.uniform(low=-2, high=2, size=1000)
    x = np.hstack([x1.reshape(-1, 1), x2.reshape(-1, 1)])
    y = six_hump_camel_function(x1, x2)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)

    x1_bo = np.random.uniform(-1, 1, 10000)
    x2_bo = np.random.uniform(-2, 2, 10000)
    x_bo = np.hstack([x1_bo.reshape(-1, 1), x2_bo.reshape(-1, 1)])
    y_bo = six_hump_camel_function(x1_bo, x2_bo)
    return x, y, pairs, x_bo, y_bo


def get_styblinski_tang_data(seed):
    rng = default_rng(seed)
    x = np.linspace(-5, 5, 2000).reshape(-1, 1)
    y = styblinski_tang_function(x)
    pairs = list(itertools.combinations(range(len(y)), 2))
    random.seed(config.seed)
    random.shuffle(pairs)

    x_bo = rng.uniform(-4, 0, size=(10000, 1))
    y_bo = styblinski_tang_function(x_bo).reshape(-1)

    return x, y, pairs, x_bo, y_bo


# preference loss function for neural network
class PrefLoss_Forrester(nn.Module):
    def __init__(self):
        super(PrefLoss_Forrester, self).__init__()

    def forward(self, x1, x2, pref):
        diff = x1 - x2
        diff = diff.squeeze(1)
        indic = torch.pow(-1, pref)
        sigmoid = nn.Sigmoid()

        loss = indic * sigmoid(diff)
        return torch.sum(loss)


def plot_acc_trend(nn_list, acc_nn_std, gp_list, acc_gp_std, fig_name):
    nb = [i for i in range(len(nn_list[0]))]
    plt.plot(nb, gp_list[0], c="orange", label="gp_random")
    plt.scatter(nb, gp_list[0], c="orange", marker='.', s=120)
    plt.plot(nb, gp_list[1], c="green", label="gp_US")
    plt.scatter(nb, gp_list[1], c="green", marker='.', s=120)
    plt.plot(nb, gp_list[2], c="yellow", label="gp_BALD")
    plt.scatter(nb, gp_list[2], c="yellow", marker='.', s=120)

    plt.plot(nb, nn_list[0], c="blue", label="nn_random")
    plt.scatter(nb, nn_list[0], c="blue", marker=',')
    plt.plot(nb, nn_list[1], c="black", label="nn_US")
    plt.scatter(nb, nn_list[1], c="black", marker=',')
    plt.plot(nb, nn_list[2], c="red", label="nn_BALD")
    plt.scatter(nb, nn_list[2], c="red", marker=',')
    plt.gca().fill_between(nb,
                           nn_list[0]-acc_nn_std[0]/10,
                           nn_list[0]+acc_nn_std[0]/10, color="blue", alpha=0.2)
    plt.gca().fill_between(nb,
                           nn_list[1]-acc_nn_std[1]/10,
                           nn_list[1]+acc_nn_std[1]/10, color="grey", alpha=0.2)
    plt.gca().fill_between(nb,
                           nn_list[2] - acc_nn_std[2] / 10,
                           nn_list[2] + acc_nn_std[2] / 10, color="red", alpha=0.2)

    plt.gca().fill_between(nb,
                           gp_list[0] - acc_gp_std[0] / 10,
                           gp_list[0] + acc_gp_std[0] / 10, color="orange", alpha=0.2)
    plt.gca().fill_between(nb,
                           gp_list[1] - acc_gp_std[1] / 10,
                           gp_list[1] + acc_gp_std[1] / 10, color="green", alpha=0.2)
    plt.gca().fill_between(nb,
                           gp_list[2] - acc_gp_std[2] / 10,
                           gp_list[2] + acc_gp_std[2] / 10, color="yellow", alpha=0.2)
    plt.legend()
    plt.savefig(fig_name)
    plt.close()
    # plt.show()


def plot_function_shape(x, y, pred):
    plt.plot(x, pred)
    plt.plot(x, y, c="red", label="True")
    plt.scatter(x[np.argmin(pred)], np.min(pred), marker="*", c="black")
    plt.scatter(x[np.argmin(y)], np.min(y), marker="^", c="blue")
    plt.show()


if __name__ == "__main__":
    print(min(get_branin_data(1)[-1]))