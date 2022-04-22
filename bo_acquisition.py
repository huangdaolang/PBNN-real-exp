import numpy as np
import torch


def choose_criterion(criterion):
    if criterion == "random":
        return random_sampling
    elif criterion == "EI_nn":
        return ei_mc
    elif criterion == "EI_gp":
        return ei


def random_sampling(query, model, y_best):
    n = len(query['y'])
    return np.random.randint(0, n)


def ei_mc(query, model, y_best):
    N_mc = 30
    x = torch.tensor(query['x'])

    score = np.zeros_like(query['y'])
    for i in range(N_mc):
        pred = model.forward_bo(x).reshape(-1)
        score += np.maximum((y_best - pred).detach().numpy(), np.zeros_like(query['y']))
    return np.argmax(score)


def ei():
    return 1