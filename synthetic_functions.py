import numpy as np


def forrester_function(x):
    return (6 * x - 2) ** 2 * np.sin(12 * x - 4)


def styblinski_tang_function(x):
    return 0.5 * np.sum(x ** 4 - 16 * x ** 2 + 5 * x, axis=1)


def branin_function(x1, x2):
    a = 1
    b = 5.1 / (4 * (np.pi ** 2))
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    y = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    return y


def hartmann_function(x):
    y = np.zeros(len(x))
    for a in range(len(x)):
        outer = 0
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]])
        P = 1e-4 * np.array([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]])
        for i in range(4):
            inner = 0
            for j in range(6):
                xj = x[a, j]
                Aij = A[i, j]
                Pij = P[i, j]
                inner = inner + Aij * (xj - Pij) ** 2

            new = alpha[i] * np.exp(-inner)
            outer += new

        y[a] = -(2.58+outer)/1.94
    return y


def levy_function(x):
    pi = np.pi

    x = 1 + (x - 1) / 4

    part1 = np.power(np.sin(pi * x[:, 0]), 2)

    part2 = np.sum(np.power(x[:, :-1] - 1, 2) * (1 + 10 * np.power(np.sin(pi * x[:, :-1] + 1), 2)), axis=1)

    part3 = np.power(x[:, -1] - 1, 2) * (1 + np.power(np.sin(2 * pi * x[:, -1]), 2))

    y = part1 + part2 + part3
    return y


def six_hump_camel_function(x1, x2):
    y = (4 - 2.1 * (x1 ** 2) + (x1 ** 4) / 3) * (x1 ** 2) + x1 * x2 + (-4 + 4 * (x2 ** 2)) * (x2 ** 2)
    return y
