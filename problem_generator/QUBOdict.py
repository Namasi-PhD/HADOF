import random
from collections import defaultdict


def generate_random_qubo_dict(num_vars, seed=None):

    if seed is not None:
        random.seed(seed)


    Q = [[random.uniform(-10, 10) if j >= i else 0.0 for j in range(num_vars)] for i in range(num_vars)]

    # density = 0.1  # 10% non-zero entries
    # Q = [
    #     [
    #         random.uniform(-10, 10) if (j >= i and random.random() < density) else 0.0
    #         for j in range(num_vars)
    #     ]
    #     for i in range(num_vars)
    # ]

    # density = 0.3  # 10% non-zero entries
    # Q = [
    #     [
    #         1.0 if (j >= i and random.random() < density) else 0.0
    #         for j in range(num_vars)
    #     ]
    #     for i in range(num_vars)
    # ]
    # for i in range(len(Q)):
    #     Q[i][i] = -1

    qubodict = defaultdict(int)
    for i in range(num_vars):
        for j in range(i, num_vars):
            if i==j:
                qubodict[(i,)] = Q[i][j]
            else:
                qubodict[(i, j)] = Q[i][j]

    return qubodict