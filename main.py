import config
from utils import *
import solver
import numpy as np
import sys
import os


def main(train_pair, query_pair, test_pair, n_acq_al, n_acq_bo, seed):
    train_al, query_al, test_al, query_bo = get_al_data(config.dataset, train_pair, query_pair, test_pair, seed)

    min_nn = np.zeros((2, n_acq_bo))
    # min_nn[0, :] = solver.solver_nn(train_al, query_al, test_al, query_bo, n_acq_al, n_acq_bo, al_acq=None, bo_acq="EI_nn")
    min_nn[1, :] = solver.solver_nn(train_al, query_al, test_al, query_bo, n_acq_al, n_acq_bo, al_acq="uncertainty_nn", bo_acq="EI_nn")

    print('Saving results...')
    root_name = 'Sim/' + config.dataset + "_60_3"
    if not os.path.exists(root_name):
        os.mkdir(root_name)
    np.save(root_name + '/nn_' + str(seed) + '.npy', min_nn)
    print(min_nn)


if __name__ == "__main__":
    n_train_pairs = config.N_train_pair
    n_query_pairs = config.N_query_pair
    n_test_pairs = config.N_test_pair
    n_acquire_al = config.N_acquire_al
    n_acquire_bo = config.N_acquire_bo
    sim = int(sys.argv[1])
    main(n_train_pairs, n_query_pairs, n_test_pairs, n_acquire_al, n_acquire_bo, sim)
