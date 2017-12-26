from __future__ import division
import numpy as np
import numpy.random as rd

def _sample_oat(k, N, num_levels, grid_jump):
    """Generate trajectories without groups
    Arguments
    ---------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size
    """

    D = k

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D + 1, D], dtype=int), -1) + \
        np.triu(-1 * np.ones([D + 1, D], dtype=int))

    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.zeros([N * (D + 1), D])

    # Create N trajectories. Each trajectory contains D+1 parameter sets.
    # (Starts at a base point, and then changes one parameter at a time)
    for j in range(N):

        # directions matrix DM - diagonal matrix of either +1 or -1
        DM = np.diag([rd.choice([-1, 1]) for _ in range(D)])

        # permutation matrix P
        perm = rd.permutation(D)
        P = np.zeros([D, D])
        for i in range(D):
            P[i, perm[i]] = 1

        # starting point for this trajectory
        x_base = np.zeros([D + 1, D])
        for i in range(D):
            x_base[:, i] = (
                rd.choice(np.arange(num_levels - grid_jump))) \
                / (num_levels - 1)

        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D + 1) + j * (D + 1)
        delta_diag = np.diag([delta for _ in range(D)])

        X[index_list, :] = 0.5 * \
            (np.mat(B) * np.mat(P) * np.mat(DM) + 1) * \
            np.mat(delta_diag) + np.mat(x_base)

    return X

print _sample_oat(5, 1, 4, 1)