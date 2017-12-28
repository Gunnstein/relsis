# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import relsis


def ls(x):
    return x[0] - x[1]




if __name__ == '__main__':
    rvs = [relsis.NormalRandomVariable(30., 4.),
           relsis.NormalRandomVariable(5., 3.),]
    X, y = relsis.monte_carlo_simulation(ls, rvs, 1e6,
                                         sampling_method='sobol')
    pf = relsis.get_probability(y, y <= 0)
    beta = relsis.get_reliability_index(pf)

    res_form = relsis.form_solver(ls, rvs)
    beta_form = res_form['beta']
    alpha_form = res_form['alpha']
    print beta, beta_form
    S1, ST = relsis.find_sensitivity_sobol(ls, X, y)
    print S1, ST
    print sum(S1)
    print alpha_form / np.linalg.norm(alpha_form, 2)
    # c = plt.hist(y, 50, fc='w', ec='k', normed=True)
    # plt.vlines(0, 0, c[0].max())
    # plt.show(block=True)

