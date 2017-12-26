# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import relsis

if __name__ == '__main__':
    ls = lambda x: x[0] - x[1]/x[2] + x[3]
    rvs = [relsis.NormalRandomVariable(20., 4.),
           relsis.NormalRandomVariable(10., 3.),
           relsis.UniformRandomVariable(0.1, 3.),
           relsis.NormalRandomVariable(1., 0.1)]
    X, y = relsis.monte_carlo_simulation(ls, rvs, 1e6)
    print relsis.get_reliability_index(float(y[y<=0].size)/float(y.size))

    S1, ST = relsis.find_sobol_sensitivity(ls, X, y)
    print S1.sum()

    res = relsis.find_sensitivity_morris(ls, rvs, 10, 4, 2)

    s = "{0:>8s} {1:>8s}".format("Morris", "Sobol")
    print "{0:17s}".format("First order")
    print "=" * (len(s)+4)
    print s
    print "=" * (len(s)+4)
    for S1m, S1s in zip(res['mu_star'], S1):
        print "{0:>8.3f} {1:>8.3f}".format(S1m, S1s)
    s = "{0:>8s} {1:>8s}".format("Morris", "Sobol")
    print "{0:17s}".format("Total")
    print "=" * (len(s)+4)
    print s
    print "=" * (len(s)+4)
    for S1m, S1s in zip(res['sigma'], ST):
        print "{0:>8.3f} {1:>8.3f}".format(S1m, S1s)
