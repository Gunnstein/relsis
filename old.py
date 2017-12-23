# -*- coding: utf-8 -*-
"""

Need to implement the following functionality:
1) Transform the integrands (X-space) into the standard normal space (U-space).

2)

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt

def find_gradient(func, x, epsilon=1e-8):
    y0 = func(x)
    J = np.zeros((x.size, ), dtype=np.double)
    x1 = x.copy()
    for i, xi in enumerate(x):
        x1[i] = xi + epsilon
        y1 = func(x1)
        J[i] = (y1 - y0) / epsilon
        x1[i] = xi
    return J.T

def func(u):
    Phi = lambda x: stats.norm.cdf(x, loc=0, scale=1)
    D0, E = 3., 30e6
    L, w, t  = 100., 2., 4.
    upx, upy = 500., 1000.
    spx, spy = 100., 100.
    px = stats.norm.ppf(Phi(u[0]), loc=upx, scale=spx)
    py = stats.norm.ppf(Phi(u[1]), loc=upy, scale=spy)
    g = D0 - 4. * L**3/(E*w*t)*np.sqrt((px/w**2)**2 + (py/t**2)**2)
    return g

def gradient(u):
    D0, E = 3., 30e6
    L, w, t  = 100., 2., 4.
    upx, upy = 500., 1000.
    spx, spy = 100., 100.
    den_sqrt = np.sqrt(((upx+u[0]*spx)/w**2)**2 + ((upy+u[1]*spy)/t**2)**2)
    dg = 4.*L**3/(E*w*t) * np.array([(upx+u[0]*spx)*spx/(w**4*den_sqrt),
                                     (upy+u[1]*spy)*spy/(t**4*den_sqrt)])
    return dg

def norm(x):
    return np.linalg.norm(x, ord=2)


def form(func, u0, nmax=100):
    u0 = np.array([0., 0.])
    g0 = func(u0)
    dg0 = find_gradient(func, u0)
    b0 = norm(u0)
    u = u0.copy()
    dg1 = dg0.copy()

    eps1 = 1e-3
    eps2 = 1e-3
    eps3 = 1e-3
    n = 0
    while True:
        dg1 = find_gradient(func, u)
        g1 = func(u)

        a = dg1 / norm(dg1)
        b1 = b0 + g1 / norm(dg1)

        u = -a * b1
        stop = ((norm(u-u0) <= eps1)
                and (norm(dg1 - dg0) <= eps2)
                and (abs(b1 - b0) <= eps3))
        if stop:
            break
        else:
            u0 = u*1.
            g0 = g1*1.
            dg0 = dg1.copy()
            b0 = b1
        n += 1
        if n > nmax:
            break
    return b1, a, u, n

class RandomVariable:

    @staticmethod
    def Phi(u):
        return stats.norm.cdf(u, loc=0, scale=1)


class NormalRandomVariable(RandomVariable):
    def __init__(self, mean, std):
        """Normally distributed random variable.

        Defined by the mean and standard deviation.

        Arguments
        ---------
        mean, std : float
            Mean and standard deviation of the variable
        """
        self.mean = mean
        self.std = std

    def __call__(self, u):
        return stats.norm.ppf(self.Phi(u), loc=self.mean, scale=self.std)

class LogNormalRandomVariable(RandomVariable):
    def __init__(self, mean, std):
        """Lognormal distributed random variable.



        """



X = np.array([NormalRandomVariable(500., 100.),
              NormalRandomVariable(1000., 100.)])

class LimitStateFunction:
    def __init__(self, X):
        self.X = X

    def get_x(self, u):
        return np.array([Xi(ui) for Xi, ui in zip(self.X, u)])

    def __call__(self, u):
        D0, E = 3., 30e6
        L, w, t  = 100., 2., 4.
        x = self.get_x(u)
        g = D0 - 4. * L**3/(E*w*t)*np.sqrt((x[0]/w**2)**2 + (x[1]/t**2)**2)
        return abs(g)

class LimitStateFunction:
    def __init__(self, X):
        self.X = X

    def get_x(self, u):
        return np.array([Xi(ui) for Xi, ui in zip(self.X, u)])

    def __call__(self, u):
        x = self.get_x(u)

        g = x[0] - x[1]
        return g

g = LimitStateFunction(X)

# print form(g, np.array([0., 0.]))


def forms(ls_func, u0):
    """
    Minimize ||u||_2
        subject to g(u) = 0

    """

    cons = (dict(type='eq', fun=ls_func))
    res = opt.minimize(lambda u: norm(u), u0,
                       constraints=cons, method='SLSQP')
    u = res['x']
    g = ls_func(u)
    dg = find_gradient(ls_func, u)
    beta = res['fun']
    a = res['jac']
    return dict(alpha=a, beta=beta, u=u, x=ls_func.get_x(u))





if __name__ == '__main__':
    print """Beta: {beta:.2f}\nAlpha: {alpha}
        """.format(**forms(g, np.array([0., 0.])))
    print (1000.-500.) / np.sqrt(100**2+100**2)
