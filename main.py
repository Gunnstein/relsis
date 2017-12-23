# -*- coding: utf-8 -*-
"""

First

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt


def _norm(x):
    return np.linalg.norm(x, ord=2)

def reliability_index(pf):
    return stats.norm.ppf(1-pf, loc=0., scale=1.)

class RandomVariable:
    """Base class for random variables.

    Interface to any random variable derived from a scipy.stats.rv_continous
    class. Sets the state of the random variable and provides a interface for
    solvers in Relsis.
    """
    def __init__(self, *args, **kwargs):
        self._rv = None
        self._kw = None

    def ppf(self, probability):
        """Returns the value of x which yields the probabilty P(X<=x)

        The inverse of the cdf also known as the percent point function.
        """
        return self._rv.ppf(probability, **self._kw)

    def cdf(self, x):
        """Returns P(X <= x) of the random variable X.

        The cumulative probability distribution function of the random
        variable.

        """
        return self._rv.cdf(x, **self._kw)

    def to_u(self, x):
        return stats.norm.ppf(self.cdf(x), loc=0., scale=1.)

    def from_u(self, u):
        return self.ppf(stats.norm.cdf(u, loc=0., scale=1.))

    def rvs(self, size=1):
        return self._rv.rvs(size=size, **self._kw)


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

        self._rv = stats.norm
        self._kw = {"loc": self.mean, "scale": self.std}

    def __str__(self):
        s = "Normaly distributed random variable, N({loc}, {scale})"
        return s.format(**self._kw)


class UniformRandomVariable(RandomVariable):
    def __init__(self, lower, upper):
        """Uniformly distributed random variable.

        Defined by the lower (included) and upper (excluded) bounds.

        Arguments
        ---------
        mean, std : float
            Mean and standard deviation of the variable
        """
        self.lower = lower
        self.upper = upper

        self._rv = stats.norm
        self._kw = {"loc": self.lower, "scale": self.upper-self.lower}

    def __str__(self):
        s = "Uniformly distributed random variable, U({loc}, {loc}+{scale})"
        return s.format(**self._kw)


def form_solver(limit_state_function, random_variables):
    """
    Evaluates the integral `limit_state_function(random_variables) <= 0` by
    first order reliability method. Transforms the limit state
    function to the standard normal space and determines the integral by
    finding the most probable point with the following minimization problem:

        minimize ||u||_2 subject to g(u) = 0

    Arguments
    ---------
    limit_state_function : function
        The function should return a scalar value and take a array of variables
        corresponding to realizations of the random variables given in the
        array `random_variables`.

    random_variables : array
        An array of RandomVariable instances.

    Returns
    -------
        dict
            A dictionary with solution.
    """
    u0 = np.zeros(len(random_variables))

    def _ls(u):
        """Transforms: u -> x space and returns limit_state_function.

        """
        x = np.array([Xi.from_u(ui) for Xi, ui in zip(random_variables, u)])
        return limit_state_function(x)


    cons = dict(type='eq', fun=_ls)
    res = opt.minimize(lambda u: _norm(u), u0,
                       constraints=cons, method='SLSQP')
    alpha = res['jac']
    beta = res['fun']

    if _ls(u0) <= 0:
        beta *= -1.

    result = dict(alpha=alpha, beta=beta)
    return result

def monte_carlo_solver(limit_state_function, random_variables, Nmax=1e5):

    n = 0
    fails = 0
    y = []
    while n < Nmax:
        n += 1
        x = [Xi.rvs() for Xi in random_variables]
        g = limit_state_function(x)
        if g <=0:
            fails += 1
    pf = float(fails) / (float(n))
    beta = reliability_index(pf)
    return beta

def lhs(num_samples, lhs_method='none'):
    n = int(num_samples)
    if lhs_method == 'none' or lhs_method is None:
        x = stats.uniform.rvs(size=n, loc=0., scale=1.)
        return x
    else:
        x = np.arange(0., n, dtype=np.double)

    if lhs_method == 'random':
        x += stats.uniform.rvs(size=x.size, loc=0., scale=1.)
    elif lhs_method == 'central':
        x += .5

    x /= float(num_samples)
    np.random.shuffle(x)
    return x


def monte_carlo_solver2(limit_state_function, random_variables,
                        num_samples=1e3, lhs_method='random'):
    X = np.array([Xi.ppf(lhs(num_samples, lhs_method))
                 for Xi in random_variables]).T
    y = np.array(map(limit_state_function, X))
    pf = float(y[y <= 0.].size) / float(y.size)
    beta = reliability_index(pf)
    return beta


def limit_state_function(x):
    D0, E = 3., 30e6
    L, w, t  = 100., 2., 4.
    g = D0 - 4. * L**3/(E*w*t)*np.sqrt((x[0]/w**2)**2 + (x[1]/t**2)**2)
    return abs(g)

if __name__ == '__main__':
    # print """Beta: {beta:.2f}\nAlpha: {alpha}
    #     """.format(**forms(g, np.array([0., 0.])))
    # print (1000.-500.) / np.sqrt(100**2+100**2)

    # print lhs(4, 3)




    ls = lambda x:  x[2] + x[0] - x[1]
    print range(1)
    X = [NormalRandomVariable(10., 4.),
         NormalRandomVariable(5., 2.),
         UniformRandomVariable(3., 5.)]

    for sts in [1e1, 1e2, 1e3, 1e4, 1e5, 1e6]:

        # res_form = form_solver(ls, X)
        # alpha_form = res_form['alpha']
        # beta_form = res_form['beta']
        beta_central = monte_carlo_solver2(ls, X, sts, 'central')
        beta_random = monte_carlo_solver2(ls, X, sts, 'random')
        beta_none = monte_carlo_solver2(ls, X, sts, 'none')

        # print alpha_form

        print "10**{3:<3.0f} RANDOM: {0:<5.2f} NONE: {1:<5.2f} CENTRAL: {2:<5.2f}".format(
            beta_random, beta_none, beta_central, np.log10(sts), )