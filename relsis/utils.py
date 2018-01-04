# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import multiprocessing

def norm(x):
    return np.linalg.norm(x, ord=2)


def get_reliability_index(pf):
    return stats.norm.ppf(1-pf, loc=0., scale=1.)


def _map(func, X, n_cpu=1):
    """Standard map function capable of multiple processors.
    """
    if n_cpu == 'max':
        n_cpu = multiprocessing.cpu_count()
    if n_cpu == 1:
        y = np.asfarray(map(func, X))
    elif n_cpu > 1:
        pool = multiprocessing.Pool(n_cpu)
        y = np.asfarray(pool.map(func, X))
        pool.close()
        pool.join()
    else:
        raise ValueError("Could not determine the number of cpus to apply")
    return y

def get_probability(sample, cond):
    """Find the probability that the sample fullfills conditions.

    Example
    -------
    >>> y = np.array([-1., 0., 1., 2.])
    >>> print get_probability(y, y<=0)
    0.5

    Arguments
    ---------
    sample : ndarray
        A sample to check the condition on.

    cond : ndarray
        An array of bools with equal size to sample which decides wether or not
        the sample point fulfills the conditions or not.

    """
    return float(sample[cond].size) / float(sample.size)


def truncate_prob_dist(x):
    dtype = x.dtype
    epsneg = np.finfo(dtype).epsneg
    eps = np.finfo(dtype).eps
    x[x<epsneg] = epsneg
    x[x>1.-eps] = 1.-eps


def q2x(X, random_variables):
    """Transforms the matrix X from [0, 1]-space to the random variable space.

    `X` is a (m x n)-array and `random_variables` is a (1xn)-array.
    """
    return np.array([Xi.ppf(X[:, n])
                 for n, Xi in enumerate(random_variables)]).T


class SobolTestFunction:
    def __init__(self, a=None):
        """Creates an instance of the sobol test function

        The Sobol test function is defined by

            product_{i=1}^{k} g_i(X_i)
                where g(X_i) = frac{|4X_i-2| + a_i}{1+a_i}

        and is highly nonlinear and nonmonotic. The partial variances and the
        total variance can be determined analytically such that the sensitivity
        indices can be determined analytically.

        Arguments
        ---------
        a : Optional[ndarray]
            The coefficients in the test function. If these are not given, the
            default is set to the one test case given in Saltellis Primer book.


        """
        self.a = a or np.array([78., 12., 0.5, 2., 97., 34.])

    def __call__(self, x):
        """Returns the Sobol analytical

        """
        a = self.a
        return np.product((np.abs(4 * x - 2) + a)/(1 + a))

if __name__ == "__main__":
    print "ready to go"