# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats

__all__ = ["RandomVariable", "NormalRandomVariable", "UniformRandomVariable"]

class RandomVariable:
    """Base class for random variables.

    Interface to any random variable derived from a scipy.stats.rv_continous
    class. Sets the state of the random variable and provides a interface for
    solvers in Relsis.
    """
    def __init__(self, *args, **kwargs):
        self._rv = None

    def ppf(self, probability):
        """Returns the value of x which yields the probabilty P(X<=x)

        The inverse of the cdf also known as the percent point function.
        """
        return self._rv.ppf(probability)

    def cdf(self, x):
        """Returns P(X <= x) of the random variable X.

        The cumulative probability distribution function of the random
        variable.

        """
        return self._rv.cdf(x)

    def to_u(self, x):
        return stats.norm.ppf(self.cdf(x), loc=0., scale=1.)

    def from_u(self, u):
        return self.ppf(stats.norm.cdf(u, loc=0., scale=1.))

    def rvs(self, size=1):
        return self._rv.rvs(size=size)


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

        self._rv = stats.norm(loc=mean, scale=std)
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

        self._rv = stats.uniform(loc=lower, scale=upper-lower)
        self._kw = {"loc": self.lower, "scale": self.upper-self.lower}

    def __str__(self):
        s = "Uniformly distributed random variable, U({loc}, {loc}+{scale})"
        return s.format(**self._kw)


class LogNormalRandomVariable(RandomVariable):
    def __init__(self, mean_ln_x, std_ln_x):
        """Lognormally distributed random variable.

        Defined by the mean and standard deviation.

        Arguments
        ---------
        mean_ln_x, std_ln_x : float
            Mean and standard deviation of the variable logarithm of the random
            variable.
        """
        self.mean = mean_ln_x
        self.std = std_ln_x
        var = np.log(1.+(std_ln_x/mean_ln_x)**2)
        mean = np.log(mean_ln_x / np.sqrt(1+(std_ln_x/mean_ln_x)**2))
        print mean, np.sqrt(var)
        self._rv = stats.lognorm(s=std_ln_x, scale=np.exp(mean_ln_x))
        self._kw = {"loc": self.mean, "scale": self.std}

    def __str__(self):
        s = "Normaly distributed random variable, N({loc}, {scale})"
        return s.format(**self._kw)

def SN_curve(S, dSc=71., m=5):
    return 2e6 * (dSc/S) ** m

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    Vx = 0.30
    std_ln_x = np.log(1.+(Vx)**2)
    S = np.linspace(1, 200)
    N = SN_curve(S)
    for s in [100., ]:
        mean_ln_x = np.log(SN_curve(s))
        std_ln_x = Vx * mean_ln_x
        lnorm = LogNormalRandomVariable(mean_ln_x, std_ln_x)
        np.logspace()

    # y = lnorm.rvs(size=100000)
    # print np.log(y).std()
    # print np.log(y).mean()
    # plt.hist(np.log(y), 100)
    plt.semilogx(N, S)
    plt.show(block=True)

