# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats

__all__ = ["RandomVariable", "NormalRandomVariable", "UniformRandomVariable",
           "LognormalRandomVariable", "DiscreteUniformRandomVariable",
           "BetaDistribution"]


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

    def pdf(self, x):
        return self._rv.pdf(x)

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
        lower, upper : float
            Bounds of the distribution
        """
        self.lower = lower
        self.upper = upper

        self._rv = stats.uniform(loc=lower, scale=upper-lower)
        self._kw = {"loc": self.lower, "scale": self.upper-self.lower}

    def __str__(self):
        s = "Uniformly distributed random variable, U({loc}, {loc}+{scale})"
        return s.format(**self._kw)


class LognormalRandomVariable(RandomVariable):
    def __init__(self, mean_ln_x, std_ln_x):
        """Lognormally distributed random variable.

        Defined by the mean (mu_ln_x) and standard deviation (std_ln_x) of
        the logarithm of the lognormal varible. Let X be a lognormal variable,
        Z be a standard normal distributed variable, then

            X = exp(mu_ln_x + std_ln_x * Z)

        Arguments
        ---------
        mean_ln_x, std_ln_x : float
            Mean and standard deviation of the variable logarithm of the random
            variable.
        """
        self.mean_ln_x = mean_ln_x
        self.std_ln_x = std_ln_x
        self.mean = np.exp(mean_ln_x)*np.sqrt(1. + (std_ln_x / mean_ln_x)**2)
        self.std = mean_ln_x * np.sqrt(np.exp(std_ln_x**2) - 1.)
        self._rv = stats.lognorm(s=std_ln_x, scale=np.exp(mean_ln_x))
        self._kw = {"loc": self.mean, "scale": self.std}

    def __str__(self):
        s = "Lognormally distributed random variable, LN({loc}, {scale})"
        return s.format(**self._kw)


class DiscreteUniformRandomVariable(RandomVariable):
    def __init__(self, lower, upper):
        """Uniform discrete random variable.

        Defined by the lower (included) and upper (excluded) bounds.

        Arguments
        ---------
        lower, upper : float
            Bounds of the distribution
        """
        self._rv = stats.randint(lower, upper)
        self._kw = dict(lower=lower, upper=upper)

    def __str__(self):
        s = "Uniform discrete random variable with bounds [{lower}, {upper}>."
        return s.format(**self._kw)


class BetaDistribution(RandomVariable):
    def __init__(self, alpha, beta, lower=0., upper=1.):
        """Beta distribution in the range from lower to upper.

        The beta distribution can take on a range of different shapes, the most
        important ones may be:
            alpha > beta > 1.0 : positively skewed
            beta > alpha > 1.0 : negatively skewed
            alpha = beta > 1.0 : symmetric
            alpha = beta = 1.0 : uniform distribution

        Arguments
        ---------
        alpha, beta : float
            The shape parameters of the beta distributions.
        upper, lower : float
            Bounds of the distribution
        """
        self._rv = stats.beta(alpha, beta, loc=lower, scale=upper-lower)
        self._kw = dict(a=alpha, b=beta, loc=lower, scale=upper-lower,
                        upper=upper, lower=lower)

    def __str__(self):
        s = "Beta distribution a={a}, b={b} and bounds [{lower}, {upper}]."
        return s.format(**self._kw)


class TriangularDistribution(RandomVariable):
    def __init__(self, lower, upper, mode):
        """Triangular distribution from lower to upper with specified mode.

        Arguments
        ---------
        upper, lower, mode : float
            Bounds and mode of the distribution
        """
        c = (mode - lower) / (upper - lower)
        self._rv = stats.triang(c, loc=lower, scale=upper-lower)
        self._kw = dict(loc=lower, scale=upper-lower, c=mode, mode=mode,
                        lower=lower, upper=upper)

    def __str__(self):
        s = "Triangular distribution, mode={mode} and bounds[{lower}, {upper}]"
        return s.format(**self._kw)
