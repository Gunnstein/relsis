# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import relsis


# Define a limit state function to evaluate. Here we assume a additive model
# with normally distributed variables
def limit_state_func(x):
    return np.sum(2*x)


# Define the random variables
random_variables = [relsis.NormalRandomVariable(10., 1.),
                    relsis.NormalRandomVariable(10., 2.),
                    relsis.NormalRandomVariable(-5., 3.),
                    relsis.NormalRandomVariable(-5., 4.)]

# The reliability index of the above problem can be determined analytically
beta_true = (10. + 10. - 5. - 5.) / np.sqrt(1.**2 + 2.**2 + 3**2 + 4**2)

# Perform a Monte Carlo simulation with LHS sampling and 4 cpus
X, y = relsis.monte_carlo_simulation(limit_state_func, random_variables, 1e5,
                                     sampling_method='latin_random', n_cpu=4)

# Calculate probability of failure and the estimated reliability index
pf = relsis.get_probability(y, y <= 0)
beta_mc = relsis.get_reliability_index(pf)

# Assess problem with FORM
res_form = relsis.form_solver(limit_state_func, random_variables)
beta_form = res_form['beta']
alpha_form = res_form['alpha']


# Print results
s = lambda mtd, beta: "  {0:<14s}: {1:>4.2f}".format(mtd, beta)
print "Reliability indices"
print s('True', beta_true)
print s('FORM', beta_form)
print s('Monte Carlo', beta_mc)

# Perform Sobol sensitivity analysis on results from Monte Carlo simulation
# using 4 cpus
S1, ST, S1conf, STconf = relsis.find_sensitivity_sobol(limit_state_func, X, y,
                                                    n_cpu=4, n_resample=1000)


# Analytical first order sensitivity indices
S1true = np.array([0.03, 0.13, 0.30, 0.53])

# Print results from sensitivity analysis
print "\nSensitivity analysis"
s = "{0:<3s}  {1:^7s}  {2:^7s}  {3:^7s}  {4:^7s}"
s1 = s.format("", "True", "S1", "ST", "alpha^2")
print s1

for n, S1t, S1i, STi, ai in zip(range(S1.size), S1true, S1, ST, alpha_form**2):
    print "X{0:<2n}  {1:>6.3f}  {2:>6.3f} {3:>7.3f} {4:>7.3f}".format(
                                                        n+1, S1t, S1i, STi, ai)

fig, [ax1, ax2] = plt.subplots(ncols=2, dpi=300)
S1ci = S1conf.T
S1ci[0] *= -1.
ax1.errorbar(range(1, 1+S1.size), S1, yerr=S1ci, fmt='o', capsize=2, ms=3)
ax1.set(xlim=(0.5, S1.size+.5), ylim=(-.1, .75), ylabel="$S_1$",)
plt.sca(ax1)
plt.xticks(xrange(1, 1+S1.size),
           ["$X_{0:n}$".format(n+1) for n in xrange(S1.size)])

STci = STconf.T
STci[0] *= -1.
ax2.errorbar(range(1, 1+ST.size), ST, yerr=STci, fmt='o', capsize=2, ms=3)
ax2.set(xlim=(0.5, ST.size+.5), ylim=(-.1, .75), ylabel='$S_T$')
plt.sca(ax2)
plt.xticks(xrange(1, 1+S1.size),
           ["$X_{0:n}$".format(n+1) for n in xrange(S1.size)])
fig.tight_layout()
plt.show(block=True)
