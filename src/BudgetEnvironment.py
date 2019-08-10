import numpy as np


def n(x):
    # The true function to estimate
    return (1.0 - np.exp(-5.0*x)) * 100


def fun(x):
    return (100 * (1.0 - np.exp(-4 * x + 3 * x ** 3))) / 5


class BudgetEnvironment:

    def __init__(self, budgets, sigma):
        self.budgets = budgets
        self.realfunc = n
        self.means = self.realfunc(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        return np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])