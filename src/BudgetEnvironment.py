import numpy as np


class BudgetEnvironment:

    def __init__(self, budgets, sigma, function):
        self.budgets = budgets
        self.realfunc = function
        self.means = function(budgets)
        self.sigmas = np.ones(len(budgets)) * sigma

    def round(self, pulled_arm):
        r = np.random.normal(self.means[pulled_arm], self.sigmas[pulled_arm])
        return r if r >= 0 else 0
        #return r