import numpy as np

from .Learner import Learner
from src.BudgetEnvironment import BudgetEnvironment

class GTSLearner(Learner):
    
    def __init__(self, n_arms, env: BudgetEnvironment):
        super(GTSLearner, self).__init__(n_arms=n_arms)
        self.env = env
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 1e3

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.means[pulled_arm] = np.mean(self.rewards_per_arm[pulled_arm])
        n_samples = len(self.rewards_per_arm[pulled_arm])
        if n_samples > 1:
            self.sigmas[pulled_arm] = np.std(self.rewards_per_arm[pulled_arm]) / n_samples

    def pull_arms(self):
        return np.random.normal(self.means, self.sigmas)