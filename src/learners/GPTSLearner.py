import numpy as np
from src.learners.Learner import Learner
from src.BudgetEnvironment import BudgetEnvironment
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTSLearner(Learner):

    """
    Gaussian Process Thompson Sampling Learner
    """

    def __init__(self, n_arms, arms, env: BudgetEnvironment, alpha=10):
        super(GPTSLearner, self).__init__(n_arms)
        self.arms = arms
        self.means = np.zeros(n_arms)
        self.sigmas = np.ones(n_arms) * 10
        self.alpha = alpha
        self.pulled_arms = []
        self.env = env
        self._create_gp()

    def _create_gp(self):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1, (1e-3, 1e3))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=self.alpha ** 2, normalize_y=True, n_restarts_optimizer=10)

    def update_observations(self, arm_idx, reward):
        super(GPTSLearner, self).update_observations(arm_idx, reward)
        self.pulled_arms.append(self.arms[arm_idx])

    def update_model(self):
        """
        Update the model in according to the previous pulled arms
        and their collected rewards
        """
        x = np.atleast_2d(self.pulled_arms).T  # matrix where the first column is the pulled_arms list
        y = self.collected_rewards
        self._create_gp()
        self.gp.fit(x, y)
        self.means, self.sigmas = self.gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.sigmas = np.maximum(self.sigmas, 1e-2)

    def update(self, pulled_arm, reward):
        """
        Update the observations and the GP model
        :param pulled_arm: index of the pulled arm
        :param reward: reward obtained by the pulled arm from the environment
        """
        self.t += 1
        self.update_observations(pulled_arm, reward)
        self.update_model()

    def pull_arms(self):
        """
        Pull all arms, in according to their means and std
        :return: an array of sampled values
        """
        return np.random.normal(self.means, self.sigmas)
