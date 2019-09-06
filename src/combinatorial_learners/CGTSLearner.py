from typing import List

import numpy as np

from src.learners.GTSLearner import GTSLearner


class CGTSLearner():

    def __init__(self, name, sub_campaigns: List[GTSLearner], budgets):
        self.name = name
        self.n_sub_campaigns = len(sub_campaigns)
        self.sub_campaigns = sub_campaigns  # list of GPTS learners
        self.sampled_values = None  # store for each sub-campaign one sample for each arm
        self.budgets = budgets
        self.budget = 0

    def pull_arms(self):
        sampled_values = [[] for _ in range(self.n_sub_campaigns)]
        for idx, gts in enumerate(self.sub_campaigns):
            sampled_values[idx] = gts.pull_arms()

        return sampled_values

    def update(self, pulled_arms, rewards):
        for idx, pulled_arm in enumerate(pulled_arms):
            self.sub_campaigns[idx].update(pulled_arm, rewards[idx])

    def get_collected_rewards(self):
        """
        retrieve the collected rewards from all sub-campaign, and sum the rewards at same position
        since they refer to the same day
        :return: list of collected rewards
        """
        collected_rewards = np.zeros(shape=self.sub_campaigns[0].collected_rewards.shape)
        for sc in self.sub_campaigns:
            collected_rewards += sc.collected_rewards

        return collected_rewards
