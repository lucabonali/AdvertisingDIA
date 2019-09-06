from typing import List

import numpy as np

from src.GPTSLearner import GPTSLearner


class CGPTSLearner:

    """ Combinatorial Gaussian Process Thompson Sampling Learner """

    def __init__(self, name, sub_campaigns: List[GPTSLearner], budgets):
        self.name = name
        self.n_sub_campaigns = len(sub_campaigns)
        self.sub_campaigns = sub_campaigns  # list of GPTS learners
        self.sampled_values = None  # store for each sub-campaign one sample for each arm
        self.budgets = budgets
        self.budget = 0
        self.sampled_values_matrix = None

    def remove_sub_campaign(self, sub_campaign):
        self.sub_campaigns.remove(sub_campaign)
        self.n_sub_campaigns -= 1

    def add_sub_campaigns(self, new_sub_campaigns):
        self.sub_campaigns.extend(new_sub_campaigns)
        self.n_sub_campaigns += len(new_sub_campaigns)

    def add_sub_campaign(self, new_sub_campaign):
        self.sub_campaigns.append(new_sub_campaign)
        self.n_sub_campaigns += 1

    def pull_arms(self, new_budget=0):
        """
        Pull all arms of all sub-campaign and then apply a combinatorial algorithms
        in order to find the best set of arm, one per each sub-campaign that satisfies
        the cumulative budget
        :param new_budget daily budget that has to be added in the cumulative one
        :return: @see combinatorial_optimization
        """
        self.budget += new_budget
        sampled_values = [[] for _ in range(self.n_sub_campaigns)]  #
        for idx, gpts in enumerate(self.sub_campaigns):
            sampled_values[idx] = gpts.pull_arms()

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
