import numpy as np


class CGPTSLearner:

    """ Combinatorial Gaussian Process Thompson Sampling Learner """

    def __init__(self, n_sub_campaigns, sub_campaigns):
        assert n_sub_campaigns == len(sub_campaigns)
        self.n_sub_campaigns = n_sub_campaigns
        self.sub_campaigns = sub_campaigns  # list of GPTS learners
        self.sampled_values = None  # store for each sub-campaign one sample for each arm
        self.budget = 0

    def add_sub_campaign(self, new_sub_campaign):
        self.sub_campaigns.append(new_sub_campaign)
        self.n_sub_campaigns += 1

    def pull_arms(self, new_budget):
        """
        Pull all arms of all sub-campaign and then apply a combinatorial algorithms
        in order to find the best set of arm, one per each sub-campaign that satisfies
        the cumulative budget
        :param new_budget daily budget that has to be added in the cumulative one
        :return: @see combinatorial_optimization
        """
        self.budget += new_budget
        self.sampled_values = [[] for _ in range(self.n_sub_campaigns)]  #
        for idx, gpts in enumerate(self.sub_campaigns):
            self.sampled_values[idx] = gpts.pull_arms()

        return self.combinatorial_optimization()

    def combinatorial_optimization(self):
        """
        In according to the sampled values solve a combinatorial problem
        that finds one arm per sub-campaign such that maximize the rewards
        and such that satisfies the given budget
        :return: list of arm idx (1 per sub-campaign) founded by the combinatorial algorithm
        """
        # TODO: change
        return [np.argmax(r) for r in self.sampled_values]

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

    def get_samples(self, campaign_idx=0):
        """
        :param campaign_idx: index of the campaign
        :return: pulled arms and their rewards -> x_obs, y_obs
        """
        assert 0 >= campaign_idx < self.n_sub_campaigns
        return self.sub_campaigns[campaign_idx].pulled_arms, self.sub_campaigns[campaign_idx].collected_rewards

    def predict(self, campaign_idx=0):
        """
        Make the curve prediction for the specified sub-campaign
        :param campaign_idx:
        :return: y_pred, sigma. @see GPTSLearner.predict()
        """
        assert 0 <= campaign_idx < self.n_sub_campaigns
        return self.sub_campaigns[campaign_idx].means, self.sub_campaigns[campaign_idx].sigmas
