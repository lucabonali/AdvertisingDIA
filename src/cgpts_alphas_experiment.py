import time
from typing import List

import numpy as np

import src.data as curves
import src.plotting as plotting
from src.BudgetEnvironment import BudgetEnvironment
from src.combinatorial_learners.CGPTSLearner import CGPTSLearner
from src.learners.GPTSLearner import GPTSLearner
from src.optimization import combinatorial_optimization

n_arms = 20
min_budget = 0
max_budget = 19

T = 100
n_experiments = 3
# 100 x 100 -> ~ 11 hours
# 100 x 80 -> 9.84 hours

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0

allow_empty = True

alphas = [1, 5, 10]

cgpts_rewards_per_experiment = [[] for _ in alphas]

sub_campaigns = []

if __name__ == '__main__':
    tot_time = time.time()
    for idx, alpha in enumerate(alphas):
        for e in range(n_experiments):
            print('Experiment with alpha = {} #{}'.format(alpha, e + 1), end='')
            start_time = time.time()

            sub_campaigns: List[GPTSLearner] = [
                GPTSLearner(n_arms=n_arms, arms=budgets, alpha=alpha, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, alpha=alpha, env=BudgetEnvironment(budgets, sigma, curves.facebook_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, alpha=alpha, env=BudgetEnvironment(budgets, sigma, curves.instagram_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, alpha=alpha, env=BudgetEnvironment(budgets, sigma, curves.youtube_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, alpha=alpha, env=BudgetEnvironment(budgets, sigma, curves.bing_agg))
            ]

            cgpts = CGPTSLearner("CGPTS", sub_campaigns, budgets)

            for t in range(T):
                reward_matrix = cgpts.pull_arms()
                pulled_arms, _ = combinatorial_optimization(reward_matrix, budgets.tolist(), allow_empty=allow_empty)
                rewards = [sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
                cgpts.update(pulled_arms, rewards)

            print(": {:.2f} sec".format(time.time() - start_time))

            cgpts_rewards_per_experiment[idx].append(cgpts.get_collected_rewards())

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
    best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist(), allow_empty=allow_empty)
    print("Best budgets => {}".format(best_budgets))
    print("Optimum      => {}".format(optimum))

    plotting.plot_multiple_rewards(cgpts_rewards_per_experiment, optimum, T, _names=["CGPTS with " + str(a) for a in alphas])
    plotting.plot_multiple_regret(np.array(cgpts_rewards_per_experiment), optimum, names=["CGPTS with " + str(a) for a in alphas])

