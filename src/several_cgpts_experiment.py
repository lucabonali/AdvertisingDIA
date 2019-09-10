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

n_algorithms = 2
T = 20
n_experiments = 1
# 100 x 100 -> ~ 11 hours
# 100 x 80 -> 9.84 hours

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0

cgpts_rewards_per_experiment = [[] for _ in range(n_algorithms)]

sub_campaigns = []


if __name__ == '__main__':
    tot_time = time.time()
    for a in range(n_algorithms):
        allow_empty = True if a == 0 else False
        for e in range(n_experiments):
            print('Experiment #{}'.format(e + 1), end='')
            start_time = time.time()

            sub_campaigns: List[GPTSLearner] = [
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.facebook_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.instagram_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.youtube_agg)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.bing_agg))
            ]

            cgpts = CGPTSLearner("CGPTS", sub_campaigns, budgets)

            for t in range(T):
                reward_matrix = cgpts.pull_arms()
                pulled_arms, _ = combinatorial_optimization(reward_matrix, budgets.tolist(), allow_empty=allow_empty)
                rewards = [sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
                cgpts.update(pulled_arms, rewards)

            print(": {:.2f} sec".format(time.time() - start_time))

            cgpts_rewards_per_experiment[a].append(cgpts.get_collected_rewards())

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
    best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist(), allow_empty=True)
    print("Best budgets => {}".format(best_budgets))
    print("Optimum      => {}".format(optimum))

    plotting.plot_multiple_rewards(cgpts_rewards_per_experiment, optimum, T, _names=["CGPTS allowing 0-budget",
                                                                                     "CGPTS_forcing not 0-budget"])
    plotting.plot_multiple_regret(np.array(cgpts_rewards_per_experiment), optimum, names=["CGPTS allowing 0-budget",
                                                                                          "CGPTS_forcing not 0-budget"])

