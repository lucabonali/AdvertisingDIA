import time
from typing import List

import numpy as np

import src.data as curves
import src.plotting as plotting
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.CGTSLearner import CGTSLearner
from src.GTSLearner import GTSLearner
from src.GPTSLearner import GPTSLearner
from src.optimization import combinatorial_optimization

n_arms = 20
min_budget = 0
max_budget = 19

T = 10
n_experiments = 2
# 100 x 100 -> ~ 11 hours
# 100 x 80 -> 9.84 hours

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0

allow_empty = True

cgpts_rewards_per_experiment = []
cgts_rewards_per_experiment = []
errs_per_experiment = []

gpts_sub_campaigns = []

if __name__ == '__main__':
    tot_time = time.time()
    for e in range(n_experiments):
        print('Experiment #{}'.format(e + 1), end='')
        start_time = time.time()

        gpts_sub_campaigns: List[GPTSLearner] = [
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.facebook_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.instagram_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.youtube_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.bing_agg))
        ]

        gts_sub_campaigns: List[GTSLearner] = [
            GTSLearner(n_arms=n_arms, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
            GTSLearner(n_arms=n_arms, env=BudgetEnvironment(budgets, sigma, curves.facebook_agg)),
            GTSLearner(n_arms=n_arms, env=BudgetEnvironment(budgets, sigma, curves.instagram_agg)),
            GTSLearner(n_arms=n_arms, env=BudgetEnvironment(budgets, sigma, curves.youtube_agg)),
            GTSLearner(n_arms=n_arms, env=BudgetEnvironment(budgets, sigma, curves.bing_agg))
        ]

        cgpts = CGPTSLearner("CGPTS", gpts_sub_campaigns, budgets)
        cgts = CGTSLearner("CGTS", gts_sub_campaigns, budgets)
        errs = [[] for _ in range(len(gpts_sub_campaigns))]

        for t in range(T):
            # GAUSSIAN PROCESS THOMPSON SAMPLING
            reward_matrix = cgpts.pull_arms()
            pulled_arms, _ = combinatorial_optimization(reward_matrix, budgets.tolist(), allow_empty=allow_empty)
            rewards = [gpts_sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
            cgpts.update(pulled_arms, rewards)

            # GAUSSIAN THOMPSON SAMPLING
            reward_matrix = cgts.pull_arms()
            pulled_arms, _ = combinatorial_optimization(reward_matrix, budgets.tolist(), allow_empty=allow_empty)
            rewards = [gpts_sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]

            cgts.update(pulled_arms, rewards)


        print(": {:.2f} sec".format(time.time() - start_time))

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())
        cgts_rewards_per_experiment.append(cgts.get_collected_rewards())
        errs_per_experiment.append(errs)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in gpts_sub_campaigns]
    best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist(), allow_empty=allow_empty)
    print("Best budgets => {}".format(best_budgets))
    print("Optimum      => {}".format(optimum))

    plotting.plot_rewards(cgpts_rewards_per_experiment, optimum, T)
    plotting.plot_rewards(cgts_rewards_per_experiment, optimum, T)
    plotting.plot_comparison_regret(np.array(cgpts_rewards_per_experiment), np.array(cgts_rewards_per_experiment), optimum)

