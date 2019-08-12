import time
from typing import List

import numpy as np

import src.data as curves
import src.plotting as plotting
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
from src.optimization import combinatorial_optimization

n_arms = 20
min_budget = 0
max_budget = 19

T = 50
n_experiments = 10
# 100 x 100 -> ~ 11 hours

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0

cgpts_rewards_per_experiment = []
errs_per_experiment = []

sub_campaigns = []

if __name__ == '__main__':
    tot_time = time.time()
    for e in range(n_experiments):
        print('Experiment #{}'.format(e + 1), end='')
        start_time = time.time()

        sub_campaigns: List[GPTSLearner] = [
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.true2)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.true3)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.true4)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.true5))
        ]

        cgpts = CGPTSLearner("CGPTS", sub_campaigns, budgets)
        errs = [[] for _ in range(len(sub_campaigns))]

        for t in range(T):
            reward_matrix = cgpts.pull_arms()
            pulled_arms, _ = combinatorial_optimization(reward_matrix, budgets.tolist())
            rewards = [sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
            cgpts.update(pulled_arms, rewards)

            for idx, sub_campaign in enumerate(sub_campaigns):
                y_pred = sub_campaign.means
                true = sub_campaign.env.realfunc(budgets)
                errs[idx].append(np.max(np.abs(true - y_pred)))

                if e == round(n_experiments/2) and t == T-1:
                    x_obs, y_obs = sub_campaign.pulled_arms, sub_campaign.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_pred,
                                                x_obs=x_obs, y_obs=y_obs,
                                                sigma=sigma,
                                                true_function=sub_campaign.env.realfunc)

        print(": {:.2f} sec".format(time.time() - start_time))

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    plotting.plot_regression_error(errs_per_experiment, len(sub_campaigns))

    true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
    best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist())
    print("Best budgets => {}".format(best_budgets))
    print("Optimum      => {}".format(optimum))

    plotting.plot_rewards(cgpts_rewards_per_experiment, optimum, T)
    plotting.plot_regret(np.array(cgpts_rewards_per_experiment), optimum)

