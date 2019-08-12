import time
from typing import List

import numpy as np

import src.data as curves
import src.plotting as plotting
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
from src.data import p_c1, p_c2, p_c3
from src.optimization import combinatorial_optimization

n_arms = 20
min_budget = 0
max_budget = 19

T = 15#57
n_experiments = 5

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0
sigma_medium = 2.5
sigma_low = 1.0

cgpts_rewards_per_experiment = []
errs_per_experiment = []

sub_campaigns = []
ctc = 0  # Index of the sub_campaign that has to be checked for disaggregation

bench_rewards_per_experiment = []

update = True

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

        bench: List[CGPTSLearner] = [
            CGPTSLearner(name="CGPTS_disagg", budgets=budgets, sub_campaigns=[
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c1)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c2)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c3))
            ])
        ]

        cgpts = CGPTSLearner("CGPTS", sub_campaigns, budgets)
        errs = [[] for _ in range(len(sub_campaigns))]

        for t in range(T):
            reward_matrix = cgpts.pull_arms()
            pulled_arms, best_reward = combinatorial_optimization(reward_matrix, budgets.tolist())

            if update: #and t > 22 and (t % 7) == 0:
                bench_reward_matrix = bench[0].pull_arms()
                bench_reward_matrix.extend(reward_matrix[1:])
                bench_pulled_arms, bench_reward = combinatorial_optimization(bench_reward_matrix, budgets.tolist())
                bench[0].update(bench_pulled_arms[0:3], [c.env.round(bench_pulled_arms[idx]) for idx, c in enumerate(bench[0].sub_campaigns)])

                #print("\nAGGREGATE REWARD    = {}".format(best_reward))
                #print(pulled_arms)
                #print("DISAGGREGATE REWARD = {}".format(bench_reward))
                #print(bench_pulled_arms)

            rewards = [sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
            cgpts.update(pulled_arms, rewards)

            """
            if update:
                for bench_cgpts in bench:
                    bench_pulled_arms = [pulled_arms[ctc]] * bench_cgpts.n_sub_campaigns
                    bench_rewards = np.array([p_c1, p_c2, p_c3]) * rewards[ctc]
                    bench_cgpts.update(pulled_arms=bench_pulled_arms, rewards=bench_rewards)
            """

            for idx, bench_campaign in enumerate(bench[0].sub_campaigns):
                if e == 0 and t == T-1:
                    y_pred = bench_campaign.means
                    x_obs, y_obs = bench_campaign.pulled_arms, bench_campaign.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_pred,
                                                x_obs=x_obs, y_obs=y_obs,
                                                sigma=sigma,
                                                true_function=bench_campaign.env.realfunc,
                                                fc='g')

            for idx, sub_campaign in enumerate(sub_campaigns):
                y_pred = sub_campaign.means
                true = sub_campaign.env.realfunc(budgets)
                errs[idx].append(np.max(np.abs(true - y_pred)))

                if e == 0 and t == T-1:
                    x_obs, y_obs = sub_campaign.pulled_arms, sub_campaign.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_pred,
                                                x_obs=x_obs, y_obs=y_obs,
                                                sigma=sigma,
                                                true_function=sub_campaign.env.realfunc)

        print(": {:.2f} sec".format(time.time() - start_time))

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

        bench_rewards_per_experiment.append(bench[0].get_collected_rewards() + cgpts.get_collected_rewards() - cgpts.sub_campaigns[0].collected_rewards)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    plotting.plot_regression_error(errs_per_experiment, len(sub_campaigns))

    true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
    best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist())
    print("Best budgets => {}".format(best_budgets))
    print("Optimum      => {}".format(optimum))

    plotting.plot_rewards(cgpts_rewards_per_experiment, optimum, T)
    plotting.plot_rewards(bench_rewards_per_experiment, optimum, T)
    plotting.plot_regret(np.array(cgpts_rewards_per_experiment), optimum)

