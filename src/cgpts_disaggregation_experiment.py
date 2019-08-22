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

T = 50
n_experiments = 1
# 100 x 80 ~12.5 hours

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5.0

allow_empty = False

disagg_day_per_experiment = []
disagg_learner_per_experiment = []
disagg_learner_counts = [0, 0, 0, 0]

cgpts_rewards_per_experiment = []

sub_campaigns = []
optimums = []
best_partitions_per_experiment = []
ctc = 0  # Index of the sub_campaign that has to be checked for disaggregation

if __name__ == '__main__':
    tot_time = time.time()
    for e in range(n_experiments):
        update = True
        print('Experiment #{}'.format(e + 1), end='')
        start_time = time.time()

        sub_campaigns: List[GPTSLearner] = [
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.facebook_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.instagram_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.youtube_agg)),
            GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.bing_agg))
        ]

        bench: List[CGPTSLearner] = [
            CGPTSLearner(name="CGPTS_disagg", budgets=budgets, sub_campaigns=[
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c1)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c2)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c3))
            ]),
            CGPTSLearner(name="CGPTS_disagg_c3", budgets=budgets, sub_campaigns=[
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, lambda x: curves.google_c1(x)*p_c1 + curves.google_c2(x)*p_c2)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c3))
            ]),
            CGPTSLearner(name="CGPTS_disagg_c2", budgets=budgets, sub_campaigns=[
                GPTSLearner(n_arms=n_arms, arms=budgets,
                            env=BudgetEnvironment(budgets, sigma, lambda x: curves.google_c1(x)*p_c1 + curves.google_c3(x)*p_c3)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c2))
            ]),
            CGPTSLearner(name="CGPTS_disagg_c1", budgets=budgets, sub_campaigns=[
                GPTSLearner(n_arms=n_arms, arms=budgets,
                            env=BudgetEnvironment(budgets, sigma, lambda x: curves.google_c2(x)*p_c2 + curves.google_c3(x)*p_c3)),
                GPTSLearner(n_arms=n_arms, arms=budgets, env=BudgetEnvironment(budgets, sigma, curves.google_c1))
            ])
        ]

        cgpts = CGPTSLearner("CGPTS", sub_campaigns, budgets)

        for t in range(T):
            reward_matrix = cgpts.pull_arms()
            pulled_arms, online_reward = combinatorial_optimization(reward_matrix, budgets.tolist(), allow_empty=allow_empty)

            rewards = [sub_campaigns[idx].env.round(arm) for idx, arm in enumerate(pulled_arms)]
            cgpts.update(pulled_arms, rewards)

            if update:
                if t > 8 and (t % 7) == 0:
                    bench_pulled_arms = [[] for _ in bench]
                    bench_rewards = [0 for _ in bench]
                    print("\nDay{}\n".format(t+1))
                    for idx, bench_cgpts in enumerate(bench):
                        bench_reward_matrix = bench_cgpts.pull_arms()
                        bench_reward_matrix.extend(reward_matrix[1:].copy())
                        tmp_pulled_arms, tmp_reward = combinatorial_optimization(bench_reward_matrix, budgets.tolist())
                        bench_pulled_arms[idx].extend(tmp_pulled_arms)
                        bench_rewards[idx] = tmp_reward
                        print("{} reward -> {} with {}".format(bench_cgpts.name, tmp_reward, tmp_pulled_arms))
                    # bench[0].update(bench_pulled_arms[0:3], [c.env.round(bench_pulled_arms[idx]) for idx, c in enumerate(bench[0].sub_campaigns)])

                    print("{} reward -> {} with {}".format(cgpts.name, online_reward, pulled_arms))

                    best_bench_reward_idx = np.argmax(np.array(bench_rewards))
                    best_bench_reward = bench_rewards[best_bench_reward_idx]
                    print("best bench reward {} at {}".format(best_bench_reward, best_bench_reward_idx))

                    if best_bench_reward >= (online_reward + sigma):
                        print("Performing disaggregation..")
                        update = False
                        disagg_day_per_experiment.append(t)
                        disagg_learner_counts[best_bench_reward_idx] += 1
                        disagg_learner_per_experiment.append(bench[best_bench_reward_idx].name)
                        cgpts.remove_sub_campaign(sub_campaigns[ctc])
                        cgpts.add_sub_campaigns(bench[best_bench_reward_idx].sub_campaigns)

                for idx, bench_cgpts in enumerate(bench):
                    if idx == 0:
                        # all classes disaggregated
                        bench_pulled_arms = [round(pulled_arms[ctc]/3)] * bench_cgpts.n_sub_campaigns
                        bench_rewards = np.array([p_c1, p_c2, p_c3]) * rewards[ctc]
                    elif idx == 1:
                        # only c3 disaggregated
                        bench_pulled_arms = [round(pulled_arms[ctc] / 2)] * bench_cgpts.n_sub_campaigns
                        bench_rewards = np.array([(p_c1 + p_c2), p_c3]) * rewards[ctc]
                    elif idx == 2:
                        # only c2 disaggregated
                        bench_pulled_arms = [round(pulled_arms[ctc] / 2)] * bench_cgpts.n_sub_campaigns
                        bench_rewards = np.array([(p_c1 + p_c3), p_c2]) * rewards[ctc]
                    elif idx == 3:
                        # only c1 disaggregated
                        bench_pulled_arms = [round(pulled_arms[ctc] / 2)] * bench_cgpts.n_sub_campaigns
                        bench_rewards = np.array([(p_c2 + p_c3), p_c1]) * rewards[ctc]
                    else:
                        raise RuntimeError("Idx not handled")

                    bench_cgpts.update(pulled_arms=bench_pulled_arms, rewards=bench_rewards)

            if e == round(n_experiments / 2) and t == T - 1:
                for b in bench:
                    for idx, bench_campaign in enumerate(b.sub_campaigns):
                        y_pred = bench_campaign.means
                        x_obs, y_obs = bench_campaign.pulled_arms, bench_campaign.collected_rewards
                        plotting.plot_gp_regression(n_samples=t,
                                                    x_pred=budgets, y_pred=y_pred,
                                                    x_obs=x_obs, y_obs=y_obs,
                                                    sigma=sigma,
                                                    true_function=bench_campaign.env.realfunc,
                                                    fc='g')

            for idx, sub_campaign in enumerate(sub_campaigns):
                if e == round(n_experiments/2) and t == T-1:
                    y_pred = sub_campaign.means
                    x_obs, y_obs = sub_campaign.pulled_arms, sub_campaign.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_pred,
                                                x_obs=x_obs, y_obs=y_obs,
                                                sigma=sigma,
                                                true_function=sub_campaign.env.realfunc)
        if update:
            disagg_day_per_experiment.append(None)
            disagg_learner_per_experiment.append(None)

        true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
        best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist(),
                                                           allow_empty=allow_empty)
        optimums.append(optimum)
        best_partitions_per_experiment.append(best_budgets)

        print(": {:.2f} sec".format(time.time() - start_time))

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    #true_rewards_matrix = [c.env.realfunc(budgets).tolist() for c in sub_campaigns]
    #best_budgets, optimum = combinatorial_optimization(true_rewards_matrix, budgets.tolist(), allow_empty=allow_empty)
    opt_idx = np.argmax(np.array(optimums))
    print("Optimums     => {}".format(optimums))
    print("Best budgets => {}".format(best_partitions_per_experiment[opt_idx]))
    print("Optimum      => {}".format(optimums[opt_idx]))

    print("Disaggregation days: {}".format(disagg_day_per_experiment))
    print("Best Learners:       {}".format(disagg_learner_per_experiment))
    print("Best Learners:       {}".format(disagg_learner_counts))

    plotting.plot_rewards(cgpts_rewards_per_experiment, optimums[opt_idx], T)
    plotting.plot_regret(np.array(cgpts_rewards_per_experiment), optimums[opt_idx])

