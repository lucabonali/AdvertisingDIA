import time
import numpy as np
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
from src.optimization import get_optimized_reward, get_optimized_arms
from typing import List

import src.plotting as plotting
import src.curves as curves


def initialize_cgpts():
    learners = []
    first1 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.google_c1)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.google_c2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.google_c3)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true3)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true4)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true5))]

    first2 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma,
                                                             lambda x: curves.google_c1(x) + curves.google_c2(
                                                                 x) + curves.google_c3(x))),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true3)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true4)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma, curves.true5))]
    learners.append(CGPTSLearner("CGPTS1", first1, budgets))
    learners.append(CGPTSLearner("CGPTS2", first2, budgets))
    return learners


n_arms = 20
min_budget = 0
max_budget = 19

T = 50
n_experiments = 1
n_learners = 2

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 5

cgpts_learners: List[CGPTSLearner] = []
rewards_per_cgpts_per_experiment = [[] for _ in range(n_learners)]
errs_per_experiment = []
current_cgpts = None

if __name__ == '__main__':
    tot_time = time.time()
    for e in range(n_experiments):
        print('Experiment #{}'.format(e + 1), end='')
        start_time = time.time()

        # Initialize all cgpts learners with their own environments
        cgpts_learners = initialize_cgpts()
        current_cgpts = cgpts_learners[0]

        if e == 0:
            print("\n")
            for cgpts in cgpts_learners:
                print("{} -> # sub campaigns = {}".format(cgpts.name, cgpts.n_sub_campaigns))

        errs = [[] for _ in range(cgpts_learners[0].n_sub_campaigns)]

        for t in range(T):
            for cgpts in cgpts_learners:
                pulled_arms = cgpts.pull_arms()
                rewards = [gpts.env.round(pulled_arms[idx]) for idx, gpts in enumerate(cgpts.sub_campaigns)]
                cgpts.update(pulled_arms, rewards)

            # Compute additional information for the current cgpts
            for idx, gpts in enumerate(cgpts_learners[0].sub_campaigns):
                if t == T - 1:
                    # plot the GP regression
                    y_predicted = gpts.means
                    x_observed = gpts.pulled_arms
                    y_observed = gpts.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_predicted,
                                                x_obs=x_observed, y_obs=y_observed,
                                                sigma=sigma, true_function=gpts.env.realfunc)
                # compute GP regression error
                y_predicted = gpts.means
                y_true = gpts.env.realfunc(budgets)
                err = np.abs(y_true - y_predicted)
                errs[idx].append(np.max(err))

        print(": {:.2f} sec".format(time.time() - start_time))

        for idx, cgpts in enumerate(cgpts_learners):
            rewards_per_cgpts_per_experiment[idx].append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    plotting.plot_regression_error(errs_per_experiment, cgpts_learners[0].n_sub_campaigns)

    true_rewards_matrix = [gpts.env.realfunc(budgets).tolist() for gpts in current_cgpts.sub_campaigns]
    optimum = get_optimized_reward(true_rewards_matrix, budgets.tolist())

    print(get_optimized_arms(true_rewards_matrix, budgets.tolist()))

    plotting.plot_multiple_rewards(rewards_per_cgpts_per_experiment, optimum, T, [c.name for c in cgpts_learners])
    plotting.plot_multiple_regret(np.array(rewards_per_cgpts_per_experiment), optimum, [c.name for c in cgpts_learners])
