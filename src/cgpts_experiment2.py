import time
import numpy as np
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
from src.optimization import get_optimized_reward, get_optimized_arms, combinatorial_optimization
from typing import List

import src.plotting as plotting
import src.curves as curves


def initialize_cgpts():
    learners = []
    agg2 = GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true2))
    agg3 = GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true3))
    agg4 = GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_medium, curves.true4))
    agg5 = GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true5))

    first2 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.google_c1)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_medium, curves.google_c2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_low, curves.google_c3)),
              agg2,
              agg3,
              agg4,
              agg5]

    first1 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high+sigma_medium+sigma_low,
                                                             lambda x: curves.google_c1(x) + curves.google_c2(
                                                                 x) + curves.google_c3(x))),
              agg2,
              agg3,
              agg4,
              agg5]

    first3 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high+sigma_medium,
                                                             lambda x: curves.google_c1(x) + curves.google_c2(
                                                                 x))),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_low, curves.google_c3)),
              agg2,
              agg3,
              agg4,
              agg5]

    first4 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high+sigma_low,
                                                             lambda x: curves.google_c1(x) + curves.google_c3(
                                                                 x))),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_medium, curves.google_c2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true2)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true3)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_medium, curves.true4)),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.true5))]

    first5 = [GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_medium + sigma_low,
                                                             lambda x: curves.google_c2(x) + curves.google_c3(
                                                                 x))),
              GPTSLearner(n_arms, budgets, BudgetEnvironment(budgets, sigma_high, curves.google_c1)),
              agg2,
              agg3,
              agg4,
              agg5]

    learners.append(CGPTSLearner("CGPTS_agg", first1, budgets))
    if update:
        learners.append(CGPTSLearner("CGPTS_disagg", first2, budgets))
        learners.append(CGPTSLearner("CGPTS_disagg_c3", first3, budgets))
        learners.append(CGPTSLearner("CGPTS_disagg_c2", first4, budgets))
        learners.append(CGPTSLearner("CGPTS_disagg_c1", first5, budgets))
    return learners


n_arms = 20
min_budget = 0
max_budget = 19

T = 50
n_experiments = 20
n_learners = 1

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma_high = 5.0
sigma_medium = 2.5
sigma_low = 1.0

cgpts_learners: List[CGPTSLearner] = []
rewards_per_cgpts_per_experiment = [[] for _ in range(n_learners)]
best_rewards_per_experiment = []
errs_per_experiment = []
current_cgpts = None

update = False

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
        best_rewards = []
        for t in range(T):
            last_week_rewards = []
            for cgpts in cgpts_learners:
                pulled_arms = cgpts.pull_arms()
                rewards = [gpts.env.round(pulled_arms[idx]) for idx, gpts in enumerate(cgpts.sub_campaigns)]
                cgpts.update(pulled_arms, rewards)
                if update and (t % 7) == 0 and t != 0:
                    last_week_rewards.append(np.sum(cgpts.get_collected_rewards()[-7:]))

            if update and (t % 7) == 0 and t != 0:
                best_rewards.extend(current_cgpts.get_collected_rewards()[-7:].tolist())
                # update best cgpts
                best_cgpts = np.argmax(last_week_rewards)
                current_cgpts = cgpts_learners[best_cgpts]
                print("Sample {} -> best model = {}".format(t+1, current_cgpts.name))

            # Compute additional information for 1 cgpts
            for idx, gpts in enumerate(cgpts_learners[0].sub_campaigns):
                if (e % 39) == 0 and t == T - 1:
                    # plot the GP regression
                    y_predicted = gpts.means
                    x_observed = gpts.pulled_arms
                    y_observed = gpts.collected_rewards
                    plotting.plot_gp_regression(n_samples=t,
                                                x_pred=budgets, y_pred=y_predicted,
                                                x_obs=x_observed, y_obs=y_observed,
                                                sigma=sigma_high, true_function=gpts.env.realfunc)
                # compute GP regression error
                y_predicted = gpts.means
                y_true = gpts.env.realfunc(budgets)
                err = np.abs(y_true - y_predicted)
                errs[idx].append(np.max(err))

        print(": {:.2f} sec".format(time.time() - start_time))

        best_rewards_per_experiment.append(best_rewards)

        for idx, cgpts in enumerate(cgpts_learners):
            rewards_per_cgpts_per_experiment[idx].append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    plotting.plot_regression_error(errs_per_experiment, cgpts_learners[0].n_sub_campaigns)

    optimums = []
    for cgpts in cgpts_learners:
        true_rewards_matrix = [gpts.env.realfunc(budgets).tolist() for gpts in cgpts.sub_campaigns]
        opt_arms, opt = combinatorial_optimization(true_rewards_matrix, budgets.tolist())
        optimums.append(opt)
        print("{}: optimum={} ; arms={}".format(cgpts.name, opt, opt_arms))

    plotting.plot_multiple_rewards(rewards_per_cgpts_per_experiment, optimums, T, [c.name for c in cgpts_learners])
    plotting.plot_multiple_regret(np.array(rewards_per_cgpts_per_experiment), optimums, [c.name for c in cgpts_learners])
    if update:
        plotting.plot_regret(best_rewards_per_experiment, np.max(optimums))
