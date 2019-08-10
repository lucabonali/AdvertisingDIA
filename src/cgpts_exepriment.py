import time
import numpy as np
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
from src.optimization import get_optimized_reward, get_optimized_arms

import src.plotting as pl

n_arms = 10
min_budget = 0
max_budget = 9

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 2

const_budget = 100
n_sub_campaigns = 5

T = 40
n_experiments = 2

cgpts_rewards_per_experiment = []
errs_per_experiment = []
rewards_per_experiment = []
envs = None

if __name__ == '__main__':
    tot_time = time.time()
    for e in range(n_experiments):
        print("Experiment #" + str(e + 1), end='')
        start_time = time.time()

        envs = [BudgetEnvironment(budgets, sigma) for _ in range(n_sub_campaigns)]
        sub_campaigns = [GPTSLearner(n_arms, arms=budgets) for _ in range(n_sub_campaigns)]
        cgpts = CGPTSLearner(sub_campaigns, budgets)

        errs = [[] for _ in range(n_sub_campaigns)]

        for t in range(0, T):
            pulled_arms = cgpts.pull_arms(const_budget)
            rewards = [env.round(pulled_arms[idx]) for idx, env in enumerate(envs)]
            cgpts.update(pulled_arms, rewards)

            # GP regression errors
            for sc in range(n_sub_campaigns):
                p, _ = cgpts.predict(sc)
                v = envs[sc].realfunc(budgets)
                err = np.abs(v - p)
                errs[sc].append(np.max(err))

            # make prediction for 1st sub-campaign
            if e == 1 and (t % 5) == 0:
                y_preds, _ = cgpts.predict()
                x_observ, y_observ = cgpts.get_samples()
                pl.plot_gp_regression(n_samples=t, x_pred=budgets, y_pred=y_preds, x_obs=x_observ, y_obs=y_observ, sigma=sigma, true_function=envs[0].realfunc)

        print(": " + str(time.time() - start_time) + " sec")

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

    print("Algorithm ended in {:.2f} sec.".format(time.time() - tot_time))

    pl.plot_regression_error(errs_per_experiment, n_sub_campaigns)

    """
    means = np.zeros(shape=len(envs[0].means))
    for env in envs:
        means += np.array(env.means)
    opt = np.max(means)
    """

    # compute the optimum
    assert envs is not None
    true_rewards = [env.realfunc(budgets).tolist() for idx, env in enumerate(envs)]
    opt = get_optimized_reward(true_rewards, budgets.tolist())
    print("optimum: " + str(opt))

    print(get_optimized_arms(true_rewards, budgets.tolist()))

    pl.plot_rewards(rewards_per_experiment=cgpts_rewards_per_experiment, opt=opt, n_samples=T)
    pl.plot_regret(rewards_per_experiment=np.array(cgpts_rewards_per_experiment), opt=opt)
