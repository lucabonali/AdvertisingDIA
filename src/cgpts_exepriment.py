import numpy as np
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
import matplotlib.pyplot as plt

n_arms = 20
min_budget = 0.0
max_budget = 1.0

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 2

const_budget = 100
n_sub_campaigns = 5

T = 40
n_experiments = 2

cgpts_rewards_per_experiment = []
errs_per_experiment = []

if __name__ == '__main__':
    for e in range(n_experiments):
        print("Experiment #" + str(e+1))

        envs = [BudgetEnvironment(budgets, sigma) for _ in range(n_sub_campaigns)]
        sub_campaigns = [GPTSLearner(n_arms, arms=budgets) for _ in range(n_sub_campaigns)]
        cgpts = CGPTSLearner(n_sub_campaigns, sub_campaigns)

        errs = [[] for _ in range(n_sub_campaigns)]

        for t in range(0, T):
            pulled_arms = cgpts.pull_arms(const_budget)
            rewards = [env.round(pulled_arms[idx]) for idx, env in enumerate(envs)]
            cgpts.update(pulled_arms, rewards)

            # GP regression errors
            for sc in range(n_sub_campaigns):
                p, _ = cgpts.predict(sc)
                v = envs[sc].true_func(budgets)
                err = np.abs(v - p)
                errs[sc].append(np.max(err))

            # make prediction for 1st sub-campaign
            if False: #e == 2 and (t % 3) == 0:
                y_pred, sigma = cgpts.predict()
                x_obs, y_obs = cgpts.get_samples()
                x_pred = budgets

                plt.figure(t)
                plt.plot(x_pred, envs[0].true_func(x_pred), 'r:', label=r'$n(x)$')
                plt.plot(np.atleast_2d(x_obs).T.ravel().ravel(), y_obs.ravel(), 'ro', label=u'Observed Clicks')
                plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
                plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                         np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
                         alpha=.5, fc='b', ec='None', label='95% conf interval')
                plt.xlabel('$x$')
                plt.ylabel('$n(x)$')
                plt.legend(loc='lower right')
                plt.show()

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())
        errs_per_experiment.append(errs)

    plt.figure(0)
    plt.ylabel("Regression Error")
    plt.xlabel("Samples")
    colors = ["", "", "", "", ""]
    mean_errs = np.mean(errs_per_experiment, axis=0)
    for sc in range(n_sub_campaigns):
        plt.plot(mean_errs[sc])
    plt.legend([str(x+1) for x in range(n_sub_campaigns)], loc="best")
    plt.show()

    means = np.zeros(shape=len(envs[0].means))
    for env in envs:
        means += np.array(env.means)
    opt = np.max(means)
    print(opt)
    # plot regret analysis of the model
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - cgpts_rewards_per_experiment, axis=0)), 'g')
    plt.legend(["CGPTS"])
    plt.show()
