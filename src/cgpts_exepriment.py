import numpy as np
from src.BudgetEnvironment import BudgetEnvironment
from src.CGPTSLearner import CGPTSLearner
from src.GPTSLearner import GPTSLearner
import matplotlib.pyplot as plt

n_arms = 20
min_budget = 0.0
max_budget = 1.0

budgets = np.linspace(min_budget, max_budget, n_arms)
sigma = 10

const_budget = 100
n_sub_campaigns = 5

T = 20
n_experiments = 1

cgpts_rewards_per_experiment = []

if __name__ == '__main__':
    for e in range(n_experiments):
        print("Experiment #" + str(e))

        envs = [BudgetEnvironment(budgets, sigma) for _ in range(n_sub_campaigns)]
        sub_campaigns = [GPTSLearner(n_arms, arms=budgets) for _ in range(n_sub_campaigns)]
        cgpts = CGPTSLearner(n_sub_campaigns, sub_campaigns)

        for t in range(0, T):
            pulled_arms = cgpts.pull_arms(const_budget)
            rewards = [env.round(pulled_arms[idx]) for idx, env in enumerate(envs)]
            cgpts.update(pulled_arms, rewards)

        cgpts_rewards_per_experiment.append(cgpts.get_collected_rewards())

    means = np.zeros(shape=len(envs[0].means))
    for env in envs:
        means += np.array(env.means)
    opt = np.max(means)
    print(opt)
    # plot data
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - cgpts_rewards_per_experiment, axis=0)), 'g')
    plt.legend(["CGPTS"])
    plt.show()
