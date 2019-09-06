import matplotlib.pyplot as plt
import numpy as np


def plot_gp_regression(n_samples, x_pred, y_pred, x_obs, y_obs, sigma, true_function, fc='b'):
    plt.figure(n_samples)
    plt.plot(x_pred, true_function(x_pred), 'r:', label=r'$True clicks$')
    plt.plot(np.atleast_2d(x_obs).T.ravel().ravel(), y_obs.ravel(), 'ro', label=u'Observed Clicks')
    plt.plot(x_pred, y_pred, 'b-', label=u'Predicted Clicks')
    plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
             np.concatenate([y_pred - 1.96 * sigma, (y_pred + 1.96 * sigma)[::-1]]),
             alpha=.5, fc=fc, ec='None', label='95% conf interval')
    plt.xlabel('$budgets$')
    plt.ylabel('$Expected number of clicks$')
    plt.legend(loc='lower right')
    plt.show()


def plot_regret(rewards_per_experiment, opt):
    # plot regret analysis of the model
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - rewards_per_experiment, axis=0)), 'g')
    plt.legend(["CGPTS"], loc="best")
    plt.show()


def plot_comparison_regret(rewards_per_experiment1, rewards_per_experiment2, opt):
    # plot regret analysis of the model
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    plt.plot(np.cumsum(np.mean(opt - rewards_per_experiment1, axis=0)), 'g')
    plt.plot(np.cumsum(np.mean(opt - rewards_per_experiment2, axis=0)), 'r')
    plt.legend(["CGPTS", "CGTS"], loc="best")
    plt.show()


def plot_multiple_regret(rewards_per_cgpts_per_experiment, opt, names):
    # plot regret analysis of the model
    print("plotting regrets..")
    plt.figure(0)
    plt.ylabel("Regret")
    plt.xlabel("t")
    for idx, rewards_per_experiment in enumerate(rewards_per_cgpts_per_experiment):
        plt.plot(np.cumsum(np.mean(opt[idx] - rewards_per_experiment, axis=0)))
    plt.legend(names, loc="best")
    plt.show()


def plot_multiple_rewards(rewards_per_cgpts_per_experiment, opt, n_samples, _names):
    print("plotting rewards..")
    names = ["Clairvoyant{}".format(idx) for idx, _ in enumerate(opt)]
    names.extend(_names)
    plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    for o in opt:
        plt.plot(np.ones(shape=n_samples) * o)
    for rewards_per_experiment in rewards_per_cgpts_per_experiment:
        plt.plot(np.mean(rewards_per_experiment, axis=0))
    plt.legend(names, loc="best")
    plt.show()


def plot_rewards(rewards_per_experiment, opt, n_samples):
    plt.figure(0)
    plt.ylabel("Reward")
    plt.xlabel("t")
    plt.plot(np.ones(shape=n_samples) * opt, 'r')
    plt.plot(np.mean(rewards_per_experiment, axis=0), 'g')
    plt.legend(["Clairvoyant", "CGPTS"], loc="best")
    plt.show()


def plot_regression_error(errs_per_experiment, n_sub_campaigns):
    plt.figure(0)
    plt.ylabel("Regression Error")
    plt.xlabel("Samples")
    mean_errs = np.mean(errs_per_experiment, axis=0)
    for sc in range(n_sub_campaigns):
        plt.plot(mean_errs[sc])
    plt.legend([str(x + 1) for x in range(n_sub_campaigns)], loc="best")
    plt.show()

