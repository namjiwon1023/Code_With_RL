import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn
import random

def _random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)


def _Static_plot(scores, figure_file):
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-10):(i+1)])
    plt.plot(scores, "r-", linewidth=1.5, label="Episode Reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="Avg Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend(loc="best", shadow=True)
    plt.title('Return')
    plt.savefig(figure_file)


def _Dynamic_plot(scores, eval_rewards):
    plt.subplot(121)
    z = [c+1 for c in range(len(scores))]
    running_avg = np.zeros(len(scores))
    for e in range(len(running_avg)):
        running_avg[e] = np.mean(scores[max(0, e-10):(e+1)])
    plt.cla()
    plt.title("Return")
    plt.grid(True)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.plot(scores, "r-", linewidth=1.5, label="Episode Reward")
    plt.plot(z, running_avg, "b-", linewidth=1.5, label="Avg Reward")
    plt.legend(loc="best", shadow=True)

    plt.subplot(122)
    plt.cla()
    plt.title("Return")
    plt.grid(True)
    plt.xlabel("Step (Unit 1000)")
    plt.ylabel("Total Reward")
    plt.plot(eval_rewards, "b-", linewidth=1.5, label="Step Reward")
    plt.legend(loc="best", shadow=True)
    plt.pause(0.1)

    plt.savefig('./sac.jpg')
    plt.show()

