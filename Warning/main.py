import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
from tqdm import tqdm
import time

from agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, NoisyDQNAgent
from utils import _random_seed
from arguments import get_args
from runner import Runner

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('./logs/' + args.algorithm)
    _random_seed(args.seed)

    if args.algorithm == 'DQN':
        agent = DQNAgent(args)

    if args.algorithm == 'Double_DQN':
        agent = DoubleDQNAgent(args)

    if args.algorithm == 'Dueling_DQN':
        agent = DuelingDQNAgent(args)

    if args.algorithm == 'D3QN':
        agent = D3QNAgent(args)

    if args.algorithm == 'Noisy_DQN':
        agent = NoisyDQNAgent(args)

    runner = Runner(agent, args, agent.env, writer)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        runner.gif(agent, agent.env)
    else:
        runner.run()
