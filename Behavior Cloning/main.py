import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
from tqdm import tqdm
import time

from algorithm.agent import SACAgent
from utils.utils import random_seed
from utils.arguments import get_args
from utils.runner import Runner

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter()
    random_seed(args.seed)

    agent = SACAgent(args)

    runner = Runner(agent, args, agent.env, writer)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        runner.gif(agent, agent.env)
    else:
        runner.run()