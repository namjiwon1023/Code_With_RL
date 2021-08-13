# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from test.agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, NoisyDQNAgent
from test.agent import DDPGAgent, TD3Agent, SACAgent, PPOAgent, A2CAgent, BC_SACAgent
from test.arguments import get_args
from test.runner import Runner

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('./logs/' + args.algorithm)

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

    if args.algorithm == 'DDPG':
        agent = DDPGAgent(args)

    if args.algorithm == 'TD3':
        agent = TD3Agent(args)

    if args.algorithm == 'SAC':
        agent = SACAgent(args)

    if args.algorithm == 'PPO':
        agent = PPOAgent(args)

    if args.algorithm == 'A2C':
        agent = A2CAgent(args)

    if args.algorithm == 'BC_SAC':
        agent = BC_SACAgent(args)

    runner = Runner(agent, args, writer)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        runner.gif(agent)
    else:
        if args.algorithm == 'PPO':
            runner.ppo_run()
        else:
            runner.run()
