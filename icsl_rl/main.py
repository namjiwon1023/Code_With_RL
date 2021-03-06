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

from icsl_rl.agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, NoisyDQNAgent
from icsl_rl.agent import DDPGAgent, TD3Agent, SACAgent, PPOAgent, A2CAgent, BC_SACAgent
from icsl_rl.arguments import get_args
from icsl_rl.runner import Runner

# Random Seed Settings
def _random_seed(seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True

    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    print('Using GPU : ', T.cuda.is_available() , ' |  Seed : ', seed)

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

    runner = Runner(agent, args, agent.env, writer)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        runner.gif(agent, agent.env)
    else:
        if args.algorithm == 'PPO':
            runner.ppo_run()
        else:
            runner.run()
