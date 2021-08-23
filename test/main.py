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
from test.utils import _random_seed
from test.arguments import dqn_parameters, double_dqn_parameters, dueling_dqn_parameters, d3qn_parameters, noisy_dqn_parameters
from test.arguments import ddpg_parameters, td3_parameters, sac_parameters, bc_sac_parameters, a2c_parameters, ppo_parameters
from test.runner import Runner

if __name__ == '__main__':
    algo = str(input('input algorithm name : '))

    if algo == 'dqn':
        args = dqn_parameters()
        agent = DQNAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'ddqn':
        args = double_dqn_parameters()
        agent = DoubleDQNAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'duelingdqn':
        args = dueling_dqn_parameters()
        agent = DuelingDQNAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'd3qn':
        args = d3qn_parameters()
        agent = D3QNAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'noisydqn':
        args = noisy_dqn_parameters()
        agent = NoisyDQNAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'ddpg':
        args = ddpg_parameters()
        agent = DDPGAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'td3':
        args = td3_parameters()
        agent = TD3Agent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'sac':
        args = sac_parameters()
        agent = SACAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'ppo':
        args = ppo_parameters()
        agent = PPOAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.ppo_run()

    if algo == 'a2c':
        args = a2c_parameters()
        agent = A2CAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

    if algo == 'bc':
        args = bc_sac_parameters()
        agent = BC_SACAgent(args)
        _random_seed(args.seed)
        writer = SummaryWriter('./logs/' + args.algorithm)
        runner = Runner(agent, args, agent.env, writer)
        if args.evaluate:
            returns = runner.evaluate()
            print('Average returns is', returns)
            runner.gif(agent, agent.env)
        else:
            runner.run()

