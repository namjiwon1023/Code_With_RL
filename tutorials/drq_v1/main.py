# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from agent import DrQAgent
from arguments import drq_parameters
from runner import Runner

if __name__ == '__main__':
    args = drq_parameters()
    writer = SummaryWriter('./logs/' + args.algorithm)
    agent = DrQAgent(args)
    runner = Runner(agent, args, writer)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
