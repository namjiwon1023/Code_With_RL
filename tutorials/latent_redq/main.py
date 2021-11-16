# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
from torch.utils.tensorboard import SummaryWriter

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from algorithm import latentREDQAlgorithm
from arguments import get_args
from runner import Runner



if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('./logs')
    agent = latentREDQAlgorithm(args)
    runner = Runner(agent, args, writer)

    if args.evaluate:
        returns = runner._evaluate_agent(agent)
        print('Average returns is', returns)
    else:
        runner.run()