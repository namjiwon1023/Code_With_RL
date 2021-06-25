from utils.runner import Runner
from utils.arguments import get_args
from utils.make_env import make_env
from utils.utils import _random_seed
import numpy as np
import random
import torch as T


if __name__ == '__main__':
    args = get_args()
    _random_seed(args.seed)
    env, args = make_env(args)
    runner = Runner(args, env)
    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()
