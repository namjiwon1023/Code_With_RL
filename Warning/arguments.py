import yaml
import argparse
import torch as T
from utils import _read_yaml

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser("Deep Reinforcement Learning Algorithm HyperParameters")

    parser.add_argument("-algo", "--algorithm", type=str, default="")
    parser.add_argument("-eval", "--evaluate", type=bool, default=False)
    args = parser.parse_args()
    if args.evaluate:
        if args.algorithm == 'dqn':
            params = './Hyperparameter/dqn.yaml'

        if args.algorithm == 'ddqn':
            params = './Hyperparameter/doubledqn.yaml'

        if args.algorithm == 'd3qn':
            params = './Hyperparameter/d3qn.yaml'

        if args.algorithm == 'duelingdqn':
            params = './Hyperparameter/duelingdqn.yaml'

        if args.algorithm == 'noisydqn':
            params = './Hyperparameter/noisydqn.yaml'

        cfg = _read_yaml(params)
        args.__dict__ = cfg

        args.device = device                # GPU or CPU
        args.seed = 123                     # random seed setting
        args.render = False                 # Visualization during training.
        args.time_steps = 3000000           # total training step
        args.episode = 1000000              # total episode
        args.save_dir = "./model"           # Where to store the trained model
        args.save_rate = 2000               # store rate
        args.model_dir = ""                 # Where to store the trained model
        args.evaluate_episodes = 10         # Parameters for Model Prediction
        args.evaluate = True               # Parameters for Model Prediction
        args.evaluate_rate = 1000           # Parameters for Model Prediction

        if args.is_discrete:
            args.env_name = 'CartPole-v0'
        else:
            args.env_name = 'Pendulum-v0'
    else:
        if args.algorithm == 'dqn':
            params = './Hyperparameter/dqn.yaml'

        if args.algorithm == 'ddqn':
            params = './Hyperparameter/doubledqn.yaml'

        if args.algorithm == 'd3qn':
            params = './Hyperparameter/d3qn.yaml'

        if args.algorithm == 'duelingdqn':
            params = './Hyperparameter/duelingdqn.yaml'

        if args.algorithm == 'noisydqn':
            params = './Hyperparameter/noisydqn.yaml'

        cfg = _read_yaml(params)
        args.__dict__ = cfg

        args.device = device                # GPU or CPU
        args.seed = 123                     # random seed setting
        args.render = False                 # Visualization during training.
        args.time_steps = 3000000           # total training step
        args.episode = 1000000              # total episode
        args.save_dir = "./model"           # Where to store the trained model
        args.save_rate = 2000               # store rate
        args.model_dir = ""                 # Where to store the trained model
        args.evaluate_episodes = 10         # Parameters for Model Prediction
        args.evaluate = False               # Parameters for Model Prediction
        args.evaluate_rate = 1000           # Parameters for Model Prediction

        if args.is_discrete:
            args.env_name = 'CartPole-v0'
        else:
            args.env_name = 'Pendulum-v0'

    return args