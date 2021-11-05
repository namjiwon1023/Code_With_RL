from torch.utils.tensorboard import SummaryWriter

from agent import SpikingSACAgent
from runner import Runner

from configparser import ConfigParser
from argparse import ArgumentParser

import gym
from gym.wrappers import RescaleAction

from utils import Dict, _random_seed

if __name__ == '__main__':

    parser = ArgumentParser('spikingsac parameters')
    parser.add_argument("--algorithm", type=str, default = 'spikingsac', help = 'algorithm to adjust (default : sac)')
    args = parser.parse_args()

    parser = ConfigParser()
    parser.read('config.ini')
    agent_param = Dict(parser, args.algorithm)

    writer = SummaryWriter('./logs/' + agent_param['algorithm'])

    env = gym.make(agent_param['env_name'])
    env = RescaleAction(env, -1, 1)

    test_env = gym.make(agent_param['env_name'])
    test_env = RescaleAction(test_env, -1, 1)

    agent_param['n_actions'] = env.action_space.shape[0]
    agent_param['n_states'] = env.observation_space.shape[0]
    agent_param['low_action'] = env.action_space.low[0]
    agent_param['max_action'] = env.action_space.high[0]
    agent_param['max_ep_step'] = env.spec.max_episode_steps

    agent = SpikingSACAgent(agent_param)

    _random_seed(env, test_env, agent_param['seed'])

    runner = Runner(agent, env, test_env, agent_param, writer)

    if agent_param['evaluate']:
        returns = runner.evaluate()
        print('Average returns is', returns)
    else:
        runner.run()