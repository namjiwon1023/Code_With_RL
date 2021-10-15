import argparse
import torch as T
import gym
from gym.wrappers import RescaleAction

Learner_device = T.device('cuda:0')
Actor_device = T.device('cpu')

def sac_parameters():
    parser = argparse.ArgumentParser("Soft Actor-Critic")
    parser.add_argument("--algorithm", default='sac')

    parser.add_argument("--Learner_device", default=Learner_device)
    parser.add_argument("--Actor_device", default=Actor_device)

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--env-name", type=str, default="LunarLanderContinuous-v2")


    parser.add_argument("--time_steps", type=int, default=3000000, help="number of time steps")
    parser.add_argument("--max_steps", type=int, default=500000, help="number of worker max steps")
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--n_starts", type=int, default=10)
    parser.add_argument("--render", type=bool, default=True)

    parser.add_argument("--actor-lr", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=1e-3, help="learning rate of alpha")

    parser.add_argument("--min_log_std", type=int, default=-20)
    parser.add_argument("--max_log_std", type=int, default=2)

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")
    parser.add_argument("--tau", type=float, default=5e-3, help="soft update rate")

    parser.add_argument("--hidden_units", type=int, default=256)

    parser.add_argument("--buffer_size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=100, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--num_learners", type=int, default=1)
    parser.add_argument("--restore", type=bool, default=False)

    args = parser.parse_args()

    env = gym.make(args.env_name)
    env = RescaleAction(env, -1, 1)

    args.n_states = env.observation_space.shape[0]
    args.n_actions = env.action_space.shape[0]

    args.max_action = env.action_space.high[0]
    args.low_action = env.action_space.low[0]
    args.max_ep_len = env.spec.max_episode_steps

    return args
