# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser("Stochastic Latent Actor-Critic")

    parser.add_argument("--algorithm", type=str, default="latent_redq")
    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--env_name", type=str, default="cheetah", help="env name")
    parser.add_argument("--task_name", type=str, default="run")

    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate of alpha")
    parser.add_argument("--latent-lr", type=float, default=1e-4, help="learning rate of latent model")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="soft update rate")

    parser.add_argument("--z1_dim", type=int, default=32, help="latent z1 dim")
    parser.add_argument("--z2_dim", type=int, default=256, help="latent z1 dim")
    parser.add_argument("--feature_dim", type=int, default=256, help="encoder feature dim")
    parser.add_argument("--num_sequences", type=int, default=8, help="number of sequences")
    parser.add_argument("--hidden_units", default=(256, 256), help="number of hidden units")

    parser.add_argument("--policy_update_delay", type=int, default=20, help="update rate")
    parser.add_argument("--target_entropy", default='auto', help="If you select ‘auto’, it is the same as SAC, if you select ‘mbpo’, the ‘target_entropy’ of different environments will have corresponding values")
    parser.add_argument("--delay_update_steps", default='auto', help="auto or delay_update_steps number")
    parser.add_argument("--q_target_mode", default='min', help="How to calculate target_Q(min, ave, rem)")
    parser.add_argument("--utd_ratio", type=int, default=20, help="update to data ratio")
    parser.add_argument("--num_Q", type=int, default=10, help="How many Q networks are there")
    parser.add_argument("--num_min", type=int, default=2, help="How many Q values to choose for calculation")

    parser.add_argument("--buffer_size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size_sac", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--batch_size_latent", type=int, default=32, help="number of episodes to optimize at the same time")

    parser.add_argument("--initial_collection_steps", type=int, default=10000, help="initial collection steps")
    parser.add_argument("--initial_learning_steps", type=int, default=100000, help="initial learning steps")
    parser.add_argument("--action_repeat", type=int, default=4, help="action repeat")

    parser.add_argument("--evaluate-episodes", type=int, default=5, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-rate", type=int, default=10000, help="how often to evaluate model")
    parser.add_argument("--evaluate", type=bool, default=False, help="Test?")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--file_critic", type=str, default="REDQ_critic.pth")

    return parser.parse_args()