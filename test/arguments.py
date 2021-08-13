# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def dqn_parameters():
    parser = argparse.ArgumentParser("Deep Q Network")
    parser.add_argument("--algorithm", default='dqn')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=True)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--render", type=bool, default=False)

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--update-rate", type=int, default=100, help="update rate")

    parser.add_argument("--max_epsilon", type=float, default=1.0, help="max epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0005, help="epsilon decay")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--buffer-size", type=int, default=1000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--algorithm_path", type=str, default="dqn.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def double_dqn_parameters():
    parser = argparse.ArgumentParser("Double Deep Q Network")
    parser.add_argument("--algorithm", default='ddqn')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=True)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--update-rate", type=int, default=100, help="update rate")

    parser.add_argument("--max_epsilon", type=float, default=1.0, help="max epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0005, help="epsilon decay")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--buffer-size", type=int, default=1000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--algorithm_path", type=str, default="ddqn.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def dueling_dqn_parameters():
    parser = argparse.ArgumentParser("Dueling Network With Deep Q Learning Algorithm")
    parser.add_argument("--algorithm", default='duelingdqn')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=True)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128, 128])

    parser.add_argument("--update-rate", type=int, default=100, help="update rate")

    parser.add_argument("--max_epsilon", type=float, default=1.0, help="max epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0005, help="epsilon decay")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--buffer-size", type=int, default=1000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--algorithm_path", type=str, default="duelingdqn.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def d3qn_parameters():
    parser = argparse.ArgumentParser("Dueling Double Deep Q Network")
    parser.add_argument("--algorithm", default='d3qn')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=True)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128, 128])

    parser.add_argument("--update-rate", type=int, default=100, help="update rate")

    parser.add_argument("--max_epsilon", type=float, default=1.0, help="max epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0005, help="epsilon decay")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--buffer-size", type=int, default=1000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=32, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--algorithm_path", type=str, default="d3qn.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")


    return parser.parse_args()

def noisy_dqn_parameters():
    parser = argparse.ArgumentParser("Noisy Deep Q Network")
    parser.add_argument("--algorithm", default='noisydqn')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=False)
    parser.add_argument("--is_discrete", type=bool, default=True)
    parser.add_argument("--use_noisy_layer", type=bool, default=True)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="CartPole-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--update-rate", type=int, default=150, help="update rate")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--buffer-size", type=int, default=10000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--algorithm_path", type=str, default="noisydqn.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def ddpg_parameters():
    parser = argparse.ArgumentParser("Deep Deterministic Policy Gradient")
    parser.add_argument("--algorithm", default='ddpg')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="Pendulum-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="epsilon decay")

    parser.add_argument("--Gaussian_noise", type=bool, default=False, help="use Gaussian noise")
    parser.add_argument("--exploration_noise", type=float, default=0.1, help="Gaussian noise exploration")

    parser.add_argument("--ou_noise_theta", type=float, default=1.0, help="OU noise theta")
    parser.add_argument("--ou_noise_sigma", type=float, default=0.1, help="OU noise sigma")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="parameter for updating the target network")

    parser.add_argument("--buffer-size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--file_actor", type=str, default="ddpg_actor.pth")
    parser.add_argument("--file_critic", type=str, default="ddpg_critic.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")


    return parser.parse_args()

def td3_parameters():
    parser = argparse.ArgumentParser("Twin Delayed Deep Deterministic Policy Gradients")
    parser.add_argument("--algorithm", default='td3')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="Pendulum-v0")
    parser.add_argument("--render", type=bool, default=False, help="")
    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")
    parser.add_argument("--episode", type=int, default=int(1e6), help="number of episode")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--policy_freq", type=int, default=2, help="update rate")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="epsilon decay")

    parser.add_argument("--exploration_noise", type=float, default=0.1, help="Gaussian noise exploration")
    parser.add_argument("--noise_clip", type=float, default=0.5, help="noise clip")
    parser.add_argument("--policy_noise", type=float, default=0.2, help="policy noise")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="parameter for updating the target network")

    parser.add_argument("--buffer-size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--file_actor", type=str, default="td3_actor.pth")
    parser.add_argument("--file_critic", type=str, default="td3_critic.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")


    return parser.parse_args()

def sac_parameters():
    parser = argparse.ArgumentParser("Soft Actor-Critic")
    parser.add_argument("--algorithm", default='sac')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--env-name", type=str, default="Pendulum-v0")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate of alpha")

    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="soft update rate")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="epsilon decay")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--buffer_size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--file_actor", type=str, default="sac_actor.pth")
    parser.add_argument("--file_critic", type=str, default="sac_critic.pth")

    return parser.parse_args()

def bc_sac_parameters():
    parser = argparse.ArgumentParser("Behavioral Cloning With Soft Actor-Critic Algorithm")
    parser.add_argument("--algorithm", default='bc')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=True)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=True)

    parser.add_argument("--env-name", type=str, default="Pendulum-v0")
    parser.add_argument("--render", type=bool, default=False, help="")
    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate of alpha")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="epsilon decay")

    # Behavior cloning parameters
    parser.add_argument("--lambda1", type=float, default=1e-3, help="lambda1")
    parser.add_argument("--lambda2", type=float, default=1.0, help="lambda2")
    parser.add_argument("--bc-batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="parameter for updating the target network")

    parser.add_argument("--buffer-size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--file_actor", type=str, default="bc_actor.pth")
    parser.add_argument("--file_critic", type=str, default="bc_critic.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def a2c_parameters():
    parser = argparse.ArgumentParser("Advantage Actor-Critic")

    parser.add_argument("--algorithm", default='a2c')
    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=False)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=False)

    parser.add_argument("--env-name", type=str, default="Pendulum-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--entropy_weight", type=float, default=1e-2, help="entropy weight")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--file_actor", type=str, default="a2c_actor.pth")
    parser.add_argument("--file_critic", type=str, default="a2c_critic.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def ppo_parameters():
    parser = argparse.ArgumentParser("Proximal policy optimization")
    parser.add_argument("--algorithm", default='ppo')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")

    parser.add_argument("--use_epsilon", type=bool, default=False)
    parser.add_argument("--is_discrete", type=bool, default=False)
    parser.add_argument("--use_noisy_layer", type=bool, default=False)
    parser.add_argument("--is_off_policy", type=bool, default=False)

    parser.add_argument("--env-name", type=str, default="Pendulum-v0")
    parser.add_argument("--render", type=bool, default=False, help="")

    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="learning rate of critic")

    parser.add_argument("--hidden_units", default=[128, 128])

    parser.add_argument("--value_weight", type=float, default=0.5, help="value_weight")
    parser.add_argument("--entropy_weight", type=float, default=0.01, help="could be 0.02")

    parser.add_argument("--epsilon", type=float, default=0.3, help="ratio.clamp(1 - clip, 1 + clip)")
    parser.add_argument("--tau", type=float, default=0.98, help="could be 0.95~0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)")

    parser.add_argument("--max_ep_len", type=int, default=1200, help="max epsilon length")
    parser.add_argument("--update_step", type=int, default=4800, help="max_ep_len * 4")

    parser.add_argument("--epoch", type=int, default=128, help="number of epoch")

    parser.add_argument("--use_clipped_value_loss", type=bool, default=False, help="")
    parser.add_argument("--standardize_advantage", type=bool, default=False, help="")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")

    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")

    parser.add_argument("--file_actor", type=str, default="ppo_actor.pth")
    parser.add_argument("--file_critic", type=str, default="ppo_critic.pth")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def maddpg_parameters():
    parser = argparse.ArgumentParser("Multi Agent Deep Deterministic Policy Gradient")

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--actor-lr", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden-size-1", type=int, default=64, help="hidden layer units")
    parser.add_argument("--hidden-size-2", type=int, default=64, help="hidden layer units")

    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")


    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def matd3_parameters():
    parser = argparse.ArgumentParser("Multi Agent Twin Delayed Deep Deterministic Policy Gradients")

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden-size-1", type=int, default=64, help="hidden layer units")
    parser.add_argument("--hidden-size-2", type=int, default=64, help="hidden layer units")

    parser.add_argument("--policy-noise", type=float, default=0.2, help="policy noise")
    parser.add_argument("--noise-clip", type=float, default=0.5, help="noise clip")
    parser.add_argument("--update-rate", type=int, default=2, help="update rate")

    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")


    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")


    return parser.parse_args()

def masac_parameters():
    parser = argparse.ArgumentParser("Multi Agent Soft Actor-Critic")

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument("--scenario-name", type=str, default="simple_tag", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=2000000, help="number of time steps")

    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate of alpha")

    parser.add_argument("--with-logprob", type=bool, default=True, help="Actor Network log_prob output")
    parser.add_argument("--update-rate", type=int, default=1, help="target network update rate")

    parser.add_argument("--hidden-size-1", type=int, default=64, help="hidden layer units")
    parser.add_argument("--hidden-size-2", type=int, default=64, help="hidden layer units")

    parser.add_argument("--min-log-std", type=int, default=-20, help="min log std")
    parser.add_argument("--max-log-std", type=int, default=2, help="max log std")

    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")


    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=100, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()

def slac_parameters():
    parser = argparse.ArgumentParser("Stochastic Latent Actor-Critic")

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

    return parser.parse_args()
