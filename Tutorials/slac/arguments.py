import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
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