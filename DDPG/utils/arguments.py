import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser("Deep Q Network")

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=123, help="random seed")

    parser.add_argument("--env-name", type=str, default="Pendulum-v0", help="name of the scenario script")
    parser.add_argument("--render", type=bool, default=False, help="")
    parser.add_argument("--time-steps", type=int, default=3000000, help="number of time steps")
    parser.add_argument("--episode", type=int, default=int(1e6), help="number of episode")
    # parser.add_argument("--start_step", type=int, default=int(1e4), help="number of start")

    parser.add_argument("--actor-lr", type=float, default=1e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")

    parser.add_argument("--hidden-size", type=int, default=128, help="hidden layer units")

    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")

    parser.add_argument("--epsilon", type=float, default=1.0, help="epsilon")
    parser.add_argument("--min_epsilon", type=float, default=0.1, help="min epsilon")
    parser.add_argument("--epsilon_decay", type=float, default=0.0001, help="epsilon decay")

    parser.add_argument("--Gaussian_noise", type=bool, default=True, help="use Gaussian noise")
    parser.add_argument("--exploration_noise", type=float, default=0.1, help="Gaussian noise exploration")

    parser.add_argument("--ou_noise_theta", type=float, default=1.0, help="OU noise theta")
    parser.add_argument("--ou_noise_sigma", type=float, default=0.1, help="OU noise sigma")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=100000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=2000, help="save model once every time this many episodes are completed")
    parser.add_argument("--model-dir", type=str, default="", help="directory in which training state and model are loaded")


    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate", type=bool, default=False, help="whether to evaluate the model")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")

    return parser.parse_args()
