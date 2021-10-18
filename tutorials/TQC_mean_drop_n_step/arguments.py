import argparse
import torch as T

device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

def tqc_parameters():
    parser = argparse.ArgumentParser("Truncated Quantile Critics")
    parser.add_argument("--algorithm", default='tqc')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--env-name", type=str, default="Walker2d-v2")

    parser.add_argument("--time-steps", type=int, default=int(1e6), help="number of time steps")
    parser.add_argument("--episode", type=int, default=int(1e6), help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=3e-4, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=3e-4, help="learning rate of critic")
    parser.add_argument("--alpha-lr", type=float, default=3e-4, help="learning rate of alpha")

    parser.add_argument("--target_update_interval", type=int, default=1, help="update rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=5e-3, help="soft update rate")

    parser.add_argument("--ac_hidden_size", type=int, default=256)
    parser.add_argument("--cri_hidden_size", type=int, default=512)

    parser.add_argument("--n_nets", type=int, default=5)
    parser.add_argument("--top_quantiles_to_drop_per_net", type=int, default=2)
    parser.add_argument("--n_quantiles", type=int, default=25)

    parser.add_argument("--n_steps", type=int, default=3)

    parser.add_argument("--log_std_min", type=int, default=-20)
    parser.add_argument("--log_std_max", type=int, default=2)

    parser.add_argument("--init_random_steps", type=int, default=10000)

    parser.add_argument("--buffer_size", type=int, default=1000000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-rate", type=int, default=1000, help="how often to evaluate model")
    parser.add_argument("--evaluate", type=bool, default=False)

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--file_actor", type=str, default="tqc_actor.pth")
    parser.add_argument("--file_critic", type=str, default="tqc_critic.pth")

    return parser.parse_args()