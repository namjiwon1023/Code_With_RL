import argparse
import torch as T

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

def drq_parameters():
    parser = argparse.ArgumentParser("Data regularized Q V2")
    parser.add_argument("--algorithm", default='drq_v2')

    parser.add_argument("--device", default=device, help="GPU or CPU")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--env-name", type=str, default="quadruped_walk")

    parser.add_argument("--action_repeat", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--image_pad", type=int, default=4)
    parser.add_argument("--frame_stack", type=int, default=3)
    parser.add_argument("--init_temperature", type=float, default=0.1)


    parser.add_argument("--time-steps", type=int, default=1000000, help="number of time steps")
    parser.add_argument("--episode", type=int, default=int(1e6), help="number of time steps")

    parser.add_argument("--actor-lr", type=float, default=1e-3, help="learning rate of actor")
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="learning rate of critic")
    parser.add_argument("--encoder-lr", type=float, default=1e-3, help="learning rate of encoder")
    parser.add_argument("--alpha-lr", type=float, default=1e-3, help="learning rate of alpha")

    parser.add_argument("--update_frequency", type=int, default=2, help="update rate")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="soft update rate")

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--feature_dim", type=int, default=50)

    parser.add_argument("--log_std_min", type=int, default=-10)
    parser.add_argument("--log_std_max", type=int, default=2)

    parser.add_argument("--init_random_steps", type=int, default=5000)

    parser.add_argument("--buffer_size", type=int, default=500000, help="number of transitions can be stored in buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="number of episodes to optimize at the same time")

    parser.add_argument("--evaluate-episodes", type=int, default=10, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-rate", type=int, default=5000, help="how often to evaluate model")
    parser.add_argument("--evaluate", type=bool, default=False)

    parser.add_argument("--save-dir", type=str, default="./model", help="directory in which training state and model should be saved")

    parser.add_argument("--file_actor", type=str, default="drq_actor.pth")
    parser.add_argument("--file_critic", type=str, default="drq_critic.pth")
    parser.add_argument("--file_encoder", type=str, default="drq_encoder.pth")

    return parser.parse_args()