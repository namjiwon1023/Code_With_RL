from torch.utils.tensorboard import SummaryWriter

from agent import DQNAgent, DoubleDQNAgent, DuelingDQNAgent, D3QNAgent, NoisyDQNAgent
from agent import DDPGAgent, TD3Agent, SACAgent, PPOAgent, A2CAgent, BC_SACAgent

from utils import _random_seed
from arguments import get_args
from runner import Runner

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter('./logs/' + args.algorithm)
    _random_seed(args.seed)

    if args.algorithm == 'DQN':
        agent = DQNAgent(args)

    if args.algorithm == 'Double_DQN':
        agent = DoubleDQNAgent(args)

    if args.algorithm == 'Dueling_DQN':
        agent = DuelingDQNAgent(args)

    if args.algorithm == 'D3QN':
        agent = D3QNAgent(args)

    if args.algorithm == 'Noisy_DQN':
        agent = NoisyDQNAgent(args)

    if args.algorithm == 'DDPG':
        agent = DDPGAgent(args)

    if args.algorithm == 'TD3':
        agent = TD3Agent(args)

    if args.algorithm == 'SAC':
        agent = SACAgent(args)

    if args.algorithm == 'PPO':
        agent = PPOAgent(args)

    if args.algorithm == 'A2C':
        agent = A2CAgent(args)

    if args.algorithm == 'BC_SAC':
        agent = BC_SACAgent(args)

    runner = Runner(agent, args, agent.env, writer)

    if args.evaluate:
        returns = runner.evaluate()
        print('Average returns is', returns)
        runner.gif(agent, agent.env)
    else:
        runner.run()