import numpy as np
import torch as T
import gym
import time
from gym.wrappers import RescaleAction
import copy

import ray
from sac import SACAgent


@ray.remote
class ReplayBuffer:
    def __init__(self, n_states, n_actions, args):

        self.states = np.zeros([args.buffer_size, n_states], dtype=np.float32)
        self.next_states = np.zeros([args.buffer_size, n_states], dtype=np.float32)
        self.actions = np.zeros([args.buffer_size, n_actions], dtype=np.float32)
        self.rewards = np.zeros([args.buffer_size], dtype=np.float32)
        self.masks = np.zeros([args.buffer_size], dtype=np.float32)

        self.max_size = args.buffer_size
        self.ptr, self.cur_len, = 0, 0

    def store(self, state, action, reward, next_state, mask):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

    def sample_batch(self, batch_size):
        index = np.random.choice(self.cur_len, batch_size, replace = False)

        return dict(
                    state = self.states[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_states[index],
                    mask = self.masks[index],
                    )

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True


@ray.remote
class ParameterServer:
    def __init__(self, weights):
        self.weights = weights.copy()

    def push(self, weights):
        self.weights = weights.copy()

    def pull(self):
        return self.weights

    def save_weights(self):
        dirPath = os.getcwd() + '/sac_model.pth'
        T.save(self.weights, dirPath)

    def load_weights(self):
        dirPath = os.getcwd() + '/sac_model.pth'
        self.weights = T.load(dirPath)


@ray.remote
def worker_rollout(ps, replay_buffer, args):
    env = gym.make(args.env)
    state = env.reset()
    score = 0
    done = False
    ep_len = 0

    agent = SACAgent(args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    for t in range(args.max_steps):

        action = agent.select_exploration_action(state)

        next_state, reward, done, _ = env.step(action)
        score += reward
        ep_len += 1

        real_done = False if ep_len == args.max_ep_len else done
        mask = 0.0 if real_done else agent.GAMMA

        replay_buffer.store.remote(state, action, reward, next_state, mask)

        state = next_state

        if real_done or (ep_len == args.max_ep_len):

            state = env.reset()
            score = 0
            done = False
            ep_len = 0

            weights = ray.get(ps.pull.remote())
            agent.set_weights(weights)


@ray.remote(num_gpus=1, max_calls=1)
def worker_train(ps, replay_buffer, writer, args):
    agent = SACAgent(args)

    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)

    cnt = 1
    while True:
        agent.total_steps += 1

        agent.learn(replay_buffer, writer)

        if cnt % 300 == 0:
            weights = agent.get_weights()
            ps.push.remote(weights)

        cnt += 1


@ray.remote
def worker_test(ps, writer):
    agent = SACAgent(args)
    weights = ray.get(ps.pull.remote())
    agent.set_weights(weights)
    test_env = gym.make(args.env)
    cnt = 1
    while True:
        cnt += 1
        ave_ret = agent._evaluate_agent(test_env, agent, args)
        if cnt % 1000 == 0:
            writer.add_scalar("reward", ave_ret, cnt)
        weights = ray.get(ps.pull.remote())
        agent.set_weights(weights)

if __name__ == '__main__':
    import argparse
    from torch.utils.tensorboard import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    args = parser.parse_args()

    args.algorithm = 'sac'
    args.hidden_units = 256
    args.target_update_interval = 1
    args.seed = 0
    args.buffer_size = int(1e6)
    args.GAMMA = 0.99,
    args.tau = 5e-3
    args.max_steps = 3000000

    args.min_log_std = -20
    args.max_log_std = 2

    args.actor_lr = 1e-3
    args.critic_lr = 1e-3
    args.alpha_lr = 1e-3
    args.batch_size = 100
    args.start_steps = 10000
    args.n_starts = 10
    args.render = True

    env = gym.make(args.env)
    env = RescaleAction(env, -1, 1)
    args.n_states = env.observation_space.shape[0]
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high[0]
    args.low_action = env.action_space.low[0]
    args.max_ep_len = env.spec.max_episode_steps

    args.Learner_device = T.device('cuda:0')
    args.Actor_device = T.device('cpu')

    args.num_workers = 6
    args.num_learners = 1

    writer = SummaryWriter('./logs/' + args.algorithm)

    ray.init()

    net = SACAgent(args)
    weights = net.get_weights()
    ps = ParameterServer.remote(weights)

    replay_buffer = ReplayBuffer.remote(args.n_states, args.n_actions, args)

    # start_time = time.time()

    # Start some training tasks.
    task_rollout = [worker_rollout.remote(ps, replay_buffer, args) for i in range(args.num_workers)]

    time.sleep(20)

    task_train = [worker_train.remote(ps, replay_buffer, writer, args) for i in range(args.num_learners)]

    time.sleep(10)

    task_test = worker_test.remote(ps, writer)
    ray.wait(task_rollout)