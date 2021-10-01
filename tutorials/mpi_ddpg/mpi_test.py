import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction
import random
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter
import argparse
from utils import _store_expert_data

from datetime import timedelta
from time import sleep, time

import threading

def _make_gif(policy, env, args, maxsteps=1000):
    envname = env.spec.id
    gif_name = '_'.join([envname])
    state = env.reset()
    done = False
    steps = []
    rewards = []
    t = 0
    while (not done) & (t< maxsteps):
        s = env.render('rgb_array')
        steps.append(s)
        action = policy.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        rewards.append(reward)
        t +=1
    print('Final reward :', np.sum(rewards))
    clip = ImageSequenceClip(steps, fps=30)
    if not os.path.isdir('gifs'):
        os.makedirs('gifs')
    if not os.path.isdir('gifs' + '/' + args.algorithm):
        os.makedirs('gifs' + '/' + args.algorithm)
    clip.write_gif('gifs' + '/'  + args.algorithm + '/{}.gif'.format(gif_name), fps=30)

def _evaluate_agent(env, agent, args, n_starts=10):
    reward_sum = 0
    for _ in range(n_starts):
        done = False
        state = env.reset()
        while (not done):
            if args.evaluate:
                env.render()
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            state = next_state
    return reward_sum / n_starts

class ReplayBuffer:
    def __init__(self, n_states, n_actions, args, gpu_id, buffer_size=None):
        if buffer_size == None:
            buffer_size = args.buffer_size

        self.device = T.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.states = np.empty([buffer_size, n_states], dtype=np.float32)
        # self.states = T.empty([buffer_size, n_states], dtype=T.float32)
        self.next_states = np.empty([buffer_size, n_states], dtype=np.float32)
        self.actions = np.empty([buffer_size, n_actions],dtype=np.float32)
        self.rewards = np.empty([buffer_size], dtype=np.float32)
        self.masks = np.empty([buffer_size],dtype=np.float32)

        self.max_size = buffer_size
        self.ptr, self.cur_len, = 0, 0
        self.n_states = n_states
        self.n_actions = n_actions

        self.transitions = []
        self.lock = threading.Lock()

    def store(self, state, action, reward, next_state, mask):
        with self.lock:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.masks[self.ptr] = mask

            self.ptr = (self.ptr + 1) % self.max_size
            self.cur_len = min(self.cur_len + 1, self.max_size)

    def sample_batch(self, batch_size):
        with self.lock:
            index = np.random.choice(self.cur_len, batch_size, replace = False)

            return dict(state = self.states[index],
                        action = self.actions[index],
                        reward = self.rewards[index],
                        next_state = self.next_states[index],
                        mask = self.masks[index],
                        )

    def store_transition(self, transition):
        self.transitions.append(transition)
        np.save('bc_memo.npy', self.transitions)

    def store_for_BC_data(self, transitions):
        for t in transitions:
            self.store(*t)

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True


parameter_patgh = './icsl_rl/Hyperparameter/ddpg.yaml'   # Algorithms can be chosen by themselves
config = _read_yaml(parameter_patgh)
# print(config)

device = T.device('cpu')
# In the jupyter book, the argparse library is not easy to use, so use the following form instead
args = argparse.Namespace(algorithm='ddpg', device=device, evaluate=False)
args.__dict__ = config
args.device = device                # GPU or CPU
args.seed = 0                     # random seed setting
args.render = False                 # Visualization during training.
args.time_steps = 3000000           # total training step
args.episode = 3000000              # total episode
args.save_dir = "./model"           # Where to store the trained model
args.save_rate = 2000               # store rate
args.model_dir = ""                 # Where to store the trained model
args.evaluate_episodes = 10         # Parameters for Model Prediction
args.evaluate = False                # Parameters for Model Prediction
args.evaluate_rate = 1000           # Parameters for Model Prediction
args.is_store_transition = False    # Store expert data
args.env_name = 'Walker2d-v2'          # discrete env
args.hidden_sizes = (256, 256)
args.activation = nn.ReLU
# print(args)

def create_mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def reset_parameters(Sequential, std=1.0, bias_const=1e-6):
    for layer in Sequential:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, std)
            nn.init.constant_(layer.bias, bias_const)


class Actor(nn.Module): # Deterministic Policy Gradient(DPG), Deep Deterministic Policy Gradient(DDPG), Twin Delayed Deep Deterministic Policy Gradients(TD3)
    def __init__(self, n_states, n_actions, args, gpu_id, max_action=None):
        super(Actor, self).__init__()
        self.device = T.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.max_action = max_action

        self.pi = create_mlp([n_states] + list(args.hidden_sizes) + [n_actions], args.activation, nn.Tanh)

        reset_parameters(self.pi)

        self.to(self.device)

    def forward(self, state):
        u = self.pi(state)
        if self.max_action == None: return u
        return self.max_action*u


class CriticQ(nn.Module): # Action Value Function
    def __init__(self, n_states, n_actions, args, gpu_id):
        super(CriticQ, self).__init__()
        self.device = T.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.Value = create_mlp([n_states + n_actions] + list(args.hidden_sizes) + [1], args.activation)

        reset_parameters(self.Value)

        self.to(self.device)

    def forward(self, state, action):
        cat = T.cat((state, action), dim=-1)
        Q = self.Value(cat)
        return Q

class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = np.float64(0.0)
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))]
        )
        self.state = x + dx
        return self.state

class DDPGAgent:
    def __init__(self, env_fn, args, gpu_id):

        # setup_pytorch_for_mpi()

        self.args = args

        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_critic.pth')

        # Environment setting
        self.env = env_fn()
        self.env = RescaleAction(self.env, -1, 1)

        seed = args.seed
        seed += 10000 * MPI.COMM_WORLD.Get_rank()
        self.env.seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        T.manual_seed(seed)
        T.cuda.manual_seed(seed)

        self.episode_limit = self.env.spec.max_episode_steps

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # OU noise setting
        self.noise = OUNoise(self.n_actions, theta=self.args.ou_noise_theta, sigma=self.args.ou_noise_sigma,)

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args, self.args.buffer_size)
        self.transition = list()

        # actor-critic network setting
        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticQ(self.n_states, self.n_actions, self.args)

        # sync_params(self.actor_eval)
        # sync_params(self.critic_eval)
        sync_networks(self.actor_eval)
        sync_networks(self.critic_eval)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)
        # loss function
        self.criterion = nn.MSELoss()

        self.actor_target = Actor(self.n_states, self.n_actions, self.args)
        self.critic_target = CriticQ(self.n_states, self.n_actions, self.args)

        self.actor_target.load_state_dict(self.actor_eval.state_dict())
        self.critic_target.load_state_dict(self.critic_eval.state_dict())

        # Storage location creation
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            self.model_path = self.args.save_dir + '/' + args.algorithm
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            self.model_path = self.model_path + '/' + args.env_name
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

            if os.path.exists(self.model_path + '/ddpg_actor.pth'):
                self.load_models()

        self.total_step = 0
        self.learning_step = 0
        self.init_random_step = 10000

    def choose_action(self, state):
        with T.no_grad():
            if self.init_random_step > self.total_step and not self.args.evaluate:
                choose_action = self.env.action_space.sample()
            else :
                choose_action = self.actor_eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).detach().cpu().numpy()

            if not self.args.evaluate:
                noise = self.noise.sample()
                choose_action = np.clip(choose_action + noise, self.low_action, self.max_action)
            self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        self.learning_step += 1
        # TD error
        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # mpi_avg_grads(self.critic_eval)
        sync_grads(self.critic_eval)
        self.critic_optimizer.step()

        # actor network loss function
        actor_loss = self._policy_update(state)

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # mpi_avg_grads(self.actor_eval)
        sync_grads(self.actor_eval)
        self.actor_optimizer.step()

        # actor target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.actor_target, self.actor_eval, self.args.tau)
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)

        if self.learning_step % 1000 == 0:
            writer.add_scalar('loss/actor', actor_loss.detach().item(), self.learning_step)
            writer.add_scalar('loss/critic', critic_loss.detach().item(), self.learning_step)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor_eval, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor_eval, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)

    # target network soft update
    def _target_soft_update(self, target_net, eval_net, tau):
        for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)

            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_values = reward + next_value * mask

        eval_values = self.critic_eval(state, action)
        # TD error
        critic_loss = self.criterion(eval_values, target_values)

        return critic_loss, state

    def _policy_update(self, state):
        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()
        return actor_loss

# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))


class Runner:
    def __init__(self, agent, args, writer):
        self.args = args
        self.agent = agent
        self.episode_limit = self.agent.episode_limit
        self.writer = writer
        self.env = self.agent.env

    def run(self):
        self.start_time = time()
        best_score = self.env.reward_range[0]

        scores = []
        store_scores = []
        eval_rewards = []

        avg_score = 0

        for i in range(self.args.episode):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            done = False
            while (not done):
                if self.args.render:
                    self.env.render()
                cur_episode_steps += 1
                self.agent.total_step += 1
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                real_done = False if cur_episode_steps >= self.episode_limit else done
                mask = 0.0 if real_done else self.args.gamma
                self.agent.transition += [reward, next_state, mask]
                self.agent.memory.store(*self.agent.transition)
                state = next_state
                score += reward

                if self.agent.memory.ready(self.args.batch_size):
                    self.agent.learn(self.writer)

                if self.agent.total_step % self.args.evaluate_rate == 0 and self.agent.memory.ready(self.args.batch_size):
                    running_reward = np.mean(scores[-10:])
                    eval_reward = _evaluate_agent(self.env, self.agent, self.args, n_starts=self.args.evaluate_episodes)
                    eval_rewards.append(eval_reward)
                    self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                    if MPI.COMM_WORLD.Get_rank() == 0:
                        print('| Episode : {} | Step : {} | Train Score : {} | Predict Score : {} | Avg score : {} | Time : {} |'.format(i, self.agent.total_step, round(score, 2), round(eval_reward, 2), round(avg_score, 2), self.time))
                    scores = []

            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(store_scores[-10:])

            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/episode_return.txt', store_scores, delimiter=",")
            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/step_return.txt', eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

    def evaluate(self):
        returns = _evaluate_agent(self.env, self.agent, self.args, n_starts=1)

    def gif(self, policy, env, maxsteps=1000):
        _make_gif(policy, env, self.args, maxsteps)

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))


def launch(args):
    writer = SummaryWriter('./logs/' + args.env_name + '/' + args.algorithm)           # Tensorboard

    agent = DDPGAgent(lambda : gym.make(args.env_name), args)            # agent setting
    runner = Runner(agent, args, writer)
    runner.run()                  # Training

if __name__ == '__main__':
    # cpu = 6
    # mpi_fork(cpu)  # run parallel code with mpi
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['IN_MPI'] = '1'

    launch(args)
