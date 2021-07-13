import torch as T
import torch.optim as optim
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import QNetwork, DuelingNetwork
from network import Actor, ActorA2C, ActorPPO, ActorSAC, CriticQ, CriticV, CriticTwin
from replaybuffer import ReplayBuffer, ReplayBufferPPO

from utils import _target_net_update, _target_soft_update, compute_gae, ppo_iter, _load_model, _save_model
from utils import mse_loss, huber_loss
from utils import OUNoise

class DQNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DQN.pth')

        self.eval = QNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DQN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.randint(0, self.n_actions)
            else :
                choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
                choose_action = choose_action.detach().cpu().numpy()

            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)

    def save_models(self):
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        _load_model(self.eval, self.checkpoint)

class DoubleDQNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DoubleDQN.pth')

        self.eval = QNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DoubleDQN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.randint(0, self.n_actions)
            else :
                choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
                choose_action = choose_action.detach().cpu().numpy()

            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)

    def save_models(self):
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        _load_model(self.eval, self.checkpoint)

class DuelingDQNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'DuelingDQN.pth')

        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/DuelingDQN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.randint(0, self.n_actions)
            else :
                choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
                choose_action = choose_action.detach().cpu().numpy()

            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)

    def save_models(self):
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        _load_model(self.eval, self.checkpoint)

class D3QNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'D3QN.pth')

        self.eval = DuelingNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/D3QN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.randint(0, self.n_actions)
            else :
                choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
                choose_action = choose_action.detach().cpu().numpy()

            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).gather(1, self.eval(next_state).argmax(dim = 1, keepdim = True))
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)


    def save_models(self):
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        _load_model(self.eval, self.checkpoint)

class NoisyDQNAgent(object):
    def __init__(self, args):

        self.args = args

        self.env = gym.make(args.env_name)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.n

        self.checkpoint = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'NoisyDQN.pth')

        self.eval = QNetwork(self.n_states, self.n_actions, args)

        self.target = copy.deepcopy(self.eval)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = optim.Adam(self.eval.parameters(), lr=self.args.critic_lr)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.transition = list()

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/NoisyDQN.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state):
        with T.no_grad():
            choose_action = self.eval(T.as_tensor(state, dtype=T.float32, device=self.args.device)).argmax()
            choose_action = choose_action.detach().cpu().numpy()
            if not self.args.evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_q = self.target(next_state).max(dim=1, keepdim=True)[0]
            target_q = reward + next_q * mask

        curr_q = self.eval(state).gather(1, action)

        loss = (target_q - curr_q)**2
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.eval.reset_noise()
        self.target.reset_noise()

        if self.total_step % self.args.update_rate == 0:
            _target_net_update(self.eval, self.target)

    def save_models(self):
        _save_model(self.eval, self.checkpoint)

    def load_models(self):
        _load_model(self.eval, self.checkpoint)

class DDPGAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ddpg_critic.pth')

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.noise = OUNoise(self.n_actions, theta=self.args.ou_noise_theta, sigma=self.args.ou_noise_sigma,)

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticQ(self.n_states, self.n_actions, self.args)

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.actor_target = copy.deepcopy(self.actor_eval)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

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

    def choose_action(self, state, epsilon):
        if epsilon >= np.random.random() and not self.args.evaluate:
            choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
        else :
            choose_action = self.actor_eval(T.as_tensor(state, device=self.actor_eval.device, dtype=T.float32)).detach().cpu().numpy()
        if not self.args.evaluate:
            if self.args.Gaussian_noise:
                noise = np.random.normal(0, self.max_action*self.args.exploration_noise, size=self.n_actions)
            else:
                noise = self.noise.sample()
            choose_action = np.clip(choose_action + noise, -1, 1)
        self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.long, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action = self.actor_target(next_state)
            next_value = self.critic_target(next_state, next_action)
            target_values = reward + next_value * mask

        eval_values = self.critic_eval(state, action)
        critic_loss = mse_loss(eval_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic_eval.parameters():
                p.requires_grad = False

        actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for p in self.critic_eval.parameters():
                p.requires_grad = True

        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.actor_eval, self.actor_target, self.args)
            _target_soft_update(self.critic_eval, self.critic_target, self.args)

    def save_models(self):
        _save_model(self.actor_eval, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)

    def load_models(self):
        _load_model(self.actor_eval, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)

class TD3Agent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'td3_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'td3_critic.pth')

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor_eval = Actor(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        self.actor_optimizer = optim.Adam(self.actor_eval.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.actor_target = copy.deepcopy(self.actor_eval)
        self.actor_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad = False

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/td3_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        if epsilon >= np.random.random() and not self.args.evaluate:
            choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
        else :
            choose_action = self.actor_eval(T.as_tensor(state, device=self.actor_eval.device, dtype=T.float32)).detach().cpu().numpy()
        if not self.args.evaluate:
            noise = np.random.normal(0, self.max_action*self.args.exploration_noise, size=self.n_actions)
            choose_action = np.clip(choose_action + noise, self.low_action, self.max_action)
        self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            noise = (T.randn_like(action) * self.args.policy_noise * self.max_action).clamp(self.args.noise_clip * self.low_action, self.args.noise_clip * self.max_action)
            next_action = (self.actor_target(next_state) + noise).clamp(self.low_action, self.max_action)

            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + next_target_q * mask

        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        q1_loss = mse_loss(current_q1, target_q)
        q2_loss = mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        if self.total_step % self.args.policy_freq == 0:
            actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            _target_soft_update(self.actor_eval, self.actor_target, self.args)
            _target_soft_update(self.critic_eval, self.critic_target, self.args)

        for p in self.critic_eval.parameters():
            p.requires_grad = True

    def save_models(self):
        _save_model(self.actor_eval, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)

    def load_models(self):
        _load_model(self.actor_eval, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)

class SACAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_critic.pth')

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/sac_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        q1_loss = mse_loss(current_q1, target_q)
        q2_loss = mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic_eval.get_double_q(state, new_action)
        q = T.min(q_1, q_2)
        actor_loss = (self.alpha * new_log_prob - q).mean()
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_eval, self.critic_target, self.args)

    def save_models(self):
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

class PPOAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ppo_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'ppo_critic.pth')

        self.env = gym.make(args.env_name)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.actor = ActorPPO(self.n_states, self.n_actions, self.args)
        self.critic = CriticV(self.n_states, self.args)

        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.args.actor_lr},
                                    {'params': self.critic.parameters(), 'lr': self.args.critic_lr}])

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/ppo_actor.pth'):
            self.load_models()

        self.total_step = 0

        self.memory = ReplayBufferPPO()

    def choose_action(self, state):
        state = T.as_tensor((state,), dtype=T.float32, device=self.args.device)
        mu, std = self.actor(state)
        if self.args.evaluate and not self.args.is_discrete:
            choose_action = mu
        if not self.args.evaluate:
            value = self.critic(state)
            self.memory.values.append(value)
            self.memory.states.append(state)
            dist = Normal(mu, std)
            choose_action = dist.sample()
            self.memory.actions.append(choose_action)
            self.memory.log_probs.append(dist.log_prob(choose_action))
        return choose_action.detach().cpu().numpy()[0]

    def learn(self, next_state):
        next_state = T.as_tensor((next_state,), dtype=T.float32, device=self.args.device)
        next_value = self.critic(next_state)

        returns = compute_gae(next_value, self.memory.rewards, self.memory.masks, self.memory.values, self.args.gamma, self.args.tau)

        states = T.cat(self.memory.states)
        actions = T.cat(self.memory.actions)
        returns = T.cat(returns).detach()
        values = T.cat(self.memory.values).detach()
        log_probs = T.cat(self.memory.log_probs).detach()

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages = returns - values

        if self.args.is_discrete:
            actions = actions.unsqueeze(1)
            log_probs = log_probs.unsqueeze(1)

        # Normalize the advantages
        if self.args.standardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(epoch = self.args.epoch,
                                                                            mini_batch_size = self.args.batch_size,
                                                                            states = states,
                                                                            actions = actions,
                                                                            values = values,
                                                                            log_probs = log_probs,
                                                                            returns = returns,
                                                                            advantages = advantages,
                                                                            ):
            mu, std = self.actor(state)
            dist = Normal(mu, std)

            entropy = dist.entropy().mean()
            log_prob = dist.log_prob(action)

            ratio = (log_prob - old_log_prob).exp()

            surr1 = ratio * adv
            surr2 = T.clamp(ratio, 1.0 - self.args.epsilon, 1.0 + self.args.epsilon) * adv

            actor_loss  = - T.min(surr1, surr2).mean()

            value = self.critic(state)

            if self.args.use_clipped_value_loss:
                value_pred_clipped = old_value + T.clamp((value - old_value), - self.args.epsilon, self.args.epsilon)
                value_loss_clipped = (return_ - value_pred_clipped).pow(2)
                value_loss = (return_ - value).pow(2)
                critic_loss = T.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = mse_loss(value, return_)

            total_loss = self.args.value_weight * critic_loss + actor_loss - entropy * self.args.entropy_weight

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        self.memory.RB_clear()


    def save_models(self):
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)

    def load_models(self):
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)

class A2CAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'a2c_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'a2c_critic.pth')

        self.env = gym.make(args.env_name)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.transition = list()

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic = CriticNetwork(self.n_states, self.args)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/a2c_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state):
        state = T.as_tensor(state, dtype=T.float32, device=self.args.device)
        mu, std = self.actor(state)
        if self.args.evaluate:
            choose_action = mu
        else:
            dist = Normal(mu, std)
            choose_action = dist.sample()
            log_prob = dist.log_prob(choose_action).sum(dim=-1)
            self.transition = [state, log_prob]
        return choose_action.clamp(self.low_action, self.max_action).detach().cpu().numpy()

    def learn(self):
        state, log_prob, next_state, reward, mask = self.transition

        # Q_t   = r + gamma * V(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        next_state = T.as_tensor(next_state, dtype=T.float32, device=self.args.device)
        current_value = self.critic(state)
        next_value = self.critic(next_state).detach()
        target_value = reward + next_value * mask
        critic_loss = huber_loss(current_value, target_value)

        # update value
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (target_value - current_value).detach()  # not backpropagated
        actor_loss = -advantage * log_prob
        actor_loss += self.args.entropy_weight * -log_prob  # entropy maximization

        # update policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_models(self):
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)

    def load_models(self):
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)

class BC_SACAgent(object):
    def __init__(self, args):
        self.args = args
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_actor.pth')
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, 'sac_critic.pth')

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticTwin(self.n_states, self.n_actions, self.args)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_eval.parameters(), lr=self.args.critic_lr)

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/sac_actor.pth'):
            self.load_models()

        self.total_step = 0

        data_path = os.path.join('','bc_memo.npy')
        self.lambda1 = self.args.lambda1
        self.lambda2 = self.args.lambda2 / self.args.bc_batch_size
        self.memo = np.load(data_path, allow_pickle = True)

        self.bc_data = ReplayBuffer(self.n_states, self.n_actions, self.args, len(self.memo))
        self.bc_data.store_for_BC_data(self.memo)

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            samples_bc = self.bc_data.sample_batch(self.args.bc_batch_size)

            state_bc = T.as_tensor(samples_bc['state'], dtype=T.float32, device=self.args.device)
            next_state_bc = T.as_tensor(samples_bc['next_state'], dtype=T.float32, device=self.args.device)
            action_bc = T.as_tensor(samples_bc['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward_bc = T.as_tensor(samples_bc['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask_bc = T.as_tensor(samples_bc['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        current_q1, current_q2 = self.critic_eval.get_double_q(state, action)
        q1_loss = mse_loss(current_q1, target_q)
        q2_loss = mse_loss(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic_eval.get_double_q(state, new_action)
        q = T.min(q_1, q_2)
        pg_loss = (self.alpha * new_log_prob - q).mean()
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        pred_action, _ = self.actor(state_bc)
        q_t = T.min(*self.critic_eval.get_double_q(state_bc, action_bc))
        q_e = T.min(*self.critic_eval.get_double_q(state_bc, pred_action))
        qf_mask = T.gt(q_t, q_e).to(self.critic_eval.device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = T.zeros(1, device=self.args.device)
        else:
            bc_loss = (
                T.mul(pred_action, qf_mask) - T.mul(action_bc, qf_mask)
            ).pow(2).sum() / n_qf_mask

        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_eval, self.critic_target, self.args)

    def save_models(self):
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic_eval, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)