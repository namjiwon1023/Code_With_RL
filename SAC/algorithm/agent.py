import copy
import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from gym.wrappers import RescaleAction
import os

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from utils.ReplayBuffer import ReplayBuffer

class TD3Agent(object):
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor_eval = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.args)

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

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/TD3_actor.pth'):
            self.load_models()

        self.total_step = 0

    def choose_action(self, state, epsilon):
        # if self.total_step < self.args.start_step and not self.args.evaluate:
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
        q1_loss = self.critic_eval.loss_func(current_q1, target_q)
        q2_loss = self.critic_eval.loss_func(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        for p in self.critic_eval.parameters():
                p.requires_grad = False

        if self.total_step % self.args.policy_freq == 0:
            # actor_loss = -(T.min(*self.critic_eval.get_double_q(state, self.actor_eval(state)))).mean()
            actor_loss = -self.critic_eval(state, self.actor_eval(state)).mean()

            self.actor_eval.optimizer.zero_grad()
            actor_loss.backward()
            self.actor_eval.optimizer.step()

            self._target_soft_update(self.actor_target, self.actor_eval, self.args.tau)
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)
        else:
            actor_loss = T.zeros(1)

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        return q1_loss.item(), q2_loss.item(), critic_loss.item(), actor_loss.item()

    def _target_soft_update(self, target_net, eval_net, tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                if self.args.evaluate:
                    self.env.render()
                # time.sleep(0.02)
                action = self.choose_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        # self.env.close()
        return reward_sum / n_starts

    def save_models(self):
        print('------ Save models ------')
        self.actor_eval.save_model()
        self.critic_eval.save_model()

    def load_models(self):
        print('------ load models ------')
        self.actor_eval.load_model()
        self.critic_eval.load_model()
