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
from torch.distributions import Normal

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork


class A2CAgent(object):
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        # self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.transition = list()

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic = CriticNetwork(self.n_states, self.args)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
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
        critic_loss = F.smooth_l1_loss(current_value, target_value)

        # update value
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # advantage = Q_t - V(s_t)
        advantage = (target_value - current_value).detach()  # not backpropagated
        actor_loss = -advantage * log_prob
        actor_loss += self.args.entropy_weight * -log_prob  # entropy maximization

        # update policy
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                if self.args.evaluate:
                    self.env.render()
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / n_starts

    def save_models(self):
        print('------ Save models ------')
        self.actor.save_model()
        self.critic.save_model()

    def load_models(self):
        print('------ load models ------')
        self.actor.load_model()
        self.critic.load_model()
