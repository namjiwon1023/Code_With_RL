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
from utils.utils import ppo_iter, compute_gae
from utils.ReplayBuffer import ReplayBuffer

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork

class PPOAgent(object):
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        # self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic = CriticNetwork(self.n_states, self.args)

        self.optimizer = optim.Adam([{'params': self.actor.parameters(), 'lr': self.args.actor_lr},
                                    {'params': self.critic.parameters(), 'lr': self.args.critic_lr}])

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/ppo_actor.pth'):
            self.load_models()

        self.total_step = 0

        self.memory = ReplayBuffer()

    def choose_action(self, state):
        state = T.as_tensor((state,), dtype=T.float32, device=self.args.device)
        # print('state :', state)
        # print('state shape:', state.shape)   # state shape: torch.Size([1, 3])
        mu, std = self.actor(state)

        if self.args.evaluate and not self.args.is_discrete:
            choose_action = mu
            # choose_action = choose_action

        if not self.args.evaluate:
            value = self.critic(state)
            # print('value shape :', value.shape)   # value shape : torch.Size([1, 1])
            self.memory.values.append(value)
            self.memory.states.append(state)
            # print('state array :', self.memory.states)
            # print('choose action shape :', choose_action.shape) # choose action shape : torch.Size([1, 1])
            dist = Normal(mu, std)
            choose_action = dist.sample()
            self.memory.actions.append(choose_action)
            # print('log prob :', dist.log_prob(choose_action).shape) # log prob : torch.Size([1, 1])
            self.memory.log_probs.append(dist.log_prob(choose_action))
            # print('action :', choose_action.detach().cpu().numpy()[0]) # action : [0.01113842]

        return choose_action.detach().cpu().numpy()[0]

    def learn(self, next_state):
        next_state = T.as_tensor((next_state,), dtype=T.float32, device=self.args.device)
        # print('next_state :', next_state.shape)
        next_value = self.critic(next_state)
        # print('next_value :', next_value.shape)

        returns = compute_gae(next_value, self.memory.rewards, self.memory.masks, self.memory.values, self.args.gamma, self.args.tau)

        states = T.cat(self.memory.states)
        # print('cat states :', states.shape)
        actions = T.cat(self.memory.actions)
        # print('cat actions :', actions.shape)
        returns = T.cat(returns).detach()
        # print('cat returns :', returns.shape)
        values = T.cat(self.memory.values).detach()
        # print('cat values :', values.shape)
        log_probs = T.cat(self.memory.log_probs).detach()
        # print('cat log_probs :', log_probs.shape)

        # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
        # Returns = R + yV(s')
        advantages = returns - values

        if self.args.is_discrete:
            actions = actions.unsqueeze(1)
            log_probs = log_probs.unsqueeze(1)

        # Normalize the advantages
        if self.args.standardize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # actor_losses, critic_losses, total_losses = [], [], []

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
                critic_loss = (return_ - value).pow(2).mean()

            total_loss = self.args.value_weight * critic_loss + actor_loss - entropy * self.args.entropy_weight

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # actor_losses.append(actor_loss.item())
            # critic_losses.append(critic_loss.item())
            # total_losses.append(total_loss.item())

        self.memory.RB_clear()

        # actor_loss = sum(actor_losses) / len(actor_losses)
        # critic_loss = sum(critic_losses) / len(critic_losses)
        # total_loss = sum(total_losses) / len(total_losses)

        # return actor_loss, critic_loss, total_loss
        # return total_loss

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

