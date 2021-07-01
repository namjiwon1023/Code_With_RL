import random
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import gym
from gym.wrappers import RescaleAction
import os
from utils.utils import ppo_iter, compute_gae

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork

class PPOAgent(object):
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic = CriticNetwork(self.n_states, self.args)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/ppo_actor.pth'):
            self.load_models()

        self.total_step = 0

        self.log_probs = []
        self.values    = []
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.masks     = []

    def choose_action(self, state):
        with T.no_grad():
            state = T.as_tensor(state, dtype=T.float32, device=self.args.device)
            action, dist = self.actor(state)
            if self.args.evaluate:
                choose_action = dist.mean()
            else:
                choose_action = action
                value = self.critic(state)
                self.values.append(value)
                self.states.append(state)
                self.actions.append(choose_action)
                self.log_probs.append(dist.log_prob(choose_action))
        return choose_action.detach().cpu().numpy()

    def learn(self, next_state):
        next_state = T.as_tensor(next_state, dtype=T.float32, device=self.args.device)
        next_value = self.critic(next_state)
        returns = compute_gae(next_value, self.rewards, self.masks, self.values, self.args.gamma, self.args.tau)
        states = T.cat(self.states)
        actions = T.cat(self.actions)
        returns = T.cat(returns).detach()
        values = T.cat(self.values).detach()
        log_probs = T.cat(self.log_probs).detach()
        advantages = returns - values
        actor_losses, critic_losses, total_losses = [], [], []

        for state, action, old_value, old_log_prob, return_, adv in ppo_iter(epoch = self.args.epoch,
                                                                            mini_batch_size = self.args.batch_size,
                                                                            states = states,
                                                                            actions = actions,
                                                                            values = values,
                                                                            log_probs = log_probs,
                                                                            returns = returns,
                                                                            advantages = advantages,
                                                                            ):
            _, dist = self.actor(state)
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
                critic_loss = 0.5 * T.max(value_loss, value_loss_clipped).mean()
            else:
                critic_loss = 0.5 * (return_ - value).pow(2).mean()

            critic_loss_ = self.args.value_weight * critic_loss
            actor_loss_ = actor_loss - self.args.entropy_weight * entropy
            total_loss = critic_loss_ + actor_loss_

            self.critic.optimizer.zero_grad()
            critic_loss_.backward(retain_graph=True)
            clip_grad_norm_(self.critic.parameters(), self.args.critic_clip)
            self.critic.optimizer.step()

            self.actor.optimizer.zero_grad()
            actor_loss_.backward()
            clip_grad_norm_(self.actor.parameters(), self.args.actor_clip)
            self.actor.optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            total_losses.append(total_loss.item())

        self.states, self.actions, self.rewards = [], [], []
        self.values, self.masks, self.log_probs = [], [], []

        actor_loss = sum(actor_losses) / len(actor_losses)
        critic_loss = sum(critic_losses) / len(critic_losses)
        total_loss = sum(total_losses) / len(total_losses)

        return actor_loss, critic_loss, total_loss


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

