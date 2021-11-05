# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from networks import Actor, MultiCritic
from ReplayBuffer import ReplayBuffer

from utils import eval_mode, _save_model, _load_model, quantile_huber_loss_f
from noise import OUNoise

class DDPGAgent:
    def __init__(self, args):
        self.args = args

        self.actor_path = os.path.join(args['save_dir'] + '/' + args['algorithm'] +'/' + args['env_name'], args['file_actor'])
        self.critic_path = os.path.join(args['save_dir'] + '/' + args['algorithm'] +'/' + args['env_name'], args['file_critic'])

        # actor-critic net setting
        self.actor = Actor(self.args).to(self.args['device'])
        self.critic = MultiCritic(self.args).to(self.args['device'])

        self.actor_target = Actor(self.args).to(self.args['device'])
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = MultiCritic(self.args).to(self.args['device'])
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args['critic_lr'])

        self.memory = ReplayBuffer(self.args)

        # loss function
        self.criterion = nn.SmoothL1Loss(reduction='mean')

        self.top_quantiles_to_drop = args['top_quantiles_to_drop_per_net'] * args['n_nets']
        self.quantiles_total = args['n_quantiles'] * args['n_nets']

        self.noise = None
        if not args['gaussian_noise']:
            self.noise = OUNoise(args['n_actions'], theta=args['ou_noise_theta'], sigma=args['ou_noise_sigma'])

        self.learning_step = 0
        self.total_step = 0

        self.train()
        self.actor_target.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def select_test_action(self, state):
        with T.no_grad():
            test_action = self.actor(T.FloatTensor(state).unsqueeze(0).to(self.args['device']))
        return test_action.detach().cpu().numpy()[0]

    def select_exploration_action(self, state):
        with T.no_grad():
            if self.total_step < self.args['start_steps']:
                exploration_action = np.random.uniform(self.args['low_action'], self.args['max_action'], self.args['n_actions'])
            else:
                if self.args['gaussian_noise']:
                    noise = np.random.normal(0, self.args['max_action']*self.args['exploration_noise'], size=self.args['n_actions'])
                else:
                    noise = self.noise.sample()
                exploration_action = self.actor(T.FloatTensor(state).unsqueeze(0).to(self.args['device']))
                exploration_action = exploration_action.detach().cpu().numpy()[0]
                exploration_action = np.clip(exploration_action + noise, -1, 1)
        return exploration_action

    def learn(self, writer):
        if not self.memory.ready(self.args['batch_size']):
            return
        self.learning_step += 1

        # TD error
        # update value
        critic_loss, state = self._value_update(self.memory, self.args['batch_size'])

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # target network soft update
        if self.total_step % self.args['target_update_interval'] == 0:
            self._target_soft_update(self.critic_target, self.critic, self.args['tau'])
            self._target_soft_update(self.actor_target, self.actor, self.args['tau'])

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.detach().cpu().item(), self.learning_step)
            writer.add_scalar("loss/critic", critic_loss.detach().cpu().item(), self.learning_step)


    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            state, next_state, action, reward, mask = self._get_batch_buffer(buffer, batch_size)

            next_action = self.actor_target(next_state)
            next_z = self.critic_target(next_state, next_action)
            sorted_z, _ = T.sort(next_z.reshape(batch_size, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

            target = reward + sorted_z_part * mask

        # Twin Critic Network Loss functions
        current_z = self.critic(state, action)

        # TD error
        # update value
        critic_loss = quantile_huber_loss_f(current_z, target, self.args['device'])

        return critic_loss, state

    def _policy_update(self, state):
        action = self.actor(state)
        q = self.critic(state, action).mean(2).mean(1, keepdim=True)
        actor_loss = -q.mean()
        return actor_loss

    def _get_batch_buffer(self, buffer, batch_size):
        samples = buffer.sample_batch(batch_size)
        state = T.FloatTensor(samples['state']).to(self.args['device'])
        next_state = T.FloatTensor(samples['next_state']).to(self.args['device'])
        action = T.FloatTensor(samples['action']).reshape(-1, self.args['n_actions']).to(self.args['device'])
        reward = T.FloatTensor(samples['reward']).reshape(-1, 1).to(self.args['device'])
        mask = T.FloatTensor(samples['mask']).reshape(-1, 1).to(self.args['device'])
        return state, next_state, action, reward, mask

    def _target_soft_update(self, target_net, eval_net, tau):
        for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
            t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def _evaluate_agent(self, env, agent, args):
        reward_sum = 0
        for _ in range(args['evaluate_episodes']):
            done = False
            state = env.reset()
            while not done:
                with eval_mode(agent):
                    action = agent.select_test_action(state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / args['evaluate_episodes']

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)


    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)

