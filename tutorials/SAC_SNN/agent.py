# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

from networks import SpikingActorSAC, SpikingCriticTwin
from ReplayBuffer import ReplayBuffer

from spikingjelly.clock_driven import functional

from utils import eval_mode, _save_model, _load_model

class SpikingSACAgent:
    def __init__(self, args):
        self.args = args

        self.actor_path = os.path.join(args['save_dir'] + '/' + args['algorithm'] +'/' + args['env_name'], args['file_actor'])
        self.critic_path = os.path.join(args['save_dir'] + '/' + args['algorithm'] +'/' + args['env_name'], args['file_critic'])

        # actor-critic net setting
        self.actor = SpikingActorSAC(self.args).to(self.args['device'])
        self.critic = SpikingCriticTwin(self.args).to(self.args['device'])

        self.critic_target = SpikingCriticTwin(self.args).to(self.args['device'])
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature Coefficient
        self.target_entropy = -self.args['n_actions']
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args['device'])

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args['critic_lr'])
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args['alpha_lr'])

        self.memory = ReplayBuffer(self.args)

        # loss function
        self.criterion = nn.SmoothL1Loss(reduction='mean')

        self.learning_step = 0
        self.total_step = 0

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_test_action(self, state):
        with T.no_grad():
            test_action, _ = self.actor(T.FloatTensor(state).unsqueeze(0).to(self.args['device']), evaluate=True, with_logprob=False)
            functional.reset_net(self.actor)
        return test_action.detach().cpu().numpy()[0]

    def select_exploration_action(self, state):
        with T.no_grad():
            if self.total_step < self.args['start_steps']:
                exploration_action = np.random.uniform(self.args['low_action'], self.args['max_action'], self.args['n_actions'])
            else:
                exploration_action, _ = self.actor(T.FloatTensor(state).unsqueeze(0).to(self.args['device']))
                functional.reset_net(self.actor)
                exploration_action = exploration_action.detach().cpu().numpy()[0]
        return exploration_action

    def learn(self, writer):
        if not self.memory.ready(self.args['batch_size']):
            return
        self.learning_step += 1

        # TD error
        # update value
        q1_loss, q2_loss, state = self._value_update(self.memory, self.args['batch_size'])
        critic_loss = q1_loss + q2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss, new_log_prob = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # target network soft update
        if self.total_step % self.args['target_update_interval'] == 0:
            self._target_soft_update(self.critic_target, self.critic, self.args['tau'])

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.detach().cpu().item(), self.learning_step)
            writer.add_scalar("loss/alpha", alpha_loss.detach().cpu().item(), self.learning_step)
            writer.add_scalar("value/alpha", self.alpha.detach().cpu().item(), self.learning_step)
            writer.add_scalar("loss/critic", critic_loss.detach().cpu().item(), self.learning_step)
            writer.add_scalar("loss/q1", q1_loss.detach().cpu().item(), self.learning_step)
            writer.add_scalar("loss/q2", q2_loss.detach().cpu().item(), self.learning_step)


    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            state, next_state, action, reward, mask = self._get_batch_buffer(buffer, batch_size)

            next_action, next_log_prob = self.actor(next_state)
            functional.reset_net(self.actor)
            next_target_q1, next_target_q2 = self.critic_target(next_state, next_action)
            functional.reset_net(self.critic_target)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic(state, action)
        functional.reset_net(self.critic)
        # TD error
        # update value
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)

        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        functional.reset_net(self.actor)
        q_1, q_2 = self.critic(state, new_action)
        functional.reset_net(self.critic)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

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
        checkpoint = os.path.join(self.args['save_dir'] + '/' + self.args['algorithm'] +'/' + self.args['env_name'], 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)
        checkpoint = os.path.join(self.args['save_dir'] + '/' + self.args['algorithm'] +'/' + self.args['env_name'], 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)
