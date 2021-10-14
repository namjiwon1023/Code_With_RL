import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction
import ray

from networks import Actor, Critic

class SACAgent(nn.Module):
    def __init__(self, args):
        super(SACAgent, self).__init__()
        self.args = args

        self.n_states = args.n_states
        self.n_actions = args.n_actions

        self.max_action = args.max_action
        self.low_action = args.low_action

        # replay buffer
        # self.buffer = ReplayBuffer(self.n_states, self.n_actions, args)

        # actor-critic net setting
        self.actor = Actor(self.n_states, self.n_actions, self.args)
        self.critic = Critic(self.n_states, self.n_actions, self.args)

        self.critic_target = Critic(self.n_states, self.n_actions, self.args)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        # loss function
        self.criterion = nn.MSELoss()

        self.total_step = 0
        self.learning_step = 0

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
            test_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device), evaluate=True, with_logprob=False)
            test_action = test_action.detach().cpu().numpy()
        return test_action

    def select_exploration_action(self, state):
        with T.no_grad():
            if self.total_step <= self.start_steps:
                exploration_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else:
                exploration_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                exploration_action = exploration_action.detach().cpu().numpy()
        return exploration_action

    def learn(self, buffer, writer):
        if not buffer.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # TD error
        # update value
        q1_loss, q2_loss, state = self._value_update(buffer, self.args.batch_size)
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
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_target, self.critic, self.args.tau)

        # tensorboard
        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.detach().item(), self.learning_step)
            writer.add_scalar("loss/critic", critic_loss.detach().item(), self.learning_step)
            writer.add_scalar("loss/q1", q1_loss.detach().item(), self.learning_step)
            writer.add_scalar("loss/q2", q2_loss.detach().item(), self.learning_step)
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            state, next_state, action, reward, mask = self._get_batch_buffer(buffer, batch_size)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic(state, action)
        # TD error
        # update value
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)
        return q1_loss, q2_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic(state, new_action)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss

    def _get_batch_buffer(self, buffer, batch_size):
        assert buffer is not None
        assert batch_size >= 0
        with T.no_grad():
            samples = ray.get(buffer.sample_batch.remote(batch_size))
            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            # state = T.FloatTensor(samples['state']).to(self.args.device)
            # next_state = T.FloatTensor(samples['next_state']).to(self.args.device)
            # action = T.FloatTensor(samples['action']).reshape(-1, self.n_actions).to(self.args.device)
            # reward = T.FloatTensor(samples['reward']).reshape(-1, 1).to(self.args.device)
            # mask = T.FloatTensor(samples['mask']).reshape(-1, 1).to(self.args.device)
        return state, next_state, action, reward, mask

    def get_weights(self):
        return {k: v.cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = g

    def add_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is None :
                pass
            elif p.grad == None :
                p.grad = torch.zeros(g.shape)
            if g is not None:
                p.grad += g

    def _evaluate_agent(self, env, agent, args):
        reward_sum = 0
        for _ in range(args.n_starts):
            done = False
            state = env.reset()
            ep_len = 0
            while not (done or (ep_len == args.max_ep_len)):
                ep_len += 1
                if args.render:
                    env.render()
                action = agent.select_test_action(state)
                next_state, reward, done, _ = env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / args.n_starts