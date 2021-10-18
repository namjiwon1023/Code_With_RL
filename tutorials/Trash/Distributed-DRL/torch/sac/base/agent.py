import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import ActorSAC, CriticTwin

class SACAgent:
    def __init__(self, args):
        self.args = args
        self.critic_update_function = None
        self.if_per = args.if_per
        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

        # Environment setting
        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, args)
        self.critic_update_function = self._value_update
        self.transition = list()

        # actor-critic net setting
        self.actor = ActorSAC(self.n_states, self.n_actions, self.args)
        self.critic = CriticTwin(self.n_states, self.n_actions, self.args)

        self.critic_target = copy.deepcopy(self.critic)
        _grad_false(self.critic_target)

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        # loss function
        self.criterion = nn.MSELoss(reduction='none' if self.if_per else 'mean')

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

    def choose_action(self, state, epsilon):
        with T.no_grad():
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()
            self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        if not self.memory.ready(self.args.batch_size):
            return
        self.learning_step += 1

        # TD error
        # update value
        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            _target_soft_update(self.critic_target, self.critic, self.args.tau)

        for p in self.critic.parameters():
            p.requires_grad = False

        actor_loss, new_log_prob = self._policy_update(state)

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

        for p in self.critic.parameters():
            p.requires_grad = True

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)

    def _value_update(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic.get_double_q(state, action)
        # TD error
        # update value
        q1_loss = self.criterion(current_q1, target_q)
        q2_loss = self.criterion(current_q2, target_q)
        critic_loss = q1_loss + q2_loss
        return critic_loss, state

    def _value_update_per(self, buffer, batch_size):
        with T.no_grad():
            # Select data from ReplayBuffer with batch_size size
            samples = buffer.sample_batch(batch_size, self.args.beta)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            weights = T.as_tensor(samples["weights"], dtype=T.float32, device=self.args.device).reshape(-1, 1)
            indices = samples["indices"]

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target.get_double_q(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_q1, current_q2 = self.critic.get_double_q(state, action)

        # TD error
        elementwise_loss = self.criterion(T.min(current_q1, current_q2), target_q)
        critic_loss = T.mean((self.criterion(current_q1, target_q) + self.criterion(current_q2, target_q)) * weights)

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.args.prior_eps
        buffer.update_priorities(indices, new_priorities)

        return critic_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic.get_double_q(state, new_action)
        q = T.min(q_1, q_2)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss
