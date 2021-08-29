import torch as T
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import Actor, MultiCritic

from replaybuffer import ReplayBuffer
from utils import quantile_huber_loss_f, disable_gradients, _save_model, _load_model

class TQCAgent:
    def __init__(self, args):
        self.args = args

        self.actor_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_actor)
        self.critic_path = os.path.join(args.save_dir + '/' + args.algorithm +'/' + args.env_name, args.file_critic)

        # Environment setting
        self.env = gym.make(args.env_name)
        self.test_env = gym.make(args.env_name)

        self.env = RescaleAction(self.env, -1, 1)
        self.test_env = RescaleAction(self.test_env, -1, 1)

        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        # replay buffer
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic net setting
        self.actor = Actor(self.n_states, self.n_actions, self.args)
        self.critic = MultiCritic(self.n_states, self.n_actions, self.args)
        self.critic_target = copy.deepcopy(self.critic)
        disable_gradients(self.critic_target)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)

        # Temperature Coefficient
        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        self.total_step = 0
        self.init_random_steps = args.init_random_steps

        self.top_quantiles_to_drop = args.top_quantiles_to_drop_per_net * args.n_nets
        self.quantiles_total = args.n_quantiles * args.n_nets

        self.learning_step = 0

    def choose_action(self, state, evaluate=False):
        with T.no_grad():
            if self.total_step < self.init_random_steps and not evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                if evaluate:
                    choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device), evaluate=True, with_logprob=False)
                    choose_action = choose_action.detach().cpu().numpy()
                else:
                    choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device), evaluate=False, with_logprob=True)
                    choose_action = choose_action.detach().cpu().numpy()
            if not evaluate:
                self.transition = [state, choose_action]
        return choose_action

    def learn(self, writer):
        self.learning_step += 1
        # TD error
        # update value
        critic_loss, state = self._value_update(self.memory, self.args.batch_size)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic, self.args.tau)

        for p in self.critic.parameters():
            p.requires_grad = False

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

        for p in self.critic.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)

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

            next_action, next_log_prob = self.actor(next_state)
            next_z = self.critic_target(next_state, next_action)
            sorted_z, _ = T.sort(next_z.reshape(batch_size, -1))
            sorted_z_part = sorted_z[:, :self.quantiles_total-self.top_quantiles_to_drop]

            # Here we calculate action value Q(s,a) = R + yV(s')
            target = reward + (sorted_z_part - self.alpha * next_log_prob) * mask

        # Twin Critic Network Loss functions
        current_z = self.critic(state, action)
        # TD error
        # update value
        critic_loss = quantile_huber_loss_f(current_z, target, self.args.device)

        return critic_loss, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q = self.critic(state, new_action).mean(2).mean(1, keepdim=True)
        # update actor network
        actor_loss = (self.alpha * new_log_prob - q).mean()
        return actor_loss, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss
