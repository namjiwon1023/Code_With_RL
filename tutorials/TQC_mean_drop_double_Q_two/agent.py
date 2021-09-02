import torch as T
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import Actor, MultiCritic, CriticTwin, MultiCriticTwin

from replaybuffer import ReplayBuffer
from utils import quantile_huber_loss_f, disable_gradients, _save_model, _load_model

class TQCAgent:
    def __init__(self, args):
        self.args = args

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
        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        # actor-critic net setting
        self.actor = Actor(self.n_states, self.n_actions, self.args)

        self.critic = MultiCritic(self.n_states, self.n_actions, self.args)
        self.critic_target = copy.deepcopy(self.critic)

        # self.critic_multi_twin = MultiCriticTwin(self.n_states, self.n_actions, self.args)
        # self.critic_multi_twin_target = copy.deepcopy(self.critic_multi_twin)

        self.critic_twin = CriticTwin(self.n_states, self.n_actions, self.args)
        self.critic_twin_target = copy.deepcopy(self.critic_twin)

        # optimizer setting
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.args.critic_lr)
        self.critic_twin_optimizer = optim.Adam(self.critic_twin.parameters(), lr=self.args.critic_lr)
        # self.critic_optimizer = optim.Adam(self.critic_multi_twin.parameters(), lr=self.args.critic_lr)

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

        self.train()
        self.critic_target.train()

        self.critic_twin_target.train()
        # self.critic_multi_twin_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        # self.critic_multi_twin.train(training)
        self.critic_twin.train(training)

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
        critic_loss_1, critic_loss_2, state = self._value_update(self.memory, self.args.batch_size)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_optimizer.zero_grad()
        self.critic_twin_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_twin_optimizer.step()

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic, self.args.tau)
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_twin_target, self.critic_twin, self.args.tau)
        # if self.total_step % self.args.target_update_interval == 0:
        #     self._target_soft_update(self.critic_multi_twin_target, self.critic_multi_twin, self.args.tau)

        # actor_loss_1, actor_loss_2, new_log_prob = self._policy_update(state)
        actor_loss_1, actor_loss_2, new_log_prob = self._policy_update(state)
        actor_loss = actor_loss_1 + actor_loss_1

        # update Policy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update Temperature Coefficient
        alpha_loss = self._temperature_update(new_log_prob)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp()

        if self.learning_step % 1000 == 0:
            writer.add_scalar("loss/critic", critic_loss.item(), self.learning_step)
            writer.add_scalar("loss/actor", actor_loss.item(), self.learning_step)
            writer.add_scalar("loss/alpha", alpha_loss.item(), self.learning_step)
            writer.add_scalar("loss/doubleQ_critic", critic_loss_1.item(), self.learning_step)
            writer.add_scalar("loss/distribution_critic", critic_loss_2.item(), self.learning_step)
            writer.add_scalar("loss/doubleQ_actor", actor_loss_2.item(), self.learning_step)
            writer.add_scalar("loss/distribution_actor", actor_loss_1.item(), self.learning_step)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)
        # _save_model(self.critic_multi_twin, self.critic_path)
        checkpoint_ = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'auxiliary.pth')
        _save_model(self.critic_twin, checkpoint_)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)
        # _load_model(self.critic_multi_twin, self.critic_path)
        checkpoint_ = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'auxiliary.pth')
        _load_model(self.critic_twin, checkpoint_)
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
            # next_z_1, next_z_2 = self.critic_multi_twin_target(next_state, next_action)
            # n_q_1, n_q_2 = next_z_1.mean(), next_z_2.mean()
            # if n_q_1 == T.min(n_q_1, n_q_2):
            #     next_z = next_z_1
            # else:
            #     next_z = next_z_2
            # if n_q_1 > n_q_2:
            #     next_z = next_z_2
            # else:
            #     next_z = next_z_1
            # next_z = T.min(next_z_1, next_z_2)
            next_z_rs = next_z.reshape(batch_size, -1)
            list_ = []
            names = locals()
            for i in range(self.args.n_nets):
                names[f'net_quantiles_{i}'] = next_z_rs[:, i*self.args.n_quantiles:(i+1)*self.args.n_quantiles]
                names[f'net_quantiles_{i}_sorted'], _ = T.sort(names[f'net_quantiles_{i}'])
                names[f'net{i}_sorted_part'] = names[f'net_quantiles_{i}_sorted'][:, :self.args.n_quantiles-self.args.top_quantiles_to_drop_per_net]
                list_.append(names[f'net{i}_sorted_part'])
                names[f'sorted_z_part'] = T.cat(list_, dim=-1)

            # Here we calculate action value Q(s,a) = R + yV(s')
            target = reward + (names[f'sorted_z_part'] - self.alpha * next_log_prob) * mask
            # print('target:{}'.format(target))

            next_target_q1, next_target_q2 = self.critic_twin_target(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            # Here we calculate action value Q(s,a) = R + yV(s')
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask
            # print('target_q:{}'.format(target_q))

        # Twin Critic Network Loss functions
        current_z = self.critic(state, action)
        # current_z1, current_z2 = self.critic_multi_twin(state, action)
        current_q1, current_q2 = self.critic_twin(state, action)

        loss_1 = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        # TD error
        # update value
        loss_2 = quantile_huber_loss_f(current_z, target, self.args.device)
        # loss_1 = quantile_huber_loss_f(current_z1, target, self.args.device)
        # loss_2 = quantile_huber_loss_f(current_z2, target, self.args.device)

        return loss_1, loss_2, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        q = self.critic(state, new_action).mean(2).mean(1, keepdim=True)
        q_1, q_2 = self.critic_twin(state, new_action)
        q_ = T.min(q_1, q_2)
        # z_1, z_2 = self.critic_multi_twin(state, new_action)
        # z = T.min(z_1, z_2)
        # q = T.min(z_1.mean(2).mean(1, keepdim=True), z_2.mean(2).mean(1, keepdim=True))
        # q = z.mean(2).mean(1, keepdim=True)
        # update actor network
        loss_1 = (self.alpha * new_log_prob - q).mean()
        loss_2 = (self.alpha * new_log_prob - q_).mean()
        # return loss_1, loss_2, new_log_prob
        return loss_1, loss_2, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss
