import torch as T
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import copy
import gym
from gym.wrappers import RescaleAction

from network import Actor, MultiCriticTwin

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
        self.critic = MultiCriticTwin(self.n_states, self.n_actions, self.args)
        self.critic_target = copy.deepcopy(self.critic)

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

        self.learning_step = 0

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

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
        critic_loss.backward()
        self.critic_optimizer.step()

        # target network soft update
        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic, self.args.tau)

        actor_loss1, actor_loss2, new_log_prob = self._policy_update(state)
        actor_loss = actor_loss1+ actor_loss2
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
            writer.add_scalar("loss/distribution_critic_1", critic_loss_1.item(), self.learning_step)
            writer.add_scalar("loss/distribution_critic_2", critic_loss_2.item(), self.learning_step)
            writer.add_scalar("loss/distribution_actor_1", actor_loss1.item(), self.learning_step)
            writer.add_scalar("loss/distribution_actor_2", actor_loss2.item(), self.learning_step)

    def save_models(self):
        print('------ Save model ------')
        _save_model(self.actor, self.actor_path)
        _save_model(self.critic, self.critic_path)
        # _save_model(self.critic_multi_twin, self.critic_path)
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.algorithm +'/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load model ------')
        _load_model(self.actor, self.actor_path)
        _load_model(self.critic, self.critic_path)
        # _load_model(self.critic_multi_twin, self.critic_path)
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
            next_z_1, next_z_2 = self.critic_target(next_state, next_action) # torch.Size([256, 5, 25])
            next_z_rs_1, next_z_rs_2 = next_z_1.reshape(batch_size, -1), next_z_2.reshape(batch_size, -1)  #  torch.Size([256, 125])
            # next_z_rs = T.where(next_z_rs_1 > next_z_rs_2, next_z_rs_2, next_z_rs_1)

            list_1 = []
            names = locals()
            for i in range(self.args.n_nets):
                names[f'net_quantiles1_{i}'] = next_z_rs_1[:, i*self.args.n_quantiles:(i+1)*self.args.n_quantiles]
                names[f'net_quantiles1_{i}_sorted'], _ = T.sort(names[f'net_quantiles1_{i}'])
                names[f'net{i}_sorted_part1'] = names[f'net_quantiles1_{i}_sorted'][:, :self.args.n_quantiles-self.args.top_quantiles_to_drop_per_net]
                list_1.append(names[f'net{i}_sorted_part1'])
                names[f'sorted_z_part1'] = T.cat(list_1, dim=-1)

            # Here we calculate action value Q(s,a) = R + yV(s')
            target1 = reward + (names[f'sorted_z_part1'] - self.alpha * next_log_prob) * mask # torch.Size([256, 115])

            list_2 = []
            names = locals()
            for i in range(self.args.n_nets):
                names[f'net_quantiles2_{i}'] = next_z_rs_2[:, i*self.args.n_quantiles:(i+1)*self.args.n_quantiles]
                names[f'net_quantiles2_{i}_sorted'], _ = T.sort(names[f'net_quantiles2_{i}'])
                names[f'net{i}_sorted_part2'] = names[f'net_quantiles2_{i}_sorted'][:, :self.args.n_quantiles-self.args.top_quantiles_to_drop_per_net]
                list_2.append(names[f'net{i}_sorted_part2'])
                names[f'sorted_z_part2'] = T.cat(list_2, dim=-1)

            # Here we calculate action value Q(s,a) = R + yV(s')
            target2 = reward + (names[f'sorted_z_part2'] - self.alpha * next_log_prob) * mask # torch.Size([256, 115])

            target = (target1 + target2) /2

        # Twin Critic Network Loss functions
        current_z1, current_z2 = self.critic(state, action) # torch.Size([256, 5, 25])

        loss1 = quantile_huber_loss_f(current_z1, target, self.args.device)
        loss2 = quantile_huber_loss_f(current_z2, target, self.args.device)


        return loss1, loss2, state

    def _policy_update(self, state):
        new_action, new_log_prob = self.actor(state)
        # b = self.critic(state, new_action).mean(2) # b size : {} torch.Size([256, 5])
        z1, z2 = self.critic(state, new_action)
        # z = T.where(z1 > z2, z2, z1)
        q1 = z1.mean(2).mean(1, keepdim=True)
        q2 = z2.mean(2).mean(1, keepdim=True)

        # q = self.critic(state, new_action).mean(2).mean(1, keepdim=True) # q size : {} torch.Size([256, 1])
        # update actor network
        loss_1 = (self.alpha * new_log_prob - q1).mean() # distribution actor loss :-19.76106834411621
        loss_2 = (self.alpha * new_log_prob - q2).mean()

        return loss_1, loss_2, new_log_prob

    def _temperature_update(self, new_log_prob):
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()
        return alpha_loss
