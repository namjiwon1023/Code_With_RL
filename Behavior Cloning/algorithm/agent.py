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

from network.ActorNetwork import ActorNetwork
from network.CriticNetwork import CriticNetwork
from utils.ReplayBuffer import ReplayBuffer

class SACAgent(object):
    def __init__(self, args):
        self.args = args

        self.env = gym.make(args.env_name)
        self.env = RescaleAction(self.env, -1, 1)
        self.n_states = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]

        self.max_action = self.env.action_space.high[0]
        self.low_action = self.env.action_space.low[0]

        self.memory = ReplayBuffer(self.n_states, self.n_actions, self.args)
        self.transition = list()

        self.actor = ActorNetwork(self.n_states, self.n_actions, self.args)
        self.critic_eval = CriticNetwork(self.n_states, self.n_actions, self.args)

        self.critic_target = copy.deepcopy(self.critic_eval)
        self.critic_target.eval()
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.target_entropy = -self.n_actions
        self.log_alpha = T.zeros(1, requires_grad=True, device=self.args.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.args.alpha_lr)

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/SAC_actor.pth'):
            self.load_models()

        self.total_step = 0

        data_path = os.path.join('','bc_memo.npy')
        self.lambda1 = self.args.lambda1
        self.lambda2 = self.args.lambda2 / self.args.bc_batch_size
        self.memo = np.load(data_path, allow_pickle = True)

        self.bc_data = ReplayBuffer(self.n_states, self.n_actions, self.args, len(self.memo))
        self.bc_data.store_for_BC_data(self.memo)

    def choose_action(self, state, epsilon):
        with T.no_grad():
            # if self.total_step < self.args.start_step and not self.args.evaluate:
            if epsilon >= np.random.random() and not self.args.evaluate:
                choose_action = np.random.uniform(self.low_action, self.max_action, self.n_actions)
            else :
                choose_action, _ = self.actor(T.as_tensor(state, dtype=T.float32, device=self.actor.device))
                choose_action = choose_action.detach().cpu().numpy()

            self.transition = [state, choose_action]
        return choose_action

    def learn(self):
        with T.no_grad():
            samples = self.memory.sample_batch(self.args.batch_size)

            state = T.as_tensor(samples['state'], dtype=T.float32, device=self.args.device)
            next_state = T.as_tensor(samples['next_state'], dtype=T.float32, device=self.args.device)
            action = T.as_tensor(samples['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward = T.as_tensor(samples['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask = T.as_tensor(samples['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            samples_bc = self.bc_data.sample_batch(self.args.bc_batch_size)

            state_bc = T.as_tensor(samples_bc['state'], dtype=T.float32, device=self.args.device)
            next_state_bc = T.as_tensor(samples_bc['next_state'], dtype=T.float32, device=self.args.device)
            action_bc = T.as_tensor(samples_bc['action'], dtype=T.float32, device=self.args.device).reshape(-1, self.n_actions)
            reward_bc = T.as_tensor(samples_bc['reward'], dtype=T.float32, device=self.args.device).reshape(-1,1)
            mask_bc = T.as_tensor(samples_bc['mask'], dtype=T.float32, device=self.args.device).reshape(-1,1)

            next_action, next_log_prob = self.actor(next_state)
            next_target_q1, next_target_q2 = self.critic_target(next_state, next_action)
            next_target_q = T.min(next_target_q1, next_target_q2)
            target_q = reward + (next_target_q - self.alpha * next_log_prob) * mask

        current_q1, current_q2 = self.critic_eval(state, action)
        q1_loss = self.critic_eval.loss_func(current_q1, target_q)
        q2_loss = self.critic_eval.loss_func(current_q2, target_q)
        critic_loss = q1_loss + q2_loss

        self.critic_eval.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_eval.optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = False

        new_action, new_log_prob = self.actor(state)
        q_1, q_2 = self.critic_eval(state, new_action)
        q = T.min(q_1, q_2)
        pg_loss = (self.alpha * new_log_prob - q).mean()
        alpha_loss = -self.log_alpha * (new_log_prob.detach() + self.target_entropy).mean()

        pred_action, _ = self.actor(state_bc)
        q_t = T.min(*self.critic_eval(state_bc, action_bc))
        q_e = T.min(*self.critic_eval(state_bc, pred_action))
        qf_mask = T.gt(q_t, q_e).to(self.critic_eval.device)
        qf_mask = qf_mask.float()
        n_qf_mask = int(qf_mask.sum().item())

        if n_qf_mask == 0:
            bc_loss = T.zeros(1, device=self.args.device)
        else:
            bc_loss = (
                T.mul(pred_action, qf_mask) - T.mul(action_bc, qf_mask)
            ).pow(2).sum() / n_qf_mask

        actor_loss = self.lambda1 * pg_loss + self.lambda2 * bc_loss

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for p in self.critic_eval.parameters():
            p.requires_grad = True

        self.alpha = self.log_alpha.exp()

        if self.total_step % self.args.target_update_interval == 0:
            self._target_soft_update(self.critic_target, self.critic_eval, self.args.tau)


        return q1_loss.item(), q2_loss.item(), critic_loss.item(), actor_loss.item(), alpha_loss.item(), pg_loss.item(), bc_loss.item()

    def _target_soft_update(self, target_net, eval_net, tau=None):
        if tau == None:
            tau = self.args.tau
        with T.no_grad():
            for t_p, l_p in zip(target_net.parameters(), eval_net.parameters()):
                t_p.data.copy_(tau * l_p.data + (1 - tau) * t_p.data)

    def evaluate_agent(self, n_starts=10):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                if self.args.evaluate:
                    self.env.render()
                # time.sleep(0.02)
                action = self.choose_action(state, 0)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        # self.env.close()
        return reward_sum / n_starts

    def save_models(self):
        print('------ Save models ------')
        self.actor.save_model()
        self.critic_eval.save_model()
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.env_name, 'log_alpha.pth')
        T.save(self.log_alpha, checkpoint)

    def load_models(self):
        print('------ load models ------')
        self.actor.load_model()
        self.critic_eval.load_model()
        checkpoint = os.path.join(self.args.save_dir + '/' + self.args.env_name, 'log_alpha.pth')
        self.log_alpha = T.load(checkpoint)
