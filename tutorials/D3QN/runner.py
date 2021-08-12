# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
from tqdm import tqdm
import time

from agent import D3QNAgent
from utils import random_seed, make_gif
from arguments import get_args


class Runner:
    def __init__(self, agent, args, env, writer):
        self.args = args
        self.epsilon = agent.epsilon
        self.episode_limit = env.spec.max_episode_steps
        self.env = env
        self.agent = agent
        self.writer = writer

    def run(self):
        best_score = self.env.reward_range[0]

        scores = []
        store_scores = []
        eval_rewards = []

        avg_score = 0
        n_updates = 0

        for i in tqdm(range(self.args.episode)):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            done = False
            while (not done):
                if self.args.render is True:
                    self.env.render()

                cur_episode_steps += 1
                self.agent.total_step += 1
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                real_done = False if cur_episode_steps >= self.episode_limit else done
                mask = 0.0 if real_done else self.args.gamma
                self.agent.transition += [reward, next_state, mask]
                self.agent.memory.store(*self.agent.transition)
                state = next_state
                score += reward

                if self.agent.memory.ready(self.args.batch_size):
                    self.agent.epsilon = max(self.args.min_epsilon, self.agent.epsilon - (self.args.max_epsilon - self.args.min_epsilon) * self.args.epsilon_decay)
                    Q_loss = self.agent.learn()
                    n_updates += 1

                if self.agent.total_step % self.args.evaluate_rate == 0 and self.agent.memory.ready(self.args.batch_size):
                    running_reward = np.mean(scores[-10:])
                    eval_reward = self.agent.evaluate_agent(n_starts=self.args.evaluate_episodes)
                    eval_rewards.append(eval_reward)
                    self.writer.add_scalar('Loss/Q', Q_loss, n_updates)
                    self.writer.add_scalar('Loss/Epsilon', self.agent.epsilon, self.agent.total_step)
                    self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                    print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                    scores = []

            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(store_scores[-10:])

            np.savetxt("./model/CartPole-v0/episode_return.txt", store_scores, delimiter=",")
            np.savetxt("./model/CartPole-v0/step_return.txt", eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps or avg_score == 200:
                print('Reach the maximum number of training steps ！')
                break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

        self.agent.env.close()


    def evaluate(self, n_starts=1):
        reward_sum = 0
        for _ in range(n_starts):
            done = False
            state = self.env.reset()
            while (not done):
                if self.args.evaluate:
                    self.env.render()
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                state = next_state
        return reward_sum / n_starts

    def gif(self, policy, env, maxsteps=1000):
        make_gif(policy, env, maxsteps)