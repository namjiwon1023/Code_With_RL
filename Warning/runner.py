import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
from tqdm import tqdm
import time

from utils import _make_gif, _evaluate_agent, _save_model


class Runner:
    def __init__(self, agent, args, env, writer):
        self.args = args
        self.epsilon = args.epsilon
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

        for i in range(self.args.episode):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            done = False
            while (not done):
                if self.args.render is True:
                    self.env.render()

                cur_episode_steps += 1
                self.agent.total_step += 1
                if self.args.use_epsilon:
                    action = self.agent.choose_action(state, self.epsilon)
                else:
                    action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                real_done = False if cur_episode_steps >= self.episode_limit else done
                mask = 0.0 if real_done else self.args.gamma
                self.agent.transition += [reward, next_state, mask]
                self.agent.memory.store(*self.agent.transition)
                state = next_state
                score += reward

                if self.agent.memory.ready(self.args.batch_size):
                    Q_loss = self.agent.learn()
                    n_updates += 1
                    self.epsilon = max(0.1, self.epsilon - self.args.epsilon_decay)

                if self.agent.total_step % self.args.evaluate_rate == 0 and self.agent.memory.ready(self.args.batch_size):
                    running_reward = np.mean(scores[-10:])
                    eval_reward = _evaluate_agent(self.env, self.agent, self.args, n_starts=self.args.evaluate_episodes)
                    eval_rewards.append(eval_reward)
                    self.writer.add_scalar('Loss/Q', Q_loss, n_updates)
                    self.writer.add_scalar('Epsilon', self.epsilon, self.agent.total_step)
                    self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                    print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                    scores = []

            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(store_scores[-10:])

            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/episode_return.txt', store_scores, delimiter=",")
            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/step_return.txt', eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                _save_model(self.agent.eval)

            if self.agent.total_step >= self.args.time_steps or avg_score == 200:
                print('Reach the maximum number of training steps ！')
                break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

        self.env.close()


    def evaluate(self):
        returns = _evaluate_agent(self.env, self.agent, self.args, n_starts=1)
        return returns

    def gif(self, policy, env, maxsteps=1000):
        _make_gif(policy, env, self.args, maxsteps)