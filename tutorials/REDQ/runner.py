# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
from utils import _evaluate_agent
import torch as T
import os
from datetime import timedelta
from time import time

class Runner:
    def __init__(self, agent, args, writer):
        self.args = args
        self.agent = agent
        self.env = self.agent.env
        self.test_env = self.agent.test_env
        self.episode_limit = self.env.spec.max_episode_steps
        self.writer = writer

        # Storage location creation
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        self.model_path = self.args.save_dir + '/' + args.algorithm
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/' + self.args.file_actor):
            self.agent.load_models()

    def run(self):
        self.start_time = time()
        best_score = self.env.reward_range[0]

        scores = []
        store_scores = []
        eval_rewards = []

        avg_score = 0

        for i in range(self.args.episode):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            done = False
            while (not done):
                cur_episode_steps += 1
                self.agent.total_step += 1

                action = self.agent.select_exploration_action(state)
                next_state, reward, done, _ = self.env.step(action)

                real_done = False if cur_episode_steps >= self.episode_limit else done
                mask = 0.0 if real_done else self.args.gamma

                self.agent.transition += [reward, next_state, mask]
                self.agent.memory.store(*self.agent.transition)
                state = next_state
                score += reward

                self.agent.learn(self.writer)

                if self.agent.total_step % self.args.evaluate_rate == 0 and self.agent.memory.ready(self.args.batch_size):
                    running_reward = np.mean(scores[-10:])
                    eval_reward = _evaluate_agent(self.test_env, self.agent, self.args, n_starts=self.args.evaluate_episodes, render=False)
                    eval_rewards.append(eval_reward)
                    print_eval_reward = np.mean(eval_rewards[-10:])
                    self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test_avg', print_eval_reward, self.agent.total_step)
                    print('| Episode : {} | Step : {} | Train Return : {} | Predict Return : {} | Predict Avg Return : {} | Critic update number : {} | Actor update number : {} | Time : {} | '.format(i, self.agent.total_step, round(running_reward, 2), round(eval_reward, 2), round(print_eval_reward, 2),  self.agent.critic_learning_step, self.agent.actor_learning_step, self.time))
                    scores = []

            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(store_scores[-10:])

            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/episode_return.txt', store_scores, delimiter=",")
            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/step_return.txt', eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

    def evaluate(self):
        returns = _evaluate_agent(self.test_env, self.agent, self.args, n_starts=1, render=True)
        return returns

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))