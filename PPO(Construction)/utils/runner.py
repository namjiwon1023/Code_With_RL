import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from tqdm import tqdm
import time

from algorithm.agent import PPOAgent
from utils.utils import make_gif, _plot
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, agent, args, env, writer):
        self.args = args
        self.env = env
        self.agent = agent
        self.writer = writer
        self.episode_limit = env.spec.max_episode_steps

    def run(self):
        plt.ion()
        plt.figure()

        best_score = self.env.reward_range[0]
        steps = 0

        scores = []
        # store_scores = []
        # eval_rewards = []

        # actor_losses = []
        # critic_losses = []
        # total_losses = []

        avg_score = 0
        # n_updates = 0

        for i in tqdm(range(self.args.episode)):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            for t in range(self.args.max_ep_len):
                if self.args.render is True:
                    self.env.render()
                self.agent.total_step += 1
                cur_episode_steps += 1
                steps += 1
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                real_done = False if cur_episode_steps >= self.episode_limit else done
                if not self.args.evaluate:
                    self.agent.memory.rewards.append(T.as_tensor((reward,), dtype=T.float32, device=self.args.device))
                    self.agent.memory.masks.append(T.as_tensor((1. - float(real_done)), dtype=T.float32, device=self.args.device))

                if steps % self.args.update_step == 0:
                    self.agent.learn(next_state)
                    steps = 0
                state = next_state
                score += reward
                if done:
                    break
            # avg_length += t



            # actor_loss, critic_loss, total_loss = self.agent.learn(next_state)
            # self.agent.learn(next_state)
            # actor_losses.append(actor_loss)
            # critic_losses.append(critic_loss)
            # total_losses.append(total_losses)
            # n_updates += 1
            # if self.args.episode % 4 == 0:
            #     # running_reward = np.mean(scores[-10:])
            #     eval_reward = self.agent.evaluate_agent(n_starts=self.args.evaluate_episodes)
            #     eval_rewards.append(eval_reward)
            #     print('| Episode : {} | Predict Score : {} | Run score : {} |'.format(i, round(eval_reward, 2), round(running_reward, 2)))

            # if self.agent.total_step % self.args.evaluate_rate == 0 and n_updates > 0:
            #     running_reward = np.mean(scores[-10:])
            #     eval_reward = self.agent.evaluate_agent(n_starts=self.args.evaluate_episodes)
            #     eval_rewards.append(eval_reward)
            #     self.writer.add_scalar('Loss/Critic', critic_loss, n_updates)
            #     self.writer.add_scalar('Loss/Actor', actor_loss, n_updates)
            #     self.writer.add_scalar('Loss/Total', total_loss, n_updates)
            #     self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
            #     self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
            #     print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
            #     scores = []


            scores.append(score)
            # store_scores.append(score)
            avg_score = np.mean(scores[-10:])

            # np.savetxt("./model/Pendulum-v0/episode_return.txt", store_scores, delimiter=",")
            # np.savetxt("./model/Pendulum-v0/step_return.txt", eval_rewards, delimiter=",")
            np.savetxt("./model/Pendulum-v0/episode_return.txt", scores, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

            if avg_score >= -140:
                print('Stop Training')
                break

            if i % 10 == 0:
                _plot(scores)
                print('Episode : {} | Avg score : {} | Time_Step : {} |'.format(i, round(avg_score, 2), self.agent.total_step))

            # print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

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