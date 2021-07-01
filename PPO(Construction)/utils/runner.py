import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from tqdm import tqdm
import time

from algorithm.agent import PPOAgent
from utils.utils import make_gif


class Runner:
    def __init__(self, agent, args, env, writer):
        self.args = args
        self.episode_limit = env.spec.max_episode_steps
        self.env = env
        self.agent = agent
        self.writer = writer

    def run(self):
        best_score = self.env.reward_range[0]

        scores = []
        store_scores = []
        eval_rewards = []
        episode = 0

        avg_score = 0
        n_updates = 0

        for time_step in tqdm(range(self.args.time_steps)):
            score = 0
            state = self.env.reset()
            for _ in tqdm(range(self.args.rollout_len)):
                if self.args.render:
                    self.env.render()
                self.agent.total_step += 1
                action = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                if not self.args.evaluate:
                    self.agent.rewards.append(T.as_tensor(reward, dtype=T.float32, device=self.args.device))
                    self.agent.masks.append(T.as_tensor(1 - done, dtype=T.float32, device=self.args.device))

                state = next_state
                score += reward

                if done:
                    episode += 1
                    state = self.env.reset()
                    scores.append(score)
                    store_scores.append(score)
                    avg_score = np.mean(store_scores[-10:])
                    score = 0


            actor_loss, critic_loss, total_loss = self.agent.learn(next_state)
            n_updates += 1

            if self.agent.total_step % self.args.evaluate_rate == 0:
                running_reward = np.mean(scores[-10:])
                eval_reward = self.agent.evaluate_agent(n_starts=self.args.evaluate_episodes)
                eval_rewards.append(eval_reward)
                self.writer.add_scalar('Loss/Critic', critic_loss, n_updates)
                self.writer.add_scalar('Loss/Actor', actor_loss, n_updates)
                self.writer.add_scalar('Loss/Total', total_loss, n_updates)
                self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(episode, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                scores = []

            np.savetxt("./model/Pendulum-v0/episode_return.txt", store_scores, delimiter=",")
            np.savetxt("./model/Pendulum-v0/step_return.txt", eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break
            if avg_score >= -140:
                print('Stop Training')
                break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(episode, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

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