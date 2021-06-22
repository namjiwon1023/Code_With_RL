import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import random
from tqdm import tqdm
import time

from agent import DQNAgent
from utils import random_seed
from arguments import get_args

if __name__ == '__main__':
    args = get_args()
    writer = SummaryWriter()
    random_seed(args.seed)

    agent = DQNAgent(args)

    if args.evaluate:
        agent.evaluate_agent()
    else:
        best_score = agent.env.reward_range[0]

        scores = []
        store_scores = []
        eval_rewards = []

        avg_score = 0
        n_updates = 0

        for i in range(1, args.episode + 1):
            state = agent.env.reset()
            cur_episode_steps = 0
            score = 0
            done = False

            while not done:

                if args.render is True:
                    agent.env.render()

                cur_episode_steps += 1
                agent.total_step += 1
                action = agent.choose_action(state)
                next_state, reward, done, _ = agent.env.step(action)
                # time.sleep(0.02)
                real_done = False if cur_episode_steps >= agent.env.spec.max_episode_steps else done
                mask = 0.0 if real_done else args.gamma
                agent.transition += [reward, next_state, mask]
                agent.memory.store(*agent.transition)
                state = next_state
                score += reward

                if agent.memory.ready(args.batch_size):
                    Q_loss = agent.learn()
                    n_updates += 1

                if agent.total_step % args.evaluate_rate == 0 and agent.memory.ready(args.batch_size):
                    running_reward = np.mean(scores)
                    eval_reward = agent.evaluate_agent(n_starts=args.evaluate_episodes)
                    eval_rewards.append(eval_reward)
                    writer.add_scalar('Loss/Q', Q_loss, n_updates)
                    writer.add_scalar('Loss/Epsilon', agent.epsilon, n_updates)
                    writer.add_scalar('Reward/Train', running_reward, agent.total_step)
                    writer.add_scalar('Reward/Test', eval_reward, agent.total_step)
                    print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                    scores = []

                agent.epsilon = max(args.min_epsilon, agent.epsilon - (args.max_epsilon - args.min_epsilon) * args.epsilon_decay)

            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(store_scores)

            np.savetxt("./episode_return.txt", store_scores, delimiter=",")
            np.savetxt("./step_return.txt", eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            if agent.total_step >= args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), agent.total_step, n_updates))

        agent.env.close()