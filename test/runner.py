# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
from test.utils import _make_gif, _evaluate_agent, _store_expert_data
import torch as T

class Runner:
    def __init__(self, agent, args, writer):
        self.args = args
        if self.args.use_epsilon:
            self.epsilon = args.epsilon
        else:
            self.epsilon = None
        self.episode_limit = env.spec.max_episode_steps

        self.env = gym.make(args.env_name)
        self.env.seed(args.seed)

        self.env_test = gym.make(args.env_name)
        self.env_test.seed(2 ** 31 - args.seed)

        self.agent = agent
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
        if args.is_discrete:
            if os.path.exists(self.model_path + '/' + self.args.algorithm_path):
                self.agent.load_models()
        else:
            if os.path.exists(self.model_path + '/' + self.args.file_actor):
                self.agent.load_models()

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
                if self.args.render:
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
                if self.args.is_off_policy:
                    self.agent.memory.store(*self.agent.transition)
                state = next_state
                score += reward

                if self.args.is_off_policy:
                    if self.agent.memory.ready(self.args.batch_size):
                        self.agent.learn()
                        n_updates += 1
                        if self.args.use_epsilon:
                            self.epsilon = max(0.1, self.epsilon - self.args.epsilon_decay)
                        else:
                            self.epsilon = None

                    if self.agent.total_step % self.args.evaluate_rate == 0 and self.agent.memory.ready(self.args.batch_size):
                        running_reward = np.mean(scores[-10:])
                        eval_reward = _evaluate_agent(self.env, self.agent, self.args, n_starts=self.args.evaluate_episodes)
                        eval_rewards.append(eval_reward)
                        self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                        self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                        print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                        scores = []
                else:
                    self.agent.learn()
                    n_updates += 1

                    if self.agent.total_step % self.args.evaluate_rate == 0:
                        running_reward = np.mean(scores[-10:])
                        eval_reward = _evaluate_agent(self.env, self.agent, self.args, n_starts=self.args.evaluate_episodes)
                        eval_rewards.append(eval_reward)
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
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

            if self.args.is_discrete:
                if avg_score == 200:            # early stopping
                    print('Stop Training')
                    break
            else:
                if avg_score >= -140:
                    print('Stop Training')
                    break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

        self.env.close()

    def ppo_run(self):
        best_score = self.env.reward_range[0]
        steps = 0
        scores = []
        store_scores = []
        eval_rewards = []

        avg_score = 0
        n_updates = 0

        for i in range(self.args.episode):
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
                    n_updates += 1

                state = next_state
                score += reward

                # if self.agent.total_step % self.args.evaluate_rate == 0 and n_updates > 0:
                #     running_reward = np.mean(scores[-10:])
                #     eval_reward = _evaluate_agent(self.env, self.agent, self.args, n_starts=self.args.evaluate_episodes)
                #     eval_rewards.append(eval_reward)
                #     self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                #     self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                #     print('| Episode : {} | Score : {} | Predict Score : {} | Avg score : {} |'.format(i, round(score, 2), round(eval_reward, 2), round(avg_score, 2)))
                #     scores = []
                if done:
                    break


            scores.append(score)
            store_scores.append(score)
            avg_score = np.mean(scores[-10:])

            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/episode_return.txt', store_scores, delimiter=",")
            np.savetxt(self.args.save_dir + '/' + self.args.algorithm + '/' + self.args.env_name + '/step_return.txt', eval_rewards, delimiter=",")


            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args.time_steps:
                print('Reach the maximum number of training steps ！')
                break

            if avg_score >= -140:
                print('Stop Training')
                break

            print('Episode : {} | Score : {} | Avg score : {} | Time_Step : {} |  update number : {} |'.format(i, round(score, 2), round(avg_score, 2), self.agent.total_step, n_updates))

        self.agent.env.close()


    def evaluate(self):
        if not self.args.is_store_transition:
            returns = _evaluate_agent(self.env, self.agent, self.args, n_starts=1)
        else:
            returns = _store_expert_data(self.env, self.agent, self.args, n_starts=1000)
        return returns

    def gif(self, policy, maxsteps=1000):
        _make_gif(policy, self.env, self.args, maxsteps)