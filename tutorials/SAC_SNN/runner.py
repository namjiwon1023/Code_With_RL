import numpy as np
import torch as T
import os

from utils import eval_mode

class Runner:
    def __init__(self, agent, env, test_env, args, writer):
        self.args = args
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.writer = writer

        # Storage location creation
        if not os.path.exists(self.args['save_dir']):
            os.mkdir(self.args['save_dir'])

        self.model_path = self.args['save_dir'] + '/' + args['algorithm']
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        self.model_path = self.model_path + '/' + args['env_name']
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)

        if os.path.exists(self.model_path + '/' + self.args['file_actor']):
            self.agent.load_models()

    def run(self):
        best_score = - np.inf

        scores = []
        train_rewards = []
        eval_rewards = []

        avg_score = 0

        for i in range(self.args['episode']):
            state = self.env.reset()
            cur_episode_steps = 0
            score = 0
            real_done = False
            while (not real_done) and (not cur_episode_steps == self.args['max_ep_step']):
                cur_episode_steps += 1
                self.agent.total_step += 1
                with eval_mode(self.agent):
                    action = self.agent.select_exploration_action(state)
                next_state, reward, done, _ = self.env.step(action)

                real_done = False if cur_episode_steps == self.args['max_ep_step'] else done
                mask = 0.0 if real_done else self.args['gamma']

                self.agent.memory.store(state, action, reward, next_state, mask)

                state = next_state
                score += reward

                if self.agent.total_step > self.args['start_steps']:
                    self.agent.learn(self.writer)

                if self.agent.total_step % self.args['evaluate_rate'] == 0 and self.agent.total_step > self.args['start_steps']:
                    running_reward = np.mean(scores[-10:])
                    eval_reward = self.agent._evaluate_agent(self.test_env, self.agent, self.args)
                    eval_rewards.append(eval_reward)
                    self.writer.add_scalar('Reward/Train', running_reward, self.agent.total_step)
                    self.writer.add_scalar('Reward/Test', eval_reward, self.agent.total_step)
                    print('| Episode : {} | Step : {} | Eval_Score : {} | Avg_Score : {} | update number : {} |'.format(i, self.agent.total_step, round(eval_reward, 2), round(avg_score, 2), self.agent.learning_step))
                    scores = []

            scores.append(score)
            train_rewards.append(score)
            avg_score = np.mean(train_rewards[-10:])

            np.savetxt(self.args['save_dir'] + '/' + self.args['algorithm'] + '/' + self.args['env_name'] + '/train_rewards.txt', train_rewards, delimiter=",")
            np.savetxt(self.args['save_dir'] + '/' + self.args['algorithm'] + '/' + self.args['env_name'] + '/eval_rewards.txt', eval_rewards, delimiter=",")

            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_models()

            if self.agent.total_step >= self.args['time_steps']:
                print('Reach the maximum number of training steps ！')
                break

    def evaluate(self):
        returns = self.agent._evaluate_agent(self.test_env, self.agent, self.args)
        return returns
