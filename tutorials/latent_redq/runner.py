# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
from collections import deque
from datetime import timedelta
from time import sleep, time

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import eval_mode


class SlacObservation:
    """
    Observation for SLAC.
    """

    def __init__(self, state_shape, action_shape, num_sequences):
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.num_sequences = num_sequences

    def reset_episode(self, state):
        self._state = deque(maxlen=self.num_sequences)
        self._action = deque(maxlen=self.num_sequences - 1)
        for _ in range(self.num_sequences - 1):
            self._state.append(np.zeros(self.state_shape, dtype=np.uint8))
            self._action.append(np.zeros(self.action_shape, dtype=np.float32))
        self._state.append(state)

    def append(self, state, action):
        self._state.append(state)
        self._action.append(action)

    @property
    def state(self):
        return np.array(self._state)[None, ...]

    @property
    def action(self):
        return np.array(self._action).reshape(1, -1)


class Runner:
    def __init__(self, agent, args, writer):
        self.args = args
        self.agent = agent
        self.writer = writer
        # Env to collect samples.
        self.env = agent.env
        self.env.seed(args.seed)

        # Env for evaluation.
        self.env_test = agent.env_test
        self.env_test.seed(2 ** 31 - args.seed)

        # Observations for training and evaluation.
        self.ob = SlacObservation(self.env.observation_space.shape, self.env.action_space.shape, self.args.num_sequences)
        self.ob_test = SlacObservation(self.env.observation_space.shape, self.env.action_space.shape, self.args.num_sequences)

        # Log setting.
        self.log = {"step": [], "return": []}
        self.csv_path = os.path.join(args.save_dir + '/' + args.env_name, "log.csv")

        # Other parameters.
        self.action_repeat = self.env.action_repeat
        self.time_steps = args.time_steps
        self.initial_collection_steps = args.initial_collection_steps
        self.initial_learning_steps = args.initial_learning_steps
        self.evaluate_rate = args.evaluate_rate
        self.evaluate_episodes = args.evaluate_episodes

        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        self.model_path = self.args.save_dir + '/' + args.env_name
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        if os.path.exists(self.model_path + '/REDQ_actor.pth'):
            self.load_models()

    def run(self):
        # Time to start training.
        self.start_time = time()
        # Episode's timestep.
        t = 0
        # Initialize the environment.
        state = self.env.reset()
        self.ob.reset_episode(state)
        self.agent.buffer.reset_episode(state)

        # Collect trajectories using random policy.
        for step in range(1, self.initial_collection_steps + 1):
            t = self.agent.step(self.agent, self.ob, t)

        # Update latent variable model first so that SLAC can learn well using (learned) latent dynamics.
        bar = tqdm(range(self.initial_learning_steps))
        for _ in bar:
            bar.set_description("Updating latent variable model.")
            self.agent.update_latent(self.writer)

        # Iterate collection, update and evaluation.
        for step in range(self.initial_collection_steps + 1, self.time_steps // self.action_repeat + 1):
            t = self.agent.step(self.agent, self.ob, t)

            # Update the algorithm.
            self.agent.update_latent(self.writer)
            self.agent.update_sac(self.writer)

            # Evaluate regularly.
            step_env = step * self.action_repeat
            if step_env % self.evaluate_rate == 0:
                self.evaluate(self.agent, step_env)
                self.agent.save_models()

        # Wait for logging to be finished.
        sleep(10)

    def evaluate(self, agent, step_env):
        mean_return = 0.0

        for i in range(self.evaluate_episodes):
            state = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                with eval_mode(agent):
                    action = agent.select_test_action(self.ob_test)
                state, reward, done, _ = self.env_test.step(action)
                self.ob_test.append(state, action)
                episode_return += reward

            mean_return += episode_return / self.evaluate_episodes

        # Log to CSV.
        self.log["step"].append(step_env)
        self.log["return"].append(mean_return)
        pd.DataFrame(self.log).to_csv(self.csv_path, index=False)

        # Log to TensorBoard.
        self.writer.add_scalar("return/test", mean_return, step_env)
        print(f"Steps: {step_env:<6}   " f"Return: {mean_return:<5.1f}   " f"Time: {self.time}")

    def _evaluate_agent(self, agent, evaluate_episodes=1):
        mean_return = 0.0

        for i in range(evaluate_episodes):
            state = self.env_test.reset()
            self.ob_test.reset_episode(state)
            episode_return = 0.0
            done = False

            while not done:
                with eval_mode(agent):
                    action = agent.select_test_action(self.ob_test)
                state, reward, done, _ = self.env_test.step(action)
                self.ob_test.append(state, action)
                episode_return += reward

            mean_return += episode_return / self.evaluate_episodes
        return mean_return

    @property
    def time(self):
        return str(timedelta(seconds=int(time() - self.start_time)))
