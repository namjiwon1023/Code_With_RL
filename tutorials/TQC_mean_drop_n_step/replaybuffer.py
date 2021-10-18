# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import random
from collections import deque

class NStepReplayBuffer:
    def __init__(self, n_states, n_actions, buffer_size, n_step= 3, gamma= 0.99):
        self.states = np.zeros([buffer_size, n_states], dtype=np.float32)
        self.next_states = np.zeros([buffer_size, n_states], dtype=np.float32)
        self.actions = np.zeros([buffer_size, n_actions], dtype=np.float32)
        self.rewards = np.zeros([buffer_size], dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.max_size = buffer_size
        self.ptr, self.cur_len, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        reward, next_state, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        state, action = self.n_step_buffer[0][:2]

        self.states[self.ptr] = state
        self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self, batch_size):
        indices = np.random.choice(self.cur_len, batch_size, replace=False)

        return dict(
            state=self.states[indices],
            next_state=self.next_states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            done=self.dones[indices],
            # for N-step Learning
            indices=indices,
        )

    def sample_batch_from_idxs(self, indices):
        # for N-step Learning
        return dict(
            state=self.states[indices],
            next_state=self.next_states[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            done=self.dones[indices],
        )

    def _get_n_step_info(self, n_step_buffer, gamma):
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        reward, next_state, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            reward = r + gamma * reward * (1 - d)
            next_state, done = (n_o, d) if d else (next_state, done)

        return reward, next_state, done

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True