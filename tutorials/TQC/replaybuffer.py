# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import numpy as np
import random

class ReplayBuffer:
    def __init__(self, n_states, n_actions, args):
        self.device = args.device

        self.states = np.empty((args.buffer_size, n_states), dtype=np.float32)
        self.next_states = np.empty((args.buffer_size, n_states), dtype=np.float32)
        self.actions = np.empty((args.buffer_size, n_actions), dtype=np.float32)
        self.rewards = np.empty((args.buffer_size), dtype=np.float32)
        self.masks = np.empty((args.buffer_size), dtype=np.float32)

        self.max_size = args.buffer_size
        self.ptr, self.cur_len, = 0, 0

    def store(self, state, action, reward, next_state, mask):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

    def sample_batch(self, batch_size):
        index = np.random.choice(self.cur_len, batch_size, replace = False)

        return dict(state = self.states[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_states[index],
                    mask = self.masks[index],
                    )

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True