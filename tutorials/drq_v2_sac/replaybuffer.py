import numpy as np

import kornia
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, buffer_size, device):
        self.buffer_size = buffer_size
        self.device = device

        self.states = np.empty((buffer_size, *obs_shape), dtype=np.uint8)
        self.next_states = np.empty((buffer_size, *obs_shape), dtype=np.uint8)

        self.actions = np.empty((buffer_size, *action_shape), dtype=np.float32)
        self.rewards = np.empty((buffer_size, 1), dtype=np.float32)

        self.masks = np.empty((buffer_size, 1), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def __len__(self):
        return self.buffer_size if self.full else self.ptr

    def store(self, state, action, reward, next_state, mask):
        np.copyto(self.states[self.ptr], state)
        np.copyto(self.actions[self.ptr], action)
        np.copyto(self.rewards[self.ptr], reward)
        np.copyto(self.next_states[self.ptr], next_state)
        np.copyto(self.masks[self.ptr], mask)

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.full = self.full or self.ptr == 0

    def sample_batch(self, batch_size):
        index = np.random.randint(0,
                                self.buffer_size if self.full else self.ptr,
                                size=batch_size)

        states = T.as_tensor(self.states[index], device=self.device).float()
        next_states = T.as_tensor(self.next_states[index], device=self.device).float()
        actions = T.as_tensor(self.actions[index], device=self.device)
        rewards = T.as_tensor(self.rewards[index], device=self.device)
        masks = T.as_tensor(self.masks[index], device=self.device)

        return states, actions, rewards, next_states, masks

    def ready(self, batch_size):
        if self.__len__() > batch_size:
            return True