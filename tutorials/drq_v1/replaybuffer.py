import numpy as np

import kornia
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import utils


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, buffer_size, image_pad, device):
        self.buffer_size = buffer_size
        self.device = device
        '''
        Data enhancement is performed on the collected pictures.
        In DrQ_v1, the data enhancement method is random translation,
        padding 4 pixels around the picture, and random cropping using the kornia library.
        '''
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

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

        states = self.states[index]
        next_states = self.next_states[index]

        states_aug = states.copy()
        next_states_aug = next_states.copy()

        states = T.as_tensor(states, device=self.device).float()
        next_states = T.as_tensor(next_states, device=self.device).float()

        states_aug = T.as_tensor(states_aug, device=self.device).float()
        next_states_aug = T.as_tensor(next_states_aug, device=self.device).float()

        actions = T.as_tensor(self.actions[index], device=self.device)
        rewards = T.as_tensor(self.rewards[index], device=self.device)

        masks = T.as_tensor(self.masks[index], device=self.device)

        states = self.aug_trans(states)
        next_states = self.aug_trans(next_states)

        states_aug = self.aug_trans(states_aug)
        next_states_aug = self.aug_trans(next_states_aug)

        return states, actions, rewards, next_states, masks, states_aug, next_states_aug

    def ready(self, batch_size):
        if self.__len__() > batch_size:
            return True