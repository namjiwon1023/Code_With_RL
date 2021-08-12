# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This file is modified from <https://github.com/ku2482/slac.pytorch>:
# Copyright (c) 2020 ku2482
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from collections import deque

import numpy as np
import torch as T


class LazyFrames:
    """
    Stacked frames which never allocate memory to the same frame.
    从不为同一帧分配内存的堆叠帧。
    只是个很简单的
    """

    def __init__(self, frames):
        self._frames = list(frames)

    def __array__(self, dtype):
        return np.array(self._frames, dtype=dtype)

    def __len__(self):
        return len(self._frames)


class SequenceBuffer:
    """
    Buffer for storing sequence data.
    用于存储序列数据的缓冲区。
    """

    def __init__(self, num_sequences=8):
        self.num_sequences = num_sequences
        self._reset_episode = False
        self.state_ = deque(maxlen=self.num_sequences + 1)
        self.action_ = deque(maxlen=self.num_sequences)
        self.reward_ = deque(maxlen=self.num_sequences)
        self.done_ = deque(maxlen=self.num_sequences)

    def reset(self):
        self._reset_episode = False
        self.state_.clear()
        self.action_.clear()
        self.reward_.clear()
        self.done_.clear()

    def reset_episode(self, state):
        assert not self._reset_episode
        self._reset_episode = True
        self.state_.append(state)

    def append(self, action, reward, done, next_state):
        assert self._reset_episode
        self.action_.append(action)
        self.reward_.append([reward])
        self.done_.append([done])
        self.state_.append(next_state)

    def get(self):
        state_ = LazyFrames(self.state_)
        action_ = np.array(self.action_, dtype=np.float32)
        reward_ = np.array(self.reward_, dtype=np.float32)
        done_ = np.array(self.done_, dtype=np.float32)
        return state_, action_, reward_, done_

    def is_empty(self):
        return len(self.reward_) == 0

    def is_full(self):
        return len(self.reward_) == self.num_sequences

    def __len__(self):
        return len(self.reward_)


class ReplayBuffer:
    """
    Replay Buffer.
    """

    def __init__(self, buffer_size, num_sequences, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.num_sequences = num_sequences
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.device = device

        # Store the sequence of images as a list of LazyFrames on CPU. It can store images with 9 times less memory.
        # 将图像序列存储为 CPU 上的 LazyFrames 列表。 它可以以少 9 倍的内存存储图像。
        self.state_ = [None] * buffer_size
        # Store other data on GPU to reduce workloads.
        # 将其他数据存储在 GPU 上以减少工作量。
        self.action_ = T.empty(buffer_size, num_sequences, *action_shape, device=device)
        self.reward_ = T.empty(buffer_size, num_sequences, 1, device=device)
        self.done_ = T.empty(buffer_size, num_sequences, 1, device=device)
        # Buffer to store a sequence of trajectories.
        # 用于存储轨迹序列的缓冲区。
        self.buff = SequenceBuffer(num_sequences=num_sequences)

    def reset_episode(self, state):
        """
        Reset the buffer and set the initial observation. This has to be done before every episode starts.
        重置缓冲区并设置初始观察。 这必须在每集开始之前完成。
        """
        self.buff.reset_episode(state)

    def append(self, action, reward, done, next_state, episode_done):
        """
        Store trajectory in the buffer. If the buffer is full, the sequence of trajectories is stored in replay buffer.
        Please pass 'masked' and 'true' done so that we can assert if the start/end of an episode is handled properly.

        将轨迹存储在缓冲区中。 如果缓冲区已满，轨迹序列将存储在重播缓冲区中。请传递 'masked' 和 'true' done 以便我们可以断言剧集的开始/结束是否正确处理。
        """
        self.buff.append(action, reward, done, next_state)

        if self.buff.is_full():
            state_, action_, reward_, done_ = self.buff.get()
            self._append(state_, action_, reward_, done_)

        if episode_done:
            self.buff.reset()

    def _append(self, state_, action_, reward_, done_):
        self.state_[self._p] = state_
        self.action_[self._p].copy_(T.from_numpy(action_))
        self.reward_[self._p].copy_(T.from_numpy(reward_))
        self.done_[self._p].copy_(T.from_numpy(done_))

        self._n = min(self._n + 1, self.buffer_size)
        self._p = (self._p + 1) % self.buffer_size

    def sample_latent(self, batch_size):
        """
        Sample trajectories for updating latent variable model.
        用于更新潜在变量模型的样本轨迹。
        image.div_(255.0) -> Normalize the image
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = T.tensor(state_, dtype=T.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes], self.done_[idxes]

    def sample_sac(self, batch_size):
        """
        Sample trajectories for updating SAC.
        用于更新 SAC 的示例轨迹。
        image.div_(255.0) -> Normalize the image
        """
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        state_ = np.empty((batch_size, self.num_sequences + 1, *self.state_shape), dtype=np.uint8)
        for i, idx in enumerate(idxes):
            state_[i, ...] = self.state_[idx]
        state_ = T.tensor(state_, dtype=T.uint8, device=self.device).float().div_(255.0)
        return state_, self.action_[idxes], self.reward_[idxes, -1], self.done_[idxes, -1]

    def __len__(self):
        return self._n
