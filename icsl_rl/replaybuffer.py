import numpy as np
import random

from icsl_rl.segment_tree import MinSegmentTree, SumSegmentTree

class ReplayBuffer:
    def __init__(self, n_states, n_actions, args, buffer_size=None):
        if buffer_size == None:
            buffer_size = args.buffer_size

        self.device = args.device

        self.states = np.empty([buffer_size, n_states], dtype=np.float32)
        self.next_states = np.empty([buffer_size, n_states], dtype=np.float32)
        # If the dimension of the action exceeds 1 dimension, then self.actions = np.empty([buffer_size, action_dim], dtype=np.float32)
        self.actions = np.empty([buffer_size],dtype=np.float32)
        self.rewards = np.empty([buffer_size], dtype=np.float32)
        self.masks = np.empty([buffer_size],dtype=np.float32)

        self.max_size = buffer_size
        self.ptr, self.cur_len, = 0, 0
        self.n_states = n_states
        self.n_actions = n_actions

        self.transitions = []

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

        '''
        Spinning up style
        from https://github.com/openai/spinningup
        batch = dict(state = self.states[index],
                    action = self.actions[index],
                    reward = self.rewards[index],
                    next_state = self.next_states[index],
                    mask = self.masks[index],
                    )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}
        '''

    def clear(self):

        self.states = np.empty([self.max_size, self.n_states], dtype=np.float32)
        self.next_states = np.empty([self.max_size, self.n_states], dtype=np.float32)
        self.actions = np.empty([self.max_size], dtype=np.float32)
        self.rewards = np.empty([self.max_size], dtype=np.float32)
        self.masks = np.empty([self.max_size], dtype=np.float32)

        self.ptr, self.cur_len, = 0, 0

    def store_transition(self, transition):
        self.transitions.append(transition)
        np.save('bc_memo.npy', self.transitions)

    def store_for_BC_data(self, transitions):
        for t in transitions:
            self.store(*t)

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True

class ReplayBufferPPO:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []

    def RB_clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.values[:]
        del self.masks[:]
        del self.log_probs[:]

class PrioritizedReplayBuffer:
    def __init__(self, n_states, args, buffer_size, alpha=0.6):
        assert alpha >= 0

        self.states = np.empty([buffer_size, n_states], dtype=np.float32)
        self.next_states = np.empty([buffer_size, n_states], dtype=np.float32)

        # If the dimension of the action exceeds 1 dimension, then self.actions = np.empty([buffer_size, action_dim], dtype=np.float32)
        self.actions = np.empty([buffer_size], dtype=np.float32)

        self.rewards = np.empty([buffer_size], dtype=np.float32)
        self.masks = np.empty([buffer_size], dtype=np.float32)

        self.max_size = buffer_size
        self.ptr, self.cur_len, = 0, 0

        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self, state , action, reward, next_state, mask):

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr] = mask

        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_len = min(self.cur_len + 1, self.max_size)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, batch_size, beta=0.4):

        assert len(self) >= batch_size
        assert beta > 0

        indices = self._sample_proportional(batch_size)

        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
                    state=self.states[indices],
                    next_state=self.next_states[indices],
                    action=self.actions[indices],
                    reward=self.rewards[indices],
                    mask=self.masks[indices],
                    weights=weights,
                    indices=indices,
                    )

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self, batch_size):
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx, beta):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

    def __len__(self):
        return self.cur_len

    def ready(self, batch_size):
        if self.cur_len >= batch_size:
            return True



# class PrioritizedReplayBuffer(ReplayBuffer):
#     """Prioritized Replay buffer.

#     Attributes:
#         max_priority (float): max priority
#         tree_ptr (int): next index of tree
#         alpha (float): alpha parameter for prioritized replay buffer
#         sum_tree (SumSegmentTree): sum tree for prior
#         min_tree (MinSegmentTree): min tree for min prior to get max weight

#     """
#     def __init__(self, n_states, n_actions, args, alpha=0.6, buffer_size=None):
#         """Initialization."""
#         assert alpha >= 0
#         super(PrioritizedReplayBuffer, self).__init__(n_states, n_actions, args, buffer_size)
#         self.max_priority, self.tree_ptr = 1.0, 0
#         self.alpha = alpha

#         # capacity must be positive and a power of 2.
#         tree_capacity = 1
#         while tree_capacity < self.max_size:
#             tree_capacity *= 2

#         self.sum_tree = SumSegmentTree(tree_capacity)
#         self.min_tree = MinSegmentTree(tree_capacity)

#     def store(self, state , action, reward, next_state, mask):
#         """Store experience and priority."""
#         super().store(state, action, reward, next_state, mask)

#         self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
#         self.tree_ptr = (self.tree_ptr + 1) % self.max_size

#     def sample_batch(self, batch_size, beta=0.4):
#         """Sample a batch of experiences."""
#         assert len(self) >= batch_size
#         assert beta > 0

#         indices = self._sample_proportional(batch_size)

#         state = self.states[indices]
#         next_state = self.next_states[indices]
#         action = self.actions[indices]
#         reward = self.rewards[indices]
#         mask = self.masks[indices]
#         weights = np.array([self._calculate_weight(i, beta) for i in indices])

#         return dict(
#                     state=state,
#                     next_state=next_state,
#                     action=action,
#                     reward=reward,
#                     mask=mask,
#                     weights=weights,
#                     indices=indices,
#                     )

#     def update_priorities(self, indices, priorities):
#         """Update priorities of sampled transitions."""
#         assert len(indices) == len(priorities)

#         for idx, priority in zip(indices, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self)

#             self.sum_tree[idx] = priority ** self.alpha
#             self.min_tree[idx] = priority ** self.alpha

#             self.max_priority = max(self.max_priority, priority)

#     def _sample_proportional(self, batch_size):
#         """Sample indices based on proportions."""
#         indices = []
#         p_total = self.sum_tree.sum(0, len(self) - 1)
#         segment = p_total / batch_size

#         for i in range(batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             upperbound = random.uniform(a, b)
#             idx = self.sum_tree.retrieve(upperbound)
#             indices.append(idx)

#         return indices

#     def _calculate_weight(self, idx, beta):
#         """Calculate the weight of the experience at idx."""
#         # get max weight
#         p_min = self.min_tree.min() / self.sum_tree.sum()
#         max_weight = (p_min * len(self)) ** (-beta)

#         # calculate weights
#         p_sample = self.sum_tree[idx] / self.sum_tree.sum()
#         weight = (p_sample * len(self)) ** (-beta)
#         weight = weight / max_weight

#         return weight


# class ReplayBuffer_N_Step:
#     """A simple numpy replay buffer."""

#     def __init__(
#         self,
#         obs_dim: int,
#         size: int,
#         batch_size: int = 32,
#         n_step: int = 1,
#         gamma: float = 0.99
#     ):
#         self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
#         self.acts_buf = np.zeros([size], dtype=np.float32)
#         self.rews_buf = np.zeros([size], dtype=np.float32)
#         self.done_buf = np.zeros(size, dtype=np.float32)
#         self.max_size, self.batch_size = size, batch_size
#         self.ptr, self.size, = 0, 0

#         # for N-step Learning
#         self.n_step_buffer = deque(maxlen=n_step)
#         self.n_step = n_step
#         self.gamma = gamma

#     def store(
#         self,
#         obs: np.ndarray,
#         act: np.ndarray,
#         rew: float,
#         next_obs: np.ndarray,
#         done: bool,
#     ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
#         transition = (obs, act, rew, next_obs, done)
#         self.n_step_buffer.append(transition)

#         # single step transition is not ready
#         if len(self.n_step_buffer) < self.n_step:
#             return ()

#         # make a n-step transition
#         rew, next_obs, done = self._get_n_step_info(
#             self.n_step_buffer, self.gamma
#         )
#         obs, act = self.n_step_buffer[0][:2]

#         self.obs_buf[self.ptr] = obs
#         self.next_obs_buf[self.ptr] = next_obs
#         self.acts_buf[self.ptr] = act
#         self.rews_buf[self.ptr] = rew
#         self.done_buf[self.ptr] = done
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)

#         return self.n_step_buffer[0]

#     def sample_batch(self) -> Dict[str, np.ndarray]:
#         idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

#         return dict(
#             obs=self.obs_buf[idxs],
#             next_obs=self.next_obs_buf[idxs],
#             acts=self.acts_buf[idxs],
#             rews=self.rews_buf[idxs],
#             done=self.done_buf[idxs],
#             # for N-step Learning
#             indices=idxs,
#         )

#     def sample_batch_from_idxs(
#         self, idxs: np.ndarray
#     ) -> Dict[str, np.ndarray]:
#         # for N-step Learning
#         return dict(
#             obs=self.obs_buf[idxs],
#             next_obs=self.next_obs_buf[idxs],
#             acts=self.acts_buf[idxs],
#             rews=self.rews_buf[idxs],
#             done=self.done_buf[idxs],
#         )

#     def _get_n_step_info(
#         self, n_step_buffer: Deque, gamma: float
#     ) -> Tuple[np.int64, np.ndarray, bool]:
#         """Return n step rew, next_obs, and done."""
#         # info of the last transition
#         rew, next_obs, done = n_step_buffer[-1][-3:]

#         for transition in reversed(list(n_step_buffer)[:-1]):
#             r, n_o, d = transition[-3:]

#             rew = r + gamma * rew * (1 - d)
#             next_obs, done = (n_o, d) if d else (next_obs, done)

#         return rew, next_obs, done

#     def __len__(self) -> int:
#         return self.size


# class PrioritizedReplayBuffer(ReplayBuffer_N_Step):
#     """Prioritized Replay buffer.

#     Attributes:
#         max_priority (float): max priority
#         tree_ptr (int): next index of tree
#         alpha (float): alpha parameter for prioritized replay buffer
#         sum_tree (SumSegmentTree): sum tree for prior
#         min_tree (MinSegmentTree): min tree for min prior to get max weight

#     """

#     def __init__(
#         self,
#         obs_dim: int,
#         size: int,
#         batch_size: int = 32,
#         alpha: float = 0.6,
#         n_step: int = 1,
#         gamma: float = 0.99,
#     ):
#         """Initialization."""
#         assert alpha >= 0

#         super(PrioritizedReplayBuffer, self).__init__(
#             obs_dim, size, batch_size, n_step, gamma
#         )
#         self.max_priority, self.tree_ptr = 1.0, 0
#         self.alpha = alpha

#         # capacity must be positive and a power of 2.
#         tree_capacity = 1
#         while tree_capacity < self.max_size:
#             tree_capacity *= 2

#         self.sum_tree = SumSegmentTree(tree_capacity)
#         self.min_tree = MinSegmentTree(tree_capacity)

#     def store(
#         self,
#         obs: np.ndarray,
#         act: int,
#         rew: float,
#         next_obs: np.ndarray,
#         done: bool,
#     ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
#         """Store experience and priority."""
#         transition = super().store(obs, act, rew, next_obs, done)

#         if transition:
#             self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
#             self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
#             self.tree_ptr = (self.tree_ptr + 1) % self.max_size

#         return transition

#     def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
#         """Sample a batch of experiences."""
#         assert len(self) >= self.batch_size
#         assert beta > 0

#         indices = self._sample_proportional()

#         obs = self.obs_buf[indices]
#         next_obs = self.next_obs_buf[indices]
#         acts = self.acts_buf[indices]
#         rews = self.rews_buf[indices]
#         done = self.done_buf[indices]
#         weights = np.array([self._calculate_weight(i, beta) for i in indices])

#         return dict(
#             obs=obs,
#             next_obs=next_obs,
#             acts=acts,
#             rews=rews,
#             done=done,
#             weights=weights,
#             indices=indices,
#         )

#     def update_priorities(self, indices: List[int], priorities: np.ndarray):
#         """Update priorities of sampled transitions."""
#         assert len(indices) == len(priorities)

#         for idx, priority in zip(indices, priorities):
#             assert priority > 0
#             assert 0 <= idx < len(self)

#             self.sum_tree[idx] = priority ** self.alpha
#             self.min_tree[idx] = priority ** self.alpha

#             self.max_priority = max(self.max_priority, priority)

#     def _sample_proportional(self) -> List[int]:
#         """Sample indices based on proportions."""
#         indices = []
#         p_total = self.sum_tree.sum(0, len(self) - 1)
#         segment = p_total / self.batch_size

#         for i in range(self.batch_size):
#             a = segment * i
#             b = segment * (i + 1)
#             upperbound = random.uniform(a, b)
#             idx = self.sum_tree.retrieve(upperbound)
#             indices.append(idx)

#         return indices

#     def _calculate_weight(self, idx: int, beta: float):
#         """Calculate the weight of the experience at idx."""
#         # get max weight
#         p_min = self.min_tree.min() / self.sum_tree.sum()
#         max_weight = (p_min * len(self)) ** (-beta)

#         # calculate weights
#         p_sample = self.sum_tree[idx] / self.sum_tree.sum()
#         weight = (p_sample * len(self)) ** (-beta)
#         weight = weight / max_weight

#         return weight


