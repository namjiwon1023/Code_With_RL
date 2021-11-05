import numpy as np

class ReplayBuffer:
    def __init__(self, args):
        self.args = args

        self.states = np.zeros([args['buffer_size'], args['n_states']], dtype = np.float32)
        self.next_states = np.zeros([args['buffer_size'], args['n_states']], dtype = np.float32)
        self.actions = np.zeros([args['buffer_size'], args['n_actions']], dtype = np.float32)
        self.rewards = np.zeros([args['buffer_size']], dtype = np.float32)
        self.masks = np.zeros([args['buffer_size']],dtype = np.float32)

        self.max_size = args['buffer_size']
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

        return dict(
                    state = self.states[index],
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
