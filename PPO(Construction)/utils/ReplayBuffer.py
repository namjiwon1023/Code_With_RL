import torch as T
import random
import numpy as np

class ReplayBuffer:
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