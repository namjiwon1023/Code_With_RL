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

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.masks = []
        self.log_probs = []

