# Copyright (c) 2021: Zhiyuan Nan (namjw@hanyang.ac.kr).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import torch as T
import torch.nn as nn
import numpy as np
import random

def _random_seed(env, test_env, seed):
    if T.backends.cudnn.enabled:
        T.backends.cudnn.benchmark = False
        T.backends.cudnn.deterministic = True
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)
    env.action_space.np_random.seed(seed)
    test_env.seed(seed+9999)
    test_env.action_space.np_random.seed(seed+9999)
    print('| CNN algorithm is fixed : {} | Seed : {} | Test env Seed : {} |'.format(T.backends.cudnn.enabled, seed, seed+9999))

def image_preprocessing(image, output_size=64):
    output_image = None
    # print('image:{}'.format(image.shape))
    if image.shape == (64, 64, 3):
        image_pre = np.transpose(image, (2, 0, 1))     # [256, 256, 3] -> [3, 256, 256]
        bin_size = image_pre.shape[1] // output_size
        output_image = image_pre.reshape((3, output_size, bin_size,
                                    output_size, bin_size)).max(4).max(2)      # [3, 256, 256] -> [3, 64, 64]
    elif image.shape == (3, 64, 64):
        output_image = image
    return output_image

def initialize_weight(m, std=1.0, bias_const=1e-6):
    '''
    linear layers initialization
    '''
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, std)
        nn.init.constant_(m.bias, bias_const)
    '''
    Convolution layers initialization
    '''
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_uniform_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Dict(dict):
    def __init__(self,config, section_name,location = False):
        super(Dict,self).__init__()
        self.initialize(config, section_name,location)
    def initialize(self, config, section_name,location):
        for key,value in config.items(section_name):
            if location :
                self[key] = value
            else:
                self[key] = eval(value)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False

# model save functions
def _save_model(net, dirpath):
    T.save(net.state_dict(), dirpath)

# model load functions
def _load_model(net, dirpath):
    net.load_state_dict(T.load(dirpath))


def quantile_huber_loss_f(quantiles, samples, device):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = T.abs(pairwise_delta)
    huber_loss = T.where(abs_pairwise_delta > 1,
                            abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = T.arange(n_quantiles, device=device).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (T.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss