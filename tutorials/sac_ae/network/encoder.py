import torch as T
import torch.nn as nn
from utils.utils import tie_weights

'''
OUT_DIM :
number of layers 2 -> 39
number of layers 4 -> 35
number of layers 6 -> 31
'''

class Encoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, device):
        super(Encoder, self).__init__()
        assert len(obs_shape) == 3   # C， W， H -> 3, 84, 84
        # number of layers 4 -> 35 output Wh
        self.num_layers = 4
        self.encoder_net = nn.Sequential(
                                    nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                    nn.Conv2d(32, 32, 3, stride=1),
                                    nn.Conv2d(32, 32, 3, stride=1),
                                    nn.Conv2d(32, 32, 3, stride=1),
                                    )
        # num_filters * out_dim * out_dim -> feature_dim
        self.fc = nn.Linear(32 * 35 * 35, feature_dim)
        self.ln = nn.LayerNorm(feature_dim)

        self.to(device)

        self.outputs = dict()

    def get_hidden(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        x = T.relu(self.encoder_net[0](obs))
        self.outputs['conv1'] = x

        for i in range(1, self.num_layers):
            x = T.relu(self.encoder_net[i](x))
            self.outputs['conv%s' % (i + 1)] = x

        h = x.view(x.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.get_hidden(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        out = T.tanh(h_norm)
        self.outputs['tanh'] = out

        return out

    def sharing_parameters(self, source):
        # actor encoder and critic encoder sharing parameters.
        for i in range(self.num_layers):
            tie_weights(src=source.encoder_net[i], trg=self.encoder_net[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.encoder_net[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)
