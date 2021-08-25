import torch as T
import torch.nn as nn

'''
OUT_DIM :
number of layers 2 -> 39
number of layers 4 -> 35
number of layers 6 -> 31
'''

class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, device):
        super(Decoder, self).__init__()
        self.num_layers = 4
        # feature_dim -> num_filters * out_dim * out_dim
        self.fc = nn.Linear(feature_dim, 32 * 35 * 35)
        # num_filters * out_dim * out_dim -> 3, 84, 84
        self.decoder_net = nn.Sequential(
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ConvTranspose2d(32, 32, 3, stride=1),
                        nn.ConvTranspose2d(32, obs_shape[0], 3, stride=2, output_padding=1),
                        )

        self.to(device)
        self.outputs = dict()

    def forward(self, h):
        h = T.relu(self.fc(h))
        self.outputs['fc'] = h

        x = h.view(-1, 32, 35, 35)
        self.outputs['deconv1'] = x

        for i in range(0, self.num_layers - 1):
            x = T.relu(self.decoder_net[i](x))
            self.outputs['deconv%s' % (i + 1)] = x

        obs = self.decoder_net[-1](x)
        self.outputs['obs'] = obs

        return obs

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_decoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_decoder/%s_i' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_decoder/deconv%s' % (i + 1), self.decoder_net[i], step)
        L.log_param('train_decoder/fc', self.fc, step)
