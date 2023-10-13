import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor

def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class VQDiscriminator(pl.LightningModule):
    def __init__(self, input_size, hidden_size, n_layer):
        super(VQDiscriminator, self).__init__()
        sequence = [nn.Conv1d(input_size, hidden_size, 4, 2, 1),
                    nn.BatchNorm1d(hidden_size),
                    nn.LeakyReLU(0.2, inplace=True)
                    ]
        layer_size = hidden_size
        for i in range(n_layer-1):
            sequence += [
                    nn.Conv1d(layer_size, layer_size//2, 4, 2, 1),
                    nn.BatchNorm1d(layer_size//2),
                    nn.LeakyReLU(0.2, inplace=True)
            ]
            layer_size = layer_size // 2

        self.out_net = nn.Conv1d(layer_size, 1, 3, 1, 1)
        self.main = nn.Sequential(*sequence)

        self.out_net.apply(init_weight)
        self.main.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        feats = self.main(inputs)
        outs = self.out_net(feats)
        return feats.permute(0, 2, 1), outs.permute(0, 2, 1)