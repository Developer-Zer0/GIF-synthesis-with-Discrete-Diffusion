import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class TextEncoderBiGRUCo(pl.LightningModule):

    def __init__(self,
                 word_size: int = 300,  # GloVe embedding size
                 pos_size: int = 15,  # One hot encoding size of POS embeddings (len(POS embedding))
                 hidden_size: int = 512,
                 output_size: int = 512, **kwargs) -> None:
        super(TextEncoderBiGRUCo, self).__init__()
        # self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        # self.linear2.apply(init_weight)
        # self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self,
                word_embs: Tensor,  # Word embeddings from GloVe etc.
                pos_onehot: Tensor,  # One-hot encodings for positional embeddings
                cap_lens  # Length of texts for pack_padded_sequence
                ) -> Tensor:

        word_embs = torch.tensor(word_embs, device=self.hidden.device).float()
        pos_onehot = torch.tensor(pos_onehot, device=self.hidden.device).float()
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        # cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True, enforce_sorted=False)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)
