#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import torch
import torch.nn as nn
import random
# private
from .encoder import GRURNNEncoder
from .attention import GRURNNDecoderAttention


class End2EndModelGraph(nn.Module): 
    """docstring for End2EndModelGraph""" 
    def __init__(self, config): 
        super(End2EndModelGraph, self).__init__() 
        self.config = config
        self.encoder = GRURNNEncoder(config)
        self.attn = GRURNNDecoderAttention(config)
        self.embedding = nn.Embedding(
            num_embeddings=2, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.gru = nn.GRUCell(
            input_size=self.config.en_hidden_size, 
            hidden_size=self.config.de_hidden_size)
        self.gru_dropout = nn.Dropout(self.config.de_drop_rate)

    def gen_masks(self, lens, max_len): 
        ranges = torch.arange(
            max_len, 
            dtype=torch.float, 
            device=self.config.device)
        ranges = ranges.expand(lens.shape[0], max_len, max_len)
        lens = lens.view(-1, 1, 1).expand(lens.shape[0], max_len, max_len)
        row_mask = (ranges < lens)
        col_mask = row_mask.transpose(1, 2)
        masks = row_mask * col_mask
        return masks

    def forward(self, xs, x_lens):
        # xs: batch_size, max_len
        batch_size = xs.shape[0]
        # for ptr net, x_len = y_len
        max_len = xs.shape[1]
        # encoder_output: batch_size, max_len, en_hidden_size
        # encoder_hidden: 1, batch_size, en_hidden_size
        encoder_output, encoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size, 
            dtype=torch.int64, 
            device=self.config.device)
        decoder_input.fill_(self.config.start_idx)
        # batch_size, embedding_dim
        decoder_input = self.embedding(decoder_input)
        # batch_size, en_hidden_size
        decoder_hidden = encoder_hidden.squeeze(0)
        # max_len, batch_size, max_len
        decoder_outputs = torch.zeros(
            max_len, 
            batch_size, 
            max_len, 
            device=self.config.device)
        # generate mask
        # batch_size, max_len, max_len
        masks = self.gen_masks(x_lens, max_len)
        for i in range(max_len):
            # batch_size, max_len
            mask = masks[:, i, :]
            # batch_size, de_hidden_size
            decoder_hidden = self.gru(decoder_input, decoder_hidden)
            decoder_hidden = self.gru_dropout(decoder_hidden)
            # batch_size, max_len
            attn_w = self.attn(decoder_hidden, encoder_output, x_lens).squeeze(1)
            # batch_size, max_len
            decoder_outputs[i] = attn_w.squeeze(1)
            # batch_size, 1
            attn_w.masked_fill(~mask, float('-inf'))
            ptr_idxes = attn_w.max(-1, keepdim=True)[1]
            # batch_size, 1, en_hidden_size
            ptr_idxes = ptr_idxes.unsqueeze(-1).expand(-1, 1, self.config.en_hidden_size)
            # batch_size, en_hidden_size
            decoder_input = torch.gather(encoder_output, dim=1, index=ptr_idxes).squeeze(1)
        # batch_size, max_len, max_len
        return decoder_outputs.transpose(0, 1)