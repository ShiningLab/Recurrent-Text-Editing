#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import torch
import torch.nn as nn


class GRURNNEncoder(nn.Module):
    """# RNN Encoder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config): 
        super(GRURNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.gru = nn.GRU(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size, 
            num_layers=self.config.en_num_layers, 
            batch_first=False, 
            dropout=0, 
            bidirectional=False)
        self.gru_dropout = nn.Dropout(self.config.en_drop_rate)

    def init_hidden(self, batch_size): 
        # num_layers*num_directions, batch_size, en_hidden_size
        return torch.zeros(
            self.config.en_num_layers, 
            batch_size, 
            self.config.en_hidden_size, 
            device=self.config.device)

    def forward(self, x, h):
        # x: 1, batch_size
        # h: num_layers*num_directions, batch_size, en_hidden_size
        # 1, batch_size, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # 1, batch_size, en_hidden_size
        # num_layers*num_directions, batch_size, en_hidden_size
        x, h = self.gru(x, h)
        x = self.gru_dropout(x)
        h = self.gru_dropout(h)
        # batch_size, en_hidden_size
        x = x.squeeze(0)
        return x, h
