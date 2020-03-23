#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import torch
import torch.nn as nn
import torch.nn.functional as F


class GRURNNDecoder(nn.Module):
    """# RNN Encoder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config): 
        super(GRURNNDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.gru = nn.GRU(
            input_size=self.config.en_hidden_size, 
            hidden_size=self.config.de_hidden_size, 
            num_layers=self.config.de_num_layers, 
            batch_first=False, 
            dropout=0, 
            bidirectional=False)
        self.gru_dropout = nn.Dropout(self.config.de_drop_rate)
        self.out = nn.Linear(
            self.config.de_hidden_size, 
            self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h): 
        # x: 1, batch_size
        # h: 1, batch_size, en_hidden_size
        # 1, batch_size, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        x = F.relu(x)
        # 1, batch_size, de_hidden_size
        # num_layers*num_directions, batch_size, de_hidden_size
        x, h = self.gru(x, h)
        x = self.gru_dropout(x)
        h = self.gru_dropout(h)
        # batch_size, de_hidden_size
        x = x.squeeze(0)
        # batch_size, vocab_size
        x = self.out(x)
        # batch_size, vocab_size
        x = self.softmax(x)

        return x, h
