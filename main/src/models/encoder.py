#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


import torch
import torch.nn as nn


class GRURNNEncoder(nn.Module):
    """RNN Encoder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config): 
        super(GRURNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
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


class LSTMRNNEncoder(nn.Module):
    """RNN Enocder with Long Short Term Memory"""
    def __init__(self, config):
        super(LSTMRNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.en_hidden_size, 
            num_layers=self.config.en_num_layers, 
            batch_first=False, 
            dropout=0, 
            bidirectional=False)
        self.lstm_dropout = nn.Dropout(self.config.en_drop_rate)

    def init_hidden(self, batch_size):
        # num_layers*num_directions, batch_size, en_hidden_size
        h = torch.zeros(
            self.config.en_num_layers, 
            batch_size, 
            self.config.en_hidden_size, 
            device=self.config.device)
        # num_layers*num_directions, batch_size, en_hidden_size
        c = torch.zeros(
            self.config.en_num_layers, 
            batch_size, 
            self.config.en_hidden_size, 
            device=self.config.device)
        return (h, c)

    def forward(self, x, hidden):
        # x: 1, batch_size
        # hidden: (h, c)
        # h, c: 1, batch_size, hidden_size
        # 1, batch_size, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # 1, batch_size, en_hidden_size
        x, (h, c) = self.lstm(x, hidden)
        x = self.lstm_dropout(x)
        h = self.lstm_dropout(h)
        c = self.lstm_dropout(c)
        # batch_size, en_hidden_size
        x = x.squeeze(0)
        return x, (h, c)

class BiLSTMRNNEncoder(nn.Module):
    """Bidirectional RNN Enocder with Long Short Term Memory"""
    def __init__(self, config):
        super(BiLSTMRNNEncoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.src_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=int(self.config.en_hidden_size/2), 
            num_layers=self.config.en_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=True)
        self.lstm_dropout = nn.Dropout(self.config.en_drop_rate)

    def forward(self, x):
        # x: batch_size, max_xs_seq_len
        # hidden: (h, c)
        # h, c: 2, batch_size, hidden_size
        # batch_size, max_xs_seq_len, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        # x: batch_size, max_xs_seq_len, en_hidden_size
        x, (h, c) = self.lstm(x)
        # h, c: 1, batch_size, en_hidden_size
        h = torch.unsqueeze(torch.cat(torch.unbind(h, 0), 1), 0)
        c = torch.unsqueeze(torch.cat(torch.unbind(c, 0), 1), 0)
        x = self.lstm_dropout(x)
        h = self.lstm_dropout(h)
        c = self.lstm_dropout(c)

        return x, (h, c)