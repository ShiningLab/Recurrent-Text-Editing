#!/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'Shining'
__email__ = 'mrshininnnnn@gmail.com'


# dependency
# public
import torch
import torch.nn as nn
import torch.nn.functional as F
# private
from .attention import *


class GRURNNDecoder(nn.Module):
    """RNN Decoder with Gated Recurrent Unit (GRU)"""
    def __init__(self, config): 
        super(GRURNNDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.tgt_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout = nn.Dropout(self.config.embedding_drop_rate)
        self.gru = nn.GRU(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.de_hidden_size, 
            num_layers=self.config.de_num_layers, 
            batch_first=False, 
            dropout=0, 
            bidirectional=False)
        self.gru_dropout = nn.Dropout(self.config.de_drop_rate)
        self.out = nn.Linear(
            self.config.de_hidden_size, 
            self.config.tgt_vocab_size)
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


class LSTMRNNDecoder(nn.Module):
    """RNN Decoder with Long Short Term Memory (LSTM) Unit"""
    def __init__(self, config):
        super(LSTMRNNDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.tgt_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout=nn.Dropout(self.config.embedding_drop_rate)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.de_hidden_size, 
            num_layers=self.config.de_num_layers, 
            batch_first=False, 
            dropout=0, 
            bidirectional=False)
        self.lstm_dropout = nn.Dropout(self.config.de_drop_rate)
        self.out = nn.Linear(
            in_features=self.config.de_hidden_size, 
            out_features=self.config.tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        # x: 1, batch_size
        # hidden: (h, c)
        # h, c: 1, batch_size, hidden_size
        # 1, batch_size, embedding_dim
        x = self.embedding(x)
        x = self.em_dropout(x)
        x = F.relu(x)
        # 1, batch_size, de_hidden_size
        # h, c: 1, batch_size, de_hidden_size
        x, (h, c) = self.lstm(x, hidden)
        x = self.lstm_dropout(x)
        h = self.lstm_dropout(h)
        c = self.lstm_dropout(c)
        # batch_size, de_hidden_size
        x = x.squeeze(0)
        # batch_size, vocab_size
        x = self.out(x)
        # batch_size, vocab_size
        x = self.softmax(x)
        return x, (h, c)

        
class AttBiLSTMRNNDecoder(nn.Module):
    """Bidirectional RNN Decoder with Long Short Term Memory (LSTM) Unit and Attention"""
    def __init__(self, config):
        super(AttBiLSTMRNNDecoder, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(
            num_embeddings=self.config.tgt_vocab_size, 
            embedding_dim=self.config.embedding_size, 
            padding_idx=self.config.pad_idx)
        self.em_dropout=nn.Dropout(self.config.embedding_drop_rate)
        self.attn = RNNDecoderAttention(self.config)
        self.attn_combine = torch.nn.Linear(
            self.config.embedding_size + self.config.en_hidden_size, 
            self.config.de_hidden_size)
        self.lstm = nn.LSTM(
            input_size=self.config.embedding_size, 
            hidden_size=self.config.de_hidden_size, 
            num_layers=self.config.de_num_layers, 
            batch_first=True, 
            dropout=0, 
            bidirectional=False)
        self.lstm_dropout = nn.Dropout(self.config.de_drop_rate)
        self.out = torch.nn.Linear(self.config.de_hidden_size, self.config.tgt_vocab_size)
        self.softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, hidden, encoder_output, src_lens):
        # x: batch_size
        # hidden: (h, c)
        # h, c: 1, batch_size, de_hidden_size
        # encoder_output: batch_size, max_src_seq_len, en_hidden_size
        # src_lens: batch_size
        # batch_size, 1, embedding_dim
        x = self.embedding(x).unsqueeze(1)
        x = self.em_dropout(x)
        # batch_size, 1, max_src_seq_len
        attn_w = self.attn(hidden, encoder_output, src_lens)
        # batch_size, 1, en_hidden_size
        context = attn_w.bmm(encoder_output)
        # batch_size, 1, de_hidden_size
        x = self.attn_combine(torch.cat((x, context), 2))
        x, (h, c) = self.lstm(x, hidden)
        # batch_size, de_hidden_size
        x = self.lstm_dropout(x).squeeze(1)
        # batch_size, tgt_vocab_size
        x = self.out(x)
        # batch_size, vocab_size
        x = self.softmax(x)
        return x, (h, c)



        
        