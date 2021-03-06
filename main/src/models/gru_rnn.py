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
from .decoder import GRURNNDecoder


class E2EModelGraph(nn.Module): 
    """docstring for E2EModelGraph""" 
    def __init__(self, config): 
        super(E2EModelGraph, self).__init__() 
        self.config = config
        self.encoder = GRURNNEncoder(config)
        self.decoder = GRURNNDecoder(config)

    def forward(self, xs, x_lens, ys, max_ys_seq_len=None, teacher_forcing_ratio=0.5):
        # xs: batch_size, max_xs_seq_len
        # x_lens: batch_size
        # ys: batch_size, max_ys_seq_len
        batch_size = xs.shape[0]
        if max_ys_seq_len == None:
            max_ys_seq_len = ys.shape[1]
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size, 
            dtype=torch.int64, 
            device=self.config.device)
        decoder_input.fill_(self.config.start_idx)
        # max_ys_seq_len, batch_size, vocab_size
        decoder_outputs = torch.zeros(
            max_ys_seq_len, 
            batch_size, 
            self.config.tgt_vocab_size, 
            device=self.config.device)
        # greedy search with teacher forcing
        for i in range(max_ys_seq_len):
            # decoder_output: batch_size, vocab_size
            # decoder_hidden: 1, batch_size, de_hidden_size
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # batch_size, vocab_size
            decoder_outputs[i] = decoder_output
            # batch_size
            decoder_input = ys[:, i] if random.random() < teacher_forcing_ratio \
            else decoder_output.max(1)[1]
        # batch_size, max_ys_seq_len, vocab_size
        return decoder_outputs.transpose(0, 1)


class RecModelGraph(nn.Module): 
    """docstring for RecModelGraph""" 
    def __init__(self, config): 
        super(RecModelGraph, self).__init__() 
        self.config = config
        self.encoder = GRURNNEncoder(config)
        self.decoder = GRURNNDecoder(config)

    def forward(self, xs, x_lens):
        # xs: batch_size, max_xs_seq_len
        # x_lens: batch_size
        batch_size = xs.shape[0]
        max_ys_seq_len = 4 # action, position, token, EOS
        # encoder_output: batch_size, max_xs_seq_len, en_hidden_size
        # decoder_hidden: 1, batch_size, en_hidden_size
        encoder_output, decoder_hidden = self.encoder(xs, x_lens)
        # batch_size
        decoder_input = torch.empty(
            batch_size, 
            dtype=torch.int64, 
            device=self.config.device)
        decoder_input.fill_(self.config.start_idx)
        # max_ys_seq_len, batch_size, vocab_size
        decoder_outputs = torch.zeros(
            max_ys_seq_len, 
            batch_size, 
            self.config.tgt_vocab_size, 
            device=self.config.device)
        # greedy search with teacher forcing
        for i in range(max_ys_seq_len):
            # batch_size, vocab_size
            # num_layers*num_directions, batch_size, de_hidden_size
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            # batch_size, vocab_size
            decoder_outputs[i] = decoder_output
            # batch_size
            decoder_input = decoder_output.max(1)[1]
        # batch_size, max_ys_seq_len, vocab_size
        return decoder_outputs.transpose(0, 1)